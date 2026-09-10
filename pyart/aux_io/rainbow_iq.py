"""Read Leonardo GDRX6 IQRecord archives into MeteoSwiss Py-ART RadarSpectra.

Sources: GDRX6 IQ Data Format v1.3 (20 April 2026), sections 1--4, and
IQRecording Archiveformat v4.0 (5 June 2024), section 2. Requires NumPy;
the RadarSpectra entry points additionally require MeteoSwiss Py-ART.

RX types 14/15/16 are supported, inside archive item 0x0122 or as raw
concatenated GDRX6 frames. TX/legacy receiver formats are not decoded.
Samples remain time-domain I + 1j*Q, with no calibration or extra TX
correction. See README_rainbow_iq_reader.md for assumptions and examples.
"""

from __future__ import annotations

import copy
import inspect
import math
import operator
import os
import struct
import tempfile
import warnings
import zlib
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

__all__ = [
    "rainbowFormatError",
    "rainbowPulse",
    "read_rainbow_header",
    "iter_rainbow_iq",
    "read_rainbow_iq",
    "iter_rainbow_spectra",
    "read_rainbow_iq_streams",
    "iter_rainbow_spectra_streams",
    "combine_rainbow_iq_streams",
    "read_rainbow_iq_scan",
]

START = 0xDD0DDD0D
STOP = 0xCC0CCC0C
RX_TYPES = {14: ("h",), 15: ("v",), 16: ("h", "v")}
FIELD_NAMES = {
    "IQ_H": "IQ_hh_ADU",
    "IQ_V": "IQ_vv_ADU",
    "RX_CHANNEL_H": "IQ_hh_high_sensitivity",
    "RX_CHANNEL_V": "IQ_vv_high_sensitivity",
}
_GPS_VALID = 1 << 5
_BAD_ANGLES = (1 << 3) | (1 << 4)
_CPI_START = 1 << 7
_RESET = (1 << 0) | (1 << 10)
_IO_CHUNK = 1024 * 1024


class rainbowFormatError(ValueError):
    """An archive or pulse frame is malformed or has an unsupported layout."""


@dataclass
class rainbowPulse:
    """One RX pulse: header values, complex samples and sensitivity bits.

    ``iq`` and ``high_sensitivity`` map ``h``/``v`` to arrays of gate_count
    samples. High sensitivity means channel selection, NOT sample validity.
    Metadata keeps source units: range km, pulse width us, ENBW MHz, frequency
    Hz, phase radians, magnitude ADU, matched-filter losses linear.
    """

    metadata: dict
    iq: dict
    high_sensitivity: dict


def _integer(value, name, minimum=0):
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _endian(value):
    try:
        return {"auto": None, "little": "<", "big": ">", "<": "<", ">": ">"}[value]
    except (KeyError, TypeError) as exc:
        raise ValueError("byte order must be 'auto', 'little', or 'big'") from exc


def _exact(stream, size, description):
    value = stream.read(size)
    if len(value) != size:
        raise rainbowFormatError(
            f"Truncated {description}: wanted {size}, got {len(value)} bytes"
        )
    return value


def _archive_header(stream):
    stream.seek(0)
    if stream.read(22) != b"[ARCHIVE_HEADER_START]":
        # The marker is 22 bytes including its closing bracket.
        stream.seek(0)
        return None
    stream.seek(0)
    header = {}
    limit = 16 * 1024 * 1024
    while stream.tell() < limit:
        line = stream.readline(min(_IO_CHUNK, limit - stream.tell()))
        if not line:
            break
        try:
            text = line.decode("utf-8").strip()
        except UnicodeDecodeError as exc:
            raise rainbowFormatError("Non-UTF-8 archive header") from exc
        if text == "[ARCHIVE_HEADER_END]":
            try:
                offset, length = int(header["body.offset"]), int(header["body.length"])
            except (KeyError, ValueError) as exc:
                raise rainbowFormatError(
                    "Missing/invalid body.offset or body.length"
                ) from exc
            if offset < stream.tell() or length < 0:
                raise rainbowFormatError(
                    "Archive body overlaps its header or has negative length"
                )
            stream.seek(0, os.SEEK_END)
            if offset + length > stream.tell():
                raise rainbowFormatError(
                    "Archive body extends beyond file size (incomplete recording)"
                )
            stream.seek(offset)
            return header
        if "=" in text:
            key, value = text.split("=", 1)
            header[key] = value
        elif text != "[ARCHIVE_HEADER_START]" and text:
            raise rainbowFormatError("Unexpected line in archive header")
    raise rainbowFormatError(
        "Missing ARCHIVE_HEADER_END or archive header exceeds 16 MiB"
    )


def read_rainbow_header(filename):
    """Read all UTF-8 archive properties as strings, without reading IQ data."""
    with open(filename, "rb") as stream:
        header = _archive_header(stream)
    if header is None:
        raise rainbowFormatError("Not a rainbow IQRecord archive")
    return header


def _archive_byteorder(stream, end, requested):
    chosen = _endian(requested)
    if chosen is not None:
        return chosen
    position = stream.tell()
    raw = _exact(stream, 12, "archive item header")
    stream.seek(position)
    candidates = []
    for order in (">", "<"):
        kind, size = struct.unpack(order + "Iq", raw)
        if 0 <= size <= end - position - 12 and 0 < kind <= 0xFFFF:
            candidates.append(order)
    if len(candidates) != 1:
        raise rainbowFormatError(
            "Cannot determine archive byte order; set archive_byteorder explicitly"
        )
    return candidates[0]


@contextmanager
def _inflate(stream, length, order, max_uncompressed_bytes):
    if length < 8:
        raise rainbowFormatError("Compressed item lacks uncompressed-size field")
    expected = struct.unpack(order + "q", _exact(stream, 8, "uncompressed size"))[0]
    if not 0 <= expected <= max_uncompressed_bytes:
        raise rainbowFormatError(
            "Compressed item exceeds max_uncompressed_bytes or has negative size"
        )
    # Large containers spill to disk rather than occupying all available RAM.
    with tempfile.SpooledTemporaryFile(max_size=8 * _IO_CHUNK) as expanded:
        decoder = zlib.decompressobj()
        remaining, total = length - 8, 0
        try:
            while remaining:
                data = _exact(stream, min(remaining, _IO_CHUNK), "compressed payload")
                remaining -= len(data)
                while data:
                    piece = decoder.decompress(
                        data, min(_IO_CHUNK, expected - total + 1)
                    )
                    total += len(piece)
                    if total > expected:
                        raise rainbowFormatError(
                            "ZLIB output exceeds its declared length"
                        )
                    expanded.write(piece)
                    if decoder.unused_data:
                        raise rainbowFormatError("Trailing bytes after ZLIB stream")
                    data = decoder.unconsumed_tail
            if not decoder.eof or total != expected:
                raise rainbowFormatError(
                    "Incomplete ZLIB stream or wrong uncompressed length"
                )
        except zlib.error as exc:
            raise rainbowFormatError(f"Invalid ZLIB stream: {exc}") from exc
        expanded.seek(0)
        yield expanded, expected


def _items(stream, end, order, max_frame_bytes, max_uncompressed_bytes, depth=0):
    if depth > 16:
        raise rainbowFormatError("Archive nesting exceeds 16 levels")
    while stream.tell() < end:
        offset = stream.tell()
        if end - offset < 12:
            raise rainbowFormatError(
                f"Incomplete archive item header at offset {offset}"
            )
        kind, length = struct.unpack(order + "Iq", _exact(stream, 12, "item header"))
        item_end = stream.tell() + length
        if length < 0 or item_end > end:
            raise rainbowFormatError(
                f"Invalid item length at offset {offset}: {length}"
            )
        if kind == 0x0122:
            if length > max_frame_bytes:
                raise rainbowFormatError("IQ frame exceeds max_frame_bytes")
            yield _exact(stream, length, "GDRX6 RX item")
        elif kind == 0x0F00:
            yield from _items(
                stream,
                item_end,
                order,
                max_frame_bytes,
                max_uncompressed_bytes,
                depth + 1,
            )
        elif kind == 0xF000:
            with _inflate(stream, length, order, max_uncompressed_bytes) as (
                expanded,
                size,
            ):
                yield from _items(
                    expanded,
                    size,
                    order,
                    max_frame_bytes,
                    max_uncompressed_bytes,
                    depth + 1,
                )
        else:
            # Includes timestamp/RCL/TX/legacy formats and undocumented types.
            stream.seek(item_end)
        if stream.tell() != item_end:
            raise rainbowFormatError(f"Item boundary mismatch at offset {offset}")


def _frame_order(raw, requested):
    order = _endian(requested)
    matches = [b for b in ("<", ">") if raw == struct.pack(b + "I", START)]
    if not matches or (order is not None and order != matches[0]):
        raise rainbowFormatError("Missing GDRX6 start pattern or wrong frame_byteorder")
    return matches[0]


def _frames(
    stream,
    header,
    archive_byteorder,
    frame_byteorder,
    max_frame_bytes,
    max_uncompressed_bytes,
):
    if header is not None:
        end = int(header["body.offset"]) + int(header["body.length"])
        if stream.tell() == end:
            return
        order = _archive_byteorder(stream, end, archive_byteorder)
        yield from _items(stream, end, order, max_frame_bytes, max_uncompressed_bytes)
        return
    while True:
        start = stream.read(4)
        if not start:
            return
        order = _frame_order(start, frame_byteorder)
        base = _exact(stream, 20, "GDRX6 base header")
        _, _, _, nheader, ndata = struct.unpack(order + "5I", base)
        size = 28 + 4 * (nheader + ndata)
        if size > max_frame_bytes:
            raise rainbowFormatError("Raw frame exceeds max_frame_bytes")
        yield start + base + _exact(stream, size - 24, "GDRX6 type header/data/stop")


def _decode_frame(raw, frame_byteorder="auto", angle_scale=360.0 / 65536):
    if len(raw) < 28:
        raise rainbowFormatError("GDRX6 frame shorter than its base header")
    order = _frame_order(raw[:4], frame_byteorder)
    kind, revision, extension, nheader, ndata = struct.unpack_from(order + "5I", raw, 4)
    expected = 28 + 4 * (nheader + ndata)
    if len(raw) != expected:
        raise rainbowFormatError(
            f"GDRX6 frame length {len(raw)} != declared {expected}"
        )
    if struct.unpack_from(order + "I", raw, len(raw) - 4)[0] != STOP:
        raise rainbowFormatError("Missing GDRX6 stop pattern")
    if kind not in RX_TYPES:
        if kind in (6, 7, 8):
            return None
        raise rainbowFormatError(f"Unsupported GDRX6 frame type {kind}")
    pols = RX_TYPES[kind]
    dual = len(pols) == 2
    required = 31 if dual else 24
    if revision != 6 or nheader < required:
        raise rainbowFormatError(
            f"RX type {kind}: expected revision 6 and >= {required} header words; "
            f"got revision {revision}, {nheader} words"
        )
    # Extension fields are permitted by the ICD and skipped using header length.
    u = np.frombuffer(raw, dtype=order + "u4", count=nheader, offset=24)
    f = u.view(order + "f4")
    m = {
        "data_type": kind,
        "header_revision": revision,
        "header_extension": extension,
        "header_words": nheader,
        "tracking_number": int(u[0]),
        "pulse_counter": int(u[1]),
        "antenna_angle_tags": int(u[2]),
        "range_start_km": float(f[3]),
        "range_step_km": float(f[4]),
        "range_stop_km": float(f[5]),
        "gate_count": int(u[6]),
        "prf_hz": float(f[7]),
    }
    shift = 5 if dual else 0
    for key, index in [
        ("rx_flags", 13),
        ("staggering_flags", 14),
        ("timestamp_sec", 15),
        ("timestamp_nsec", 16),
        ("timestamp_status", 17),
        ("matched_filter_mode", 19),
    ]:
        m[key] = int(u[index + shift])
    for j, pol in enumerate(pols):
        for key, single, double in [
            ("tx_average_frequency_hz", 8, 8),
            ("tx_pulse_frequency_hz", 9, 10),
            ("tx_magnitude_adu", 10, 12),
            ("tx_phase_rad", 11, 14),
            ("dds_frequency_hz", 12, 16),
            ("matched_filter_loss_linear", 20, 25),
            ("equivalent_noise_bandwidth_mhz", 21, 27),
        ]:
            m[key + "_" + pol] = float(f[double + j if dual else single])
    m["channel_mask"] = int(u[29 if dual else 22])
    m["pulse_width_us"] = float(f[30 if dual else 23])
    m["azimuth"] = (m["antenna_angle_tags"] & 0xFFFF) * angle_scale % 360.0
    m["elevation"] = (((m["antenna_angle_tags"] >> 16) * angle_scale + 180) % 360) - 180
    if not (
        m["gate_count"] > 0
        and math.isfinite(m["range_start_km"])
        and math.isfinite(m["range_step_km"])
        and m["range_step_km"] > 0
        and math.isfinite(m["prf_hz"])
        and m["prf_hz"] > 0
    ):
        raise rainbowFormatError("Invalid gate count, range geometry or PRF")
    if m["rx_flags"] & _GPS_VALID and m["timestamp_nsec"] >= 1_000_000_000:
        raise rainbowFormatError("GPS-valid timestamp_nsec is not below 1e9")

    nsamples = m["gate_count"] * len(pols)
    nfull, remainder = divmod(nsamples, 32)
    packed_words = nfull * 67 + (3 + 2 * remainder if remainder else 0)
    padded_words = ((nsamples + 31) // 32) * 67
    if ndata not in (packed_words, padded_words):
        raise rainbowFormatError(
            f"RX data length {ndata} words incompatible with {m['gate_count']} gates "
            f"and {len(pols)} polarizations; expected {packed_words} or {padded_words}"
        )
    words = np.frombuffer(raw, dtype=order + "u4", count=ndata, offset=24 + nheader * 4)
    samples = np.empty(nsamples, np.complex64)
    high = np.empty(nsamples, np.uint8)
    # Full 67-word blocks: mask, reserved, reserved, 32 interleaved I/Q pairs.
    full = words[: nfull * 67].reshape(nfull, 67)
    if nfull:
        pairs = full[:, 3:].copy().view(order + "f4").reshape(-1, 2)
        samples[: nfull * 32].real = pairs[:, 0]
        samples[: nfull * 32].imag = pairs[:, 1]
        high[: nfull * 32] = (
            (full[:, 0, None] >> np.arange(32, dtype=np.uint32)) & 1
        ).ravel()
    if remainder:
        last = words[nfull * 67 :]
        pairs = last[3 : 3 + remainder * 2].view(order + "f4").reshape(-1, 2)
        samples[nfull * 32 :].real = pairs[:, 0]
        samples[nfull * 32 :].imag = pairs[:, 1]
        high[nfull * 32 :] = (last[0] >> np.arange(remainder, dtype=np.uint32)) & 1
    samples, high = samples.reshape(-1, len(pols)), high.reshape(-1, len(pols))
    return rainbowPulse(
        m,
        {p: samples[:, j] for j, p in enumerate(pols)},
        {p: high[:, j] for j, p in enumerate(pols)},
    )


def iter_rainbow_iq(
    filename,
    *,
    archive_byteorder="auto",
    frame_byteorder="auto",
    angle_scale=360.0 / 65536,
    max_frame_bytes=64 * 1024**2,
    max_uncompressed_bytes=4 * 1024**3,
):
    """Yield decoded RX pulses using bounded memory; Py-ART is not required.

    Both an IQRecord archive and a raw GDRX6 frame stream are auto-detected.
    Nested containers and ZLIB streams are supported. Unknown archive items
    are skipped. Byte orders of archive envelopes and embedded frames are
    detected independently. An explicit order overrides detection.
    """
    _endian(archive_byteorder)
    _endian(frame_byteorder)
    if not math.isfinite(angle_scale) or angle_scale <= 0:
        raise ValueError("angle_scale must be finite and positive")
    _integer(max_frame_bytes, "max_frame_bytes", 28)
    _integer(max_uncompressed_bytes, "max_uncompressed_bytes", 1)
    with open(filename, "rb") as stream:
        header = _archive_header(stream)
        for raw in _frames(
            stream,
            header,
            archive_byteorder,
            frame_byteorder,
            max_frame_bytes,
            max_uncompressed_bytes,
        ):
            pulse = _decode_frame(raw, frame_byteorder, angle_scale)
            if pulse is not None:
                yield pulse


def _start_ns(value):
    if value is None:
        return None
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError("start_time must have an explicit timezone")
        value = value.timestamp()
    if not math.isfinite(float(value)):
        raise ValueError("start_time must be finite")
    return round(float(value) * 1_000_000_000)


class _Clock:
    def __init__(self, header, start_time, mode, counter_policy="mask"):
        if mode not in ("auto", "gps", "reconstruct"):
            raise ValueError("time_mode must be 'auto', 'gps', or 'reconstruct'")
        if counter_policy not in ("mask", "raise"):
            raise ValueError("counter_policy must be 'mask' or 'raise'")
        self.mode = mode
        self.counter_policy = counter_policy
        self.anchor = _start_ns(start_time)
        if self.anchor is None and header and "data.time.ms.from" in header:
            self.anchor = int(header["data.time.ms.from"]) * 1_000_000
        self.previous = None
        self.warned = False
        self.warned_events = set()
        self.elapsed_ns = 0.0
        self.absolute_known = self.anchor is not None
        self.origin = self.anchor
        self.segment = 0

    def apply(self, pulse):
        m = pulse.metadata
        prev = self.previous
        delta = (
            ((m["pulse_counter"] - prev["pulse_counter"]) & 0xFFFFFFFF) if prev else 1
        )
        repeated = prev is not None and delta == 0
        discontinuity = prev is not None and delta > 1_000_000
        processing_reset = bool(m["rx_flags"] & (1 << 10))
        if discontinuity or processing_reset:
            self.segment += 1
        gps = bool(m["rx_flags"] & _GPS_VALID) and m["timestamp_sec"] > 0
        if self.mode != "reconstruct" and gps:
            timestamp = m["timestamp_sec"] * 1_000_000_000 + m["timestamp_nsec"]
            source = 0
            self.anchor, self.elapsed_ns = timestamp, 0.0
            self.absolute_known = True
            if self.origin is None:
                self.origin = timestamp
        elif self.mode == "gps":
            raise rainbowFormatError(
                "Pulse lacks valid GPS time; use auto/reconstruct with start_time"
            )
        else:
            if self.anchor is None:
                raise rainbowFormatError(
                    "No valid GPS time or archive time anchor; supply start_time"
                )
            detail = (
                f"RX frame {m.get('source_pulse_index', '?')}: "
                f"counter {prev['pulse_counter'] if prev else 'none'} -> {m['pulse_counter']}, "
                f"delta={delta}, tracking={m['tracking_number']}, "
                f"rx_flags=0x{m['rx_flags']:08x}"
            )
            if discontinuity:
                if self.counter_policy == "raise":
                    raise rainbowFormatError(
                        "Cannot reconstruct timing across counter discontinuity; "
                        + detail
                    )
                self.absolute_known = False
                if "discontinuity" not in self.warned_events:
                    warnings.warn(
                        "Counter discontinuity without valid GPS time; "
                        "absolute times are masked until GPS time becomes valid. "
                        "IQ is retained and a new ray starts. " + detail,
                        stacklevel=3,
                    )
                    self.warned_events.add("discontinuity")
            elif repeated:
                if self.counter_policy == "raise":
                    raise rainbowFormatError(
                        "Repeated pulse counter without valid GPS time; " + detail
                    )
                if "repeated" not in self.warned_events:
                    warnings.warn(
                        "Repeated pulse counter without valid GPS time; treating the "
                        "frame as another representation of the same transmitted pulse "
                        "and reusing its time. Frames are retained, not merged. "
                        + detail,
                        stacklevel=3,
                    )
                    self.warned_events.add("repeated")
            elif prev is not None and self.absolute_known:
                self.elapsed_ns += delta * 1_000_000_000 / prev["prf_hz"]
            # A processing-chain reset flag does not itself reset the pulse
            # counter/clock. With a continuous counter, keep reconstructing time.
            timestamp = (
                self.anchor + round(self.elapsed_ns) if self.absolute_known else None
            )
            source = (3 if repeated else 1) if self.absolute_known else 2
            if not self.warned:
                warnings.warn(
                    "Pulse timing reconstructed from an anchor, pulse counters and preceding PRF; "
                    "absolute pulse time and variable-PRT timing are approximate.",
                    stacklevel=3,
                )
                self.warned = True
        if (
            prev is not None
            and timestamp is not None
            and prev["time_ns"] is not None
            and timestamp < prev["time_ns"]
        ):
            raise rainbowFormatError(
                "Pulse times run backwards; check clock/epoch and time_mode"
            )
        m["time_ns"], m["time_source"] = timestamp, source
        m["time_origin_ns"] = self.origin
        m["clock_segment"] = self.segment
        m["timing_event"] = (
            2 if discontinuity else 1 if repeated else 3 if processing_reset else 0
        )
        m["counter_gap"] = int(prev is not None and delta != 1)
        self.previous = m


def _new_ray(previous, current):
    a, b = previous.metadata, current.metadata
    sa, sb = a["staggering_flags"], b["staggering_flags"]
    mode = sb & 15
    # Compare SELECTED counters: intervening sibling frames can legitimately
    # repeat the same transmitted-pulse counter and be removed by pulse_filter.
    delta = (b["pulse_counter"] - a["pulse_counter"]) & 0xFFFFFFFF
    if delta != 1 or b["clock_segment"] != a["clock_segment"]:
        return True
    if (a["time_ns"] is None) != (b["time_ns"] is None):
        return True
    if b["rx_flags"] & (_CPI_START | _RESET):
        return True
    if (sa ^ sb) & (1 << 5) or (sa & 15) != mode:
        return True
    if a["data_type"] != b["data_type"] or not np.isclose(
        a["pulse_width_us"], b["pulse_width_us"], rtol=1e-6, equal_nan=True
    ):
        return True
    if a["matched_filter_mode"] != b["matched_filter_mode"]:
        return True
    if (a["rx_flags"] ^ b["rx_flags"]) & ((1 << 1) | (1 << 2)):
        return True
    if mode in (3, 4, 5) and ((sa >> 19) & 4095) != ((sb >> 19) & 4095):
        return True
    if mode not in (1, 2) and not np.isclose(a["prf_hz"], b["prf_hz"], rtol=1e-6):
        return True
    return False


def _rays(
    filename,
    header,
    pulses_per_ray,
    start_pulse,
    stop_pulse,
    pulse_filter,
    rng_min,
    rng_max,
    range_reference,
    start_time,
    time_mode,
    counter_policy,
    max_pulses_per_ray,
    max_memory_bytes,
    reader_options,
):
    clock = _Clock(header, start_time, time_mode, counter_policy)
    ray, geometry, gates, ranges = [], None, None, None
    pulses = iter_rainbow_iq(filename, **reader_options)
    try:
        for index, pulse in enumerate(pulses):
            if stop_pulse is not None and index >= stop_pulse:
                break
            m = pulse.metadata
            m["source_pulse_index"] = index
            clock.apply(pulse)
            if index < start_pulse or (
                pulse_filter is not None and not pulse_filter(m)
            ):
                continue
            this_geometry = (m["gate_count"], m["range_start_km"], m["range_step_km"])
            if geometry is None:
                geometry = this_geometry
                offset = 0.5 if range_reference == "start" else 0.0
                all_ranges = (
                    m["range_start_km"]
                    + (np.arange(m["gate_count"]) + offset) * m["range_step_km"]
                ) * 1000
                gates = np.flatnonzero(
                    (all_ranges >= rng_min) & (all_ranges <= rng_max)
                )
                if not gates.size:
                    raise ValueError("No range gates within rng_min/rng_max")
                ranges = all_ranges[gates]
            elif geometry[0] != this_geometry[0] or not np.allclose(
                geometry[1:], this_geometry[1:], rtol=1e-7, atol=1e-8
            ):
                raise rainbowFormatError(
                    "Selected pulses have changing range geometry; use "
                    "read_rainbow_iq_streams to separate SSPA short/long "
                    "pulse streams, or pulse_filter to select one stream"
                )
            if ray and (
                _new_ray(ray[-1], pulse)
                or (pulses_per_ray is not None and len(ray) >= pulses_per_ray)
            ):
                yield ray, ranges
                ray = []
            if len(ray) >= max_pulses_per_ray:
                raise rainbowFormatError(
                    "No ray boundary before max_pulses_per_ray; set pulses_per_ray "
                    "or raise the limit for a known long CPI"
                )
            # Copy only retained gates so views do not keep full pulse buffers alive.
            pulse.iq = {p: a[gates].copy() for p, a in pulse.iq.items()}
            pulse.high_sensitivity = {
                p: a[gates].copy() for p, a in pulse.high_sensitivity.items()
            }
            ray.append(pulse)
            if (len(ray) * (gates.size * len(pulse.iq) * 24 + 4096)) > max_memory_bytes:
                raise MemoryError(
                    "One ray exceeds max_memory_bytes; reduce ranges or pulses_per_ray"
                )
        if ray:
            yield ray, ranges
    finally:
        pulses.close()


def _circular_mean(values):
    angle = np.angle(np.mean(np.exp(1j * np.deg2rad(values))), deg=True) % 360
    return 0.0 if np.isclose(angle, 360) else float(angle)


def _pyart():
    try:
        from pyart.config import FileMetadata
        from pyart.core.radar_spectra import RadarSpectra
    except ImportError as exc:
        raise ImportError(
            "This reader requires MeteoSwiss Py-ART with "
            "pyart.core.radar_spectra.RadarSpectra; the low-level "
            "iter_rainbow_iq only needs NumPy"
        ) from exc
    return FileMetadata, RadarSpectra


def _build_spectra(rays, ranges, header, filename, options):
    FileMetadata, RadarSpectra = _pyart()
    names = FIELD_NAMES if options["field_names"] is None else options["field_names"]
    fm = FileMetadata(
        "rainbow_IQ",
        names,
        options["additional_metadata"],
        options["file_field_names"],
        options["exclude_fields"],
        options["include_fields"],
    )

    def variable(name, data, units=None, **extra):
        result = fm(name)
        if units is not None:
            result.setdefault("units", units)
        result.update(extra)
        result["data"] = data
        return result

    nrays, ngates = len(rays), len(ranges)
    npulses = np.array([len(ray) for ray in rays], dtype=np.int32)
    npmax = int(npulses.max())
    shape = (nrays, ngates, npmax)
    present = set().union(*(pulse.iq for ray in rays for pulse in ray))
    fields = {}
    for pol in ("h", "v"):
        if pol not in present:
            continue
        for source, channel in [
            ("IQ_" + pol.upper(), False),
            ("RX_CHANNEL_" + pol.upper(), True),
        ]:
            if channel and not options["include_channel_masks"]:
                continue
            name = fm.get_field_name(source)
            if name is None:
                continue
            if name in fields:
                raise ValueError(f"Field mapping produces duplicate name {name!r}")
            data = np.ma.masked_all(shape, dtype=np.uint8 if channel else np.complex64)
            for i, ray in enumerate(rays):
                for j, pulse in enumerate(ray):
                    if pol in pulse.iq:
                        values = (
                            pulse.high_sensitivity[pol] if channel else pulse.iq[pol]
                        )
                        data[i, :, j] = np.ma.masked_invalid(values)
                        if pulse.metadata["rx_flags"] & options["mask_rx_flags"]:
                            data[i, :, j] = np.ma.masked
            fields[name] = variable(name, data, "1" if channel else "ADU")
            fields[name].setdefault(
                "long_name",
                "RX high-sensitivity channel selection"
                if channel
                else f"{pol.upper()} received time-domain IQ samples",
            )
            if channel:
                fields[name].update(
                    flag_values=np.array([0, 1], np.uint8),
                    flag_meanings="low_sensitivity high_sensitivity",
                )
    if not fields:
        raise ValueError("No output fields selected")

    # Reader-specific per-pulse metadata live outside CF/Radial instrument_parameters.
    keys = set().union(*(p.metadata for ray in rays for p in ray))
    pulse_metadata = {}
    for key in sorted(keys):
        example = next(
            (
                p.metadata[key]
                for ray in rays
                for p in ray
                if p.metadata.get(key) is not None
            ),
            None,
        )
        dtype = np.int64 if isinstance(example, int) or key == "time_ns" else np.float64
        array = np.ma.masked_all((nrays, npmax), dtype=dtype)
        for i, ray in enumerate(rays):
            for j, pulse in enumerate(ray):
                if pulse.metadata.get(key) is not None:
                    array[i, j] = pulse.metadata[key]
        pulse_metadata[key] = {"data": array}

    first_ns = next(
        (
            p.metadata["time_ns"]
            for ray in rays
            for p in ray
            if p.metadata["time_ns"] is not None
        ),
        rays[0][0].metadata["time_origin_ns"],
    )
    epoch_seconds = first_ns // 1_000_000_000
    epoch_ns = epoch_seconds * 1_000_000_000
    epoch = datetime.fromtimestamp(epoch_seconds, timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    time_units = "seconds since " + epoch
    times = np.ma.masked_all(nrays, dtype=np.float64)
    for i, ray in enumerate(rays):
        if all(p.metadata["time_ns"] is not None for p in ray):
            times[i] = np.mean([(p.metadata["time_ns"] - epoch_ns) / 1e9 for p in ray])
    pulse_metadata["time"] = {
        "data": (pulse_metadata["time_ns"]["data"] - epoch_ns) / 1e9,
        "units": time_units,
        "calendar": "gregorian",
    }
    pulse_metadata["time_source"].update(
        flag_values=np.array([0, 1, 2, 3]),
        flag_meanings="gps reconstructed unavailable repeated_counter",
    )
    pulse_metadata["timing_event"].update(
        flag_values=np.array([0, 1, 2, 3]),
        flag_meanings="none repeated_counter counter_discontinuity processing_reset",
    )
    azimuth, elevation = np.ma.masked_all(nrays), np.ma.masked_all(nrays)
    for i, ray in enumerate(rays):
        good = [p.metadata for p in ray if not p.metadata["rx_flags"] & _BAD_ANGLES]
        if good:
            azimuth[i] = _circular_mean([m["azimuth"] for m in good])
            elevation[i] = np.mean([m["elevation"] for m in good])

    prt, width, nyquist = (np.ma.masked_all(nrays) for _ in range(3))
    frequencies, velocities = np.ma.masked_all((nrays, npmax)), np.ma.masked_all(
        (nrays, npmax)
    )
    uniform = np.zeros(nrays, dtype=np.uint8)
    # RF frequency is deliberately explicit: receiver TX measurements may be IF.
    rf = options["frequency"]
    modes = []
    for i, ray in enumerate(rays):
        m = [p.metadata for p in ray]
        n = len(ray)
        prfs = np.array([p["prf_hz"] for p in m])
        mode = m[0]["staggering_flags"] & 15
        regular = mode in (0, 3, 4, 5) and np.allclose(prfs, prfs[0], rtol=1e-6)
        if n > 1 and all(p["time_source"] == 0 for p in m):
            dt = np.diff(np.array([p["time_ns"] for p in m], dtype=np.int64)) / 1e9
            regular = regular and np.allclose(dt, 1.0 / prfs[0], rtol=1e-3, atol=1e-8)
        if regular:
            prt[i] = 1.0 / prfs[0]
            frequencies[i, :n] = np.fft.fftshift(np.fft.fftfreq(n, d=prt[i]))
            uniform[i] = 1
            if rf is not None:
                velocities[i, :n] = frequencies[i, :n] * 299792458.0 / (2 * rf)
                nyquist[i] = 299792458.0 * prfs[0] / (4 * rf)
        widths = [p["pulse_width_us"] for p in m]
        if np.all(np.isfinite(widths)) and np.allclose(widths, widths[0]):
            width[i] = widths[0] * 1e-6
        modes.append("fixed" if regular else "staggered")
    instrument = {
        "prt": variable("prt", prt, "seconds"),
        "pulse_width": variable("pulse_width", width, "seconds"),
    }
    if rf is not None:
        instrument["frequency"] = variable("frequency", np.array([rf]), "Hz")
        instrument["nyquist_velocity"] = variable("nyquist_velocity", nyquist, "m/s")
    instrument["prt_mode"] = variable(
        "prt_mode", np.array([modes[0] if len(set(modes)) == 1 else "staggered"]), "1"
    )
    range_dict = variable(
        "range",
        ranges,
        "meters",
        meters_to_center_of_first_gate=float(ranges[0]),
        meters_between_gates=rays[0][0].metadata["range_step_km"] * 1000,
        spacing_is_constant="true",
    )
    scan_type = options["scan_type"]
    sweep_mode = {
        "ppi": "azimuth_surveillance",
        "sector": "sector",
        "rhi": "rhi",
        "other": "other",
        "vpt": "vertical_pointing",
    }[scan_type]
    fixed = options["fixed_angle"]
    if fixed is None:
        angles = azimuth if scan_type == "rhi" else elevation
        fixed = (
            _circular_mean(angles.compressed())
            if scan_type == "rhi" and angles.count()
            else float(np.ma.median(angles))
            if angles.count()
            else np.nan
        )
    metadata = fm("metadata")
    metadata.update(
        original_container="rainbow GDRX6 IQRecord",
        source=os.fspath(filename),
        iq_data_domain="time",
        iq_convention="I + 1j*Q",
        rainbow_range_reference=options["range_reference"],
        rainbow_angle_scale_deg_per_adu=options["angle_scale"],
        rainbow_reader="GDRX6 ICD v1.3; archive v4.0",
        comment="One user-selected acquisition segment per sweep; "
        "no automatic interpretation of proprietary states.* scan settings.",
    )
    site = []
    for key, units in [
        ("latitude", "degrees_north"),
        ("longitude", "degrees_east"),
        ("altitude", "meters"),
    ]:
        value = options[key]
        site.append(
            variable(
                key, np.ma.masked_all(1) if value is None else np.array([value]), units
            )
        )
    time_dict = variable("time", times)
    # FileMetadata's default is only "seconds"; the epoch is data-dependent
    # and must override that default (and any stale supplied time metadata).
    time_dict.update(units=time_units, calendar="gregorian")
    # ``Doppler_frequency`` and ``Doppler_velocity`` are attributes of every
    # RadarSpectra object, but older Py-ART releases did not accept them as
    # constructor keywords.  Pass them at construction only when that version
    # explicitly supports the corresponding argument, then attach them below.
    doppler_frequency = variable("Doppler_frequency", frequencies, "Hz")
    doppler_velocity = (
        None if rf is None else variable("Doppler_velocity", velocities, "m/s")
    )
    constructor_parameters = inspect.signature(RadarSpectra.__init__).parameters
    optional_constructor_arguments = {"instrument_parameters": instrument}
    if "Doppler_frequency" in constructor_parameters:
        optional_constructor_arguments["Doppler_frequency"] = doppler_frequency
    if "Doppler_velocity" in constructor_parameters:
        optional_constructor_arguments["Doppler_velocity"] = doppler_velocity
    radar = RadarSpectra(
        time_dict,
        range_dict,
        fields,
        metadata,
        scan_type,
        *site,
        variable("sweep_number", np.array([0], np.int32)),
        variable("sweep_mode", np.array([sweep_mode])),
        variable("fixed_angle", np.ma.masked_invalid(np.array([fixed])), "degrees"),
        variable("sweep_start_ray_index", np.array([0], np.int32)),
        variable("sweep_end_ray_index", np.array([nrays - 1], np.int32)),
        variable("azimuth", azimuth, "degrees"),
        variable("elevation", elevation, "degrees"),
        variable("number_of_pulses", npulses, "1"),
        **optional_constructor_arguments,
    )
    if "Doppler_frequency" not in constructor_parameters:
        radar.Doppler_frequency = doppler_frequency
    if "Doppler_velocity" not in constructor_parameters:
        radar.Doppler_velocity = doppler_velocity
    radar.rainbow_archive_header = dict(header or {})
    radar.rainbow_pulse_metadata = pulse_metadata
    radar.rainbow_uniform_prt = {
        "data": uniform,
        "comment": "1: a uniform FFT frequency grid can be assigned",
    }
    return radar


def read_rainbow_iq(
    filename,
    field_names=None,
    additional_metadata=None,
    file_field_names=False,
    exclude_fields=None,
    include_fields=None,
    *,
    pulses_per_ray=None,
    max_rays=None,
    start_pulse=0,
    stop_pulse=None,
    pulse_filter=None,
    rng_min=None,
    rng_max=None,
    latitude=None,
    longitude=None,
    altitude=None,
    frequency=None,
    scan_type="other",
    fixed_angle=None,
    range_reference="start",
    time_mode="auto",
    start_time=None,
    counter_policy="mask",
    archive_byteorder="auto",
    frame_byteorder="auto",
    angle_scale=360.0 / 65536,
    include_channel_masks=False,
    mask_rx_flags=0,
    max_pulses_per_ray=4096,
    max_memory_bytes=10 * 1024**3,
    max_frame_bytes=64 * 1024**2,
    max_uncompressed_bytes=4 * 1024**3,
    combine_streams=False,
    transition_weighting="linear",
    sigmoid_steepness=8.0,
):
    """Read selected GDRX6 RX pulses into one MeteoSwiss RadarSpectra.

    Parameters
    ----------
    filename : str or pathlib.Path
        IQRecord .iqr archive, or raw concatenated GDRX6 pulse frames.
    field_names : dict, optional
        Map IQ_H/IQ_V to Py-ART names (defaults: IQ_hh_ADU/IQ_vv_ADU), and
        optionally RX_CHANNEL_H/RX_CHANNEL_V. None mappings omit fields.
    additional_metadata : dict of dict, optional
        Passed to Py-ART FileMetadata; keyed by output variable/field names.
    file_field_names : bool
        Use IQ_H/IQ_V and RX_CHANNEL_H/RX_CHANNEL_V as output field names.
    exclude_fields, include_fields : sequence of str, optional
        Filter final field names using Py-ART FileMetadata conventions.
    pulses_per_ray : int, optional
        Maximum consecutive pulses per output ray. None uses CPI toggle bit
        5, RX CPI-start bit 7 and homogeneous PRF-batch boundaries. Explicit
        counts still respect those boundaries and discontinuities. The final
        short ray is retained; missing padding is masked.
    max_rays : int, optional
        Limit the number of output rays. Recommended for initial inspection.
    start_pulse, stop_pulse : int
        Half-open interval of decoded RX pulse ordinals, before pulse_filter.
        Earlier pulses are visited for clock reconstruction, not retained.
    pulse_filter : callable(dict) -> bool, optional
        Select pulse headers, for example by pulse_width_us or range_step_km.
        Missing transmitted pulses split runs. Removing sibling frames with
        repeated counters does not split a consecutive selected sequence.
    rng_min, rng_max : float, optional
        Inclusive range limits in meters, applied after range centering.
    latitude, longitude, altitude : float, optional
        Site location in degrees/degrees/meters MSL. Missing values are masked.
        Proprietary states.* keys are not guessed.
    frequency : float, optional
        Actual RF carrier frequency in Hz, needed for a Doppler velocity axis.
        Header TX frequencies are preserved but may describe receiver IF;
        they are not silently used as the transmitted RF carrier.
    scan_type : {'other', 'ppi', 'sector', 'rhi', 'vpt'}
        One selected acquisition segment is represented as one sweep. Split
        a volume into sweeps externally using known scan settings.
    fixed_angle : float, optional
        Sweep target angle in degrees; otherwise derived from valid ray angles.
    range_reference : {'start', 'center'}
        'start': header range_start describes the near edge (the archive
        convention); add half a gate for RadarSpectra center coordinates.
        'center': header range_start already describes gate centers.
    time_mode : {'auto', 'gps', 'reconstruct'}
        auto uses GPS-valid pulse timestamps, otherwise an approximate clock
        from archive start time, pulse counters and preceding pulse PRF.
        gps requires every pulse timestamp to be valid. reconstruct ignores
        GPS and uses the anchor for all pulses. GDRX6 seconds are treated as
        Unix UTC seconds. Reconstructed times are explicitly marked.
    start_time : aware datetime, ISO string or float, optional
        Unix UTC anchor of the FIRST RX pulse, before all selection. Overrides
        the archive start anchor, but not valid GPS in auto mode.
    counter_policy : {'mask', 'raise'}
        Without valid GPS, 'mask' reuses the preceding time for repeated
        counters (assuming sibling representations of one transmitted pulse),
        retaining all frames. Backward/very large counter jumps mask absolute
        time until valid GPS returns. Rays split across discontinuities; no
        missing interval is invented. 'raise' rejects either event with details.
        A processing-reset flag with continuous counters only splits the ray.
    archive_byteorder, frame_byteorder : {'auto', 'little', 'big'}
        Independently choose envelope and GDRX6 word byte orders.
    angle_scale : float
        Degrees per angle ADU, default 360/65536 (a conventional assumption).
    include_channel_masks : bool
        Add per-sample high/low receiver-channel selection fields.
    mask_rx_flags : int
        Mask entire IQ pulses whose RX flags intersect this bitmask; default
        zero preserves the received data, apart from non-finite values.
    max_pulses_per_ray : int
        Guard for absent CPI markers. Raises instead of inventing boundaries.
    max_memory_bytes : int
        Conservative working-array budget, not an OS-level RAM guarantee.
        For larger files use iter_rainbow_spectra; dense padding counts too.
    max_frame_bytes, max_uncompressed_bytes : int
        Format/allocation limits. Large inflated items are spooled to disk.

    Returns
    -------
    radar : pyart.core.radar_spectra.RadarSpectra
        Complex64 masked IQ fields shaped (nrays, ngates, npulses_max).
        No FFT is applied. Uniform rays have fftshift frequency coordinates;
        nonuniform/staggered rays have masked coordinates. Extra attributes:
        rainbow_archive_header, rainbow_pulse_metadata (ray x pulse), and
        rainbow_uniform_prt. Generic Py-ART writers/subsetters may not preserve
        these extra attributes: keep the source archive.
    """
    if combine_streams:
        kwargs = locals().copy()
        kwargs.pop("filename")
        kwargs.pop("combine_streams")
        kwargs.pop("transition_weighting")
        kwargs.pop("sigmoid_steepness")
        streams = read_rainbow_iq_streams(filename, **kwargs)
        return combine_rainbow_iq_streams(
            streams,
            transition_weighting=transition_weighting,
            sigmoid_steepness=sigmoid_steepness,
        )
    options = locals().copy()
    options.pop("filename")
    iterator = _spectra_chunks(filename, None, options)
    try:
        return next(iterator)
    finally:
        iterator.close()


def iter_rainbow_spectra(filename, rays_per_chunk=128, **kwargs):
    """Yield bounded-size RadarSpectra chunks; keywords match read_rainbow_iq.

    Each chunk is one sweep segment, not necessarily a complete physical scan.
    ``max_rays`` limits the total output across all chunks. Chunks have their
    own time origins. Fully consume, or explicitly close, this generator to
    release its input file and any temporary decompressed container.
    """
    import inspect

    parameters = inspect.signature(read_rainbow_iq).bind(filename, **kwargs)
    parameters.apply_defaults()
    options = dict(parameters.arguments)
    options.pop("filename")
    if options.pop("combine_streams"):
        raise ValueError(
            "combine_streams is only available through read_rainbow_iq; "
            "stream chunks cannot be combined until both pulse streams are complete"
        )
    options.pop("transition_weighting")
    options.pop("sigmoid_steepness")
    rays_per_chunk = _integer(rays_per_chunk, "rays_per_chunk", 1)
    yield from _spectra_chunks(filename, rays_per_chunk, options)


def _prepare_read(filename, options):
    _pyart()  # Fail before scanning a large input when the required class is absent.
    o = options
    for key in (
        "max_pulses_per_ray",
        "max_memory_bytes",
        "max_frame_bytes",
        "max_uncompressed_bytes",
    ):
        o[key] = _integer(o[key], key, 1)
    for key in ("pulses_per_ray", "max_rays"):
        if o[key] is not None:
            o[key] = _integer(o[key], key, 1)
    if (
        o["pulses_per_ray"] is not None
        and o["pulses_per_ray"] > o["max_pulses_per_ray"]
    ):
        raise ValueError("pulses_per_ray exceeds max_pulses_per_ray")
    o["start_pulse"] = _integer(o["start_pulse"], "start_pulse")
    if o["stop_pulse"] is not None:
        o["stop_pulse"] = _integer(o["stop_pulse"], "stop_pulse", o["start_pulse"] + 1)
    o["mask_rx_flags"] = _integer(o["mask_rx_flags"], "mask_rx_flags")
    if o["range_reference"] not in ("start", "center"):
        raise ValueError("range_reference must be 'start' or 'center'")
    if o["scan_type"] not in ("other", "ppi", "rhi", "sector", "vpt"):
        raise ValueError("Unsupported scan_type")
    for key in ("frequency", "latitude", "longitude", "altitude", "fixed_angle"):
        if o[key] is not None and not math.isfinite(o[key]):
            raise ValueError(f"{key} must be finite")
    if o["frequency"] is not None and o["frequency"] <= 0:
        raise ValueError("frequency must be positive (RF carrier in Hz)")
    if o["latitude"] is not None and not -90 <= o["latitude"] <= 90:
        raise ValueError("latitude must be between -90 and 90 degrees")
    if o["longitude"] is not None and not -180 <= o["longitude"] <= 180:
        raise ValueError("longitude must be between -180 and 180 degrees")
    lower = -np.inf if o["rng_min"] is None else float(o["rng_min"])
    upper = np.inf if o["rng_max"] is None else float(o["rng_max"])
    if np.isnan(lower) or np.isnan(upper) or lower > upper:
        raise ValueError("Invalid range limits")
    with open(filename, "rb") as stream:
        header = _archive_header(stream)
    return header, lower, upper


def _spectra_chunks(filename, chunk_size, options):
    o = options
    header, lower, upper = _prepare_read(filename, o)
    reader_keys = (
        "archive_byteorder",
        "frame_byteorder",
        "angle_scale",
        "max_frame_bytes",
        "max_uncompressed_bytes",
    )
    ray_iter = _rays(
        filename,
        header,
        o["pulses_per_ray"],
        o["start_pulse"],
        o["stop_pulse"],
        o["pulse_filter"],
        lower,
        upper,
        o["range_reference"],
        o["start_time"],
        o["time_mode"],
        o["counter_policy"],
        o["max_pulses_per_ray"],
        o["max_memory_bytes"],
        {k: o[k] for k in reader_keys},
    )
    rays, total, maximum = [], 0, 0
    try:
        for ray, ranges in ray_iter:
            maximum = max(maximum, len(ray))
            # Include original IQ/masks, padded output, pulse dictionaries and
            # metadata arrays. Deliberately overestimate the common dual-pol case.
            estimate = (len(rays) + 1) * maximum * (len(ranges) * 64 + 8192)
            if estimate > o["max_memory_bytes"]:
                raise MemoryError(
                    "Selection exceeds max_memory_bytes; use iter_rainbow_spectra "
                    "with fewer rays_per_chunk, or reduce gates/pulses"
                )
            rays.append(ray)
            total += 1
            last = o["max_rays"] is not None and total >= o["max_rays"]
            if (chunk_size is not None and len(rays) >= chunk_size) or last:
                radar = _build_spectra(rays, ranges, header, filename, o)
                rays, maximum = [], 0
                yield radar
                del radar
            if last:
                return
        if rays:
            yield _build_spectra(rays, ranges, header, filename, o)
        elif total == 0:
            raise rainbowFormatError(
                "No selected GDRX6 RX pulses; expected archive item 0x0122 "
                "with RX types 14/15/16"
            )
    finally:
        ray_iter.close()


_STREAM_KEYS = (
    "pulse_width_us",
    "gate_count",
    "range_start_km",
    "range_step_km",
    "data_type",
    "channel_mask",
)


def _stream_signature(metadata):
    # NaN must not generate a new dictionary key for every unknown pulse width.
    return tuple(
        None
        if isinstance(metadata[k], float) and not math.isfinite(metadata[k])
        else metadata[k]
        for k in _STREAM_KEYS
    )


def _stream_options(filename, kwargs):
    import inspect

    parameters = inspect.signature(read_rainbow_iq).bind(filename, **kwargs)
    parameters.apply_defaults()
    options = dict(parameters.arguments)
    options.pop("filename")
    return options


def read_rainbow_iq_streams(filename, *, max_streams=16, **kwargs):
    """Read interleaved SSPA short/long pulses into separate RadarSpectra objects.

    Decodes the selected recording in ONE pass. Streams are separated by
    nominal pulse width, full native range grid, RX data type and header channel
    mask. Each stream has its own clock, CPI grouping, range axis and PRF.
    Identical counters in different streams are never treated as duplicate
    samples. IQ arrays are neither interpolated nor stitched across streams.

    Parameters
    ----------
    filename : str or pathlib.Path
        IQRecord archive or raw GDRX6 frame stream.
    max_streams : int
        Guard against an unexpectedly large number of changing configurations.
    **kwargs
        Same options as read_rainbow_iq, with these differences:
        max_rays caps rays PER STREAM; start_pulse/stop_pulse still count raw RX
        frames across all streams. The selected file interval is scanned even
        when known streams reach max_rays, so later streams are not overlooked.
        Use stop_pulse for a bounded initial inspection. A range selection that
        misses one stream simply omits that stream. max_memory_bytes covers the
        combined estimated buffers across all streams.

    Returns
    -------
    streams : dict[str, RadarSpectra]
        If exactly two output streams have different, positive finite pulse
        widths, the keys are 'short' and 'long' according to those widths.
        Otherwise keys are stable discovery IDs ('stream_0', 'stream_1', ...).
        Every object has a rainbow_stream dictionary containing its discovery
        ID and native header configuration. Equal widths with different grids
        remain distinct streams; the reader does not guess their physical role.

    Notes
    -----
    In the absence of valid GPS, the first frame of EACH stream uses the supplied
    start_time or archive anchor and subsequent times are reconstructed within
    that stream. This gives an approximate local time axis, NOT a measurement
    of the relative timing between short and long transmissions. If one stream
    begins later in the recording its unknown start delay is also not recovered.
    Counter increments of one within each stream are assumed; gaps split rays.
    This assumption matches interleaved streams sharing pulse/cycle counters,
    but not every possible alternating-transmission counter convention.

    See iter_rainbow_spectra_streams for large recordings. Never combine raw
    short/long complex samples without a known timing/calibration relationship.
    """
    options = _stream_options(filename, kwargs)
    iterator = _stream_chunks(filename, None, max_streams, options)
    streams = {}
    try:
        for name, radar in iterator:
            streams[name] = radar
    finally:
        iterator.close()
    if len(streams) == 2:
        ordered = sorted(
            streams.values(),
            key=lambda r: r.rainbow_stream["pulse_width_us"]
            if r.rainbow_stream["pulse_width_us"] is not None
            else math.inf,
        )
        widths = [r.rainbow_stream["pulse_width_us"] for r in ordered]
        if all(w is not None and w > 0 for w in widths) and not np.isclose(
            widths[0], widths[1]
        ):
            return {"short": ordered[0], "long": ordered[1]}
    return streams


def _get_transition_weights(
    ng0, ng1, radar_ngates, mode="linear", sigmoid_steepness=8.0
):
    """Return near-stream/far-stream weights on a common range grid."""
    if mode not in (
        "mean",
        "linear",
        "sigmoid",
        "only_compressed",
        "only_uncompressed",
    ):
        raise ValueError(f"Unknown transition_weighting {mode!r}")
    w0 = np.zeros(radar_ngates, dtype=np.float64)
    w1 = np.zeros(radar_ngates, dtype=np.float64)
    ind0 = np.arange(ng0, dtype=int)
    ind1 = np.arange(radar_ngates - ng1, radar_ngates, dtype=int)
    w0[ind0] = 1.0
    w1[ind1] = 1.0
    first, last = max(0, radar_ngates - ng1), min(ng0, radar_ngates)
    if last <= first:
        return w0, w1
    size = last - first
    if mode == "mean":
        near = np.full(size, 0.5)
    elif mode == "linear":
        near = np.array([0.5]) if size == 1 else np.linspace(1.0, 0.0, size)
    elif mode == "sigmoid":
        if size == 1:
            near = np.array([0.5])
        else:
            value = 1.0 / (
                1.0 + np.exp(sigmoid_steepness * np.linspace(-1.0, 1.0, size))
            )
            near = (value - value.min()) / (value.max() - value.min())
    elif mode == "only_uncompressed":
        near = np.ones(size)
    else:
        near = np.zeros(size)
    w0[first:last] = near
    w1[first:last] = 1.0 - near
    return w0, w1


def _common_range_grid(first, second):
    """Make the finest constant grid containing two native gate-centre axes."""
    low, high = min(first[0], second[0]), max(first[-1], second[-1])
    step = min(np.min(np.diff(first)), np.min(np.diff(second)))
    # A half-gate offset is common for the two rainbow waveforms.  Refine,
    # without interpolation, until both native grids fall on output centres.
    for _ in range(20):
        count = int(round((high - low) / step)) + 1
        grid = low + np.arange(count) * step
        indices = [
            np.rint((source - low) / step).astype(int) for source in (first, second)
        ]
        if all(
            np.allclose(grid[index], source, rtol=0.0, atol=step * 1e-2)
            for source, index in zip((first, second), indices)
        ):
            return grid, indices
        step /= 2.0
    raise rainbowFormatError(
        "Short/long range grids are not commensurate; cannot combine without interpolation"
    )


def combine_rainbow_iq_streams(
    streams, *, transition_weighting="linear", sigmoid_steepness=8.0
):
    """Combine the short and long rainbow IQ streams on a shared range grid.

    The shorter pulse is used at near range and the longer pulse at far range.
    In common gates, complex IQ is weighted with the same transition choices as
    :func:`read_rainbow_iq` (``mean``, ``linear``, ``sigmoid``,
    ``only_uncompressed``, or ``only_compressed``).  No range interpolation is
    performed.  Both streams must have matching rays and pulses per ray.
    """
    if set(streams) == {"short", "long"}:
        near_key, far_key = "short", "long"
    elif len(streams) == 2:
        # Some SSPA modes report the same nominal TX width for both products.
        # In that case the physically meaningful discriminator is their range
        # coverage: the near product begins earlier and ends earlier.
        (near_key, near), (far_key, far) = sorted(
            streams.items(), key=lambda item: item[1].range["data"][0]
        )
        if not (near.range["data"][-1] < far.range["data"][-1]):
            raise rainbowFormatError(
                "Automatic combination requires two streams with ordered near/far range coverage"
            )
    else:
        raise rainbowFormatError(
            "Automatic combination requires exactly two compatible streams"
        )
    near, far = streams[near_key], streams[far_key]
    if near.nrays != far.nrays or not np.array_equal(
        near.npulses["data"], far.npulses["data"]
    ):
        raise rainbowFormatError(
            "Short/long streams do not have matching ray/CPI structure"
        )
    counters0 = near.rainbow_pulse_metadata["pulse_counter"]["data"]
    counters1 = far.rainbow_pulse_metadata["pulse_counter"]["data"]
    if not np.ma.allequal(counters0, counters1):
        raise rainbowFormatError("Short/long streams cannot be paired by pulse counter")
    ranges, (ind0, ind1) = _common_range_grid(near.range["data"], far.range["data"])
    present0, present1 = np.zeros(len(ranges), bool), np.zeros(len(ranges), bool)
    present0[ind0], present1[ind1] = True, True
    overlap = np.flatnonzero(present0 & present1)
    w0, w1 = present0.astype(float), present1.astype(float)
    if len(overlap):
        transition0, transition1 = _get_transition_weights(
            len(overlap),
            len(overlap),
            len(overlap),
            mode=transition_weighting,
            sigmoid_steepness=sigmoid_steepness,
        )
        w0[overlap], w1[overlap] = transition0, transition1

    fields = {}
    for name in set(near.fields) | set(far.fields):
        first, second = near.fields.get(name), far.fields.get(name)
        if first is None or second is None:
            raise rainbowFormatError(
                "Short/long streams have different output fields; select a common field mapping"
            )
        a0, a1 = np.ma.asarray(first["data"]), np.ma.asarray(second["data"])
        shape = (near.nrays, len(ranges), a0.shape[2])
        if (
            a0.shape[0] != shape[0]
            or a1.shape[0] != shape[0]
            or a0.shape[2] != a1.shape[2]
        ):
            raise rainbowFormatError("Short/long field shapes are incompatible")
        output = np.ma.masked_all(shape, dtype=np.result_type(a0.dtype, a1.dtype))
        for source, indices, weights in ((a0, ind0, w0[ind0]), (a1, ind1, w1[ind1])):
            valid = ~np.ma.getmaskarray(source)
            for local, target in enumerate(indices):
                old_valid = ~np.ma.getmaskarray(output[:, target, :])
                if not old_valid.any():
                    output[:, target, :] = source[:, local, :]
                elif np.iscomplexobj(output):
                    numerator = (
                        output[:, target, :].filled(0) * w0[target] * old_valid
                        + source[:, local, :].filled(0)
                        * weights[local]
                        * valid[:, local, :]
                    )
                    denominator = (
                        w0[target] * old_valid + weights[local] * valid[:, local, :]
                    )
                    output[:, target, :] = np.ma.masked_where(
                        denominator == 0, numerator / denominator
                    )
                elif weights[local] >= w0[target]:
                    output[:, target, :] = source[:, local, :]
        result = dict(first)
        result["data"] = output
        fields[name] = result

    near.range = dict(near.range)
    near.range.update(
        data=ranges,
        meters_to_center_of_first_gate=float(ranges[0]),
        meters_between_gates=float(ranges[1] - ranges[0]),
        spacing_is_constant="true",
    )
    # RadarSpectra caches this scalar at construction; updating range alone is
    # insufficient in older and current Py-ART implementations.
    near.ngates = len(ranges)
    near.fields = fields
    near.metadata = dict(near.metadata)
    near.metadata.update(
        rainbow_stream_combination="short/long range-domain IQ combination",
        rainbow_transition_weighting=transition_weighting,
    )
    near.rainbow_combined_streams = {
        "near_stream_id": near_key,
        "far_stream_id": far_key,
        "near": dict(streams[near_key].rainbow_stream),
        "far": dict(streams[far_key].rainbow_stream),
        "near_weight": w0,
        "far_weight": w1,
        "range_interpolation": False,
    }
    return near


def _angular_difference(first, second):
    """Smallest signed azimuth difference in degrees."""
    return (first - second + 180.0) % 360.0 - 180.0


def _copy_ray_dictionary(sources, attribute, npulses=None):
    """Concatenate a RadarSpectra dictionary whose first axis is ray."""
    dictionaries = [getattr(radar, attribute) for radar, _ in sources]
    if any(value is None for value in dictionaries):
        return None
    template = dict(dictionaries[0])
    arrays = [
        np.ma.asarray(value["data"])[ray]
        for value, (_, ray) in zip(dictionaries, sources)
    ]
    if npulses is not None and arrays[0].ndim == 1:
        width = int(npulses.max())
        data = np.ma.masked_all(
            (len(arrays), width), dtype=np.result_type(*[a.dtype for a in arrays])
        )
        for index, array in enumerate(arrays):
            data[index, : len(array)] = array
    else:
        data = np.ma.asarray(arrays)
    template["data"] = data
    return template


def _concatenate_iq_rays(sources, moment_radar, moment_indices):
    """Build one RadarSpectra from selected source rays in moment-ray order."""
    template = sources[0][0]
    if any(
        not np.allclose(r.range["data"], template.range["data"], rtol=0.0, atol=1e-6)
        for r, _ in sources
    ):
        raise rainbowFormatError(
            "Selected IQ files have different range grids; read compatible scan settings separately"
        )
    if any(set(r.fields) != set(template.fields) for r, _ in sources):
        raise rainbowFormatError(
            "Selected IQ files have different fields; use identical reader options"
        )
    npulses = np.array([r.npulses["data"][ray] for r, ray in sources], dtype=np.int32)
    output = copy.copy(template)
    output.nrays = len(sources)
    output.npulses = {"data": npulses}
    output.time = _copy_ray_dictionary(sources, "time")
    # The moment file defines the requested scan geometry and output order.
    output.azimuth = dict(template.azimuth)
    output.azimuth["data"] = np.ma.asarray(moment_radar.azimuth["data"])[moment_indices]
    output.elevation = dict(template.elevation)
    output.elevation["data"] = np.ma.asarray(moment_radar.elevation["data"])[
        moment_indices
    ]
    output.sweep_number = {"data": np.array([0], dtype=np.int32)}
    output.sweep_start_ray_index = {"data": np.array([0], dtype=np.int32)}
    output.sweep_end_ray_index = {"data": np.array([len(sources) - 1], dtype=np.int32)}
    output.fixed_angle = {"data": np.ma.array([np.ma.median(output.elevation["data"])])}

    output.fields = {}
    for name, field in template.fields.items():
        width = int(npulses.max())
        data = np.ma.masked_all(
            (len(sources), template.ngates, width), dtype=field["data"].dtype
        )
        for target, (radar, ray) in enumerate(sources):
            values = np.ma.asarray(radar.fields[name]["data"])[ray]
            data[target, :, : values.shape[-1]] = values
        copied = dict(field)
        copied["data"] = data
        output.fields[name] = copied
    for attribute in ("Doppler_frequency", "Doppler_velocity"):
        value = _copy_ray_dictionary(sources, attribute, npulses)
        if value is not None:
            setattr(output, attribute, value)
    output.instrument_parameters = dict(template.instrument_parameters or {})
    for key in ("prt", "pulse_width", "nyquist_velocity"):
        if key in output.instrument_parameters:
            values = []
            for radar, ray in sources:
                values.append(
                    np.ma.asarray(radar.instrument_parameters[key]["data"])[ray]
                )
            copied = dict(output.instrument_parameters[key])
            copied["data"] = np.ma.asarray(values)
            output.instrument_parameters[key] = copied
    output.rainbow_pulse_metadata = {
        key: {
            **value,
            "data": _copy_pulse_metadata(sources, value["data"], key, npulses),
        }
        for key, value in template.rainbow_pulse_metadata.items()
    }
    return output


def _copy_pulse_metadata(sources, template_data, key, npulses):
    width = int(npulses.max())
    data = np.ma.masked_all(
        (len(sources), width), dtype=np.ma.asarray(template_data).dtype
    )
    for target, (radar, ray) in enumerate(sources):
        values = np.ma.asarray(radar.rainbow_pulse_metadata[key]["data"])[ray]
        data[target, : len(values)] = values
    return data


def read_rainbow_iq_scan(
    moment_filename,
    iqr_filenames,
    *,
    azimuth_tolerance=0.25,
    elevation_tolerance=0.25,
    combine_streams=True,
    moment_reader=None,
    **iq_kwargs,
):
    """Read IQRecord files for one Rainbow scan, using its moment rays as reference.

    Only IQ rays that match an actually present Rainbow moment ray are retained.
    The returned RadarSpectra follows the moment-file ray order; its azimuth and
    elevation are copied from that file, while time and IQ remain from GDRX6.

    Parameters
    ----------
    moment_filename : str or path-like
        One Rainbow moment file readable by :func:`pyart.aux_io.read_rainbow_wrl`.
    iqr_filenames : iterable of str or path-like
        IQRecord files covering portions of the same scan.
    azimuth_tolerance, elevation_tolerance : float
        Maximum independent angular mismatch in degrees.
    combine_streams : bool
        Combine the near/far SSPA products within each IQR before matching.
    moment_reader : callable, optional
        Test/integration override. The default is ``pyart.aux_io.read_rainbow_wrl``.
    **iq_kwargs
        Passed to :func:`read_rainbow_iq` for every IQR file.
    """
    if azimuth_tolerance <= 0 or elevation_tolerance <= 0:
        raise ValueError("Angular tolerances must be positive")
    if isinstance(iqr_filenames, (str, os.PathLike)):
        iqr_filenames = [iqr_filenames]
    iqr_filenames = list(iqr_filenames)
    if not iqr_filenames:
        raise ValueError("iqr_filenames is empty")
    if moment_reader is None:
        try:
            from pyart.aux_io import read_rainbow_wrl
        except ImportError as exc:
            raise ImportError(
                "read_rainbow_iq_scan requires Py-ART Rainbow support"
            ) from exc
        moment_reader = read_rainbow_wrl
    moment = moment_reader(moment_filename)
    candidates = []
    for filename in iqr_filenames:
        radar = read_rainbow_iq(filename, combine_streams=combine_streams, **iq_kwargs)
        candidates.extend((radar, ray) for ray in range(radar.nrays))
    if not candidates:
        raise rainbowFormatError("No IQ rays were read")

    matched, indices, used = [], [], set()
    for target in range(moment.nrays):
        azimuth, elevation = (
            moment.azimuth["data"][target],
            moment.elevation["data"][target],
        )
        possible = []
        for index, (radar, ray) in enumerate(candidates):
            if index in used:
                continue
            da = abs(_angular_difference(radar.azimuth["data"][ray], azimuth))
            de = abs(radar.elevation["data"][ray] - elevation)
            if da <= azimuth_tolerance and de <= elevation_tolerance:
                possible.append(
                    (da / azimuth_tolerance + de / elevation_tolerance, index)
                )
        if possible:
            _, index = min(possible)
            used.add(index)
            matched.append(candidates[index])
            indices.append(target)
    if not matched:
        raise rainbowFormatError(
            "No IQ ray matches a ray present in the Rainbow moment file"
        )
    result = _concatenate_iq_rays(matched, moment, np.asarray(indices, dtype=int))
    result.rainbow_moment_reference = {
        "filename": os.fspath(moment_filename),
        "matched_ray_indices": np.asarray(indices, dtype=np.int32),
        "moment_ray_count": moment.nrays,
        "iq_ray_count": len(candidates),
        "azimuth_tolerance_degrees": float(azimuth_tolerance),
        "elevation_tolerance_degrees": float(elevation_tolerance),
    }
    return result


def iter_rainbow_spectra_streams(
    filename, rays_per_chunk=128, *, max_streams=16, **kwargs
):
    """Yield (stream_id, RadarSpectra) chunks from interleaved pulse streams.

    Reads the selected file interval once. Each stream accumulates its own rays
    and produces a chunk when it reaches rays_per_chunk. Final partial chunks
    are retained. Output follows chunk readiness, not globally sorted time.
    IDs remain stream_0/stream_1/etc.; no short/long aliases are assigned because
    additional pulse widths may appear later. Inspect radar.rainbow_stream.

    Options and timing assumptions match read_rainbow_iq_streams. max_rays is
    a PER-STREAM cap across all chunks. Fully consume or close this iterator to
    release the input file and temporary decompression files. Process chunks
    as they arrive; retaining all of them defeats the memory bound.
    """
    rays_per_chunk = _integer(rays_per_chunk, "rays_per_chunk", 1)
    options = _stream_options(filename, kwargs)
    yield from _stream_chunks(filename, rays_per_chunk, max_streams, options)


def _stream_chunks(filename, chunk_size, max_streams, options):
    o = options
    header, lower, upper = _prepare_read(filename, o)
    max_streams = _integer(max_streams, "max_streams", 1)
    # Validate clock options even if no stream is selected.
    _Clock(header, o["start_time"], o["time_mode"], o["counter_policy"])
    states = {}
    reader_keys = (
        "archive_byteorder",
        "frame_byteorder",
        "angle_scale",
        "max_frame_bytes",
        "max_uncompressed_bytes",
    )
    pulses = iter_rainbow_iq(filename, **{k: o[k] for k in reader_keys})
    timing_warned = False
    yielded = False

    def finish_ray(state):
        if state["ray"]:
            state["rays"].append(state["ray"])
            state["maximum"] = max(state["maximum"], len(state["ray"]))
            state["ray"] = []
            state["count"] += 1

    def check_memory():
        estimate = sum(
            (len(s["rays"]) + bool(s["ray"]))
            * max(s["maximum"], len(s["ray"]))
            * (len(s["ranges"]) * 64 + 8192)
            for s in states.values()
        )
        if estimate > o["max_memory_bytes"]:
            raise MemoryError(
                "Combined pulse-stream buffers exceed max_memory_bytes; "
                "use iter_rainbow_spectra_streams with fewer rays_per_chunk, "
                "or reduce gates/pulses"
            )

    def emit(state):
        radar = _build_spectra(state["rays"], state["ranges"], header, filename, o)
        radar.rainbow_stream = dict(state["description"])
        radar.metadata["rainbow_stream_id"] = state["id"]
        radar.metadata["rainbow_time_alignment"] = (
            "Use valid GPS for cross-stream alignment; reconstructed times have "
            "an unknown offset for the first pulse of each stream"
        )
        state["rays"], state["maximum"] = [], 0
        return state["id"], radar

    try:
        for index, pulse in enumerate(pulses):
            if o["stop_pulse"] is not None and index >= o["stop_pulse"]:
                break
            m = pulse.metadata
            m["source_pulse_index"] = index
            signature = _stream_signature(m)
            if signature not in states:
                if len(states) >= max_streams:
                    raise rainbowFormatError(
                        "More configurations than max_streams; restrict the "
                        "recording interval or inspect the native frame headers"
                    )
                stream_id = f"stream_{len(states)}"
                offset = 0.5 if o["range_reference"] == "start" else 0.0
                all_ranges = (
                    m["range_start_km"]
                    + (np.arange(m["gate_count"]) + offset) * m["range_step_km"]
                ) * 1000
                gates = np.flatnonzero((all_ranges >= lower) & (all_ranges <= upper))
                clock = _Clock(
                    header, o["start_time"], o["time_mode"], o["counter_policy"]
                )
                clock.warned = True  # Use the more specific multi-stream warning below.
                states[signature] = {
                    "id": stream_id,
                    "description": dict(zip(_STREAM_KEYS, signature)),
                    "gates": gates,
                    "ranges": all_ranges[gates],
                    "clock": clock,
                    "ray": [],
                    "rays": [],
                    "maximum": 0,
                    "count": 0,
                }
                states[signature]["description"]["stream_id"] = stream_id
            state = states[signature]
            if not len(state["gates"]):
                continue
            if o["max_rays"] is not None and state["count"] >= o["max_rays"]:
                continue
            # Separate counters/clocks before evaluating selection: identical
            # short/long counters are not consecutive samples in either stream.
            state["clock"].apply(pulse)
            if m["time_source"] != 0 and not timing_warned:
                warnings.warn(
                    "Pulse timing reconstructed independently for each stream. "
                    "The first frame of each stream uses the recording/start_time anchor; "
                    "relative offsets between streams are unknown without valid GPS. "
                    "Equal short/long counters do not establish simultaneous transmission.",
                    stacklevel=3,
                )
                timing_warned = True
            if index < o["start_pulse"] or (
                o["pulse_filter"] is not None and not o["pulse_filter"](m)
            ):
                continue
            ray = state["ray"]
            if ray and _new_ray(ray[-1], pulse):
                finish_ray(state)
            if chunk_size is not None and len(state["rays"]) >= chunk_size:
                check_memory()
                yield emit(state)
                yielded = True
            if o["max_rays"] is not None and state["count"] >= o["max_rays"]:
                continue
            if len(state["ray"]) >= o["max_pulses_per_ray"]:
                raise rainbowFormatError(
                    f"{state['id']}: no ray boundary before "
                    "max_pulses_per_ray; set pulses_per_ray"
                )
            gates = state["gates"]
            pulse.iq = {p: a[gates].copy() for p, a in pulse.iq.items()}
            pulse.high_sensitivity = {
                p: a[gates].copy() for p, a in pulse.high_sensitivity.items()
            }
            state["ray"].append(pulse)
            if (
                o["pulses_per_ray"] is not None
                and len(state["ray"]) >= o["pulses_per_ray"]
            ):
                finish_ray(state)
            check_memory()
            if chunk_size is not None and len(state["rays"]) >= chunk_size:
                yield emit(state)
                yielded = True
        for state in states.values():
            finish_ray(state)
            if state["rays"]:
                check_memory()
                yield emit(state)
                yielded = True
        if not yielded:
            raise rainbowFormatError(
                "No selected GDRX6 IQ streams within the requested ranges/interval"
            )
    finally:
        pulses.close()
