"""
pyart.io.common
===============

Input/output routines common to many file formats.

.. autosummary::
    :toctree: generated/

    read_files
    prepare_for_read
    stringarray_to_chararray
    _test_arguments
    make_time_unit_str

"""

import bz2
import glob
import gzip

import fsspec
import netCDF4
import numpy as np

import pyart


def read_files(wildcard, reader, **kwargs):
    """
    Read multiple radar files matching a wildcard and merge them into a single
    Py-ART Radar object.

    The files are read in sorted order. The first file initializes the radar
    object, and subsequent files are merged into it using
    ``pyart.util.radar_utils.join_radar``.

    Parameters
    ----------
    wildcard : str
        File path pattern (including wildcards) used to locate radar files,
        e.g. ``"/path/to/data/MLD1816221000U.*"``.
    reader : callable
        Py-ART radar reader function used to read a single file, for example
        ``pyart.aux_io.read_metranet`` or ``pyart.io.read``.

    Returns
    -------
    radar : pyart.core.Radar
        A Py-ART Radar object containing the merged data from all files
        matching the wildcard.

    Raises
    ------
    FileNotFoundError
        If no files match the provided wildcard pattern.

    Notes
    -----
    This function assumes that all radar files are compatible for merging,
    i.e. they represent consecutive sweeps or volumes that can be combined
    using ``pyart.util.radar_utils.join_radar``.
    """
    files = sorted(glob.glob(wildcard))

    if not files:
        raise FileNotFoundError(f"No files found matching wildcard: {wildcard}")

    radar = reader(files[0], **kwargs)

    for fi in files[1:]:
        radar = pyart.util.radar_utils.join_radar(radar, reader(fi, **kwargs))

    return radar


def prepare_for_read(filename, storage_options={"anon": True}):
    """
    Return a file like object read for reading.

    Open a file for reading in binary mode with transparent decompression of
    Gzip and BZip2 files. The resulting file-like object should be closed.

    Parameters
    ----------
    filename : str or file-like object
        Filename or file-like object which will be opened. File-like objects
        will not be examined for compressed data.

    storage_options : dict, optional
        Parameters passed to the backend file-system such as Google Cloud Storage,
        Amazon Web Service S3.

    Returns
    -------
    file_like : file-like object
        File like object from which data can be read.

    """
    # if a file-like object was provided, return
    if hasattr(filename, "read"):  # file-like object
        return filename

    # look for compressed data by examining the first few bytes
    fh = fsspec.open(filename, mode="rb", compression="infer", **storage_options).open()
    magic = fh.read(3)
    fh.close()

    # If the data is still compressed, use gunzip/bz2 to uncompress the data
    if magic.startswith(b"\x1f\x8b"):
        return gzip.GzipFile(filename, "rb")

    if magic.startswith(b"BZh"):
        return bz2.BZ2File(filename, "rb")

    return fsspec.open(
        filename, mode="rb", compression="infer", **storage_options
    ).open()


def stringarray_to_chararray(arr, numchars=None):
    """
    Convert a string array to a character array with one extra dimension.

    Implementation that falls back to pure-numpy conversion if
    netCDF4.stringtochar is unavailable or fails.

    Parameters
    ----------
    arr : array-like
        String or bytes array
    numchars : int, optional
        Fixed character width. Must be >= actual max string length.

    Returns
    -------
    ndarray
        Character array with dtype 'S1' and shape (*arr.shape, numchars)
    """
    arr = np.asarray(arr)

    # Handle scalar
    scalar = arr.ndim == 0
    if scalar:
        arr = arr.reshape((1,))

    # Handle masked arrays
    if np.ma.isMaskedArray(arr):
        arr = arr.filled("")

    # Try netCDF4 first
    carr = None
    try:
        carr = netCDF4.stringtochar(arr)
    except (ImportError, AttributeError, Exception):
        pass  # Fall through to manual conversion

    # Manual fallback
    if carr is None:
        carr = _manual_string_to_char(arr, numchars)

    # Validate and pad if numchars specified
    if numchars is not None:
        arr_numchars = carr.shape[-1]
        if numchars < arr_numchars:
            raise ValueError(
                f"numchars ({numchars}) must be >= actual width ({arr_numchars})"
            )
        if numchars > arr_numchars:
            out = np.zeros(arr.shape + (numchars,), dtype="S1")
            out[..., :arr_numchars] = carr
            carr = out

    # Restore scalar shape
    if scalar:
        carr = carr[0]

    return carr


def _manual_string_to_char(arr, numchars=None):
    """Manual string-to-char conversion."""
    # Handle empty arrays
    if arr.size == 0:
        width = numchars if numchars is not None else 1
        return np.zeros(arr.shape + (width,), dtype="S1")

    # Encode to bytes
    flat = arr.ravel()
    encoded = []
    for x in flat:
        if x is None or x == "":
            encoded.append(b"")
        elif isinstance(x, bytes):
            encoded.append(x)
        else:
            encoded.append(str(x).encode("utf-8"))

    # Determine width
    max_bytes = max(len(b) for b in encoded)
    width = numchars if numchars is not None else max(max_bytes, 1)

    # Allocate and fill
    chararr = np.zeros(arr.shape + (width,), dtype="S1")
    for idx, b in zip(np.ndindex(arr.shape), encoded):
        if len(b) > width:
            b = b[:width]  # Truncate
        # Use null padding for netCDF compatibility
        padded = b + b"\x00" * (width - len(b))
        chararr[idx] = np.frombuffer(padded, dtype="S1", count=width)

    return chararr


def _test_arguments(dic):
    """Issue a warning if receive non-empty argument dict."""
    if dic:
        import warnings

        warnings.warn(f"Unexpected arguments: {dic.keys()}")


def make_time_unit_str(dtobj):
    """Return a time unit string from a datetime object."""
    return "seconds since " + dtobj.strftime("%Y-%m-%dT%H:%M:%SZ")
