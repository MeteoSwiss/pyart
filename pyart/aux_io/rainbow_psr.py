"""
pyart.aux_io.rainbow_psr
========================

Routines for reading RAINBOW PSR files (Used by SELEX)

.. autosummary::
    :toctree: generated/

    read_rainbow_psr
    read_rainbow_psr_spectra
    read_psr_header
    read_psr_cpi_header
    read_psr_spectra
    get_item_numbers
    get_field
    get_spectra_field
    get_Doppler_info
    get_noise_field
    convert_data
    get_library
    get_library_path

"""

# specific modules for this function
import ctypes
import importlib
import os
from copy import deepcopy
from warnings import warn

import numpy as np

from ..config import FileMetadata
from ..core.radar_spectra import RadarSpectra
from ..exceptions import MissingOptionalDependency
from ..io.common import _test_arguments
from ..util import ma_broadcast_to, subset_radar
from .rainbow_wrl import read_rainbow_wrl

# Check existence of required libraries
# Lint compatible version (wod, 20.07.2023)
if importlib.util.find_spec("wradlib"):
    _WRADLIB_AVAILABLE = True
else:
    _WRADLIB_AVAILABLE = False

PSR_FIELD_NAMES = [
    "transmitted_signal_power_h",
    "transmitted_signal_power_v",
    "noisedBADU_hh",
    "noisedBADU_vv",
    "noisedBm_hh",
    "noisedBm_vv",
    "noisedBZ_hh",
    "noisedBZ_vv",
]

PSR_FIELD_NAMES_SPECTRA = [
    "spectral_noise_power_hh_ADU",
    "spectral_noise_power_vv_ADU",
    "complex_spectra_hh_ADU",
    "complex_spectra_vv_ADU",
]


def read_rainbow_psr(
    filename,
    filenames_psr,
    field_names=None,
    additional_metadata=None,
    file_field_names=False,
    exclude_fields=None,
    include_fields=None,
    undo_txcorr=True,
    cpi="mean",
    ang_tol=0.5,
    azi_min=None,
    azi_max=None,
    ele_min=None,
    ele_max=None,
    rng_min=None,
    rng_max=None,
    transition_weighting="linear",
    **kwargs,
):
    """
    Read transmitted power and noise levels from a PSR file
    (to read the complex spectrum use read_rainbow_psr_spectra)

    Parameters
    ----------
    filename : str
        Name of the rainbow file to be used as reference.
    filenames_psr : list of str
        Name of the PSR files
    field_names : dict, optional
        Dictionary mapping RAINBOW field names to radar field names. If a
        data type found in the file does not appear in this dictionary or has
        a value of None it will not be placed in the radar.fields dictionary.
        A value of None, the default, will use the mapping defined in the
        Py-ART configuration file.
    additional_metadata : dict of dicts, optional
        Dictionary of dictionaries to retrieve metadata during this read.
        This metadata is not used during any successive file reads unless
        explicitly included.  A value of None, the default, will not
        introduct any addition metadata and the file specific or default
        metadata as specified by the Py-ART configuration file will be used.
    file_field_names : bool, optional
        True to use the MDV data type names for the field names. If this
        case the field_names parameter is ignored. The field dictionary will
        likely only have a 'data' key, unless the fields are defined in
        `additional_metadata`.
    exclude_fields : list or None, optional
        List of fields to exclude from the radar object. This is applied
        after the `file_field_names` and `field_names` parameters. Set
        to None to include all fields specified by include_fields.
    include_fields : list or None, optional
        List of fields to include from the radar object. This is applied
        after the `file_field_names` and `field_names` parameters. Set
        to None to include all fields not specified by exclude_fields.
    undo_txcorr: Bool
        If True the correction of the transmitted power is removed from the
        noise signal
    cpi : str
        The CPI to use. Can be 'low_prf', 'intermediate_prf', 'high_prf',
        'mean', 'all'. If 'mean' the mean within the angle step is taken
    ang_tol : float
        Tolerated angle distance between nominal radar angle and angle in
        PSR files
    azi_min, azi_max, ele_min, ele_max : float or None
        The minimum and maximum angles to keep (deg)
    rng_min, rng_max : float or None
        The minimum and maximum ranges to keep (m)
    transition_weighting : str
        For pulse-compressed radars only.
        Defines how the spectrum from the uncompressed and compressed spectra
        should be combined in the transition region.
        Available options are: "mean", "linear", "sigmoid",
        "only_compressed", "only_uncompressed"
        "mean": both spectra are averaged with weight of 0.5 each
        "linear": the weight of the uncompressed spectra is 1 at the beg. of
        the transition region, 0 at the end.
        "sigmoid": Similar to linear but a sigmoid (derivable) function is
        used instead.
        "only_compressed": only uses compressed spectrum in the transition
        (for testing purposes)
        "only_uncompressed": only uses uncompressed spectrum in the transition
        (for testing purposes)
        Default is "linear".
    Returns
    -------
    radar : Radar
        Radar object containing data from PSR file.
    """
    # check that wradlib is available
    if not _WRADLIB_AVAILABLE:
        raise MissingOptionalDependency(
            "wradlib is required to use read_rainbow_psr but is not installed"
        )

    # test for non empty kwargs
    _test_arguments(kwargs)

    # create radar object used as reference
    try:
        radar = read_rainbow_wrl(filename)
        radar.fields = dict()
        radar = subset_radar(
            radar,
            None,
            rng_min=rng_min,
            rng_max=rng_max,
            ele_min=ele_min,
            ele_max=ele_max,
            azi_min=azi_min,
            azi_max=azi_max,
        )
        if radar is None:
            warn("No data within specified azimuth, elevation and range limits")
            return None
    except OSError as ee:
        warn(str(ee))
        warn("Unable to read file " + filename)
        return None

    # create metadata retrieval object
    if field_names is None:
        field_names = PSR_FIELD_NAMES

    filemetadata = FileMetadata(
        "PSR",
        field_names,
        additional_metadata,
        file_field_names,
        exclude_fields,
        include_fields,
    )
    dBADU_to_dBm_hh = filemetadata("dBADU_to_dBm_hh")
    dBADU_to_dBm_vv = filemetadata("dBADU_to_dBm_vv")
    mfloss_h = filemetadata("matched_filter_loss_h")
    mfloss_v = filemetadata("matched_filter_loss_v")
    pathatt = filemetadata("path_attenuation")

    cpi_header, header = read_psr_cpi_headers(filenames_psr)
    if cpi_header is None:
        return None

    # Check whether radar uses pulse compression
    pulse_compression = "states.spbpctfmchanneltransitionranges" in header

    # keep only valid items
    prfs = np.sort(np.unique(cpi_header["prfs"]))
    items, radar = get_item_numbers(
        radar,
        cpi_header["azi_start"],
        cpi_header["azi_stop"],
        cpi_header["ele_start"],
        cpi_header["ele_stop"],
        cpi_header["prfs"],
        prfs,
        pulse_compression=pulse_compression,
        ngates=cpi_header["ngates"],
        cpi=cpi,
        ang_tol=ang_tol,
    )

    has_items = items[0].size if pulse_compression else items.size
    if not has_items:
        warn("No items matching radar object")
        return None

    if pulse_compression:
        ng0 = cpi_header["ngates"][0]
        ng1 = cpi_header["ngates"][1]

        w0_1d, w1_1d = _get_transition_weights(
            ng0,
            ng1,
            radar.ngates,
            mode=transition_weighting,
            sigmoid_steepness=8.0,
        )

    for field_name in field_names:
        if not pulse_compression:
            field_data = get_field(
                radar,
                cpi_header,
                header,
                items,
                prfs.size,
                field_name,
                undo_txcorr=undo_txcorr,
                cpi=cpi,
            )
        else:
            weighted_sum = None
            weight_sum = None

            # first item group starts at gate 0
            # second item group is mapped from the end, like in spectra reader
            ind_rng_groups = [
                np.arange(0, ng0, dtype=int),
                np.arange(0, ng1, dtype=int),
            ]

            for i, (item_group, ind_rng_aux) in enumerate(zip(items, ind_rng_groups)):
                field_part = get_field(
                    radar,
                    cpi_header,
                    header,
                    item_group,
                    prfs.size,
                    field_name,
                    undo_txcorr=undo_txcorr,
                    cpi=cpi,
                )

                field_part = np.ma.asarray(field_part)
                valid = ~np.ma.getmaskarray(field_part)
                filled = field_part.filled(field_part.dtype.type(0))

                if weighted_sum is None:
                    out_shape = list(field_part.shape)
                    out_shape[1] = radar.ngates
                    weighted_sum = np.zeros(out_shape, dtype=np.float64)
                    weight_sum = np.zeros(out_shape, dtype=np.float64)

                if i == 0:
                    # first field occupies the beginning of the radial
                    out_idx = ind_rng_aux
                    weights_1d = w0_1d[out_idx]
                else:
                    # second field is flipped to the end of the radial
                    out_idx = (radar.ngates - ind_rng_aux)[::-1] - 1
                    weights_1d = w1_1d[out_idx]

                weights = weights_1d[None, :]

                weighted_sum[:, out_idx] += (
                    filled[:, ind_rng_aux] * weights * valid[:, ind_rng_aux]
                )
                weight_sum[:, out_idx] += weights * valid[:, ind_rng_aux]

            valid_out = weight_sum > 0
            field_data = np.ma.masked_all(weighted_sum.shape, dtype=field_part.dtype)
            field_data[valid_out] = (
                weighted_sum[valid_out] / weight_sum[valid_out]
            ).astype(field_part.dtype)

        if field_name in (
            "noisedBADU_hh",
            "noisedBADU_vv",
            "noisedBm_hh",
            "noisedBm_vv",
            "noisedBZ_hh",
            "noisedBZ_vv",
        ):
            field_data = get_noise_field(radar, field_data, header, field_name)

        field_dict = filemetadata(field_name)
        field_dict["data"] = field_data
        radar.add_field(field_name, field_dict)

    # if pulse compression used, flatten items for any later per-item metadata
    if pulse_compression:
        items = np.sort(np.concatenate([np.asarray(item) for item in items]))

    # get further metadata
    pw_ind = header["states.spbpwidth"]

    if not hasattr(radar, "radar_calibration") or radar.radar_calibration is None:
        radar.radar_calibration = dict()

    dBADU_to_dBm_hh["data"] = np.array(
        [np.atleast_1d(header["states.spbdbmtologoffset"])[pw_ind]]
    )
    radar.radar_calibration.update({"dBADU_to_dBm_hh": dBADU_to_dBm_hh})

    dBADU_to_dBm_vv["data"] = np.array(
        [np.atleast_1d(header["states.spbdbmtologoffset"])[pw_ind]]
    )
    radar.radar_calibration.update({"dBADU_to_dBm_vv": dBADU_to_dBm_vv})

    if "states.gdrxmfloss" in header:
        mfloss_h["data"] = np.array([header["states.gdrxmfloss"][pw_ind]])
        mfloss_v["data"] = np.array([header["states.gdrxmfloss"][pw_ind]])
    elif "states.spbdphmflossforbatch" in header:
        mfloss_h["data"] = np.array([header["states.spbdphmflossforbatch"]])
        mfloss_v["data"] = np.array([header["states.spbdpvmflossforbatch"]])

    radar.radar_calibration.update({"matched_filter_loss_h": mfloss_h})
    radar.radar_calibration.update({"matched_filter_loss_v": mfloss_v})

    pathatt["data"] = np.array([header["states.rspathatt"]])
    radar.radar_calibration.update({"path_attenuation": pathatt})
    return radar


def read_rainbow_psr_spectra(
    filename,
    filenames_psr,
    field_names=None,
    additional_metadata=None,
    file_field_names=False,
    exclude_fields=None,
    include_fields=None,
    undo_txcorr=True,
    fold=True,
    positive_away=True,
    cpi="low_prf",
    ang_tol=0.5,
    azi_min=None,
    azi_max=None,
    ele_min=None,
    ele_max=None,
    rng_min=None,
    rng_max=None,
    transition_weighting="linear",
    **kwargs,
):
    """
    Read a PSR file to get the complex spectra

    Parameters
    ----------
    filename : str
        Name of the rainbow file to be used as reference.
    filenames_psr : list of str
        list of PSR file names
    field_names : dict, optional
        Dictionary mapping RAINBOW field names to radar field names. If a
        data type found in the file does not appear in this dictionary or has
        a value of None it will not be placed in the radar.fields dictionary.
        A value of None, the default, will use the mapping defined in the
        Py-ART configuration file.
    additional_metadata : dict of dicts, optional
        Dictionary of dictionaries to retrieve metadata during this read.
        This metadata is not used during any successive file reads unless
        explicitly included.  A value of None, the default, will not
        introduct any addition metadata and the file specific or default
        metadata as specified by the Py-ART configuration file will be used.
    file_field_names : bool, optional
        True to use the MDV data type names for the field names. If this
        case the field_names parameter is ignored. The field dictionary will
        likely only have a 'data' key, unless the fields are defined in
        `additional_metadata`.
    exclude_fields : list or None, optional
        List of fields to exclude from the radar object. This is applied
        after the `file_field_names` and `field_names` parameters. Set
        to None to include all fields specified by include_fields.
    include_fields : list or None, optional
        List of fields to include from the radar object. This is applied
        after the `file_field_names` and `field_names` parameters. Set
        to None to include all fields not specified by exclude_fields.
    undo_txcorr: Bool
        If True the correction of the transmitted power is removed from the
        noise signal
    fold: Bool
        If True the spectra is folded so that 0-Doppler is in the middle
    positive_away: Bool
        If True the spectra is reversed so that positive velocities are
        away from the radar
    cpi : str
        The CPI to use. Can be 'low_prf', 'intermediate_prf', 'high_prf' or
        'all'
    ang_tol : float
        Tolerated angle distance between nominal radar angle and angle in
        PSR files
    azi_min, azi_max, ele_min, ele_max : float or None
        The minimum and maximum angles to keep (deg)
    rng_min, rng_max : float or None
        The minimum and maximum ranges to keep (m)
    transition_weighting : str
        For pulse-compressed radars only.
        Defines how the spectrum from the uncompressed and compressed spectra
        should be combined in the transition region.
        Available options are: "mean", "linear" and "sigmoid", "only_compressed", "only_uncompressed"
        "mean": both spectra are averaged with weight of 0.5 each
        "linear": the weight of the uncompressed spectra is 1 at the beg. of
        the transition region, 0 at the end.
        "sigmoid": Similar to linear but a sigmoid (derivable) function is used instead.
        "only_compressed": only uses compressed spectrum in the transition (for testing purposes)
        "only_uncompressed": only uses uncompressed spectrum in the transition (for testing purposes)
        Default is "linear".
    Returns
    -------
    radar : Radar
        Radar object containing data from PSR file.

    """
    # check that wradlib is available
    if not _WRADLIB_AVAILABLE:
        raise MissingOptionalDependency(
            "wradlib is required to use read_rainbow_psr but is not installed"
        )

    # test for non empty kwargs
    _test_arguments(kwargs)

    # create radar object used as reference
    try:
        radar = read_rainbow_wrl(filename)
        radar.fields = dict()
        rng_orig = radar.range["data"]
        radar = subset_radar(
            radar,
            None,
            rng_min=rng_min,
            rng_max=rng_max,
            ele_min=ele_min,
            ele_max=ele_max,
            azi_min=azi_min,
            azi_max=azi_max,
        )
        if radar is None:
            warn("No data within specified azimuth, elevation and" " range limits")
            return None
        ind_rng_start = np.where(rng_orig == radar.range["data"][0])[0]
        ind_rng_end = np.where(rng_orig == radar.range["data"][-1])[0]
        start = np.asarray(ind_rng_start).item()
        end = np.asarray(ind_rng_end).item()
        ind_rng = np.arange(start, end + 1, dtype=int)
    except OSError as ee:
        warn(str(ee))
        warn("Unable to read file " + filename)
        return None

    # create metadata retrieval object
    if field_names is None:
        field_names = PSR_FIELD_NAMES_SPECTRA

    filemetadata = FileMetadata(
        "PSR",
        field_names,
        additional_metadata,
        file_field_names,
        exclude_fields,
        include_fields,
    )
    dBADU_to_dBm_hh = filemetadata("dBADU_to_dBm_hh")
    dBADU_to_dBm_vv = filemetadata("dBADU_to_dBm_vv")
    mfloss_h = filemetadata("matched_filter_loss_h")
    mfloss_v = filemetadata("matched_filter_loss_v")
    pathatt = filemetadata("path_attenuation")

    cpi_header, header = read_psr_cpi_headers(filenames_psr)
    if cpi_header is None:
        return None

    # Check whether radar uses pulse compression
    pulse_compression = "states.spbpctfmchanneltransitionranges" in header

    # keep only valid items
    prfs = np.sort(np.unique(cpi_header["prfs"]))
    items, radar = get_item_numbers(
        radar,
        cpi_header["azi_start"],
        cpi_header["azi_stop"],
        cpi_header["ele_start"],
        cpi_header["ele_stop"],
        cpi_header["prfs"],
        prfs,
        pulse_compression=pulse_compression,
        ngates=cpi_header["ngates"],
        cpi=cpi,
        ang_tol=ang_tol,
    )

    has_items = items[0].size if pulse_compression else items.size
    if not has_items:
        warn("No items matching radar object")

    if pulse_compression:
        ng0 = cpi_header["ngates"][0]
        ng1 = cpi_header["ngates"][1]

        w0_1d, w1_1d = _get_transition_weights(
            ng0,
            ng1,
            radar.ngates,
            mode=transition_weighting,
            sigmoid_steepness=8.0,
        )

    fields = {}
    for field_name in field_names:
        if not pulse_compression:
            if field_name in (
                "spectral_noise_power_hh_ADU",
                "spectral_noise_power_vv_ADU",
                "spectral_noise_power_hv_ADU",
                "spectral_noise_power_vh_ADU",
            ):
                field_data = get_spectral_noise(
                    radar, cpi_header, header, items, undo_txcorr=undo_txcorr
                )
            else:
                field_data = get_spectra_field(
                    radar,
                    filenames_psr,
                    cpi_header["npulses"],
                    header["items_per_file"],
                    items,
                    ind_rng,
                    fold=fold,
                    positive_away=positive_away,
                )
        else:
            weighted_sum = None
            weight_sum = None

            ind_rng_groups = [
                np.arange(0, ng0, dtype=int),
                np.arange(0, ng1, dtype=int),
            ]

            for i, (item_group, ind_rng_aux) in enumerate(zip(items, ind_rng_groups)):
                if field_name in (
                    "spectral_noise_power_hh_ADU",
                    "spectral_noise_power_vv_ADU",
                    "spectral_noise_power_hv_ADU",
                    "spectral_noise_power_vh_ADU",
                ):
                    field_part = get_spectral_noise(
                        radar,
                        cpi_header,
                        header,
                        item_group,
                        undo_txcorr=undo_txcorr,
                    )
                else:
                    field_part = get_spectra_field(
                        radar,
                        filenames_psr,
                        cpi_header["npulses"],
                        header["items_per_file"],
                        item_group,
                        ind_rng_aux,
                        fold=fold,
                        positive_away=positive_away,
                    )
                # ensure masked array
                field_part = np.ma.asarray(field_part)
                valid = ~np.ma.getmaskarray(field_part)
                filled = field_part.filled(field_part.dtype.type(0))
                if weighted_sum is None:
                    out_shape = list(field_part.shape)
                    out_shape[1] = radar.ngates
                    weighted_sum = np.zeros(out_shape, dtype=filled.dtype)
                    weight_sum = np.zeros(out_shape, dtype=np.float64)

                if i == 0:
                    # first spectrum goes directly to beginning of radial
                    out_idx = ind_rng_aux
                    weights_1d = w0_1d[out_idx]

                else:
                    # second spectrum is flipped to the end of the radial
                    out_idx = (radar.ngates - ind_rng_aux)[::-1] - 1
                    weights_1d = w1_1d[out_idx]

                # expand weights to (1, range, 1) for broadcasting
                weights = weights_1d[None, :, None]
                weighted_sum[:, out_idx, :] += (
                    filled[:, ind_rng_aux, :] * weights * valid[:, ind_rng_aux, :]
                )
                weight_sum[:, out_idx, :] += weights * valid[:, ind_rng_aux, :]

            valid_out = weight_sum > 0
            field_data = np.ma.masked_all(weighted_sum.shape, dtype=field_part.dtype)
            field_data[valid_out] = (
                weighted_sum[valid_out] / weight_sum[valid_out]
            ).astype(field_part.dtype)

        field_dict = filemetadata(field_name)
        field_dict["data"] = field_data
        fields[field_name] = field_dict

    if pulse_compression:
        items = np.sort([ii for item in items for ii in item])  # flatten items

    Doppler_velocity = filemetadata("Doppler_velocity")
    Doppler_frequency = filemetadata("Doppler_frequency")
    npulses = filemetadata("number_of_pulses")

    vel_data, freq_data = get_Doppler_info(
        cpi_header["prfs"][items],
        cpi_header["npulses"][items],
        header["states.rsplambda"] * 1e-2,
        fold=fold,
    )

    Doppler_velocity["data"] = vel_data
    Doppler_frequency["data"] = freq_data
    npulses["data"] = cpi_header["npulses"][items]

    # get further metadata
    pw_ind = header["states.spbpwidth"]
    if not hasattr(radar, "radar_calibration") or radar.radar_calibration is None:
        radar.radar_calibration = dict()

    dBADU_to_dBm_hh["data"] = np.array(
        [np.atleast_1d(header["states.spbdbmtologoffset"])[pw_ind]]
    )
    radar.radar_calibration.update({"dBADU_to_dBm_hh": dBADU_to_dBm_hh})

    dBADU_to_dBm_vv["data"] = np.array(
        [np.atleast_1d(header["states.spbdbmtologoffset"])[pw_ind]]
    )
    radar.radar_calibration.update({"dBADU_to_dBm_vv": dBADU_to_dBm_vv})

    if "states.gdrxmfloss" in header:
        mfloss_h["data"] = np.array([header["states.gdrxmfloss"][pw_ind]])
        mfloss_v["data"] = np.array([header["states.gdrxmfloss"][pw_ind]])
    elif "states.spbdphmflossforbatch" in header:
        mfloss_h["data"] = np.array([header["states.spbdphmflossforbatch"]])
        mfloss_v["data"] = np.array([header["states.spbdpvmflossforbatch"]])

    radar.radar_calibration.update({"matched_filter_loss_h": mfloss_h})
    radar.radar_calibration.update({"matched_filter_loss_v": mfloss_v})

    pathatt["data"] = np.array([header["states.rspathatt"]])
    radar.radar_calibration.update({"path_attenuation": pathatt})
    return RadarSpectra(
        radar.time,
        radar.range,
        fields,
        radar.metadata,
        radar.scan_type,
        radar.latitude,
        radar.longitude,
        radar.altitude,
        radar.sweep_number,
        radar.sweep_mode,
        radar.fixed_angle,
        radar.sweep_start_ray_index,
        radar.sweep_end_ray_index,
        radar.azimuth,
        radar.elevation,
        npulses,
        Doppler_velocity=Doppler_velocity,
        Doppler_frequency=Doppler_frequency,
        rays_are_indexed=radar.rays_are_indexed,
        ray_angle_res=radar.ray_angle_res,
        instrument_parameters=radar.instrument_parameters,
        radar_calibration=radar.radar_calibration,
    )


def read_psr_cpi_headers(filenames):
    """
    Reads the CPI data headers contained in multiple PSR files

    Parameters
    ----------
    filenames : list of str
        Name of the PSR files

    Returns
    -------
    cpi_header, header : dict
        Dictionary containing the PSR header data and the CPI headers data

    """
    cpi_header = {
        "azi_start": [],
        "azi_stop": [],
        "ele_start": [],
        "ele_stop": [],
        "npulses": [],
        "prfs": [],
        "ngates": [],
        "tx_pwr": [],
        "noise": [],
    }

    header = None
    for filename in filenames:
        cpi_header_aux, header_aux = read_psr_cpi_header(filename)
        if cpi_header_aux is None:
            warn("File " + filename + " could not be read")
            continue

        if header is None:
            header = header_aux
            header.update({"items_per_file": [header_aux["item.count"]]})
        else:
            header["item.count"] = header["item.count"] + header_aux["item.count"]
            header["items_per_file"].append(header_aux["item.count"])

        cpi_header["azi_start"].extend(cpi_header_aux["azi_start"])
        cpi_header["azi_stop"].extend(cpi_header_aux["azi_stop"])
        cpi_header["ele_start"].extend(cpi_header_aux["ele_start"])
        cpi_header["ele_stop"].extend(cpi_header_aux["ele_stop"])
        cpi_header["npulses"].extend(cpi_header_aux["npulses"])
        cpi_header["prfs"].extend(cpi_header_aux["prfs"])
        cpi_header["ngates"].extend(cpi_header_aux["ngates"])
        cpi_header["tx_pwr"].extend(cpi_header_aux["tx_pwr"])

        noise_aux = cpi_header_aux.get("noise")
        if cpi_header["noise"] is not None:
            if isinstance(noise_aux, np.ma.MaskedArray):
                value = noise_aux.item() if noise_aux.shape == () else noise_aux

                if value is None:
                    pass
                elif isinstance(value, np.ma.MaskedArray):
                    cpi_header["noise"].extend(value.compressed().tolist())
                else:
                    cpi_header["noise"].extend(np.atleast_1d(value).tolist())

            elif noise_aux is not None:
                cpi_header["noise"].extend(np.atleast_1d(noise_aux).tolist())
    cpi_header["azi_start"] = np.array(cpi_header["azi_start"])
    cpi_header["azi_stop"] = np.array(cpi_header["azi_stop"])
    cpi_header["ele_start"] = np.array(cpi_header["ele_start"])
    cpi_header["ele_stop"] = np.array(cpi_header["ele_stop"])
    cpi_header["npulses"] = np.array(cpi_header["npulses"])
    cpi_header["prfs"] = np.array(cpi_header["prfs"])
    cpi_header["ngates"] = np.array(cpi_header["ngates"])
    cpi_header["tx_pwr"] = np.ma.array(cpi_header["tx_pwr"])
    cpi_header["noise"] = (
        np.ma.array(cpi_header["noise"])
        if len(cpi_header["noise"])
        else cpi_header["tx_pwr"] + np.nan
    )
    header["items_per_file"] = np.array(header["items_per_file"])

    return cpi_header, header


def read_psr_header(filename):
    """
    Read a PSR file header.

    Parameters
    ----------
    filename : str
        Name of the PSR file

    Returns
    -------
    header : dict
        Dictionary containing the PSR header data

    """
    header = dict()
    try:
        with open(filename, newline=None, encoding="latin-1") as txtfile:
            # read first line: ARCHIVE_HEADER_START
            line = txtfile.readline()
            while 0 == 0:
                line = txtfile.readline()
                if "ARCHIVE_HEADER_END" in line:
                    break

                strings = line.split("=")
                key = strings[0]
                val = convert_data(strings[1].replace("\n", ""))
                header.update({key: val})

            # read noise
            line = txtfile.readline()
            if "PSR REDUCED" not in line:
                warn("Header does not contain noise data")
                return header

            noise = np.array([])
            while 0 == 0:
                line = txtfile.readline()
                strings = line.split()
                try:
                    if line.startswith("i"):  # old format
                        noise = np.append(noise, np.float32(strings[3].split("=")[1]))
                    else:
                        noise = np.append(noise, np.float32(strings[2]))
                except (IndexError, ValueError):
                    break
            header.update({"noise": noise})

            return header
    except OSError as ee:
        warn(str(ee))
        warn("Unable to read file " + filename)
        return header


def read_psr_cpi_header(filename):
    """
    Reads the CPI data headers contained in a PSR file

    Parameters
    ----------
    filename : str
        Name of the PSR file

    Returns
    -------
    cpi_header, header : dict
        Dictionary containing the PSR header data and the CPI headers data

    """
    # load PSR library
    psr_lib = get_library()

    c_float_p = ctypes.POINTER(ctypes.c_float)
    c_long_p = ctypes.POINTER(ctypes.c_long)

    header = read_psr_header(filename)
    nitems = header["item.count"]

    azi_start = np.empty(nitems, dtype=np.float32)
    azi_stop = np.empty(nitems, dtype=np.float32)
    ele_start = np.empty(nitems, dtype=np.float32)
    ele_stop = np.empty(nitems, dtype=np.float32)
    npulses = np.empty(nitems, dtype=np.int32)
    prfs = np.empty(nitems, dtype=np.float32)
    ngates = np.empty(nitems, dtype=np.int32)
    tx_pwr = np.empty(nitems, dtype=np.float32)

    c_filename = ctypes.c_char_p(filename.encode("utf-8"))

    try:
        psr_lib.psr_getValueArrays(
            c_filename,
            azi_start.ctypes.data_as(c_float_p),
            azi_stop.ctypes.data_as(c_float_p),
            ele_start.ctypes.data_as(c_float_p),
            ele_stop.ctypes.data_as(c_float_p),
            npulses.ctypes.data_as(c_long_p),
            prfs.ctypes.data_as(c_float_p),
            ngates.ctypes.data_as(c_long_p),
            tx_pwr.ctypes.data_as(c_float_p),
        )
    except OSError as ee:
        warn(str(ee))
        warn("Unable to read file " + filename)
        return None, None

    noise = None
    if "noise" in header:
        noise = np.ma.array(header["noise"]) * npulses

    cpi_header = {
        "azi_start": azi_start,
        "azi_stop": azi_stop,
        "ele_start": ele_start,
        "ele_stop": ele_stop,
        "npulses": npulses,
        "prfs": prfs,
        "ngates": ngates,
        "tx_pwr": np.ma.array(tx_pwr, dtype=float),
        "noise": np.ma.array(noise),
    }

    return cpi_header, header


def read_psr_spectra(filename):
    """
    Reads the complex spectral data contained in a PSR file

    Parameters
    ----------
    filename : str
        Name of the PSR file

    Returns
    -------
    spectra : 3D complex ndArray
       The complex spectra

    """
    # load PSR library
    psr_lib = get_library()

    c_filename = ctypes.c_char_p(filename.encode("utf-8"))
    c_complex_p = np.ctypeslib.ndpointer(dtype=np.complex64, ndim=1, flags="C")

    cpi_header, _ = read_psr_cpi_header(filename)
    if cpi_header is None:
        return None

    nitems = cpi_header["ngates"].size
    ngates = np.max(cpi_header["ngates"])
    npulses_max = np.max(cpi_header["npulses"])

    spectra = np.ma.masked_all((nitems, ngates, npulses_max), dtype=np.complex128)
    for item in range(nitems):
        npulses = cpi_header["npulses"][item]
        for gate in range(cpi_header["ngates"][item]):
            spectrum = np.empty(npulses, dtype=np.complex64)
            try:
                psr_lib.psr_getPowerSpectrum(
                    c_filename,
                    ctypes.c_int(item),
                    ctypes.c_int(gate),
                    spectrum.ctypes.data_as(c_complex_p),
                )
                spectra[item, gate, 0:npulses] = spectrum
            except OSError as ee:
                warn(str(ee))
                warn(
                    "Unable to get CPI element "
                    + str(item)
                    + " at range gate "
                    + str(gate)
                    + " from file "
                    + filename
                )

    return spectra


def _get_item_numbers_single_group(
    radar,
    azi_start,
    azi_stop,
    ele_start,
    ele_stop,
    prf_array,
    prfs,
    cpi="low_prf",
    ang_tol=0.5,
):
    """
    Internal helper: original get_item_numbers logic for a single CPI subset.
    """
    if radar.scan_type == "ppi":
        cpi_ang_center = azi_start + (azi_stop - azi_start) / 2.0
        cpi_fixed_angle = ele_start + (ele_stop - ele_start) / 2.0

        ref_ang = radar.azimuth["data"]
        fixed_ang = radar.elevation["data"]

    elif radar.scan_type == "rhi" or radar.scan_type == "other":
        cpi_ang_center = ele_start + (ele_stop - ele_start) / 2.0
        cpi_fixed_angle = azi_start + (azi_stop - azi_start) / 2.0

        ref_ang = radar.elevation["data"]
        fixed_ang = radar.azimuth["data"]

    else:
        raise ValueError(f"Unsupported scan type {radar.scan_type}")

    if prfs.size > 1 and cpi == "all":
        items = np.array([], dtype=int)
        sweep_start = radar.sweep_start_ray_index["data"]
        sweep_end = radar.sweep_end_ray_index["data"]
        items_per_sweep = np.array([], dtype=int)

        items_aux = np.arange(cpi_ang_center.size)
        for i, (s_start, s_end) in enumerate(zip(sweep_start, sweep_end)):
            fixed = radar.fixed_angle["data"][i]
            ind = np.where(np.abs(cpi_fixed_angle - fixed) < ang_tol)[0]

            cpi_ang_center_aux = cpi_ang_center[ind]
            items_aux2 = items_aux[ind]

            ind = np.where(
                np.logical_and(
                    cpi_ang_center_aux >= ref_ang[s_start] - ang_tol,
                    cpi_ang_center_aux <= ref_ang[s_end] + ang_tol,
                )
            )[0]

            items = np.append(items, items_aux2[ind])
            items_per_sweep = np.append(items_per_sweep, items_aux2[ind].size)

        radar = change_rays(
            radar, cpi_ang_center[items], cpi_fixed_angle[items], items_per_sweep
        )
        return items, radar

    items_aux = np.arange(cpi_ang_center.size)
    if prfs.size > 1:
        if cpi == "low_prf":
            mask = prf_array == prfs[0]
            cpi_ang_center = cpi_ang_center[mask]
            cpi_fixed_angle = cpi_fixed_angle[mask]
            items_aux = items_aux[mask]

        elif cpi == "intermediate_prf":
            if prfs.size == 3:
                mask = prf_array == prfs[1]
                cpi_ang_center = cpi_ang_center[mask]
                cpi_fixed_angle = cpi_fixed_angle[mask]
                items_aux = items_aux[mask]
            else:
                warn("Less than 3 different prfs. Low prf data will be used")
                mask = prf_array == prfs[0]
                cpi_ang_center = cpi_ang_center[mask]
                cpi_fixed_angle = cpi_fixed_angle[mask]
                items_aux = items_aux[mask]

        elif cpi == "high_prf":
            mask = prf_array == prfs[-1]
            cpi_ang_center = cpi_ang_center[mask]
            cpi_fixed_angle = cpi_fixed_angle[mask]
            items_aux = items_aux[mask]

        elif cpi == "mean":
            cpi_ang_center_aux = []
            cpi_fixed_angle_aux = []
            items_aux2 = []
            for prf in prfs:
                mask = prf_array == prf
                cpi_ang_center_aux.append(cpi_ang_center[mask])
                cpi_fixed_angle_aux.append(cpi_fixed_angle[mask])
                items_aux2.append(items_aux[mask])

            cpi_ang_center = cpi_ang_center_aux
            cpi_fixed_angle = cpi_fixed_angle_aux
            items_aux = items_aux2

    if prfs.size > 1 and cpi == "mean":
        items = []

        if radar.scan_type == "other":
            for j, prf in enumerate(prfs):
                cpi_fixed_angle_aux = cpi_fixed_angle[j]
                cpi_ang_center_aux = cpi_ang_center[j]
                items_aux2 = items_aux[j]

                ind = np.where(np.abs(cpi_fixed_angle_aux - fixed_ang[0]) < ang_tol)[0]
                if ind.size < radar.nrays:
                    return np.array([], dtype=int), radar
                items_aux4 = items_aux2[ind[: radar.nrays]]
                items.append(items_aux4)
            return items, radar

        for j, prf in enumerate(prfs):
            cpi_fixed_angle_aux = cpi_fixed_angle[j]
            cpi_ang_center_aux = cpi_ang_center[j]
            items_aux2 = items_aux[j]
            items_aux4 = np.array([], dtype=int)

            for i, ang in enumerate(ref_ang):
                fixed = fixed_ang[i]
                ind = np.where(np.abs(cpi_fixed_angle_aux - fixed) < ang_tol)[0]
                cpi_ang_center_aux2 = cpi_ang_center_aux[ind]
                items_aux3 = items_aux2[ind]

                if cpi_ang_center_aux2.size == 0:
                    continue

                ind2 = np.argmin(np.abs(cpi_ang_center_aux2 - ang))
                items_aux4 = np.append(items_aux4, items_aux3[ind2])

            items.append(items_aux4)

        return items, radar

    if radar.scan_type == "other":
        ind = np.where(np.abs(cpi_fixed_angle - fixed_ang[0]) < ang_tol)[0]
        if ind.size < radar.nrays:
            return np.array([], dtype=int), radar

        return items_aux[ind[: radar.nrays]], radar

    items = np.array([], dtype=int)
    for i, ang in enumerate(ref_ang):
        fixed = fixed_ang[i]
        ind = np.where(np.abs(cpi_fixed_angle - fixed) < ang_tol)[0]

        if ind.size == 0:
            continue

        cpi_ang_center_aux = cpi_ang_center[ind]
        items_aux2 = items_aux[ind]

        ind2 = np.argmin(np.abs(cpi_ang_center_aux - ang))
        items = np.append(items, items_aux2[ind2])

    return items, radar


def get_item_numbers(
    radar,
    azi_start,
    azi_stop,
    ele_start,
    ele_stop,
    prf_array,
    prfs,
    pulse_compression=False,
    ngates=None,
    cpi="low_prf",
    ang_tol=0.5,
):
    """
    Gets the item numbers to be used and eventually modify the radar object
    to accomodate more angles.

    If pulse_compression is True, the CPI entries are split by unique ngates and
    the selection is run independently for each gate group.

    Returns
    -------
    items : int array, list of int arrays, or list of lists
        - normal case: same behavior as before
        - pulse_compression=True:
            one item selection per unique ngates group
    radar : radar object
        possibly modified radar object
    """
    if pulse_compression:
        if ngates is None:
            warn(
                "Please provide number of gates for every CPI element when reading "
                "pulse compressed data"
            )
            return None

        ngates = np.asarray(ngates)
        if ngates.size != prf_array.size:
            raise ValueError(
                "ngates and prf_array must have the same length when "
                "pulse_compression=True"
            )

        unique_ngates = list(set(ngates))
        all_items = []
        radar_out = radar

        for gates in unique_ngates:
            mask = ngates == gates

            items_group, radar_out = _get_item_numbers_single_group(
                radar_out,
                azi_start[mask],
                azi_stop[mask],
                ele_start[mask],
                ele_stop[mask],
                prf_array[mask],
                np.unique(prf_array[mask]),
                cpi=cpi,
                ang_tol=ang_tol,
            )

            # map back to original CPI indices
            original_idx = np.where(mask)[0]

            if isinstance(items_group, list):
                # cpi == "mean": list of arrays
                items_group_global = [
                    original_idx[np.asarray(it)] for it in items_group
                ]
            else:
                items_group_global = original_idx[np.asarray(items_group)]

            all_items.append(items_group_global)

        return all_items, radar_out

    return _get_item_numbers_single_group(
        radar,
        azi_start,
        azi_stop,
        ele_start,
        ele_stop,
        prf_array,
        prfs,
        cpi=cpi,
        ang_tol=ang_tol,
    )


def get_field(
    radar, cpi_header, header, items, nprfs, field_name, undo_txcorr=True, cpi="low_prf"
):
    """
    Gets the field corresponding to the reference radar

    Parameters
    ----------
    radar: radar object
        the reference radar
    cpi_header, header : dict
        dictionaries containing the PSR file header and CPI headers data
    items : int array
        array containing the items to select
    nprfs: float
        The number of different prfs in the file
    field_name : str
        The name of the field to filter
    undo_txcorr : bool
        If True and field is a noise field the correction of the received
        signal by the transmitted power is undone
    cpi : str
        The CPI to use. Can be 'low_prf', 'intermediate_prf', 'high_prf',
        'all'. If 'all' the mean within the angle step is taken

    Returns
    -------
    field_data : 2D float array
        The PSR data in the format of the reference radar fields

    """

    if "noise" in field_name:
        field = cpi_header["noise"]
        if undo_txcorr:
            if "states.gdrxmaxpowerkwpw" in header:  # old systems
                maxpow = header.get("states.gdrxmaxpowerkwpw")[
                    header["states.spbpwidth"]
                ]
            else:
                maxpow = header.get("states.gdrx5caltxpowkwpwh")
            field[cpi_header["tx_pwr"] > 0.0] *= (
                cpi_header["tx_pwr"][cpi_header["tx_pwr"] > 0.0] / maxpow
            )
    elif field_name in ("transmitted_signal_power_h", "transmitted_signal_power_v"):
        field = cpi_header["tx_pwr"]

    if nprfs > 1 and cpi == "mean":
        field_filt = np.ma.masked_all((radar.nrays, nprfs))
        npulses_filt = np.ma.masked_all((radar.nrays, nprfs), dtype=int)
        npulses = cpi_header["npulses"]
        for i in range(nprfs):
            items_aux = items[i]
            for j, item in enumerate(items_aux):
                field_filt[j, i] = field[item] * npulses[item]
                npulses_filt[j, i] = npulses[item]
        field_filt = np.transpose(
            np.atleast_2d(
                np.ma.sum(field_filt, axis=-1) / np.ma.sum(npulses_filt, axis=-1)
            )
        )
    else:
        field_filt = np.ma.masked_all((radar.nrays, 1))
        for i, item in enumerate(items):
            field_filt[i, 0] = field[item]

    field_data = ma_broadcast_to(field_filt, (radar.nrays, radar.ngates))

    return field_data


def get_spectral_noise(radar, cpi_header, header, items, undo_txcorr=True):
    """
    Gets the field corresponding to the reference radar

    Parameters
    ----------
    radar: radar object
        the reference radar
    cpi_header, header : dict
        dictionaries containing the PSR file header and CPI headers data
    items : int array
        array containing the items to select
    undo_txcorr : bool
        If True and field is a noise field the correction of the received
        signal by the transmitted power is undone

    Returns
    -------
    field_data : 2D float array
        The PSR data in the format of the reference radar fields

    """
    field = cpi_header["noise"] / cpi_header["npulses"]
    if undo_txcorr:
        if "states.gdrxmaxpowerkwpw" in header:  # old systems
            maxpow = header.get("states.gdrxmaxpowerkwpw")[header["states.spbpwidth"]]
        else:
            maxpow = header.get("states.gdrx5caltxpowkwpwh")

        field[cpi_header["tx_pwr"] > 0.0] *= (
            cpi_header["tx_pwr"][cpi_header["tx_pwr"] > 0.0] / maxpow
        )

    field_filt = np.ma.masked_all((radar.nrays, 1, 1))
    for i, item in enumerate(items):
        field_filt[i, 0, 0] = field[item]

    npulses_max = np.max(cpi_header["npulses"][items])
    field_data = ma_broadcast_to(field_filt, (radar.nrays, radar.ngates, npulses_max))

    return field_data


def get_spectra_field(
    radar,
    filenames,
    npulses,
    items_per_file,
    items,
    ind_rng,
    fold=True,
    positive_away=True,
):
    """
    Gets the field corresponding to the reference radar


    Parameters
    ----------
    radar: radar object
        the reference radar
    filename : str
        name of the PSR file
    npulses : int array
        array containing the number of pulses for each item
    items_per_file : int array
        array containing the number of items in each PSR file
    items : int array
        array containing the items to select
    ind_rng : int array
        array containing the indices to the range gates to select
    fold : Bool
        If True the spectra is folded
    positive_away : Bool
        If True positive Doppler velocities are way from the radar

    Returns
    -------
    spectra : 3D complex float array
        The complex spectra field

    """
    # load PSR library
    psr_lib = get_library()

    c_complex_p = np.ctypeslib.ndpointer(dtype=np.complex64, ndim=1, flags="C")

    npulses_max = np.max(npulses[items])

    spectra = np.ma.masked_all(
        (radar.nrays, radar.ngates, npulses_max), dtype=np.complex64
    )
    accu_items = np.cumsum(items_per_file)
    for ray, item in enumerate(items):
        # get file to read and item number within file
        ind = np.where(accu_items >= item)[0]
        if ind.size == 0 or ind[0] == 0:
            item_aux = item
            ind = 0
        else:
            ind = ind[0]
            item_aux = item - accu_items[ind - 1]

        c_filename = ctypes.c_char_p(filenames[ind].encode("utf-8"))
        for rng, gate in enumerate(ind_rng):
            npulses_item = npulses[item]
            spectrum = np.empty(npulses_item, dtype=np.complex64)
            try:
                psr_lib.psr_getPowerSpectrum(
                    c_filename,
                    ctypes.c_int(item_aux),
                    ctypes.c_int(gate),
                    spectrum.ctypes.data_as(c_complex_p),
                )
                if fold:
                    nfold = int(np.ceil(npulses_item / 2.0))
                    spectrum = np.append(spectrum[nfold:], spectrum[0:nfold])
                if positive_away:
                    spectrum = spectrum[::-1]

                spectra[ray, rng, 0:npulses_item] = spectrum
            except OSError as ee:
                warn(str(ee))
                warn(
                    "Unable to get CPI element "
                    + str(item)
                    + " at range gate "
                    + str(gate)
                    + " from file "
                    + filenames[ind]
                )

    spectra = np.ma.masked_equal(spectra, 0.0)

    return spectra


def get_Doppler_info(prfs, npulses, wavelength, fold=True):
    """
    Gets the Doppler information


    Parameters
    ----------
    prfs: float array
        the PRF at each ray
    npulses : float array
        the number of pulses per ray
    wavelength : float
        the radar wavelength [m]
    fold : Bool
        If True the spectra is folded

    Returns
    -------
    Doppler_velocity, Doppler_frequency : 2D float array
        The Doppler velocity and Doppler frequency bins for each ray

    """
    nrays = npulses.size
    npulses_max = np.max(npulses)
    freq_res = prfs / npulses
    vel_res = freq_res * wavelength / 2.0
    Doppler_frequency = np.ma.masked_all((nrays, npulses_max))
    Doppler_velocity = np.ma.masked_all((nrays, npulses_max))
    for ray in range(nrays):
        pulses_ray = np.arange(npulses[ray])
        npulses_ray = pulses_ray.size
        if fold:
            nfold = int(np.ceil(npulses_ray / 2.0))
            pulses_ray = np.append(
                pulses_ray[nfold:] - npulses_ray, pulses_ray[0:nfold]
            )

        Doppler_frequency[ray, 0:npulses_ray] = pulses_ray * freq_res[ray]
        Doppler_velocity[ray, 0:npulses_ray] = pulses_ray * vel_res[ray]

    return Doppler_velocity, Doppler_frequency


def get_noise_field(radar, field_data, header, field_name):
    """
    Puts the noise field in the desired units

    Parameters
    ----------
    radar: radar object
        the reference radar
    field_data : 2D float array
        The PSR data in the format of the reference radar fields
    header : dict
        Dictionary containing the PSR file metadata
    field_name : str
        The name of the field

    Returns
    -------
    field_data : 2D float array
        The PSR data in the format of the reference radar fields

    """
    field_data = 10.0 * np.ma.log10(field_data)

    if field_name in ("noisedBADU_hh", "noisedBADU_vv"):
        return field_data

    pw_ind = header["states.spbpwidth"]
    if field_name in ("noisedBZ_hh", "noisedBm_hh"):
        dBadu2dBm = np.atleast_1d(header["states.spbdbmtologoffset"])[pw_ind]
    else:
        dBadu2dBm = np.atleast_1d(header["states.spbdpvdbmtologoffset"])[pw_ind]

    field_data = field_data + dBadu2dBm

    if field_name in ("noisedBm_hh", "noisedBm_vv"):
        return field_data

    if field_name == "noisedBZ_hh":
        radconst = np.atleast_1d(header["states.rspdphradconst"])[pw_ind]
    else:
        radconst = np.atleast_1d(header["states.rspdpvradconst"])[pw_ind]
    if "states.gdrxmfloss" in header:
        mfloss = np.array([header["states.gdrxmfloss"][pw_ind]])
    elif "states.spbdphmflossforbatch" in header:
        mfloss = np.array([header["states.spbdphmflossforbatch"]])

    pathatt = header["states.rspathatt"]

    rangeKm = ma_broadcast_to(
        np.atleast_2d(radar.range["data"] / 1000.0), (radar.nrays, radar.ngates)
    )

    field_data += radconst + mfloss + pathatt * rangeKm + 20.0 * np.log10(rangeKm)

    return field_data


def convert_data(values):
    """
    Converts an string of values into the corresponding format

    Parameters
    ----------
    values: str
        string containg the values to convert

    Returns
    -------
    values : int, float, str or 1D array of int, float or str
        The converted values

    """
    if " " in values[1:-1]:
        values = values.split()
        try:
            return np.array(values, dtype=int)
        except ValueError:
            try:
                return np.array(values, dtype=float)
            except ValueError:
                return np.array(values, dtype=str)

    try:
        return int(values)
    except ValueError:
        try:
            return float(values)
        except ValueError:
            return values


def get_library():
    """
    return the link to C-shared library

    Returns
    -------
    psr_lib : link
        loaded PSR C-library

    """
    try:
        library_path = get_library_path()
    except MissingOptionalDependency:
        raise MissingOptionalDependency(" PSR library path NOT defined")

    try:
        psr_lib = ctypes.cdll.LoadLibrary(library_path + "/" + "libDX50.so")
    except OSError as ee:
        warn(str(ee))
        raise MissingOptionalDependency("Unable to load PSR library")

    return psr_lib


def get_library_path():
    """
    find valid library path

    Returns
    -------
    psr_lib_path : str
        library path

    """
    # Check if path is correct
    library_paths = [
        os.environ.get("PSRLIB_PATH"),
        os.path.expanduser("~") + "/pyrad/src/libDX50/lib/",
    ]
    library_path = ""
    for p in library_paths:
        if p is not None:
            if os.path.isdir(p):
                library_path = p
                break

    if not library_path:
        raise MissingOptionalDependency(" PSR library path NOT defined")

    return library_path


def change_rays(radar, moving_angle, fixed_angle, rays_per_sweep):
    """
    Modify the radar object to accomodate new rays

    Parameters
    ----------
    radar : radar object
        the radar to modify
    moving_angle : float array
        The moving angles
    fixed_angle : float array
        The fixed angles
    rays_per_sweep : array of ints
        The number of rays per sweep

    Returns
    -------
    new_radar : radar object
        The modified radar

    """
    new_radar = deepcopy(radar)
    new_radar.nrays = moving_angle.size
    if radar.scan_type == "ppi":
        new_radar.azimuth["data"] = moving_angle
        new_radar.elevation["data"] = fixed_angle
    elif radar.scan_type == "rhi":
        new_radar.azimuth["data"] = fixed_angle
        new_radar.elevation["data"] = moving_angle

    ray_factor = int(new_radar.nrays / radar.nrays)

    # change time
    time_res = np.append(
        (radar.time["data"][1:] - radar.time["data"][:-1]) / ray_factor,
        (radar.time["data"][-1] - radar.time["data"][-2]) / ray_factor,
    )
    time_sum = np.cumsum(
        np.reshape(np.repeat(time_res, ray_factor), (radar.nrays, ray_factor)), axis=1
    ).flatten()
    new_radar.time["data"] = np.repeat(radar.time["data"], ray_factor) + time_sum

    new_radar.sweep_start_ray_index["data"] = np.append(
        0, np.cumsum(rays_per_sweep[:-1])
    )
    new_radar.sweep_end_ray_index["data"] = np.cumsum(rays_per_sweep - 1)
    new_radar.init_rays_per_sweep()
    if new_radar.ray_angle_res is not None:
        new_radar.ray_angle_res["data"] /= ray_factor

    new_radar.init_gate_x_y_z()
    new_radar.init_gate_longitude_latitude()
    new_radar.init_gate_altitude()

    return new_radar


def _get_transition_weights(
    ng0,
    ng1,
    radar_ngates,
    mode="mean",
    sigmoid_steepness=8.0,
):
    """
    Compute 1D transition weights for two spectra combined along range.

    Parameters
    ----------
    ng0 : int
        Number of gates of first spectrum (placed at beginning of radial).
    ng1 : int
        Number of gates of second spectrum (placed at end of radial, flipped).
    radar_ngates : int
        Total number of gates in output radar.
    mode : str
        'mean', 'linear', 'sigmoid', 'only_compressed' or 'only_uncompressed'
    sigmoid_steepness : float
        Steepness of sigmoid if mode='sigmoid'

    Returns
    -------
    w0 : ndarray of shape (radar_ngates,)
        Weights for first spectrum
    w1 : ndarray of shape (radar_ngates,)
        Weights for second spectrum
    """
    w0 = np.zeros(radar_ngates, dtype=np.float64)
    w1 = np.zeros(radar_ngates, dtype=np.float64)

    # coverage in output coordinates
    ind0 = np.arange(0, ng0, dtype=int)
    ind1 = np.arange(radar_ngates - ng1, radar_ngates, dtype=int)

    # outside overlap: full contribution where only one spectrum exists
    w0[ind0] = 1.0
    w1[ind1] = 1.0

    overlap_start = max(0, radar_ngates - ng1)
    overlap_stop = min(ng0, radar_ngates)

    if overlap_stop <= overlap_start:
        return w0, w1

    overlap = np.arange(overlap_start, overlap_stop, dtype=int)
    n_overlap = overlap.size

    if mode == "mean":
        trans0 = np.full(n_overlap, 0.5, dtype=np.float64)
        trans1 = 1.0 - trans0

    elif mode == "linear":
        if n_overlap == 1:
            trans0 = np.array([0.5], dtype=np.float64)
        else:
            trans0 = np.linspace(1.0, 0.0, n_overlap, dtype=np.float64)
        trans1 = 1.0 - trans0

    elif mode == "sigmoid":
        if n_overlap == 1:
            trans0 = np.array([0.5], dtype=np.float64)
        else:
            x = np.linspace(-1.0, 1.0, n_overlap, dtype=np.float64)
            s = 1.0 / (1.0 + np.exp(sigmoid_steepness * x))
            # normalize exactly to [0,1] endpoints as well as possible
            trans0 = (s - s.min()) / (s.max() - s.min())
        trans1 = 1.0 - trans0
    elif mode == "only_uncompressed":
        trans0 = np.full(n_overlap, 1, dtype=np.float64)
        trans1 = trans1 = 1.0 - trans0
    elif mode == "only_compressed":
        trans0 = np.full(n_overlap, 0, dtype=np.float64)
        trans1 = trans1 = 1.0 - trans0
    else:
        raise ValueError(
            f"Unknown transition mode '{mode}'. Use 'mean', 'linear', or 'sigmoid'."
        )

    w0[overlap] = trans0
    w1[overlap] = trans1

    return w0, w1
