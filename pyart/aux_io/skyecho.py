"""
Utilities for reading SkyEcho files.

"""

import os
import warnings

import netCDF4
import numpy as np
from netCDF4 import num2date

from ..config import FileMetadata
from ..core.radar import Radar
from ..io.cfradial import (
    _find_all_meta_group_vars,
    _ncvar_to_dict,
    _unpack_variable_gate_field_dic,
    write_cfradial,
)
from ..io.common import _test_arguments

# Variables and dimensions in the instrument_parameter convention and
# radar_parameters sub-convention that will be read from and written to
# CfRadial files using Py-ART.
# The meta_group attribute cannot be used to identify these parameters as
# it is often set incorrectly.
_INSTRUMENT_PARAMS_DIMS = {
    # instrument_parameters sub-convention
    "frequency": ("frequency"),
    "follow_mode": ("sweep", "string_length"),
    "pulse_width": ("time",),
    "prt_mode": ("sweep", "string_length"),
    "prt": ("time",),
    "prt_ratio": ("time",),
    "polarization_mode": ("sweep", "string_length"),
    "nyquist_velocity": ("time",),
    "unambiguous_range": ("time",),
    "n_samples": ("time",),
    "sampling_ratio": ("time",),
    # radar_parameters sub-convention
    "radar_antenna_gain_h": (),
    "radar_antenna_gain_v": (),
    "radar_beam_width_h": (),
    "radar_beam_width_v": (),
    "radar_receiver_bandwidth": (),
    "radar_measured_transmit_power_h": ("time",),
    "radar_measured_transmit_power_v": ("time",),
    "radar_rx_bandwidth": (),  # non-standard
    "measured_transmit_power_v": ("time",),  # non-standard
    "measured_transmit_power_h": ("time",),  # non-standard
}


def read_skyecho(
    filename,
    sweep_end_time=None,
    field_names=None,
    additional_metadata=None,
    file_field_names=False,
    exclude_fields=None,
    include_fields=None,
    **kwargs,
):
    """
    Read a SkyEcho netCDF file.

    Parameters
    ----------
    filename : str
        Name of netCDF file to read data from.
    sweep_end_time : datetime object
        The end time of the sweep to read. If None the first sweep in the file
        will be read.
    field_names : dict, optional
        Dictionary mapping field names in the file names to radar field names.
        Unlike other read functions, fields not in this dictionary or having a
        value of None are still included in the radar.fields dictionary, to
        exclude them use the `exclude_fields` parameter. Fields which are
        mapped by this dictionary will be renamed from key to value.
    additional_metadata : dict of dicts, optional
        This parameter is not used, it is included for uniformity.
    file_field_names : bool, optional
        True to force the use of the field names from the file in which
        case the `field_names` parameter is ignored. False will use to
        `field_names` parameter to rename fields.
    exclude_fields : list or None, optional
        List of fields to exclude from the radar object. This is applied
        after the `file_field_names` and `field_names` parameters. Set
        to None to include all fields specified by include_fields.
    include_fields : list or None, optional
        List of fields to include from the radar object. This is applied
        after the `file_field_names` and `field_names` parameters. Set
        to None to include all fields not specified by exclude_fields.

    Returns
    -------
    radar : Radar
        Radar object.

    Notes
    -----
    This function has not been tested on "stream" Cfradial files.

    """
    # test for non empty kwargs
    _test_arguments(kwargs)

    # create metadata retrieval object
    filemetadata = FileMetadata(
        "cfradial", field_names, additional_metadata, file_field_names, exclude_fields
    )

    # read the data
    ncobj = netCDF4.Dataset(filename, "r")
    ncvars = ncobj.variables

    # determine number of sweeps in file
    azi = ncvars["azimuth"][:] * 180.0 / np.pi
    sweep_start, sweep_end = _get_sweep_limits(azi)

    # select which sweep to read
    epoch_unix_units = "seconds since 1970-01-01T00:00:00Z"
    end_time = num2date(ncvars["utc_unixtimestamp"][sweep_end], epoch_unix_units)
    ind_sweep = 0
    if sweep_end_time is not None:
        ind_sweep = np.where(end_time >= sweep_end_time)[0][0]

    ind_sweep_start = sweep_start[ind_sweep]
    ind_sweep_end = sweep_end[ind_sweep]

    # 4.1 Global attribute -> move to metadata dictionary
    metadata = {k: getattr(ncobj, k) for k in ncobj.ncattrs()}
    if "n_gates_vary" in metadata:
        metadata["n_gates_vary"] = "false"  # corrected below

    # 4.2 Dimensions (do nothing)

    # 4.3 Global variable -> move to metadata dictionary
    if "volume_number" in ncvars:
        if np.ma.isMaskedArray(ncvars["volume_number"][:]):
            metadata["volume_number"] = int(
                np.ma.getdata(ncvars["volume_number"][:].flatten())
            )
        else:
            metadata["volume_number"] = int(ncvars["volume_number"][:])
    else:
        metadata["volume_number"] = 0

    global_vars = {
        "platform_type": "fixed",
        "instrument_type": "radar",
        "primary_axis": "axis_z",
    }
    # ignore time_* global variables, these are calculated from the time
    # variable when the file is written.
    for var, default_value in global_vars.items():
        if var in ncvars:
            metadata[var] = str(netCDF4.chartostring(ncvars[var][:]))
        else:
            metadata[var] = default_value

    # 4.4 coordinate variables -> create attribute dictionaries
    time = filemetadata("time")
    time["units"] = epoch_unix_units
    time["data"] = ncvars["utc_unixtimestamp"][ind_sweep_start : ind_sweep_end + 1]
    _range = _ncvar_to_dict(ncvars["range"])

    # 4.5 Ray dimension variables

    # 4.6 Location variables -> create attribute dictionaries
    latitude = _ncvar_to_dict(ncvars["latitude"])
    longitude = _ncvar_to_dict(ncvars["longitude"])
    altitude = _ncvar_to_dict(ncvars["altitude"])
    if "altitude_agl" in ncvars:
        altitude_agl = _ncvar_to_dict(ncvars["altitude_agl"])
    else:
        altitude_agl = None

    # 4.8 Sensor pointing variables -> create attribute dictionaries
    azimuth = filemetadata("azimuth")
    azimuth["data"] = azi[ind_sweep_start : ind_sweep_end + 1]

    elevation = filemetadata("elevation")
    ele = ncvars["elevation"][ind_sweep_start : ind_sweep_end + 1] * 180.0 / np.pi
    elevation["data"] = ele
    if "scan_rate" in ncvars:
        scan_rate = _ncvar_to_dict(ncvars["scan_rate"])
    else:
        scan_rate = None

    if "antenna_transition" in ncvars:
        antenna_transition = _ncvar_to_dict(ncvars["antenna_transition"])
    else:
        antenna_transition = None

    # 4.7 Sweep variables -> create atrribute dictionaries
    sweep_mode = filemetadata("sweep_mode")
    sweep_mode["data"] = np.array(["ppi"])
    fixed_angle = filemetadata("fixed_angle")
    fixed_angle["data"] = np.array([ele[0]])
    sweep_start_ray_index = filemetadata("sweep_start_ray_index")
    sweep_start_ray_index["data"] = np.array([0])
    sweep_end_ray_index = filemetadata("sweep_end_ray_index")
    sweep_end_ray_index["data"] = np.array([ele.size - 1])

    if "sweep_number" in ncvars:
        sweep_number = _ncvar_to_dict(ncvars["sweep_number"])
    else:
        nsweeps = len(sweep_start_ray_index["data"])
        sweep_number = filemetadata("sweep_number")
        sweep_number["data"] = np.arange(nsweeps, dtype="float32")
        warnings.warn(
            "Warning: File violates CF/Radial convention. "
            + "Missing sweep_number variable"
        )

    if "target_scan_rate" in ncvars:
        target_scan_rate = _ncvar_to_dict(ncvars["target_scan_rate"])
    else:
        target_scan_rate = None
    if "rays_are_indexed" in ncvars:
        rays_are_indexed = _ncvar_to_dict(ncvars["rays_are_indexed"])
    else:
        rays_are_indexed = None
    if "ray_angle_res" in ncvars:
        ray_angle_res = _ncvar_to_dict(ncvars["ray_angle_res"])
    else:
        ray_angle_res = None

    scan_type = "ppi"

    # 4.9 Moving platform geo-reference variables
    # Aircraft specific varaibles
    if "rotation" in ncvars:
        rotation = _ncvar_to_dict(ncvars["rotation"])
    else:
        rotation = None

    if "tilt" in ncvars:
        tilt = _ncvar_to_dict(ncvars["tilt"])
    else:
        tilt = None

    if "roll" in ncvars:
        roll = _ncvar_to_dict(ncvars["roll"])
    else:
        roll = None

    if "drift" in ncvars:
        drift = _ncvar_to_dict(ncvars["drift"])
    else:
        drift = None

    if "heading" in ncvars:
        heading = _ncvar_to_dict(ncvars["heading"])
    else:
        heading = None

    if "pitch" in ncvars:
        pitch = _ncvar_to_dict(ncvars["pitch"])
    else:
        pitch = None

    if "georefs_applied" in ncvars:
        georefs_applied = _ncvar_to_dict(ncvars["georefs_applied"])
    else:
        georefs_applied = None

    # 4.10 Moments field data variables -> field attribute dictionary
    if "ray_n_gates" in ncvars:
        # all variables with dimensions of n_points are fields.
        keys = [k for k, v in ncvars.items() if v.dimensions == ("n_points",)]
    elif metadata["title"] == "Level2 rainfall rate file":
        # all variables with dimensions of 'azimuth', 'range' are fields
        keys = [k for k, v in ncvars.items() if v.dimensions == ("azimuth", "range")]
    else:
        # all variables with dimensions of 'time', 'range' are fields
        keys = [k for k, v in ncvars.items() if v.dimensions == ("time", "range")]

    fields = {}
    for key in keys:
        field_name = filemetadata.get_field_name(key)
        if field_name is None:
            if exclude_fields is not None and key in exclude_fields:
                if key not in include_fields:
                    continue
            if include_fields is None or key in include_fields:
                field_name = key
            else:
                continue
        fields[field_name] = _ncvar_to_dict(ncvars[key], lazydict=True)
        fields[field_name]["data"] = ncvars[key][ind_sweep_start : ind_sweep_end + 1, :]
        if fields[field_name]["units"] == "radians":
            fields[field_name]["data"] *= 180 / np.pi
            fields[field_name]["units"] = "deg"

    if "ray_n_gates" in ncvars:
        shape = (len(ncvars["time"]), len(ncvars["range"]))
        ray_n_gates = ncvars["ray_n_gates"][:]
        ray_start_index = ncvars["ray_start_index"][:]
        for dic in fields.values():
            _unpack_variable_gate_field_dic(dic, shape, ray_n_gates, ray_start_index)

    # 4.5 instrument_parameters sub-convention -> instrument_parameters dict
    # 4.6 radar_parameters sub-convention -> instrument_parameters dict
    keys = [k for k in _INSTRUMENT_PARAMS_DIMS.keys() if k in ncvars]
    instrument_parameters = {k: _ncvar_to_dict(ncvars[k]) for k in keys}
    if instrument_parameters == {}:  # if no parameters set to None
        instrument_parameters = None

    # 4.7 lidar_parameters sub-convention -> skip

    # 4.8 radar_calibration sub-convention -> radar_calibration
    keys = _find_all_meta_group_vars(ncvars, "radar_calibration")
    radar_calibration = {k: _ncvar_to_dict(ncvars[k]) for k in keys}
    if radar_calibration == {}:
        radar_calibration = None

    ncobj.close()

    return Radar(
        time,
        _range,
        fields,
        metadata,
        scan_type,
        latitude,
        longitude,
        altitude,
        sweep_number,
        sweep_mode,
        fixed_angle,
        sweep_start_ray_index,
        sweep_end_ray_index,
        azimuth,
        elevation,
        instrument_parameters=instrument_parameters,
        radar_calibration=radar_calibration,
        altitude_agl=altitude_agl,
        scan_rate=scan_rate,
        antenna_transition=antenna_transition,
        target_scan_rate=target_scan_rate,
        rays_are_indexed=rays_are_indexed,
        ray_angle_res=ray_angle_res,
        rotation=rotation,
        tilt=tilt,
        roll=roll,
        drift=drift,
        heading=heading,
        pitch=pitch,
        georefs_applied=georefs_applied,
    )


def extract_sweeps_skyecho(
    filename,
    basepath=None,
    sweep_dir=None,
    dir_timeformat="%Y/%m/%d/",
    file_prefix=None,
    file_sufix=None,
    file_timeformat="%Y-%m-%dT%H-%M-%S",
    sweep_start_time=None,
    sweep_end_time=None,
    field_names=None,
    additional_metadata=None,
    file_field_names=False,
    exclude_fields=None,
    include_fields=None,
    verbose=False,
    **kwargs,
):
    """
    Read a SkyEcho netCDF file and create new CF/Radial files with single
    sweeps

    Parameters
    ----------
    filename : str
        Name of netCDF file to read data from.
    basepath : str or None
        Base path where to store the extracted sweep files. If None it is
        stored in the current directory
    sweep_dir : str or None
        Directory where to store the extracted sweep files. If None it is
        stored on the base path
    dir_timeformat : str
        Time format for the directory
    file_prefix : str or None
        prefix to add to the name of the file containing a single sweep
    file_sufix : str or None
        sufix to add to the name of the file containing a single sweep
    file_timeformat :str
        Time format of the file name
    sweep_start_time, sweep_end_time : datetime object
        The start and the end period where we want to extract sweeps from.
    field_names : dict, optional
        Dictionary mapping field names in the file names to radar field names.
        Unlike other read functions, fields not in this dictionary or having a
        value of None are still included in the radar.fields dictionary, to
        exclude them use the `exclude_fields` parameter. Fields which are
        mapped by this dictionary will be renamed from key to value.
    additional_metadata : dict of dicts, optional
        This parameter is not used, it is included for uniformity.
    file_field_names : bool, optional
        True to force the use of the field names from the file in which
        case the `field_names` parameter is ignored. False will use to
        `field_names` parameter to rename fields.
    exclude_fields : list or None, optional
        List of fields to exclude from the radar object. This is applied
        after the `file_field_names` and `field_names` parameters. Set
        to None to include all fields specified by include_fields.
    include_fields : list or None, optional
        List of fields to include from the radar object. This is applied
        after the `file_field_names` and `field_names` parameters. Set
        to None to include all fields not specified by exclude_fields.
    verbose : bool
        Be verbose about processes

    Returns
    -------
    radar : Radar
        Radar object.

    Notes
    -----
    This function has not been tested on "stream" Cfradial files.

    """
    # test for non empty kwargs
    _test_arguments(kwargs)

    # create metadata retrieval object
    filemetadata = FileMetadata(
        "cfradial", field_names, additional_metadata, file_field_names, exclude_fields
    )

    # read the data
    if verbose:
        print(f"opening file {filename}")
    ncobj = netCDF4.Dataset(filename, "r")
    ncvars = ncobj.variables

    if verbose:
        print(f"file {filename} opened")

    # determine number of sweeps in file
    if verbose:
        print("checking the number of sweeps in the file")
    azi = ncvars["azimuth"][:] * 180.0 / np.pi
    if verbose:
        print(f"number of azimuths {azi.size}")
    sweep_start, sweep_end = _get_sweep_limits(azi)
    if verbose:
        print(f"Number of sweeps in file: {sweep_start.size}")

    # select interval of sweeps to read
    epoch_unix_units = "seconds since 1970-01-01T00:00:00Z"

    start_time = num2date(ncvars["utc_unixtimestamp"][sweep_start], epoch_unix_units)
    ind_first_sweep = 0
    if sweep_start_time is not None:
        ind_first_sweep = np.where(start_time >= sweep_start_time)[0][0]

    end_time = num2date(ncvars["utc_unixtimestamp"][sweep_end], epoch_unix_units)
    ind_last_sweep = sweep_start.size - 1
    if sweep_end_time is not None:
        ind_last_sweep = np.where(end_time >= sweep_end_time)[0][0]

    if verbose:
        print(f"Number of sweeps to read: {ind_last_sweep - ind_first_sweep+1}")

    # 4.1 Global attribute -> move to metadata dictionary
    metadata = {k: getattr(ncobj, k) for k in ncobj.ncattrs()}
    if "n_gates_vary" in metadata:
        metadata["n_gates_vary"] = "false"  # corrected below

    # 4.2 Dimensions (do nothing)

    # 4.3 Global variable -> move to metadata dictionary
    if "volume_number" in ncvars:
        if np.ma.isMaskedArray(ncvars["volume_number"][:]):
            metadata["volume_number"] = int(
                np.ma.getdata(ncvars["volume_number"][:].flatten())
            )
        else:
            metadata["volume_number"] = int(ncvars["volume_number"][:])
    else:
        metadata["volume_number"] = 0

    global_vars = {
        "platform_type": "fixed",
        "instrument_type": "radar",
        "primary_axis": "axis_z",
    }
    # ignore time_* global variables, these are calculated from the time
    # variable when the file is written.
    for var, default_value in global_vars.items():
        if var in ncvars:
            metadata[var] = str(netCDF4.chartostring(ncvars[var][:]))
        else:
            metadata[var] = default_value

    # 4.4 coordinate variables -> create attribute dictionaries
    time = filemetadata("time")
    time["units"] = epoch_unix_units
    _range = _ncvar_to_dict(ncvars["range"])

    # 4.5 Ray dimension variables

    # 4.6 Location variables -> create attribute dictionaries
    latitude = _ncvar_to_dict(ncvars["latitude"])
    longitude = _ncvar_to_dict(ncvars["longitude"])
    altitude = _ncvar_to_dict(ncvars["altitude"])
    if "altitude_agl" in ncvars:
        altitude_agl = _ncvar_to_dict(ncvars["altitude_agl"])
    else:
        altitude_agl = None

    # 4.8 Sensor pointing variables -> create attribute dictionaries
    azimuth = filemetadata("azimuth")
    elevation = filemetadata("elevation")

    if "scan_rate" in ncvars:
        scan_rate = _ncvar_to_dict(ncvars["scan_rate"])
    else:
        scan_rate = None

    if "antenna_transition" in ncvars:
        antenna_transition = _ncvar_to_dict(ncvars["antenna_transition"])
    else:
        antenna_transition = None

    # 4.7 Sweep variables -> create atrribute dictionaries
    sweep_mode = filemetadata("sweep_mode")
    sweep_mode["data"] = np.array(["ppi"])
    fixed_angle = filemetadata("fixed_angle")
    sweep_start_ray_index = filemetadata("sweep_start_ray_index")
    sweep_start_ray_index["data"] = np.array([0])
    sweep_end_ray_index = filemetadata("sweep_end_ray_index")

    if "sweep_number" in ncvars:
        sweep_number = _ncvar_to_dict(ncvars["sweep_number"])
    else:
        nsweeps = len(sweep_start_ray_index["data"])
        sweep_number = filemetadata("sweep_number")
        sweep_number["data"] = np.arange(nsweeps, dtype="float32")
        warnings.warn(
            "Warning: File violates CF/Radial convention. "
            + "Missing sweep_number variable"
        )

    if "target_scan_rate" in ncvars:
        target_scan_rate = _ncvar_to_dict(ncvars["target_scan_rate"])
    else:
        target_scan_rate = None
    if "rays_are_indexed" in ncvars:
        rays_are_indexed = _ncvar_to_dict(ncvars["rays_are_indexed"])
    else:
        rays_are_indexed = None
    if "ray_angle_res" in ncvars:
        ray_angle_res = _ncvar_to_dict(ncvars["ray_angle_res"])
    else:
        ray_angle_res = None

    scan_type = "ppi"

    # 4.9 Moving platform geo-reference variables
    # Aircraft specific variables
    if "rotation" in ncvars:
        rotation = _ncvar_to_dict(ncvars["rotation"])
    else:
        rotation = None

    if "tilt" in ncvars:
        tilt = _ncvar_to_dict(ncvars["tilt"])
    else:
        tilt = None

    if "roll" in ncvars:
        roll = _ncvar_to_dict(ncvars["roll"])
    else:
        roll = None

    if "drift" in ncvars:
        drift = _ncvar_to_dict(ncvars["drift"])
    else:
        drift = None

    if "heading" in ncvars:
        heading = _ncvar_to_dict(ncvars["heading"])
    else:
        heading = None

    if "pitch" in ncvars:
        pitch = _ncvar_to_dict(ncvars["pitch"])
    else:
        pitch = None

    if "georefs_applied" in ncvars:
        georefs_applied = _ncvar_to_dict(ncvars["georefs_applied"])
    else:
        georefs_applied = None

    # 4.10 Moments field data variables -> field attribute dictionary
    if "ray_n_gates" in ncvars:
        # all variables with dimensions of n_points are fields.
        field_keys = [k for k, v in ncvars.items() if v.dimensions == ("n_points",)]
    elif metadata["title"] == "Level2 rainfall rate file":
        # all variables with dimensions of 'azimuth', 'range' are fields
        field_keys = [
            k for k, v in ncvars.items() if v.dimensions == ("azimuth", "range")
        ]
    else:
        # all variables with dimensions of 'time', 'range' are fields
        field_keys = [k for k, v in ncvars.items() if v.dimensions == ("time", "range")]

    # 4.5 instrument_parameters sub-convention -> instrument_parameters dict
    # 4.6 radar_parameters sub-convention -> instrument_parameters dict
    keys = [k for k in _INSTRUMENT_PARAMS_DIMS.keys() if k in ncvars]
    instrument_parameters = {k: _ncvar_to_dict(ncvars[k]) for k in keys}
    if instrument_parameters == {}:  # if no parameters set to None
        instrument_parameters = None

    # 4.7 lidar_parameters sub-convention -> skip

    # 4.8 radar_calibration sub-convention -> radar_calibration
    keys = _find_all_meta_group_vars(ncvars, "radar_calibration")
    radar_calibration = {k: _ncvar_to_dict(ncvars[k]) for k in keys}
    if radar_calibration == {}:
        radar_calibration = None

    for ind_sweep_start, ind_sweep_end in zip(
        sweep_start[ind_first_sweep : ind_last_sweep + 1],
        sweep_end[ind_first_sweep : ind_last_sweep + 1],
    ):

        time["data"] = ncvars["utc_unixtimestamp"][ind_sweep_start : ind_sweep_end + 1]

        if verbose:
            print(f'reading sweep starting at time {time["data"][0]}')

        azimuth["data"] = azi[ind_sweep_start : ind_sweep_end + 1]
        ele = ncvars["elevation"][ind_sweep_start : ind_sweep_end + 1] * 180.0 / np.pi
        elevation["data"] = ele
        fixed_angle["data"] = np.array([ele[0]])
        sweep_end_ray_index["data"] = np.array([ele.size - 1])

        fields = {}
        for key in field_keys:
            field_name = filemetadata.get_field_name(key)
            if field_name is None:
                if exclude_fields is not None and key in exclude_fields:
                    if key not in include_fields:
                        continue
                if include_fields is None or key in include_fields:
                    field_name = key
                else:
                    continue
            fields[field_name] = _ncvar_to_dict(ncvars[key], lazydict=True)
            fields[field_name]["data"] = ncvars[key][
                ind_sweep_start : ind_sweep_end + 1, :
            ]

        if "ray_n_gates" in ncvars:
            shape = (len(ncvars["time"]), len(ncvars["range"]))
            ray_n_gates = ncvars["ray_n_gates"][:]
            ray_start_index = ncvars["ray_start_index"][:]
            for dic in fields.values():
                _unpack_variable_gate_field_dic(
                    dic, shape, ray_n_gates, ray_start_index
                )

        radar = Radar(
            time,
            _range,
            fields,
            metadata,
            scan_type,
            latitude,
            longitude,
            altitude,
            sweep_number,
            sweep_mode,
            fixed_angle,
            sweep_start_ray_index,
            sweep_end_ray_index,
            azimuth,
            elevation,
            instrument_parameters=instrument_parameters,
            radar_calibration=radar_calibration,
            altitude_agl=altitude_agl,
            scan_rate=scan_rate,
            antenna_transition=antenna_transition,
            target_scan_rate=target_scan_rate,
            rays_are_indexed=rays_are_indexed,
            ray_angle_res=ray_angle_res,
            rotation=rotation,
            tilt=tilt,
            roll=roll,
            drift=drift,
            heading=heading,
            pitch=pitch,
            georefs_applied=georefs_applied,
        )
        sweep_time_ref = num2date(time["data"][-1], epoch_unix_units)
        savedir = _get_save_dir(
            sweep_time_ref,
            basepath=basepath,
            sweep_dir=sweep_dir,
            dir_timeformat=dir_timeformat,
        )
        fname = _get_file_name(
            sweep_time_ref,
            file_prefix=file_prefix,
            file_sufix=file_sufix,
            timeformat=file_timeformat,
        )
        print(f"written file {savedir}{fname}")
        write_cfradial(f"{savedir}{fname}", radar)

    ncobj.close()


def get_sweep_time_coverage(filename):
    """
    Get the start time and end time for each sweep. Assumes clockwise rotation
    from north

    Parameters
    ----------
    filename : str
        Name of skyecho netCDF file to read data from.

    Returns
    -------
    tstart, tend : array of datetime objects
        the time start and end of each sweep
    sweep_start, sweep_end : array of indices
        the time indices of the start and end of each sweep

    """
    # read the data
    ncobj = netCDF4.Dataset(filename)
    ncvars = ncobj.variables

    # determine number of sweeps in file
    sweep_start, sweep_end = _get_sweep_limits(ncvars["azimuth"][:] * 180.0 / np.pi)

    # epoch_units = "seconds since " + ''.join(
    #     ncvars["time_coverage_start"][:])
    epoch_unix_units = "seconds since 1970-01-01T00:00:00Z"

    tstart = num2date(ncvars["utc_unixtimestamp"][sweep_start], epoch_unix_units)
    tend = num2date(ncvars["utc_unixtimestamp"][sweep_end], epoch_unix_units)

    ncobj.close()

    return tstart, tend, sweep_start, sweep_end


def _get_sweep_limits(azi):
    """
    Get the indices where the sweeps start and stop. Assumes clockwise
    rotation with 0 deg at north.

    Parameters
    ----------
    azi : array of floats
        azimuths

    Returns
    -------
    sweep_start, sweep_end : array of indices
        the time indices of the start and end of each sweep

    """
    delta_az = np.zeros(azi.shape)
    delta_az[0] = azi[0]
    delta_az[1:] = azi[1:] - azi[:-1]
    sweep_change = np.where(delta_az < 0)[0]
    sweep_start = np.append([0], sweep_change)
    sweep_end = np.append(sweep_change - 1, [azi.size - 1])

    return sweep_start, sweep_end


def _get_save_dir(
    sweep_end_time, basepath=None, sweep_dir=None, dir_timeformat="%Y/%m/%d/"
):
    """
    Creates the path where to store single sweep files

    Parameters
    ----------
    sweep_end_time : datetime object
        The end time of the sweep
    basepath : str or None
        The base path of the sweep. If none is going to be the current
        directory
    sweep_dir: str
        The directory where to store the sweep. If none is going to be the
        current directoy
    dir_timeformat: str
        The time format of the directory

    Returns
    -------
    savedir : str
        the directory where to store the file

    """
    if basepath is None:
        savedir = "./"
    else:
        savedir = basepath
    if dir_timeformat is not None:
        savedir = f"{savedir}{sweep_end_time.strftime(dir_timeformat)}"
    if sweep_dir is not None:
        savedir = f"{savedir}{sweep_dir}/"

    if not os.path.isdir(savedir):
        os.makedirs(savedir)
    return savedir


def _get_file_name(
    sweep_end_time, file_prefix=None, file_sufix=None, timeformat="%Y-%m-%dT%H-%M-%S"
):
    """
    Creates the path where to store single sweep files

    Parameters
    ----------
    sweep_end_time : datetime object
        The end time of the sweep
    file_prefix : str or None
        If not None, the portion of the name preceding the time stamp
    file_sufix: str or None
        If not None, the portion of the name after the time stamp
    timeformat: str
        The format of the time stamp

    Returns
    -------
    fname : str
        the file name

    """
    fname = f"{sweep_end_time.strftime(timeformat)}"
    if file_prefix is not None:
        fname = f"{file_prefix}_{fname}"
    if file_sufix is not None:
        fname = f"{fname}_{file_sufix}"
    fname = f"{fname}.nc"

    return fname
