"""
pyart.io.spectra
=================

Utilities for reading spectra netcdf files.

.. autosummary::
    :toctree: generated/

    read_spectra
    write_spectra

"""

import datetime
import getpass
import platform
import warnings

import netCDF4
import numpy as np

from ..config import FileMetadata
from ..core.radar_spectra import RadarSpectra
from ..io.cfradial import (
    _create_ncvar,
    _find_all_meta_group_vars,
    _ncvar_to_dict,
    _unpack_variable_gate_field_dic,
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


def read_spectra(
    filename,
    field_names=None,
    additional_metadata=None,
    file_field_names=False,
    exclude_fields=None,
    include_fields=None,
    delay_field_loading=False,
    **kwargs,
):
    """
    Read a spectra netCDF file.

    Parameters
    ----------
    filename : str
        Name of CF/Radial netCDF file to read data from.
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
    delay_field_loading : bool
        True to delay loading of field data from the file until the 'data'
        key in a particular field dictionary is accessed.  In this case
        the field attribute of the returned Radar object will contain
        LazyLoadDict objects not dict objects.  Delayed field loading will not
        provide any speedup in file where the number of gates vary between
        rays (ngates_vary=True) and is not recommended.

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
    ncobj = netCDF4.Dataset(filename)
    ncvars = ncobj.variables

    # 4.1 Global attribute -> move to metadata dictionary
    metadata = dict([(k, getattr(ncobj, k)) for k in ncobj.ncattrs()])
    if "n_gates_vary" in metadata:
        metadata["n_gates_vary"] = "false"  # corrected below

    # 4.2 Dimensions (do nothing)

    # 4.3 Global variable -> move to metadata dictionary
    if "volume_number" in ncvars:
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
    time = _ncvar_to_dict(ncvars["time"])
    _range = _ncvar_to_dict(ncvars["range"])

    # 4.5 Ray dimension variables

    # 4.5.b Doppler dimension variables
    npulses = _ncvar_to_dict(ncvars["number_of_pulses"])

    if "Doppler_velocity" in ncvars:
        Doppler_velocity = _ncvar_to_dict(ncvars["Doppler_velocity"])
    else:
        Doppler_velocity = None

    if "Doppler_frequency" in ncvars:
        Doppler_frequency = _ncvar_to_dict(ncvars["Doppler_frequency"])
    else:
        Doppler_frequency = None

    # 4.6 Location variables -> create attribute dictionaries
    latitude = _ncvar_to_dict(ncvars["latitude"])
    longitude = _ncvar_to_dict(ncvars["longitude"])
    altitude = _ncvar_to_dict(ncvars["altitude"])
    if "altitude_agl" in ncvars:
        altitude_agl = _ncvar_to_dict(ncvars["altitude_agl"])
    else:
        altitude_agl = None

    # 4.7 Sweep variables -> create atrribute dictionaries
    sweep_mode = _ncvar_to_dict(ncvars["sweep_mode"])
    fixed_angle = _ncvar_to_dict(ncvars["fixed_angle"])
    sweep_start_ray_index = _ncvar_to_dict(ncvars["sweep_start_ray_index"])
    sweep_end_ray_index = _ncvar_to_dict(ncvars["sweep_end_ray_index"])

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

    # first sweep mode determines scan_type
    try:
        mode = netCDF4.chartostring(sweep_mode["data"][0])[()].decode("utf-8")
    except AttributeError:
        # Python 3, all strings are already unicode.
        mode = netCDF4.chartostring(sweep_mode["data"][0])[()]

    # options specified in the CF/Radial standard
    if mode == "rhi":
        scan_type = "rhi"
    elif mode == "vertical_pointing":
        scan_type = "vpt"
    elif mode == "azimuth_surveillance":
        scan_type = "ppi"
    elif mode == "elevation_surveillance":
        scan_type = "rhi"
    elif mode == "manual_ppi":
        scan_type = "ppi"
    elif mode == "manual_rhi":
        scan_type = "rhi"

    # fallback types
    elif "sur" in mode:
        scan_type = "ppi"
    elif "sec" in mode:
        scan_type = "sector"
    elif "rhi" in mode:
        scan_type = "rhi"
    elif "ppi" in mode:
        scan_type = "ppi"
    else:
        scan_type = "other"

    # 4.8 Sensor pointing variables -> create attribute dictionaries
    azimuth = _ncvar_to_dict(ncvars["azimuth"])
    elevation = _ncvar_to_dict(ncvars["elevation"])
    if "scan_rate" in ncvars:
        scan_rate = _ncvar_to_dict(ncvars["scan_rate"])
    else:
        scan_rate = None

    if "antenna_transition" in ncvars:
        antenna_transition = _ncvar_to_dict(ncvars["antenna_transition"])
    else:
        antenna_transition = None

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
    else:
        # all variables with dimensions of 'time', 'range', 'npulses_max' are
        # fields
        keys = [
            k
            for k, v in ncvars.items()
            if v.dimensions == ("time", "range", "npulses_max")
        ]

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
        fields[field_name] = _ncvar_to_dict(ncvars[key], delay_field_loading)

    if "ray_n_gates" in ncvars:
        shape = (len(ncvars["time"]), len(ncvars["range"]))
        ray_n_gates = ncvars["ray_n_gates"][:]
        ray_start_index = ncvars["ray_start_index"][:]
        for dic in fields.values():
            _unpack_variable_gate_field_dic(dic, shape, ray_n_gates, ray_start_index)

    # 4.5 instrument_parameters sub-convention -> instrument_parameters dict
    # 4.6 radar_parameters sub-convention -> instrument_parameters dict
    keys = [k for k in _INSTRUMENT_PARAMS_DIMS.keys() if k in ncvars]
    instrument_parameters = dict((k, _ncvar_to_dict(ncvars[k])) for k in keys)
    if instrument_parameters == {}:  # if no parameters set to None
        instrument_parameters = None

    # 4.7 lidar_parameters sub-convention -> skip

    # 4.8 radar_calibration sub-convention -> radar_calibration
    keys = _find_all_meta_group_vars(ncvars, "radar_calibration")
    radar_calibration = dict((k, _ncvar_to_dict(ncvars[k])) for k in keys)
    if radar_calibration == {}:
        radar_calibration = None

    # do not close file is field loading is delayed
    if not delay_field_loading:
        ncobj.close()
    return RadarSpectra(
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
        npulses,
        Doppler_velocity=Doppler_velocity,
        Doppler_frequency=Doppler_frequency,
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


def write_spectra(
    filename,
    radar,
    format="NETCDF4",
    time_reference=None,
    arm_time_variables=False,
    physical=True,
):
    """
    Write a Radar Spectra object to a netCDF file.

    The files produced by this routine follow the `CF/Radial standard`_.
    Attempts are also made to to meet many of the standards outlined in the
    `ARM Data File Standards`_.

    .. _CF/Radial standard: http://www.ral.ucar.edu/projects/titan/docs/radial_formats/cfradial.html
    .. _ARM Data File Standards: https://docs.google.com/document/d/1gBMw4Kje6v8LBlsrjaGFfSLoU0jRx-07TIazpthZGt0/edit?pli=1

    To control how the netCDF variables are created, set any of the following
    keys in the radar attribute dictionaries.

        * _Zlib
        * _DeflateLevel
        * _Shuffle
        * _Fletcher32
        * _Continguous
        * _ChunkSizes
        * _Endianness
        * _Least_significant_digit
        * _FillValue

    See the netCDF4 documentation for details on these settings.

    Parameters
    ----------
    filename : str
        Filename to create.
    radar : Radar
        Radar object.
    format : str, optional
        NetCDF format, one of 'NETCDF4', 'NETCDF4_CLASSIC',
        'NETCDF3_CLASSIC' or 'NETCDF3_64BIT'. See netCDF4 documentation for
        details.
    time_reference : bool
        True to include a time_reference variable, False will not include
        this variable. The default, None, will include the time_reference
        variable when the first time value is non-zero.
    arm_time_variables : bool
        True to create the ARM standard time variables base_time and
        time_offset, False will not create these variables.
    physical : bool
        True to store the radar fields as physical numbers, False will store
        the radar fields as binary if the keyword '_Write_as_dtype' is in the
        field metadata. The gain and offset can be specified in the keyword
        'scale_factor' and 'add_offset' or calculated on the fly.

    """
    dataset = netCDF4.Dataset(filename, "w", format=format)

    # determine the maximum string length
    max_str_len = len(radar.sweep_mode["data"][0])
    for k in ["follow_mode", "prt_mode", "polarization_mode"]:
        if (radar.instrument_parameters is not None) and (
            k in radar.instrument_parameters
        ):
            sdim_length = len(radar.instrument_parameters[k]["data"][0])
            max_str_len = max(max_str_len, sdim_length)
    str_len = max(max_str_len, 32)  # minimum string legnth of 32

    # create time, range, npulses_max and sweep dimensions
    dataset.createDimension("time", None)
    dataset.createDimension("range", radar.ngates)
    dataset.createDimension("npulses_max", radar.npulses_max)
    dataset.createDimension("sweep", radar.nsweeps)
    dataset.createDimension("string_length", str_len)

    # global attributes
    # remove global variables from copy of metadata
    metadata_copy = dict(radar.metadata)
    global_variables = [
        "volume_number",
        "platform_type",
        "instrument_type",
        "primary_axis",
        "time_coverage_start",
        "time_coverage_end",
        "time_reference",
    ]
    for var in global_variables:
        if var in metadata_copy:
            metadata_copy.pop(var)

    # determine the history attribute if it doesn't exist, save for
    # the last attribute.
    if "history" in metadata_copy:
        history = metadata_copy.pop("history")
    else:
        user = getpass.getuser()
        node = platform.node()
        time_str = datetime.datetime.now().isoformat()
        t = (user, node, time_str)
        history = "created by {} on {} at {} using Py-ART".format(*t)

    dataset.setncatts(metadata_copy)

    if "Conventions" not in dataset.ncattrs():
        dataset.setncattr("Conventions", "CF/Radial")

    if "field_names" not in dataset.ncattrs():
        dataset.setncattr("field_names", ", ".join(radar.fields.keys()))

    # history should be the last attribute, ARM standard
    dataset.setncattr("history", history)

    # arm time variables base_time and time_offset if requested
    if arm_time_variables:
        dt = netCDF4.num2date(radar.time["data"][0], radar.time["units"])
        td = dt - datetime.datetime.fromtimestamp(0, datetime.timezone.utc)
        base_time = {
            "data": np.array([td.seconds + td.days * 24 * 3600], "int32"),
            "string": dt.strftime("%d-%b-%Y,%H:%M:%S GMT"),
            "units": "seconds since 1970-1-1 0:00:00 0:00",
            "ancillary_variables": "time_offset",
            "long_name": "Base time in Epoch",
        }
        _create_ncvar(base_time, dataset, "base_time", ())

        time_offset = {
            "data": radar.time["data"],
            "long_name": "Time offset from base_time",
            "units": radar.time["units"].replace("T", " ").replace("Z", ""),
            "ancillary_variables": "time_offset",
            "calendar": "gregorian",
        }
        _create_ncvar(time_offset, dataset, "time_offset", ("time",))

    # standard variables
    _create_ncvar(radar.time, dataset, "time", ("time",))
    _create_ncvar(radar.range, dataset, "range", ("range",))
    _create_ncvar(radar.azimuth, dataset, "azimuth", ("time",))
    _create_ncvar(radar.elevation, dataset, "elevation", ("time",))
    _create_ncvar(radar.npulses, dataset, "number_of_pulses", ("time",))

    # Optional Doppler variables
    if radar.Doppler_velocity is not None:
        _create_ncvar(
            radar.Doppler_velocity, dataset, "Doppler_velocity", ("time", "npulses_max")
        )

    if radar.Doppler_frequency is not None:
        _create_ncvar(
            radar.Doppler_frequency,
            dataset,
            "Doppler_frequency",
            ("time", "npulses_max"),
        )

    # optional sensor pointing variables
    if radar.scan_rate is not None:
        _create_ncvar(radar.scan_rate, dataset, "scan_rate", ("time",))
    if radar.antenna_transition is not None:
        _create_ncvar(
            radar.antenna_transition, dataset, "antenna_transition", ("time",)
        )

    # fields
    for field, dic in radar.fields.items():
        _create_ncvar(
            dic,
            dataset,
            field,
            ("time", "range", "npulses_max"),
            physical=physical,
            is_field=True,
        )

    # sweep parameters
    _create_ncvar(radar.sweep_number, dataset, "sweep_number", ("sweep",))
    _create_ncvar(radar.fixed_angle, dataset, "fixed_angle", ("sweep",))
    _create_ncvar(
        radar.sweep_start_ray_index, dataset, "sweep_start_ray_index", ("sweep",)
    )
    _create_ncvar(radar.sweep_end_ray_index, dataset, "sweep_end_ray_index", ("sweep",))
    _create_ncvar(radar.sweep_mode, dataset, "sweep_mode", ("sweep", "string_length"))
    if radar.target_scan_rate is not None:
        _create_ncvar(radar.target_scan_rate, dataset, "target_scan_rate", ("sweep",))
    if radar.rays_are_indexed is not None:
        _create_ncvar(
            radar.rays_are_indexed,
            dataset,
            "rays_are_indexed",
            ("sweep", "string_length"),
        )
    if radar.ray_angle_res is not None:
        _create_ncvar(radar.ray_angle_res, dataset, "ray_angle_res", ("sweep",))

    # instrument_parameters
    if (radar.instrument_parameters is not None) and (
        "frequency" in radar.instrument_parameters.keys()
    ):
        size = len(radar.instrument_parameters["frequency"]["data"])
        dataset.createDimension("frequency", size)

    if radar.instrument_parameters is not None:
        for k in radar.instrument_parameters.keys():
            if k in _INSTRUMENT_PARAMS_DIMS:
                dim = _INSTRUMENT_PARAMS_DIMS[k]
                _create_ncvar(radar.instrument_parameters[k], dataset, k, dim)
            else:
                # Do not try to write instrument parameter whose dimensions are
                # not known, rather issue a warning and skip the parameter
                message = (
                    f"Unknown instrument parameter: {k}, " + "not written to file."
                )
                warnings.warn(message)

    # radar_calibration variables
    if radar.radar_calibration is not None and radar.radar_calibration != {}:
        size = [
            len(d["data"])
            for k, d in radar.radar_calibration.items()
            if k not in ["r_calib_index", "r_calib_time"]
        ][0]
        dataset.createDimension("r_calib", size)
        for key, dic in radar.radar_calibration.items():
            if key == "r_calib_index":
                dims = ("time",)
            elif key == "r_calib_time":
                dims = ("r_calib", "string_length")
            else:
                dims = ("r_calib",)
            _create_ncvar(dic, dataset, key, dims)

    # latitude, longitude, altitude, altitude_agl
    if radar.latitude["data"].size == 1:
        # stationary platform
        _create_ncvar(radar.latitude, dataset, "latitude", ())
        _create_ncvar(radar.longitude, dataset, "longitude", ())
        _create_ncvar(radar.altitude, dataset, "altitude", ())
        if radar.altitude_agl is not None:
            _create_ncvar(radar.altitude_agl, dataset, "altitude_agl", ())
    else:
        # moving platform
        _create_ncvar(radar.latitude, dataset, "latitude", ("time",))
        _create_ncvar(radar.longitude, dataset, "longitude", ("time",))
        _create_ncvar(radar.altitude, dataset, "altitude", ("time",))
        if radar.altitude_agl is not None:
            _create_ncvar(radar.altitude_agl, dataset, "altitude_agl", ("time",))

    # time_coverage_start and time_coverage_end variables
    time_dim = ("string_length",)
    units = radar.time["units"]
    start_dt = netCDF4.num2date(radar.time["data"][0], units)
    if start_dt.microsecond != 0:
        # truncate to nearest second
        start_dt -= datetime.timedelta(microseconds=start_dt.microsecond)
    end_dt = netCDF4.num2date(radar.time["data"][-1], units)
    if end_dt.microsecond != 0:
        # round up to next second
        end_dt += datetime.timedelta(seconds=1) - datetime.timedelta(
            microseconds=end_dt.microsecond
        )
    start_dic = {
        "data": np.array(start_dt.isoformat() + "Z", dtype="S"),
        "long_name": "UTC time of first ray in the file",
        "units": "unitless",
    }
    end_dic = {
        "data": np.array(end_dt.isoformat() + "Z", dtype="S"),
        "long_name": "UTC time of last ray in the file",
        "units": "unitless",
    }
    _create_ncvar(start_dic, dataset, "time_coverage_start", time_dim)
    _create_ncvar(end_dic, dataset, "time_coverage_end", time_dim)

    # time_reference is required or requested.
    if time_reference is None:
        if radar.time["data"][0] == 0:
            time_reference = False
        else:
            time_reference = True
    if time_reference:
        ref_dic = {
            "data": np.array(radar.time["units"][-20:], dtype="S"),
            "long_name": "UTC time reference",
            "units": "unitless",
        }
        _create_ncvar(ref_dic, dataset, "time_reference", time_dim)

    # global variables
    # volume_number, required
    vol_dic = {"long_name": "Volume number", "units": "unitless"}
    if "volume_number" in radar.metadata:
        vol_dic["data"] = np.array([radar.metadata["volume_number"]], dtype="int32")
    else:
        vol_dic["data"] = np.array([0], dtype="int32")
    _create_ncvar(vol_dic, dataset, "volume_number", ())

    # platform_type, optional
    if "platform_type" in radar.metadata:
        dic = {
            "long_name": "Platform type",
            "data": np.array(radar.metadata["platform_type"], dtype="S"),
        }
        _create_ncvar(dic, dataset, "platform_type", ("string_length",))

    # instrument_type, optional
    if "instrument_type" in radar.metadata:
        dic = {
            "long_name": "Instrument type",
            "data": np.array(radar.metadata["instrument_type"], dtype="S"),
        }
        _create_ncvar(dic, dataset, "instrument_type", ("string_length",))

    # primary_axis, optional
    if "primary_axis" in radar.metadata:
        dic = {
            "long_name": "Primary axis",
            "data": np.array(radar.metadata["primary_axis"], dtype="S"),
        }
        _create_ncvar(dic, dataset, "primary_axis", ("string_length",))

    # moving platform geo-reference variables
    if radar.rotation is not None:
        _create_ncvar(radar.rotation, dataset, "rotation", ("time",))

    if radar.tilt is not None:
        _create_ncvar(radar.tilt, dataset, "tilt", ("time",))

    if radar.roll is not None:
        _create_ncvar(radar.roll, dataset, "roll", ("time",))

    if radar.drift is not None:
        _create_ncvar(radar.drift, dataset, "drift", ("time",))

    if radar.heading is not None:
        _create_ncvar(radar.heading, dataset, "heading", ("time",))

    if radar.pitch is not None:
        _create_ncvar(radar.pitch, dataset, "pitch", ("time",))

    if radar.georefs_applied is not None:
        _create_ncvar(radar.georefs_applied, dataset, "georefs_applied", ("time",))

    dataset.close()
