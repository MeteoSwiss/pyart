"""
pyart.io.cf1
============

Utilities for reading CF1 files.

.. autosummary::
    :toctree: generated/
    :template: dev_template.rst

    _NetCDFVariableDataExtractor

.. autosummary::
    :toctree: generated/

    read_cf1

"""

import datetime

import numpy as np
import netCDF4

from ..config import FileMetadata
from ..io.common import _test_arguments, make_time_unit_str
from ..core.radar import Radar
from ..io.cfradial import _find_all_meta_group_vars, _ncvar_to_dict
from ..io.cfradial import _unpack_variable_gate_field_dic


# Variables and dimensions in the instrument_parameter convention and
# radar_parameters sub-convention that will be read from and written to
# CfRadial files using Py-ART.
# The meta_group attribute cannot be used to identify these parameters as
# it is often set incorrectly.
_INSTRUMENT_PARAMS_DIMS = {
    # instrument_parameters sub-convention
    'frequency': ('frequency'),
    'follow_mode': ('sweep', 'string_length'),
    'pulse_width': ('time', ),
    'prt_mode': ('sweep', 'string_length'),
    'prt': ('time', ),
    'prt_ratio': ('time', ),
    'polarization_mode': ('sweep', 'string_length'),
    'nyquist_velocity': ('time', ),
    'unambiguous_range': ('time', ),
    'n_samples': ('time', ),
    'sampling_ratio': ('time', ),
    # radar_parameters sub-convention
    'radar_antenna_gain_h': (),
    'radar_antenna_gain_v': (),
    'radar_beam_width_h': (),
    'radar_beam_width_v': (),
    'radar_receiver_bandwidth': (),
    'radar_measured_transmit_power_h': ('time', ),
    'radar_measured_transmit_power_v': ('time', ),
    'radar_rx_bandwidth': (),           # non-standard
    'measured_transmit_power_v': ('time', ),    # non-standard
    'measured_transmit_power_h': ('time', ),    # non-standard
}


def read_cf1(filename, field_names=None, additional_metadata=None,
             file_field_names=False, exclude_fields=None,
             include_fields=None, delay_field_loading=False, **kwargs):
    """
    Read a CF-1 netCDF file.

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
    filemetadata = FileMetadata('cfradial', field_names, additional_metadata,
                                file_field_names, exclude_fields)
    latitude = filemetadata('latitude')
    longitude = filemetadata('longitude')
    altitude = filemetadata('altitude')
    altitude_agl = filemetadata('altitude_agl')
    azimuth = filemetadata('azimuth')
    elevation = filemetadata('elevation')
    sweep_start_ray_index = filemetadata('sweep_start_ray_index')
    sweep_end_ray_index = filemetadata('sweep_end_ray_index')
    sweep_number = filemetadata('sweep_number')
    sweep_mode = filemetadata('sweep_mode')
    fixed_angle = filemetadata('fixed_angle')
    _range = filemetadata('range')
    _time = filemetadata('time')

    # read the data
    ncobj = netCDF4.Dataset(filename)
    ncvars = ncobj.variables

    # 4.1 Global attribute -> move to metadata dictionary
    metadata = dict([(k, getattr(ncobj, k)) for k in ncobj.ncattrs()])
    if 'n_gates_vary' in metadata:
        metadata['n_gates_vary'] = 'false'  # corrected below

    # 4.2 Dimensions (do nothing)

    # 4.3 Global variable -> move to metadata dictionary
    if 'volume_number' in ncvars:
        metadata['volume_number'] = int(ncvars['volume_number'][:])
    else:
        metadata['volume_number'] = 0

    global_vars = {'platform_type': 'fixed', 'instrument_type': 'radar',
                   'primary_axis': 'axis_z'}
    # ignore time_* global variables, these are calculated from the time
    # variable when the file is written.
    for var, default_value in global_vars.items():
        if var in ncvars:
            metadata[var] = str(netCDF4.chartostring(ncvars[var][:]))
        else:
            metadata[var] = default_value

    # 4.4 coordinate variables -> create attribute dictionaries
    time = _ncvar_to_dict(ncvars['time'])
    microsec = _ncvar_to_dict(ncvars['microsec'])
    _time['data'] = time['data']+1e-6*microsec['data']
    _time['data'] -= time['data'][0]
    sweep_start = (
        datetime.datetime(1970, 1, 1) +
        datetime.timedelta(seconds=int(time['data'][0])))
    _time['units'] = make_time_unit_str(sweep_start)

    _range['data'] = _ncvar_to_dict(ncvars['range'])['data']

    # 4.5 Ray dimension variables

    # 4.6 Location variables -> create attribute dictionaries
    latitude['data'] = np.array([metadata['Latitude']])
    longitude['data'] = np.array([metadata['Longitude']])
    altitude['data'] = np.array([metadata['Altitude']])
    altitude_agl['data'] = np.array([metadata.get('altitude_agl', None)])

    # 4.8 Sensor pointing variables -> create attribute dictionaries
    azimuth['data'] = _ncvar_to_dict(ncvars['azi'])['data']
    elevation['data'] = _ncvar_to_dict(ncvars['elv'])['data']

    if 'scan_rate' in ncvars:
        scan_rate = _ncvar_to_dict(ncvars['scan_rate'])
    else:
        scan_rate = None

    if 'antenna_transition' in ncvars:
        antenna_transition = _ncvar_to_dict(ncvars['antenna_transition'])
    else:
        antenna_transition = None

    # 4.7 Sweep variables -> create attribute dictionaries
    sweep_start_ray_index['data'] = np.array([0])
    sweep_end_ray_index['data'] = np.array(
        [azimuth['data'].size-1], dtype=np.int32)
    sweep_number['data'] = np.array([0])

    nazi_unique = np.unique(azimuth['data'])
    nele_unique = np.unique(elevation['data'])
    if nele_unique.size == 1 and nazi_unique.size == 1:
        scan_type = 'other'
        sweep_mode['data'] = np.array(['pointing'])
        fixed_angle['data'] = nele_unique
    elif nele_unique.size == 1 and nazi_unique.size > 1:
        scan_type = 'ppi'
        sweep_mode['data'] = np.array(['azimuth_surveillance'])
        fixed_angle['data'] = nele_unique
    elif nele_unique.size > 1 and nazi_unique.size == 1:
        scan_type = 'rhi'
        sweep_mode['data'] = np.array(['elevation_surveillance'])
        fixed_angle['data'] = nazi_unique
    else:
        raise ValueError(
            'Only single sweeps PPI, RHI or pointing are supported')

    if 'target_scan_rate' in ncvars:
        target_scan_rate = _ncvar_to_dict(ncvars['target_scan_rate'])
    else:
        target_scan_rate = None
    if 'rays_are_indexed' in ncvars:
        rays_are_indexed = _ncvar_to_dict(ncvars['rays_are_indexed'])
    else:
        rays_are_indexed = None
    if 'ray_angle_res' in ncvars:
        ray_angle_res = _ncvar_to_dict(ncvars['ray_angle_res'])
    else:
        ray_angle_res = None

    # 4.9 Moving platform geo-reference variables
    # Aircraft specific varaibles
    if 'rotation' in ncvars:
        rotation = _ncvar_to_dict(ncvars['rotation'])
    else:
        rotation = None

    if 'tilt' in ncvars:
        tilt = _ncvar_to_dict(ncvars['tilt'])
    else:
        tilt = None

    if 'roll' in ncvars:
        roll = _ncvar_to_dict(ncvars['roll'])
    else:
        roll = None

    if 'drift' in ncvars:
        drift = _ncvar_to_dict(ncvars['drift'])
    else:
        drift = None

    if 'heading' in ncvars:
        heading = _ncvar_to_dict(ncvars['heading'])
    else:
        heading = None

    if 'pitch' in ncvars:
        pitch = _ncvar_to_dict(ncvars['pitch'])
    else:
        pitch = None

    if 'georefs_applied' in ncvars:
        georefs_applied = _ncvar_to_dict(ncvars['georefs_applied'])
    else:
        georefs_applied = None

    # 4.10 Moments field data variables -> field attribute dictionary
    if 'ray_n_gates' in ncvars:
        # all variables with dimensions of n_points are fields.
        keys = [k for k, v in ncvars.items()
                if v.dimensions == ('n_points', )]
    else:
        # all variables with dimensions of 'time', 'range' are fields
        keys = [k for k, v in ncvars.items()
                if v.dimensions == ('time', 'range')]

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
        if not delay_field_loading:
            db = fields[field_name].get('db', 0)
            if db:
                # warn(field_name+' data will be converted to log')
                fields[field_name]['data'] = 10.*np.ma.log10(
                    fields[field_name]['data'])

    if 'ray_n_gates' in ncvars:
        shape = (len(ncvars['time']), len(ncvars['range']))
        ray_n_gates = ncvars['ray_n_gates'][:]
        ray_start_index = ncvars['ray_start_index'][:]
        for dic in fields.values():
            _unpack_variable_gate_field_dic(
                dic, shape, ray_n_gates, ray_start_index)

    # 4.5 instrument_parameters sub-convention -> instrument_parameters dict
    # 4.6 radar_parameters sub-convention -> instrument_parameters dict
    keys = [k for k in _INSTRUMENT_PARAMS_DIMS.keys() if k in ncvars]
    instrument_parameters = dict((k, _ncvar_to_dict(ncvars[k])) for k in keys)
    if instrument_parameters == {}:  # if no parameters set to None
        instrument_parameters = None

    # 4.7 lidar_parameters sub-convention -> skip

    # 4.8 radar_calibration sub-convention -> radar_calibration
    keys = _find_all_meta_group_vars(ncvars, 'radar_calibration')
    radar_calibration = dict((k, _ncvar_to_dict(ncvars[k])) for k in keys)
    if radar_calibration == {}:
        radar_calibration = None

    # do not close file is field loading is delayed
    if not delay_field_loading:
        ncobj.close()
    return Radar(
        _time, _range, fields, metadata, scan_type,
        latitude, longitude, altitude,
        sweep_number, sweep_mode, fixed_angle, sweep_start_ray_index,
        sweep_end_ray_index,
        azimuth, elevation,
        instrument_parameters=instrument_parameters,
        radar_calibration=radar_calibration,
        altitude_agl=altitude_agl,
        scan_rate=scan_rate,
        antenna_transition=antenna_transition,
        target_scan_rate=target_scan_rate,
        rays_are_indexed=rays_are_indexed, ray_angle_res=ray_angle_res,
        rotation=rotation, tilt=tilt, roll=roll, drift=drift, heading=heading,
        pitch=pitch, georefs_applied=georefs_applied)
