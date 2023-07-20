"""
pyart.io.cfradial2
==================

Utilities for reading CF/Radial2 files.

.. autosummary::
    :toctree: generated/

    read_cfradial2

"""

from warnings import warn

import netCDF4
import numpy as np

from ..config import FileMetadata
from ..core.radar import Radar
from .cfradial import (
    _find_all_meta_group_vars,
    _ncvar_to_dict,
    _unpack_variable_gate_field_dic,
)
from .common import _test_arguments

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


def read_cfradial2(filename, field_names=None, additional_metadata=None,
                   file_field_names=False, exclude_fields=None,
                   include_fields=None, delay_field_loading=False, **kwargs):
    """
    Read a Cfradial2 netCDF file.

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
    filemetadata = FileMetadata('cfradial2', field_names, additional_metadata,
                                file_field_names, exclude_fields)

    fixed_angle = filemetadata('fixed_angle')
    sweep_start_ray_index = filemetadata('sweep_start_ray_index')
    sweep_end_ray_index = filemetadata('sweep_end_ray_index')
    sweep_number = filemetadata('sweep_number')

    # read the data
    ncobj = netCDF4.Dataset(filename)
    ncvars = ncobj.variables
    ncgroups = ncobj.groups

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
            metadata[var] = ncvars[var][:]
        else:
            metadata[var] = default_value

    # 4.6 Location variables -> create attribute dictionaries
    latitude = _ncvar_to_dict(ncvars['latitude'])
    longitude = _ncvar_to_dict(ncvars['longitude'])
    altitude = _ncvar_to_dict(ncvars['altitude'])
    if 'altitude_agl' in ncvars:
        altitude_agl = _ncvar_to_dict(ncvars['altitude_agl'])
    else:
        altitude_agl = None

    # sweep group names
    sweep_group_name = _ncvar_to_dict(ncvars['sweep_group_name'])
    fixed_angle['data'] = np.array(
        [_ncvar_to_dict(ncvars['sweep_fixed_angle'])['data'][0]])

    # get data of first sweep ------------------------------------------------
    sweep_vars = ncgroups[sweep_group_name['data'][0]].variables

    # 4.4 coordinate variables -> create attribute dictionaries
    time = _ncvar_to_dict(sweep_vars['time'])
    time_reference = _ncvar_to_dict(sweep_vars['time_reference'])
    time['units'] = time['units'].replace(
        'time_reference', time_reference['data'][0])
    _range = _ncvar_to_dict(sweep_vars['range'])

    # 4.5 Ray dimension variables

    # 4.7 Sweep variables -> create attribute dictionaries
    sweep_mode = _ncvar_to_dict(sweep_vars['sweep_mode'])
    sweep_start_ray_index['data'] = np.array(
        [_ncvar_to_dict(sweep_vars['ray_index'])['data'][0]],
        dtype=np.int)
    sweep_end_ray_index['data'] = np.array(
        [_ncvar_to_dict(sweep_vars['ray_index'])['data'][-1]],
        dtype=np.int)
    sweep_number['data'] = np.array([0])

    if 'target_scan_rate' in sweep_vars:
        target_scan_rate = _ncvar_to_dict(sweep_vars['target_scan_rate'])
    else:
        target_scan_rate = None
    if 'rays_are_indexed' in sweep_vars:
        rays_are_indexed = _ncvar_to_dict(sweep_vars['rays_are_indexed'])
    else:
        rays_are_indexed = None
    if 'ray_angle_resolution' in sweep_vars:
        ray_angle_res = _ncvar_to_dict(sweep_vars['ray_angle_resolution'])
    else:
        ray_angle_res = None

    # first sweep mode determines scan_type
    mode = sweep_mode['data'][0]

    # options specified in the CF/Radial standard
    if mode == 'rhi':
        scan_type = 'rhi'
    elif mode == 'vertical_pointing':
        scan_type = 'vpt'
    elif mode == 'azimuth_surveillance':
        scan_type = 'ppi'
    elif mode == 'elevation_surveillance':
        scan_type = 'rhi'
    elif mode == 'manual_ppi':
        scan_type = 'ppi'
    elif mode == 'manual_rhi':
        scan_type = 'rhi'

    # fallback types
    elif 'sur' in mode:
        scan_type = 'ppi'
    elif 'sec' in mode:
        scan_type = 'sector'
    elif 'rhi' in mode:
        scan_type = 'rhi'
    elif 'ppi' in mode:
        scan_type = 'ppi'
    else:
        scan_type = 'other'

    # 4.8 Sensor pointing variables -> create attribute dictionaries
    azimuth = _ncvar_to_dict(sweep_vars['azimuth'])
    elevation = _ncvar_to_dict(sweep_vars['elevation'])
    if 'scan_rate' in sweep_vars:
        scan_rate = _ncvar_to_dict(sweep_vars['scan_rate'])
    else:
        scan_rate = None

    if 'antenna_transition' in sweep_vars:
        antenna_transition = _ncvar_to_dict(sweep_vars['antenna_transition'])
    else:
        antenna_transition = None

    # 4.9 Moving platform geo-reference variables
    # Aircraft specific varaibles
    if 'rotation' in sweep_vars:
        rotation = _ncvar_to_dict(sweep_vars['rotation'])
    else:
        rotation = None

    if 'tilt' in sweep_vars:
        tilt = _ncvar_to_dict(sweep_vars['tilt'])
    else:
        tilt = None

    if 'roll' in sweep_vars:
        roll = _ncvar_to_dict(sweep_vars['roll'])
    else:
        roll = None

    if 'drift' in sweep_vars:
        drift = _ncvar_to_dict(sweep_vars['drift'])
    else:
        drift = None

    if 'heading' in sweep_vars:
        heading = _ncvar_to_dict(sweep_vars['heading'])
    else:
        heading = None

    if 'pitch' in sweep_vars:
        pitch = _ncvar_to_dict(sweep_vars['pitch'])
    else:
        pitch = None

    if 'georefs_applied' in sweep_vars:
        georefs_applied = _ncvar_to_dict(sweep_vars['georefs_applied'])
    else:
        georefs_applied = None

    # 4.10 Moments field data variables -> field attribute dictionary
    if 'ray_n_gates' in sweep_vars:
        # all variables with dimensions of n_points are fields.
        keys = [k for k, v in sweep_vars.items()
                if v.dimensions == ('n_points', )]
    else:
        # all variables with dimensions of 'time', 'range' are fields
        keys = [k for k, v in sweep_vars.items()
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
        fields[field_name] = _ncvar_to_dict(
            sweep_vars[key], delay_field_loading)

    if 'ray_n_gates' in sweep_vars:
        shape = (len(sweep_vars['time']), len(sweep_vars['range']))
        ray_n_gates = sweep_vars['ray_n_gates'][:]
        ray_start_index = sweep_vars['ray_start_index'][:]
        for dic in fields.values():
            _unpack_variable_gate_field_dic(
                dic, shape, ray_n_gates, ray_start_index)

    # 4.5 instrument_parameters sub-convention -> instrument_parameters dict
    # 4.6 radar_parameters sub-convention -> instrument_parameters dict
    keys = [k for k in _INSTRUMENT_PARAMS_DIMS.keys() if k in sweep_vars]
    instrument_parameters = dict(
        (k, _ncvar_to_dict(sweep_vars[k])) for k in keys)
    if instrument_parameters == {}:  # if no parameters set to None
        instrument_parameters = None

    # 4.7 lidar_parameters sub-convention -> skip

    # 4.8 radar_calibration sub-convention -> radar_calibration
    keys = _find_all_meta_group_vars(sweep_vars, 'radar_calibration')
    radar_calibration = dict((k, _ncvar_to_dict(sweep_vars[k])) for k in keys)
    if radar_calibration == {}:
        radar_calibration = None

    # end of getting data of first sweep -------------------------------------

    # if only one sweep finish here:
    if ncobj.dimensions['sweep'].size == 1:
        # do not close file is field loading is delayed
        if not delay_field_loading:
            ncobj.close()
        return Radar(
            time, _range, fields, metadata, scan_type,
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

    # get data from other sweeps
    for i, sweep_name in enumerate(sweep_group_name['data'][1:]):
        sweep_vars = ncgroups[sweep_name].variables

        # 4.4 coordinate variables -> create attribute dictionaries
        _range_sweep = _ncvar_to_dict(sweep_vars['range'])
        if not np.allclose(_range['data'], _range_sweep['data']):
            warn('Skipping sweep ' + sweep_name +
                 ' because range is different from the rest')
            continue

        time_reference_sweep = _ncvar_to_dict(sweep_vars['time_reference'])
        if time_reference_sweep['data'] != time_reference['data']:
            warn('Skipping sweep ' + sweep_name +
                 ' because time reference is different from the rest')
            continue
        time_sweep = _ncvar_to_dict(sweep_vars['time'])
        time['data'] = np.append(time['data'], time_sweep['data'])

        # 4.5 Ray dimension variables

        # 4.7 Sweep variables -> create attribute dictionaries
        sweep_mode['data'] = np.append(
            sweep_mode['data'],
            _ncvar_to_dict(sweep_vars['sweep_mode'])['data'])
        sweep_start_ray_index['data'] = np.append(
            sweep_start_ray_index['data'],
            _ncvar_to_dict(sweep_vars['ray_index'])['data'][0])
        sweep_end_ray_index['data'] = np.append(
            sweep_end_ray_index['data'],
            [_ncvar_to_dict(sweep_vars['ray_index'])['data'][-1]])
        sweep_number['data'] = np.append(sweep_number['data'], i + 1)

        if 'target_scan_rate' in sweep_vars:
            target_scan_rate['data'] = np.append(
                target_scan_rate['data'],
                _ncvar_to_dict(sweep_vars['target_scan_rate'])['data'])
        else:
            target_scan_rate = None
        if 'rays_are_indexed' in sweep_vars:
            rays_are_indexed['data'] = np.append(
                rays_are_indexed['data'],
                _ncvar_to_dict(sweep_vars['rays_are_indexed'])['data'])
        else:
            rays_are_indexed = None
        if 'ray_angle_resolution' in sweep_vars:
            ray_angle_res['data'] = np.append(
                ray_angle_res['data'],
                _ncvar_to_dict(sweep_vars['ray_angle_resolution'])['data'])
        else:
            ray_angle_res = None

        # first sweep mode determines scan_type

        # 4.8 Sensor pointing variables -> create attribute dictionaries
        azimuth['data'] = np.append(
            azimuth['data'],
            _ncvar_to_dict(sweep_vars['azimuth'])['data'])
        elevation['data'] = np.append(
            elevation['data'],
            _ncvar_to_dict(sweep_vars['elevation'])['data'])
        if 'scan_rate' in sweep_vars:
            scan_rate['data'] = np.append(
                scan_rate['data'],
                _ncvar_to_dict(sweep_vars['scan_rate'])['data'])
        else:
            scan_rate = None

        if 'antenna_transition' in sweep_vars:
            antenna_transition['data'] = np.append(
                antenna_transition['data'],
                _ncvar_to_dict(sweep_vars['antenna_transition'])['data'])
        else:
            antenna_transition = None

        # 4.9 Moving platform geo-reference variables
        # Aircraft specific varaibles
        if 'rotation' in sweep_vars:
            rotation['data'] = np.append(
                rotation['data'],
                _ncvar_to_dict(sweep_vars['rotation'])['data'])
        else:
            rotation = None

        if 'tilt' in sweep_vars:
            tilt['data'] = np.append(
                tilt['data'],
                _ncvar_to_dict(sweep_vars['tilt'])['data'])
        else:
            tilt = None

        if 'roll' in sweep_vars:
            roll['data'] = np.append(
                roll['data'],
                _ncvar_to_dict(sweep_vars['roll'])['data'])
        else:
            roll = None

        if 'drift' in sweep_vars:
            drift['data'] = np.append(
                drift['data'],
                _ncvar_to_dict(sweep_vars['drift'])['data'])
        else:
            drift = None

        if 'heading' in sweep_vars:
            heading['data'] = np.append(
                heading['data'],
                _ncvar_to_dict(sweep_vars['heading'])['data'])
        else:
            heading = None

        if 'pitch' in sweep_vars:
            pitch['data'] = np.append(
                pitch['data'],
                _ncvar_to_dict(sweep_vars['pitch'])['data'])
        else:
            pitch = None

        if 'georefs_applied' in sweep_vars:
            georefs_applied['data'] = np.append(
                georefs_applied['data'],
                _ncvar_to_dict(sweep_vars['georefs_applied'])['data'])
        else:
            georefs_applied = None

        # 4.10 Moments field data variables -> field attribute dictionary
        if 'ray_n_gates' in sweep_vars:
            # all variables with dimensions of n_points are fields.
            keys = [k for k, v in sweep_vars.items()
                    if v.dimensions == ('n_points', )]
        else:
            # all variables with dimensions of 'time', 'range' are fields
            keys = [k for k, v in sweep_vars.items()
                    if v.dimensions == ('time', 'range')]

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
            if field_name not in fields:
                continue
            if not delay_field_loading:
                fields[field_name]['data'] = np.ma.append(
                    fields[field_name]['data'],
                    _ncvar_to_dict(
                        sweep_vars[key], delay_field_loading)['data'],
                    axis=0)

        if 'ray_n_gates' in sweep_vars:
            shape = (len(sweep_vars['time']), len(sweep_vars['range']))
            ray_n_gates = sweep_vars['ray_n_gates'][:]
            ray_start_index = sweep_vars['ray_start_index'][:]
            for dic in fields.values():
                _unpack_variable_gate_field_dic(
                    dic, shape, ray_n_gates, ray_start_index)

    # do not close file is field loading is delayed
    if not delay_field_loading:
        ncobj.close()
    return Radar(
        time, _range, fields, metadata, scan_type,
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
