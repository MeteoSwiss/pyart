"""
pyart.aux_io.metranet_reader
============================

Routines for putting METRANET data files into radar object.
(Used by ELDES www.eldesradar.it)

.. autosummary::
    :toctree: generated/

    read_metranet
    read_metranet_c
    read_metranet_python

"""

import os
import datetime
import platform
from warnings import warn

import numpy as np
from scipy.stats import circmean

from ..config import FileMetadata, get_fillvalue
from ..io.common import make_time_unit_str, _test_arguments
from ..core.radar import Radar
from ..exceptions import MissingOptionalDependency

from .metranet_c import Selex_Angle, get_library
from .metranet_python import read_polar as read_polar_python
from .metranet_c import read_polar as read_polar_c

from .dn_to_float import nyquist_vel
# check existence of METRANET library
try:
    METRANET_LIB = get_library(momentms=False)
    if platform.system() == 'Linux':
        METRANET_LIB = get_library(momentms=True)
    _METRANETLIB_AVAILABLE = True
except SystemExit:
    _METRANETLIB_AVAILABLE = False

METRANET_FIELD_NAMES = {
    'WID': 'spectrum_width',
    'VEL': 'velocity',
    'ZH': 'reflectivity',
    'ZV': 'reflectivity_vv',  # non standard name
    'ZDR': 'differential_reflectivity',
    'RHO': 'uncorrected_cross_correlation_ratio',
    'PHI': 'uncorrected_differential_phase',
    'ST1': 'stat_test_lag1',  # statistical test on lag 1 (non standard name)
    'ST2': 'stat_test_lag2',  # statistical test on lag 2 (non standard name)
    'WBN': 'wide_band_noise',  # (non standard name)
    'MPH': 'mean_phase',  # (non standard name)
    'CLT': 'clutter_exit_code',  # (non standard name)
    'ZHC': 'reflectivity_hh_clut',  # cluttered horizontal reflectivity
    'ZVC': 'reflectivity_hh_clut',  # cluttered vertical reflectivity
}

PM_MOM = ["ZH", "ZV", "ZDR", "RHO", "PHI", "VEL", "WID", "ST1", "ST2", "WBN",
          "MPH"]
PH_MOM = ["ZH", "ZV", "ZDR", "RHO", "PHI", "VEL", "WID", "ST1", "ST2", "WBN",
          "MPH", "CLT"]
PL_MOM = ["ZH", "ZV", "ZDR", "RHO", "PHI", "VEL", "WID", "ZHC", "ZVC"]

NPM_MOM = 11
NPH_MOM = 12
NPL_MOM = 9


def read_metranet(filename, field_names=None, rmax=0.,
                  additional_metadata=None, file_field_names=False,
                  exclude_fields=None, reader='C', nbytes=4, **kwargs):

    """
    Read a METRANET file.

    Parameters
    ----------
    filename : str
        Name of the METRANET file to read.
    field_names : dict, optional
        Dictionary mapping METRANET field names to radar field names. If a
        data type found in the file does not appear in this dictionary or has
        a value of None it will not be placed in the radar.fields dictionary.
        A value of None, the default, will use the mapping defined in the
        Py-ART configuration file.
    rmax : float, optional
        Maximum radar range to store in the radar object [m]. If 0 all data
        will be stored
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
        after the `file_field_names` and `field_names` parameters.
    reader : str
        The reader library to use. Can be either 'C' or 'python'
    nbytes : int
        The number of bytes used to store the data in numpy arrays, e.g. if
        nbytes=4 then floats are going to be stored as np.float32

    Returns
    -------
    radar : Radar
        Radar object containing data from METRANET file.

    """
    # check if it is the right file
    bfile = os.path.basename(filename)

    supported_file = (bfile.startswith('PM') or bfile.startswith('PH') or
                      bfile.startswith('PL') or bfile.startswith('MS') or
                      bfile.startswith('MH') or bfile.startswith('ML'))

    if not supported_file:
        raise ValueError(
            'Only polar data files starting by ' +
            'PM, PH, PL, MS, MH or ML are supported')

    if reader == 'C' and _METRANETLIB_AVAILABLE:
        return read_metranet_c(
            filename, field_names, rmax, additional_metadata,
            file_field_names, exclude_fields, nbytes=nbytes, **kwargs)

    if reader == 'python':
        return read_metranet_python(
            filename, field_names, rmax, additional_metadata,
            file_field_names, exclude_fields, nbytes=nbytes, **kwargs)

    warn('Invalid reader name or C library not available,' +
         ' using python (default) instead')
    return read_metranet_python(
        filename, field_names, rmax, additional_metadata,
        file_field_names, exclude_fields, nbytes=nbytes, **kwargs)


def read_metranet_c(filename, field_names=None, rmax=0.,
                    additional_metadata=None, file_field_names=False,
                    exclude_fields=None, nbytes=4, **kwargs):
    """
    Read a METRANET file.

    Parameters
    ----------
    filename : str
        Name of the METRANET file to read.
    field_names : dict, optional
        Dictionary mapping METRANET field names to radar field names. If a
        data type found in the file does not appear in this dictionary or has
        a value of None it will not be placed in the radar.fields dictionary.
        A value of None, the default, will use the mapping defined in the
        Py-ART configuration file.
    rmax : float, optional
        Maximum radar range to store in the radar object [m]. If 0 all data
        will be stored
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
        after the `file_field_names` and `field_names` parameters.
    nbytes : int
        The number of bytes used to store the data in numpy arrays, e.g. if
        nbytes=4 then floats are going to be stored as np.float32

    Returns
    -------
    radar : Radar
        Radar object containing data from METRANET file.

    """
    # check that METRANET library is available
    if not _METRANETLIB_AVAILABLE:
        raise MissingOptionalDependency(
            "Metranet library is required to use read_metranet " +
            "but is not installed")

    if nbytes == 4:
        dtype = np.float32
    elif nbytes == 8:
        dtype = np.float64
    else:
        warn('Number of bytes to store the data ('+str(nbytes) +
             ') not supported. 4 bytes will be used')
        dtype = np.float32

    # test for non empty kwargs
    _test_arguments(kwargs)

    bfile = os.path.basename(filename)

    # create metadata retrieval object
    if field_names is None:
        field_names = METRANET_FIELD_NAMES
    filemetadata = FileMetadata(
        'METRANET', field_names, additional_metadata, file_field_names,
        exclude_fields)

    # get definitions from filemetadata class
    latitude = filemetadata('latitude')
    longitude = filemetadata('longitude')
    altitude = filemetadata('altitude')
    metadata = filemetadata('metadata')
    sweep_start_ray_index = filemetadata('sweep_start_ray_index')
    sweep_end_ray_index = filemetadata('sweep_end_ray_index')
    sweep_number = filemetadata('sweep_number')
    sweep_mode = filemetadata('sweep_mode')
    fixed_angle = filemetadata('fixed_angle')
    elevation = filemetadata('elevation')
    _range = filemetadata('range')
    azimuth = filemetadata('azimuth')
    _time = filemetadata('time')

    # other metadata
    frequency = filemetadata('frequency')
    beamwidth_h = filemetadata('radar_beam_width_h')
    beamwidth_v = filemetadata('radar_beam_width_v')
    pulse_width = filemetadata('pulse_width')
    rays_are_indexed = filemetadata('rays_are_indexed')
    ray_angle_res = filemetadata('ray_angle_res')
    nyquist_velocity = filemetadata('nyquist_velocity')

    ret = read_polar_c(filename, 'ZH', physic_value=True, masked_array=True)
    if ret is None:
        raise ValueError('Unable to read file '+filename)

    # total number of rays composing the sweep
    total_record = ret.header['row']
    if total_record == 0:
        raise ValueError('Number of rays in file=0.')

    # M files returning 0 pulse width. Hardcode it for the moment
    # pulse_width['data'] = np.array(
    #    [ret.header['PulseWidth']*1e-6], dtype='float64')
    pulse_width['data'] = 0.5e-6*np.ones(total_record, dtype=dtype)
    rays_are_indexed['data'] = np.array(['true'])
    ray_angle_res['data'] = np.array([1.], dtype=dtype)

    # number of gates in a ray
    num_gates = ret.header['column']

    # sweep_number (is the sweep index)
    # current sweep number (from 0 to 19)
    sweep_number['data'] = np.array([ret.header['CurrentSweep']-1])

    az_data = np.empty(total_record, dtype=dtype)
    el_data = np.empty(total_record, dtype=dtype)
    time_data = np.empty(total_record, dtype=dtype)
    ray_index_data = np.empty(total_record, dtype=dtype)

    angres = ray_angle_res['data'][0]
    valid_rays = np.ones((total_record)).astype(bool)

    ant_mode = ret.header['AntMode']  # scanning mode code
    if ant_mode == 0:
        scan_type = 'ppi'
        sweep_mode['data'] = np.array(['azimuth_surveillance'])
        # ray starting elevation angle information
        fixed_angle['data'] = np.array(
            [Selex_Angle(ret.pol_header[0].start_angle).el],
            dtype=dtype)

        # azimuth
        for i in range(total_record):
            # ray starting azimuth angle information
            if ret.pol_header[i].start_angle != 0:
                # ray starting azimuth angle information
                start_angle = Selex_Angle(ret.pol_header[i].start_angle).az
                # ray ending azimuth angle information
                end_angle = Selex_Angle(ret.pol_header[i].end_angle).az
                if end_angle > start_angle:
                    az_data[i] = start_angle + (end_angle-start_angle)/2.
                else:
                    az_data[i] = start_angle + (end_angle+360.-start_angle)/2.
                if az_data[i] > 360: # Can happen in spurious cases
                    az_data[i] -= 360
            else:
                valid_rays[i] = 0

        if not np.all(valid_rays):
            # incomplete scan

            az_full_scan = np.arange(0+angres/2, 360+angres/2, angres)
            az_closest = angres*np.floor(az_data[valid_rays]/angres)

            idx_az = np.zeros((sum(valid_rays))).astype(int)
            for i, _ in enumerate(idx_az):
                idx_az[i] = np.searchsorted(az_full_scan, az_closest[i])

            corr_az = az_full_scan
            corr_az[idx_az] = az_data[valid_rays]
            az_data = corr_az

        azimuth['data'] = az_data

        # elevation
        elevation['data'] = np.repeat(fixed_angle['data'], total_record)
    elif ant_mode == 1:
        scan_type = 'rhi'
        sweep_mode['data'] = np.array(['elevation_surveillance'])
        # ray starting azimuth angle information
        fixed_angle['data'] = np.array(
            [Selex_Angle(ret.pol_header[0].start_angle).az],
            dtype=dtype)

        # azimuth
        azimuth['data'] = np.repeat(fixed_angle['data'], total_record)

        # elevation
        for i in range(total_record):
            # ray starting elevation angle information
            start_angle = Selex_Angle(ret.pol_header[i].start_angle).el
            # ray ending elevation angle information
            end_angle = Selex_Angle(ret.pol_header[i].end_angle).el
            el_data[i] = start_angle + (end_angle-start_angle)/2.
        elevation['data'] = el_data
    elif ant_mode == 2:
        scan_type = 'sector'  # sector scan
        sweep_mode['data'] = np.array(['sector'])
        # ray starting elevation angle information
        fixed_angle['data'] = np.array(
            [Selex_Angle(ret.pol_header[0].start_angle).el],
            dtype=dtype)

        # azimuth
        for i in range(total_record):
            # ray starting azimuth angle information
            start_angle = Selex_Angle(ret.pol_header[i].start_angle).az
            # ray ending azimuth angle information
            end_angle = Selex_Angle(ret.pol_header[i].end_angle).az
            if end_angle > start_angle:
                az_data[i] = start_angle + (end_angle-start_angle)/2.
            else:
                az_data[i] = start_angle + (end_angle+360.-start_angle)/2.
        azimuth['data'] = az_data

        # elevation
        elevation['data'] = np.repeat(fixed_angle['data'], total_record)
    elif ant_mode == 3:
        scan_type = 'other'  # point of interest scan
        sweep_mode['data'] = np.array(['pointing'])
        # ray starting elevation angle information
        fixed_angle['data'] = np.array(
            [Selex_Angle(ret.pol_header[0].start_angle).el],
            dtype=dtype)

        # azimuth
        # ray starting elevation angle information
        azimuth['data'] = Selex_Angle(ret.pol_header[0].start_angle).az

        # elevation
        elevation['data'] = fixed_angle['data']
    elif ant_mode == 4:
        scan_type = 'other'  # off
        sweep_mode['data'] = np.array(['idle'])
    else:
        raise ValueError('Unknown scan type')

    if np.all(valid_rays):
        idx_az = np.arange(0, 360/angres).astype(int)

    # range (to center of beam [m])
    # distance to start of first range gate [usually 0 m]
    start_range = float(ret.header['StartRange'])
    # range resolution [m]
    gate_width = float(ret.header['GateWidth'])*1000.
    _range['data'] = np.linspace(
        start_range+gate_width/2., float(num_gates-1.) *
        gate_width+gate_width/2., num_gates, dtype=dtype)

    if rmax > 0.:
        _range['data'] = _range['data'][_range['data'] < rmax]
        nrange = len(_range['data'])

    # time (according to default_config this is the Time at the center of
    # each ray, in fractional seconds since the volume started)
    # here we find the time of end of ray since the first ray in the sweep
    for i in range(total_record):
        # time when the ray was created [s from 1.1.1970].
        # (last received pulse+processing time)
        data_time = float(ret.pol_header[i].data_time)
        # the hundreths of seconds to add to the data_time
        data_time_residue = float(ret.pol_header[i].data_time_residue)
        time_data[i] = data_time+data_time_residue/100.
        ray_index_data[i] = ret.pol_header[i].sequence

    sweep_start = min(time_data)
    start_time = datetime.datetime.utcfromtimestamp(sweep_start)
    _time['data'] = time_data-sweep_start
    _time['units'] = make_time_unit_str(start_time)

    # sweep_start_ray_index, sweep_end_ray_index
    # should be specified since start of volume but we do not have this
    # information so we specify it since start of sweep instead.
    if np.all(valid_rays):
        sweep_start_ray_index['data'] = np.array(
            [min(ray_index_data)], dtype='int32')  # ray index of first ray
        sweep_end_ray_index['data'] = np.array(
            [max(ray_index_data)], dtype='int32')   # ray index of last ray
    else:
        sweep_start_ray_index['data'] = np.array(
            [0], dtype='int32')  # ray index of first ray
        sweep_end_ray_index['data'] = np.array(
            [len(ray_index_data)], dtype='int32')   # ray index of last ray

    # ----  other information that can be obtained from metadata in file
    #       sweep information:
    #       total number of sweeps compositing the volume (i.e. 20):
    #       total_sweep=ret.pol_header[0].total_sweep
    #       total number of data bytes in the ray
    #   (num_gates*number_of_moments*(number_of_bytes in each moment)):
    #       data_bytes=ret.pol_header[0].data_bytes
    #       # time period of repetition of the volume scan:
    #       repeat_time=ret.pol_header[0].repeat_time
    #       # Nyquist velocity [m/s]:
    #       ny_quest=ret.pol_header[0].ny_quest
    #       Maximum Doppler spectrum width [m/s]:
    #       w_ny_quest=ret.pol_header[0].w_ny_quest
    #
    #       # ray specific information
    #       0 no end of sweep, 1 end of sweep, 2 end of volume scan:
    #       end_of_sweep=ret.pol_header[0].end_of_sweep
    #       number of pulses used in data integration:
    #       pulses=ret.pol_header[0].pulses
    # ------------------------------------------------------------------

    # metadata
    # get radar id
    if isinstance(ret.header["RadarName"], str):
        radar_id = ret.header["RadarName"]
    else:
        radar_id = ret.header["RadarName"].decode('utf-8')

    metadata['instrument_name'] = radar_id

    # hardcoded radar dependent metadata
    latitude['data'] = np.array([ret.header['RadarLat']], dtype=dtype)
    longitude['data'] = np.array([ret.header['RadarLon']], dtype=dtype)
    altitude['data'] = np.array([ret.header['RadarHeight']], dtype=dtype)
    frequency['data'] = np.array([ret.header['Frequency']], dtype=dtype)
    beamwidth_h['data'] = np.array([1.0], dtype=dtype)
    beamwidth_v['data'] = np.array([1.0], dtype=dtype)

    # Nyquist velocity (+-nv_value)
    nv_value = nyquist_vel(sweep_number['data'][0])

    nyquist_velocity['data'] = nv_value*np.ones(total_record, dtype=dtype)

    # fields
    fields = {}

    # ZH field
    field_name = filemetadata.get_field_name('ZH')
    if field_name is not None:
        # create field dictionary
        field_dic = filemetadata(field_name)
        # incomplete scan
        if not np.all(valid_rays):
            data = np.ma.masked_all(ret.data.shape)
            data[idx_az, :] = ret.data[idx_az, :]
        else:
            data = ret.data
        field_dic['data'] = data
        if rmax > 0:
            field_dic['data'] = field_dic['data'][:, :nrange]
        field_dic['_FillValue'] = get_fillvalue()
        fields[field_name] = field_dic

    # rest of fields
    if bfile.startswith('PM') or bfile.startswith('MS'):
        for i in range(1, NPM_MOM):
            field_name = filemetadata.get_field_name(PM_MOM[i])
            if field_name is not None:
                ret = read_polar_c(
                    filename, PM_MOM[i], physic_value=True, masked_array=True)
                # create field dictionary
                field_dic = filemetadata(field_name)
                if not np.all(valid_rays):
                    # incomplete scan
                    data = np.ma.masked_all(ret.data.shape)
                    data[idx_az, :] = ret.data[idx_az, :]
                else:
                    data = ret.data
                field_dic['data'] = data
                if rmax > 0:
                    field_dic['data'] = field_dic['data'][:, :nrange]
                field_dic['_FillValue'] = get_fillvalue()
                fields[field_name] = field_dic
    elif bfile.startswith('PH') or bfile.startswith('MH'):
        for i in range(1, NPH_MOM):
            field_name = filemetadata.get_field_name(PH_MOM[i])
            if field_name is not None:
                ret = read_polar_c(
                    filename, PH_MOM[i], physic_value=True, masked_array=True)
                # create field dictionary
                field_dic = filemetadata(field_name)
                if not np.all(valid_rays):
                    data = np.ma.masked_all(ret.data.shape)
                    data[idx_az, :] = ret.data[idx_az, :]
                else:
                    data = ret.data
                field_dic['data'] = data
                if rmax > 0:
                    field_dic['data'] = field_dic['data'][:, :nrange]
                field_dic['_FillValue'] = get_fillvalue()
                fields[field_name] = field_dic
    else:
        for i in range(1, NPL_MOM):
            field_name = filemetadata.get_field_name(PL_MOM[i])
            if field_name is not None:
                ret = read_polar_c(
                    filename, PL_MOM[i], physic_value=True, masked_array=True)
                # create field dictionary
                field_dic = filemetadata(field_name)
                if not np.all(valid_rays):
                    data = np.ma.masked_all(ret.data.shape)
                    data[idx_az, :] = ret.data[idx_az, :]
                else:
                    data = ret.data
                field_dic['data'] = data
                if rmax > 0:
                    field_dic['data'] = field_dic['data'][:, :nrange]
                field_dic['_FillValue'] = get_fillvalue()
                fields[field_name] = field_dic

    # instrument_parameters
    instrument_parameters = dict()
    instrument_parameters.update({'frequency': frequency})
    instrument_parameters.update({'radar_beam_width_h': beamwidth_h})
    instrument_parameters.update({'radar_beam_width_v': beamwidth_v})
    instrument_parameters.update({'pulse_width': pulse_width})
    instrument_parameters.update({'nyquist_velocity': nyquist_velocity})

    return Radar(_time, _range, fields, metadata, scan_type, latitude,
                 longitude, altitude, sweep_number, sweep_mode, fixed_angle,
                 sweep_start_ray_index, sweep_end_ray_index, azimuth,
                 elevation, rays_are_indexed=rays_are_indexed,
                 ray_angle_res=ray_angle_res,
                 instrument_parameters=instrument_parameters)


def read_metranet_python(filename, field_names=None, rmax=0.,
                         additional_metadata=None, file_field_names=False,
                         exclude_fields=None, nbytes=4, **kwargs):
    """
    Read a METRANET file.

    Parameters
    ----------
    filename : str
        Name of the METRANET file to read.
    field_names : dict, optional
        Dictionary mapping METRANET field names to radar field names. If a
        data type found in the file does not appear in this dictionary or has
        a value of None it will not be placed in the radar.fields dictionary.
        A value of None, the default, will use the mapping defined in the
        Py-ART configuration file.
    rmax : float, optional
        Maximum radar range to store in the radar object [m]. If 0 all data
        will be stored
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
        after the `file_field_names` and `field_names` parameters.
    nbytes : int
        The number of bytes used to store the data in numpy arrays, e.g. if
        nbytes=4 then floats are going to be stored as np.float32

    Returns
    -------
    radar : Radar
        Radar object containing data from METRANET file.

    """
    if nbytes == 4:
        dtype = np.float32
    elif nbytes == 8:
        dtype = np.float64
    else:
        warn('Number of bytes to store the data ('+str(nbytes) +
             ') not supported. 4 bytes will be used')
        dtype = np.float32

    # test for non empty kwargs
    _test_arguments(kwargs)

    bfile = os.path.basename(filename)

    # create metadata retrieval object
    if field_names is None:
        field_names = METRANET_FIELD_NAMES
    filemetadata = FileMetadata(
        'METRANET', field_names, additional_metadata, file_field_names,
        exclude_fields)

    # get definitions from filemetadata class
    latitude = filemetadata('latitude')
    longitude = filemetadata('longitude')
    altitude = filemetadata('altitude')
    metadata = filemetadata('metadata')
    sweep_start_ray_index = filemetadata('sweep_start_ray_index')
    sweep_end_ray_index = filemetadata('sweep_end_ray_index')

    sweep_number = filemetadata('sweep_number')
    sweep_mode = filemetadata('sweep_mode')
    fixed_angle = filemetadata('fixed_angle')
    elevation = filemetadata('elevation')
    _range = filemetadata('range')
    azimuth = filemetadata('azimuth')
    _time = filemetadata('time')

    # other metadata
    frequency = filemetadata('frequency')
    beamwidth_h = filemetadata('radar_beam_width_h')
    beamwidth_v = filemetadata('radar_beam_width_v')
    pulse_width = filemetadata('pulse_width')
    rays_are_indexed = filemetadata('rays_are_indexed')
    ray_angle_res = filemetadata('ray_angle_res')
    nyquist_velocity = filemetadata('nyquist_velocity')

    ret = read_polar_python(filename, physic_value=True, masked_array=True,
                            reorder_angles=True)
    if ret is None:
        raise ValueError('Unable to read file '+filename)

    # total number of rays composing the sweep
    total_record = ret.data['ZH'].shape[0]
    if total_record == 0:
        raise ValueError('Number of rays in file=0.')

    # M files returning 0 pulse width. Hardcode it for the moment
    # pulse_width['data'] = np.array(
    #    [ret.header['PulseWidth']*1e-6], dtype='float64')
    pulse_width['data'] = 0.5e-6*np.ones(total_record, dtype=dtype)
    rays_are_indexed['data'] = np.array(['true'])
    ray_angle_res['data'] = np.array([1.], dtype=dtype)
    angres = ray_angle_res['data'][0]

    # number of gates in a ray
    num_gates = ret.data['ZH'].shape[1]

    # sweep_number (is the sweep index)
    # current sweep number (from 0 to 19)
    sweep_number['data'] = np.array([ret.header['currentsweep']-1])

    time_data = np.empty(total_record, dtype=dtype)
    ray_index_data = np.empty(total_record, dtype=dtype)

    ant_mode = ret.header['antmode']  # scanning mode code

    no_missing_az = True # will be checked later on
    if ant_mode == 0:
        scan_type = 'ppi'
        sweep_mode['data'] = np.array(['azimuth_surveillance'])
        # ray starting elevation angle information
        fixed_angle['data'] = np.array([ret.pol_header[0]['startangle_el']],
                                       dtype=dtype)

        start_az = np.array([ray['startangle_az'] for ray in ret.pol_header])
        end_az = np.array([ray['endangle_az'] for ray in ret.pol_header])

        start_az = np.deg2rad(start_az)
        end_az = np.deg2rad(end_az)


        az_data = np.rad2deg(
            circmean(np.column_stack((start_az, end_az)), axis=1))

        if len(az_data) != 360:
            # incomplete scan

            az_full_scan = np.arange(0+angres/2, 360+angres/2, angres)
            az_closest = angres*np.floor(az_data/angres)

            idx_az = np.zeros((len(az_data))).astype(int)
            for i, _ in enumerate(idx_az):
                idx_az[i] = np.searchsorted(az_full_scan, az_closest[i])

            corr_az = az_full_scan
            corr_az[idx_az] = az_data
            az_data = corr_az
            no_missing_az = False
        azimuth['data'] = az_data

        # elevation
        elevation['data'] = np.repeat(fixed_angle['data'], total_record)
    elif ant_mode == 1:
        scan_type = 'rhi'
        sweep_mode['data'] = np.array(['elevation_surveillance'])
        # ray starting azimuth angle information
        fixed_angle['data'] = np.array([ret.pol_header[0]['startangle_az']],
                                       dtype=dtype)

        # azimuth
        azimuth['data'] = np.repeat(fixed_angle['data'], total_record)

        elevation['data'] = 0.5 * np.array([
            ray['startangle_el']+ray['endangle_el'] for ray in ret.pol_header])
    elif ant_mode == 2:
        scan_type = 'sector'  # sector scan
        sweep_mode['data'] = np.array(['sector'])
        # ray starting elevation angle information
        fixed_angle['data'] = np.array([ret.pol_header[0]['startangle_el']],
                                       dtype=dtype)

        # azimuth
        start_az = np.array([ray['startangle_az'] for ray in ret.pol_header])
        end_az = np.array([ray['endangle_az'] for ray in ret.pol_header])
        start_az = np.deg2rad(start_az)
        end_az = np.deg2rad(end_az)


        az_data = np.rad2deg(
            circmean(np.column_stack((start_az, end_az)), axis=1))

        if len(az_data) != 360:
            # incomplete scan

            az_full_scan = np.arange(0+angres/2, 360+angres/2, angres)
            az_closest = angres*np.floor(az_data/angres)

            idx_az = np.zeros((len(az_data))).astype(int)
            for i, _ in enumerate(idx_az):
                idx_az[i] = np.searchsorted(az_full_scan, az_closest[i])

            corr_az = az_full_scan
            corr_az[idx_az] = az_data
            az_data = corr_az
            no_missing_az = False

        azimuth['data'] = az_data

        # elevation
        elevation['data'] = np.repeat(fixed_angle['data'], total_record)
    elif ant_mode == 3:
        scan_type = 'other'  # point of interest scan
        sweep_mode['data'] = np.array(['pointing'])
        # ray starting elevation angle information
        fixed_angle['data'] = np.array([ret.pol_header[0]['startangle_el']],
                                       dtype=dtype)

        # azimuth
        # ray starting elevation angle information
        azimuth['data'] = np.array(
            [ret.pol_header[0]['startangle_az']], dtype=dtype)

        # elevation
        elevation['data'] = fixed_angle['data']
    elif ant_mode == 4:
        scan_type = 'other'  # off
        sweep_mode['data'] = np.array(['idle'])
    else:
        raise ValueError('Unknown scan type')

    # range (to center of beam [m])
    # distance to start of first range gate [usually 0 m]
    start_range = float(ret.header['startrange'])
    # range resolution [m]
    gate_width = float(ret.header['gatewidth'])*1000.
    _range['data'] = np.linspace(
        start_range+gate_width/2., float(num_gates-1.) *
        gate_width+gate_width/2., num_gates, dtype=dtype)

    if rmax > 0.:
        _range['data'] = _range['data'][_range['data'] < rmax]
        nrange = len(_range['data'])

    # time (according to default_config this is the Time at the center of
    # each ray, in fractional seconds since the volume started)
    # here we find the time of end of ray since the first ray in the sweep
    for i in range(total_record):
        # time when the ray was created [s from 1.1.1970].
        # (last received pulse+processing time)
        data_time = float(ret.pol_header[i]['datatime'])
        # the hundreths of seconds to add to the data_time
        data_time_residue = float(ret.pol_header[i]['datatime_residue'])
        time_data[i] = data_time+data_time_residue/100.

    ray_index_data = range(total_record)

    sweep_start = min(time_data)
    start_time = datetime.datetime.utcfromtimestamp(sweep_start)
    _time['data'] = time_data-sweep_start
    _time['units'] = make_time_unit_str(start_time)

    # sweep_start_ray_index, sweep_end_ray_index
    # should be specified since start of volume but we do not have this
    # information so we specify it since start of sweep instead.

    if no_missing_az:
        sweep_start_ray_index['data'] = np.array(
            [min(ray_index_data)], dtype='int32')  # ray index of first ray
        sweep_end_ray_index['data'] = np.array(
            [max(ray_index_data)], dtype='int32')   # ray index of last ray
    else:
        sweep_start_ray_index['data'] = np.array(
            [0], dtype='int32')  # ray index of first ray
        sweep_end_ray_index['data'] = np.array(
            [360], dtype='int32')   # ray index of last ray

    # ----  other information that can be obtained from metadata in file
    #       sweep information:
    #       total number of sweeps compositing the volume (i.e. 20):
    #       total_sweep=ret.pol_header[0].total_sweep
    #       total number of data bytes in the ray
    #   (num_gates*number_of_moments*(number_of_bytes in each moment)):
    #       data_bytes=ret.pol_header[0].data_bytes
    #       # time period of repetition of the volume scan:
    #       repeat_time=ret.pol_header[0].repeat_time
    #       # Nyquist velocity [m/s]:
    #       ny_quest=ret.pol_header[0].ny_quest
    #       Maximum Doppler spectrum width [m/s]:
    #       w_ny_quest=ret.pol_header[0].w_ny_quest
    #
    #       # ray specific information
    #       0 no end of sweep, 1 end of sweep, 2 end of volume scan:
    #       end_of_sweep=ret.pol_header[0].end_of_sweep
    #       number of pulses used in data integration:
    #       pulses=ret.pol_header[0].pulses
    # ------------------------------------------------------------------

    # metadata
    # get radar id
    if isinstance(ret.header["radarname"], str):
        radar_id = ret.header["radarname"]
    else:
        radar_id = ret.header["radarname"].decode('utf-8')

    metadata['instrument_name'] = radar_id

    # hardcoded radar dependent metadata
    latitude['data'] = np.array([ret.header['radarlat']], dtype=dtype)
    longitude['data'] = np.array([ret.header['radarlon']], dtype=dtype)
    altitude['data'] = np.array([ret.header['radarheight']], dtype=dtype)
    frequency['data'] = np.array([ret.header['frequency']], dtype=dtype)
    beamwidth_h['data'] = np.array([1.0], dtype=dtype)
    beamwidth_v['data'] = np.array([1.0], dtype=dtype)

    # Nyquist velocity (+-nv_value)
    nv_value = nyquist_vel(sweep_number['data'][0])

    nyquist_velocity['data'] = nv_value*np.ones(total_record, dtype=dtype)

    # fields
    fields = {}

    # rest of fields
    if bfile.startswith('MS'):
        nmoments = NPM_MOM
        momnames = PM_MOM
    elif bfile.startswith('MH'):
        nmoments = NPH_MOM
        momnames = PH_MOM
    else:
        nmoments = NPL_MOM
        momnames = PL_MOM

    for i in range(0, nmoments):
        field_name = filemetadata.get_field_name(momnames[i])
        if field_name is not None:

            # create field dictionary
            field_dic = filemetadata(field_name)
            data = ret.data[momnames[i]]

            # Check if scan is complete
            if len(data) != len(azimuth['data']):
                tmp = np.ma.masked_all((len(azimuth['data']), data.shape[1]))
                tmp[idx_az, :] = data
                data = tmp

            field_dic['data'] = data

            if rmax > 0:
                field_dic['data'] = field_dic['data'][:, :nrange]

            field_dic['_FillValue'] = get_fillvalue()
            fields[field_name] = field_dic

    # instrument_parameters
    instrument_parameters = dict()
    instrument_parameters.update({'frequency': frequency})
    instrument_parameters.update({'radar_beam_width_h': beamwidth_h})
    instrument_parameters.update({'radar_beam_width_v': beamwidth_v})
    instrument_parameters.update({'pulse_width': pulse_width})
    instrument_parameters.update({'nyquist_velocity': nyquist_velocity})

    return Radar(_time, _range, fields, metadata, scan_type, latitude,
                 longitude, altitude, sweep_number, sweep_mode, fixed_angle,
                 sweep_start_ray_index, sweep_end_ray_index, azimuth,
                 elevation, rays_are_indexed=rays_are_indexed,
                 ray_angle_res=ray_angle_res,
                 instrument_parameters=instrument_parameters)
