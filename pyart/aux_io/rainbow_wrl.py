"""
pyart.aux_io.rainbow
====================

Routines for reading RAINBOW files (Used by SELEX) using the wradlib library

.. autosummary::
    :toctree: generated/

    read_rainbow_wrl
    _get_angle
    _get_data
    _get_time

"""

# specific modules for this function
import os
from warnings import warn
import datetime

import numpy as np

from ..config import FileMetadata, get_fillvalue
from ..io.common import make_time_unit_str, _test_arguments
from ..core.radar import Radar
from ..exceptions import MissingOptionalDependency

try:
    # `read_rainbow` as of wradlib version 1.0.0
    from wradlib.io import read_Rainbow as read_rainbow
    _WRADLIB_AVAILABLE = True
except ImportError:
    try:
        from wradlib.io import read_rainbow
        _WRADLIB_AVAILABLE = True
    except ImportError:
        _WRADLIB_AVAILABLE = False

RAINBOW_FIELD_NAMES = {
    'W': 'spectrum_width',
    'Wv': 'spectrum_width_vv',  # non standard name
    'Wu': 'unfiltered_spectrum_width',  # non standard name
    'Wvu': 'unfiltered_spectrum_width_vv',  # non standard name
    'V': 'velocity',
    'Vv': 'velocity_vv',  # non standard name
    'Vu': 'unfiltered_velocity',  # non standard name
    'Vvu': 'unfiltered_velocity_vv',  # non standard name
    'dBZ': 'reflectivity',
    'dBZv': 'reflectivity_vv',       # non standard name
    'dBuZ': 'unfiltered_reflectivity',  # non standard name
    'dBuZv': 'unfiltered_reflectivity_vv',  # non standard name
    'ZDR': 'differential_reflectivity',
    'ZDRu': 'unfiltered_differential_reflectivity',  # non standard name
    'RhoHV': 'cross_correlation_ratio',
    'RhoHVu': 'unfiltered_cross_correlation_ratio',  # non standard name
    'PhiDP': 'differential_phase',
    'uPhiDP': 'uncorrected_differential_phase',  # non standard name
    'uPhiDPu':
        'uncorrected_unfiltered_differential_phase',  # non standard name
    'KDP': 'specific_differential_phase',
    'uKDP': 'uncorrected_specific_differential_phase',  # non standard name
    'uKDPu':                                            # non standard name
        'uncorrected_unfiltered_specific_differential_phase',
    'SQI': 'signal_quality_index',  # non standard name
    'SQIv': 'signal_quality_index_vv',  # non standard name
    'SQIu': 'unfiltered_signal_quality_index',  # non standard name
    'SQIvu': 'unfiltered_signal_quality_index_vv',  # non standard name
    'TEMP': 'temperature',  # non standard name
    'ISO0': 'iso0',  # non standard name
    'VIS': 'visibility'  # non standard name
}

PULSE_WIDTH_VEC = [0.33e-6, 0.5e-6, 1.2e-6, 2.0e-6]  # pulse width [s]


def read_rainbow_wrl(filename, field_names=None, additional_metadata=None,
                     file_field_names=False, exclude_fields=None,
                     include_fields=None, nbytes=4, **kwargs):
    """
    Read a RAINBOW file.
    This routine has been tested to read rainbow5 files version 5.22.3,
    5.34.16 and 5.35.1.
    Since the rainbow file format is evolving constantly there is no guaranty
    that it can work with other versions.
    If necessary, the user should adapt to code according to its own
    file version and raise an issue upstream.

    Data types read by this routine:
    Reflectivity: dBZ, dBuZ, dBZv, dBuZv
    Velocity: V, Vu, Vv, Vvu
    Spectrum width: W, Wu, Wv, Wvu
    Differential reflectivity: ZDR, ZDRu
    Co-polar correlation coefficient: RhoHV, RhoHVu
    Co-polar differential phase: PhiDP, uPhiDP, uPhiDPu
    Specific differential phase: KDP, uKDP, uKDPu
    Signal quality parameters: SQI, SQIu, SQIv, SQIvu
    Temperature: TEMP
    Position of the range bin respect to the ISO0: ISO0
    radar visibility according to Digital Elevation Model (DEM): VIS

    Parameters
    ----------
    filename : str
        Name of the RAINBOW file to read.
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
        introduce any addition metadata and the file specific or default
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
    nbytes : int
        The number of bytes used to store the data in numpy arrays, e.g. if
        nbytes=4 then floats are going to be stored as np.float32


    Returns
    -------
    radar : Radar
        Radar object containing data from RAINBOW file.

    """
    # check that wradlib is available
    if not _WRADLIB_AVAILABLE:
        raise MissingOptionalDependency(
            "wradlib is required to use read_rainbow_wrl but is not installed")

    # test for non empty kwargs
    _test_arguments(kwargs)

    # check if it is the right file. Open it and read it
    bfile = os.path.basename(filename)
    supported_file = (bfile.endswith('.vol') or bfile.endswith('.azi') or
                      bfile.endswith('.ele') or bfile.endswith('poi'))
    if not supported_file:
        raise ValueError(
            'Only data files with extension .vol, .azi or .ele are supported')

    if nbytes == 4:
        dtype = np.float32
    elif nbytes == 8:
        dtype = np.float64
    else:
        warn('Number of bytes to store the data ('+str(nbytes) +
             ') not supported. 4 bytes will be used')
        dtype = np.float32

    # create metadata retrieval object
    if field_names is None:
        field_names = RAINBOW_FIELD_NAMES
    filemetadata = FileMetadata('RAINBOW', field_names, additional_metadata,
                                file_field_names, exclude_fields,
                                include_fields)

    try:
        with open(filename, 'rb') as fid:
            rbf = read_rainbow(fid, loaddata=True)
    except OSError as ee:
        warn(str(ee))
        warn('Unable to read file '+filename)
        return None

    # check the number of slices
    nslices = int(rbf['volume']['scan']['pargroup']['numele'])
    if nslices > 1:
        single_slice = False
        common_slice_info = rbf['volume']['scan']['slice'][0]
    else:
        single_slice = True
        common_slice_info = rbf['volume']['scan']['slice']

    # check the data type
    # all slices should have the same data type
    datatype = common_slice_info['slicedata']['rawdata']['@type']
    field_name = filemetadata.get_field_name(datatype)
    if field_name is None:
        raise ValueError('Field Name Unknown')

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
    field_dic = filemetadata(field_name)

    # other metadata
    frequency = filemetadata('frequency')
    rad_cal_h = filemetadata('calibration_constant_hh')
    rad_cal_v = filemetadata('calibration_constant_vv')
    tx_pwr_h = filemetadata('transmit_power_h')
    tx_pwr_v = filemetadata('transmit_power_v')
    beamwidth_h = filemetadata('radar_beam_width_h')
    beamwidth_v = filemetadata('radar_beam_width_v')
    pulse_width = filemetadata('pulse_width')
    rays_are_indexed = filemetadata('rays_are_indexed')
    ray_angle_res = filemetadata('ray_angle_res')

    # get general file information

    # position and radar frequency
    if 'sensorinfo' in rbf['volume'].keys():
        latitude['data'] = np.array(
            [rbf['volume']['sensorinfo']['lat']], dtype=dtype)
        longitude['data'] = np.array(
            [rbf['volume']['sensorinfo']['lon']], dtype=dtype)
        altitude['data'] = np.array(
            [rbf['volume']['sensorinfo']['alt']], dtype=dtype)
        frequency['data'] = np.array(
            [3e8 / float(rbf['volume']['sensorinfo']['wavelen'])],
            dtype=dtype)
        beamwidth_h['data'] = np.array(
            [float(rbf['volume']['sensorinfo']['beamwidth'])])
        beamwidth_v['data'] = beamwidth_h['data']
    elif 'radarinfo' in rbf['volume'].keys():
        latitude['data'] = np.array(
            [rbf['volume']['radarinfo']['@lat']], dtype=dtype)
        longitude['data'] = np.array(
            [rbf['volume']['radarinfo']['@lon']], dtype=dtype)
        altitude['data'] = np.array(
            [rbf['volume']['radarinfo']['@alt']], dtype=dtype)
        frequency['data'] = np.array(
            [3e8 / float(rbf['volume']['radarinfo']['wavelen'])],
            dtype=dtype)
        beamwidth_h['data'] = np.array(
            [float(rbf['volume']['radarinfo']['beamwidth'])], dtype=dtype)
        beamwidth_v['data'] = beamwidth_h['data']

    # antenna speed
    if 'antspeed' in common_slice_info:
        ant_speed = float(common_slice_info['antspeed'])
    else:
        ant_speed = 10.
        print('WARNING: Unable to read antenna speed. Default value of ' +
              str(ant_speed) + ' deg/s will be used')

    # angle step and sampling mode
    angle_step = float(common_slice_info['anglestep'])
    rays_are_indexed['data'] = None
    ray_angle_res['data'] = None
    if 'fixselect' in common_slice_info:
        if common_slice_info['fixselect'] == 'AngleStep':
            if single_slice:
                rays_are_indexed['data'] = np.array(['true'])
                ray_angle_res['data'] = np.array([angle_step], dtype=dtype)
            else:
                rays_are_indexed['data'] = list()
                ray_angle_res['data'] = np.empty(nslices, dtype=dtype)
                for i in range(nslices):
                    rays_are_indexed['data'].append('true')
                    ray_angle_res['data'][i] = angle_step
                rays_are_indexed['data'] = np.array(rays_are_indexed['data'])
        elif common_slice_info['fixselect'] == 'TimeSamp':
            rays_are_indexed['data'] = list()
            if single_slice:
                rays_are_indexed['data'] = np.array(['false'])
            else:
                for i in range(nslices):
                    rays_are_indexed['data'].append('false')
                rays_are_indexed['data'] = np.array(rays_are_indexed['data'])
        else:
            warn('Unknown sampling mode')

    # sweep_number (is the sweep index)
    sweep_number['data'] = np.arange(nslices, dtype='int32')

    # get number of rays and number of range bins per sweep
    rays_per_sweep = np.empty(nslices, dtype='int32')

    if single_slice:
        rays_per_sweep[0] = int(
            common_slice_info['slicedata']['rawdata']['@rays'])
        nbins = int(common_slice_info['slicedata']['rawdata']['@bins'])
        ssri = np.array([0], dtype='int32')
        seri = np.array([rays_per_sweep[0] - 1], dtype='int32')
    else:
        # number of range bins per ray in sweep
        nbins_sweep = np.empty(nslices, dtype='int32')
        for i in range(nslices):
            slice_info = rbf['volume']['scan']['slice'][i]
            # number of rays per sweep
            rays_per_sweep[i] = int(
                slice_info['slicedata']['rawdata']['@rays'])

            # number of range bins per ray in sweep
            nbins_sweep[i] = int(
                slice_info['slicedata']['rawdata']['@bins'])

        # all sweeps have to have the same number of range bins
        if any(nbins_sweep != nbins_sweep[0]):
            raise ValueError('number of range bins changes between sweeps')
        nbins = nbins_sweep[0]
        ssri = np.cumsum(np.append([0], rays_per_sweep[:-1])).astype('int32')
        seri = np.cumsum(rays_per_sweep).astype('int32') - 1

    # total number of rays and sweep start ray index and end
    total_rays = sum(rays_per_sweep)
    sweep_start_ray_index['data'] = ssri
    sweep_end_ray_index['data'] = seri

    # pulse width and calibration constant
    pulse_width['data'] = None
    rad_cal_h['data'] = None
    rad_cal_v['data'] = None
    tx_pwr_h['data'] = None
    tx_pwr_v['data'] = None
    if 'pw_index' in common_slice_info:
        pw_index = int(common_slice_info['pw_index'])

        pulse_width['data'] = PULSE_WIDTH_VEC[pw_index]*np.ones(
            total_rays, dtype=dtype)

        # calibration constant
        if 'rspdphradconst' in common_slice_info:
            cal_vec = common_slice_info['rspdphradconst'].split()
            rad_cal_h['data'] = np.array(
                [float(cal_vec[pw_index])], dtype=dtype)

        if 'rspdpvradconst' in common_slice_info:
            cal_vec = common_slice_info['rspdpvradconst'].split()
            rad_cal_v['data'] = np.array(
                [float(cal_vec[pw_index])], dtype=dtype)

        # magnetron transmit power
        if 'gdrxmaxpowkw' in common_slice_info:
            tx_pwr_dBm = (
                10.*np.log10(float(common_slice_info['gdrxmaxpowkw'])*1e3)+30.)
            tx_pwr_h['data'] = np.array([tx_pwr_dBm], dtype=dtype)
            tx_pwr_v['data'] = np.array([tx_pwr_dBm], dtype=dtype)

    # range
    r_res = float(common_slice_info['rangestep']) * 1000.
    if 'start_range' in common_slice_info.keys():
        start_range = float(common_slice_info['start_range']) * 1000.
    else:
        start_range = 0.
    _range['data'] = np.linspace(
        start_range+r_res / 2., float(nbins - 1.) * r_res+r_res / 2.,
        nbins).astype(dtype)

    # containers for data
    t_fixed_angle = np.empty(nslices, dtype=dtype)
    moving_angle = np.empty(total_rays, dtype=dtype)
    static_angle = np.empty(total_rays, dtype=dtype)
    time_data = np.empty(total_rays, dtype=dtype)
    fdata = np.ma.zeros((total_rays, nbins), dtype=dtype,
                        fill_value=get_fillvalue())

    # read data from file
    if bfile.endswith('.vol') or bfile.endswith('.azi'):
        scan_type = 'ppi'
        sweep_mode['data'] = np.array(nslices * ['azimuth_surveillance'])
    elif bfile.endswith('.ele'):
        scan_type = 'rhi'
        sweep_mode['data'] = np.array(['elevation_surveillance'])
    else:
        scan_type = 'other'
        sweep_mode['data'] = np.array(['pointing'])

    # read data from file:
    for i in range(nslices):
        if single_slice:
            slice_info = common_slice_info
        else:
            slice_info = rbf['volume']['scan']['slice'][i]

        # fixed angle
        if scan_type == 'other':
            t_fixed_angle[i] = float(slice_info['posazi'])
        else:
            t_fixed_angle[i] = float(slice_info['posangle'])

        # fixed angle (repeated for each ray)
        static_angle[ssri[i]: seri[i]+1] = t_fixed_angle[i]

        # moving angle
        moving_angle[ssri[i]: seri[i]+1], angle_start, angle_stop = (
            _get_angle(slice_info['slicedata']['rayinfo'],
                       angle_step=angle_step, scan_type=scan_type,
                       dtype=dtype))

        # time
        if (isinstance(slice_info['slicedata']['rayinfo'], dict) or
                len(slice_info['slicedata']['rayinfo']) == 2):
            time_data[ssri[i]:seri[i]+1], sweep_start = _get_time(
                slice_info['slicedata']['@date'],
                slice_info['slicedata']['@time'], angle_start[0],
                angle_stop[-1], angle_step, rays_per_sweep[i], ant_speed,
                scan_type=scan_type)
        else:
            sweep_start = datetime.datetime.strptime(
                slice_info['slicedata']['@datetimehighaccuracy'],
                '%Y-%m-%dT%H:%M:%S.%f')
            time_data[ssri[i]:seri[i]+1] = np.array(
                slice_info['slicedata']['rayinfo'][2]['data']*1e-3,
                dtype=np.float64)

        if i == 0:
            start_time = sweep_start
        else:
            time_data[ssri[i]:seri[i]+1] += (
                (sweep_start-start_time).total_seconds())

        # data
        fdata[ssri[i]:seri[i]+1, :] = _get_data(
            slice_info['slicedata']['rawdata'],
            rays_per_sweep[i], nbins, dtype=dtype)

    if bfile.endswith('.vol') or bfile.endswith('.azi'):
        azimuth['data'] = moving_angle
        elevation['data'] = static_angle
    else:
        azimuth['data'] = static_angle
        elevation['data'] = moving_angle

    fixed_angle['data'] = t_fixed_angle

    _time['data'] = time_data
    _time['units'] = make_time_unit_str(start_time)

    # fields
    fields = {}
    # create field dictionary
    field_dic['_FillValue'] = get_fillvalue()
    field_dic['data'] = fdata
    fields[field_name] = field_dic

    # metadata
    # metadata['instrument_name'] = radar_id

    # instrument_parameters
    instrument_parameters = dict()
    instrument_parameters.update({'frequency': frequency})
    instrument_parameters.update({'radar_beam_width_h': beamwidth_h})
    instrument_parameters.update({'radar_beam_width_v': beamwidth_v})
    if pulse_width['data'] is not None:
        instrument_parameters.update({'pulse_width': pulse_width})

    # radar calibration parameters
    radar_calibration = None
    if ((rad_cal_h['data'] is not None) or (rad_cal_v['data'] is not None) or
            (tx_pwr_h['data'] is not None) or (tx_pwr_v['data'] is not None)):
        radar_calibration = dict()
        if rad_cal_h['data'] is not None:
            radar_calibration.update({'calibration_constant_hh': rad_cal_h})
        if rad_cal_v['data'] is not None:
            radar_calibration.update({'calibration_constant_vv': rad_cal_v})
        if tx_pwr_h['data'] is not None:
            radar_calibration.update({'transmit_power_h': tx_pwr_h})
        if tx_pwr_v['data'] is not None:
            radar_calibration.update({'transmit_power_v': tx_pwr_v})

    # angle res
    if rays_are_indexed['data'] is None:
        rays_are_indexed = None
        ray_angle_res = None

    if rays_are_indexed is not None:
        if rays_are_indexed['data'][0] == 'false':
            ray_angle_res = None

    return Radar(_time, _range, fields, metadata, scan_type, latitude,
                 longitude, altitude, sweep_number, sweep_mode, fixed_angle,
                 sweep_start_ray_index, sweep_end_ray_index, azimuth,
                 elevation, rays_are_indexed=rays_are_indexed,
                 ray_angle_res=ray_angle_res,
                 instrument_parameters=instrument_parameters,
                 radar_calibration=radar_calibration)


def _get_angle(ray_info, angle_step=None, scan_type='ppi', dtype=np.float32):
    """
    obtains the ray angle start, stop and center

    Parameters
    ----------
    ray_info : dictionary of dictionaries
        contains the ray info
    angle_step : float
        Optional. The angle step. Used in case there is no information of
        angle stop. Otherwise ignored.
    scan_type : str
        Default ppi. scan_type. Either ppi or rhi.
    dtype : numpy data type object
        The data type of the numpy array where the angles are stored

    Returns
    -------
    moving_angle : numpy array
        the central point of the angle [Deg]
    angle_start :
        the starting point of the angle [Deg]
    angle_stop :
        the end point of the angle [Deg]

    """
    bin_to_deg = 360./65536.

    def _extract_angles(data):
        angle = np.array(data * bin_to_deg, dtype=dtype)
        if scan_type == 'rhi':
            ind = (angle > 225.).nonzero()
            angle[ind] -= 360.
        return angle

    try:
        angle_start = _extract_angles(ray_info['data'])
        if angle_step is None:
            raise ValueError('Unknown angle step')
        angle_stop = angle_start + angle_step
    except TypeError:
        angle_start = _extract_angles(ray_info[0]['data'])
        angle_stop = _extract_angles(ray_info[1]['data'])

    moving_angle = np.angle((np.exp(1.j * np.deg2rad(angle_start)) +
                             np.exp(1.j * np.deg2rad(angle_stop))) / 2.,
                            deg=True)
    moving_angle[moving_angle < 0.] += 360.  # [0, 360]

    return moving_angle, angle_start, angle_stop


def _get_data(rawdata, nrays, nbins, dtype=np.float32):
    """
    Obtains the raw data

    Parameters
    ----------
    rawdata : dictionary of dictionaries
        contains the raw data information
    nrays : int
        Number of rays in sweep
    nbins : int
        Number of bins in ray
    dtype : numpy data type object
        The data type of the numpy array where the data is stored

    Returns
    -------
    data : numpy array
        the data

    """
    databin = rawdata['data']
    datamin = float(rawdata['@min'])
    datamax = float(rawdata['@max'])
    datadepth = float(rawdata['@depth'])
    datatype = rawdata['@type']

    data = np.array(
        datamin+(databin-1)*(datamax-datamin)/(2**datadepth-2),
        dtype=dtype)

    # fill invalid data with fill value
    mask = databin == 0
    data[mask.nonzero()] = get_fillvalue()

    # put phidp data in the range [-180, 180]
    if datatype in ('PhiDP', 'uPhiDP', 'uPhiDPu'):
        is_above_180 = data > 180.
        data[is_above_180.nonzero()] -= 360.

    data = np.reshape(data, [nrays, nbins])
    mask = np.reshape(mask, [nrays, nbins])

    masked_data = np.ma.array(data, mask=mask, fill_value=get_fillvalue())

    return masked_data


def _get_time(date_sweep, time_sweep, first_angle_start, last_angle_stop,
              angle_step, nrays, ant_speed, scan_type='ppi'):
    """
    Computes the time at the center of each ray

    Parameters
    ----------
    date_sweep, time_sweep : str
        the date and time of the sweep
    first_angle_start : float
        The starting point of the first angle in the sweep
    last_angle_stop : float
        The end point of the last angle in the sweep
    nrays : int
        Number of rays in sweep
    ant_speed : float
        antenna speed [deg/s]
    scan_type : str
        Default ppi. scan_type. Either ppi or rhi.

    Returns
    -------
    time_data : numpy array
        the time of each ray since sweep start
    sweep_start : datetime object
        sweep start time

    """
    sweep_start = datetime.datetime.strptime(
        date_sweep+' '+time_sweep, '%Y-%m-%d %H:%M:%S')
    if scan_type in ('ppi', 'other'):
        if (last_angle_stop > first_angle_start) and (
                np.round((last_angle_stop - first_angle_start) /
                         nrays, decimals=2) >= angle_step):
            sweep_duration = (last_angle_stop - first_angle_start) / ant_speed
        elif (last_angle_stop < first_angle_start) and (
                np.round((first_angle_start - last_angle_stop) /
                         nrays, decimals=2) >= angle_step):
            sweep_duration = (first_angle_start - last_angle_stop) / ant_speed
        else:
            sweep_duration = (
                last_angle_stop + 360. - first_angle_start) / ant_speed
    else:
        if last_angle_stop > first_angle_start:
            sweep_duration = (last_angle_stop - first_angle_start) / ant_speed
        else:
            sweep_duration = (first_angle_start - last_angle_stop) / ant_speed

    time_angle = sweep_duration/nrays

    time_data = np.linspace(
        time_angle / 2., sweep_duration-time_angle / 2., num=nrays)

    return time_data, sweep_start
