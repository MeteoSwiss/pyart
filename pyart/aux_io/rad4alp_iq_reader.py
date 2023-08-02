"""
pyart.aux_io.rainbow_psr
========================

Routines for reading RAINBOW PSR files (Used by SELEX)

.. autosummary::
    :toctree: generated/

    read_iq
    read_iq_data

"""

# specific modules for this function
import os
from copy import deepcopy
from warnings import warn

import numpy as np
from scipy.constants import speed_of_light

from ..config import FileMetadata
from ..core.radar_spectra import RadarSpectra
from ..io.common import _test_arguments
from ..util import subset_radar
from .metranet_reader import read_metranet
from .rainbow_psr import get_Doppler_info

IQ_FIELD_NAMES = {
    'IQhhADU': 'IQ_hh_ADU',
    'IQvvADU': 'IQ_vv_ADU',
}


def read_iq(filename, filenames_iq, field_names=None,
            additional_metadata=None, file_field_names=False,
            exclude_fields=None, include_fields=None, reader='C', nbytes=4,
            prf=None, ang_tol=0.4, noise_h=None, noise_v=None, rconst_h=None,
            rconst_v=None, radconst_h=None, radconst_v=None, mfloss_h=1.,
            mfloss_v=1., azi_min=None, azi_max=None, ele_min=None,
            ele_max=None, rng_min=None, rng_max=None, **kwargs):
    """
    Read a rad4alp IQ file.

    Parameters
    ----------
    filename : str
        Name of the METRANET file to be used as reference.
    filenames_iq : list of str
        Name of the IQ files
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
    reader : str
        The library used to read the METRANET reference file. Can be either
        'C' or 'python'
    nbytes : int
        The number of bytes used to store the data in numpy arrays, e.g. if
        nbytes=4 then floats are going to be stored as np.float6432
    prf : float
        The PRF of the read scan
    ang_tol : float
        Tolerated angle distance between nominal radar angle and angle in
        PSR files
    noise_h, noise_v : float
        The estimated H(V) noise power (ADU) of the scan
    rconst_h, rconst_v : float
        Dynamical factor used in the conversion from dBADU to dBm/dBZ
    radconst_h, radconst_v : float
        The H(V) radar constant
    mfloss_h, mfloss_v : float
        The H(V) matched filter losses in the receiver (dB)
    azi_min, azi_max, ele_min, ele_max : float or None
        The minimum and maximum angles to keep (deg)
    rng_min, rng_max : float or None
        The minimum and maximum ranges to keep (m)

    Returns
    -------
    radar : Radar
        Radar object containing data from PSR file.

    """
    # test for non empty kwargs
    _test_arguments(kwargs)

    # create radar object used as reference
    try:
        radar = read_metranet(
            filename, field_names={'ZH': 'reflectivity'}, reader=reader,
            nbytes=nbytes)
        radar.fields = dict()
        rng_orig = radar.range['data']
        radar = subset_radar(
            radar, None, rng_min=None, rng_max=None, ele_min=ele_min,
            ele_max=ele_max, azi_min=azi_min, azi_max=azi_max)
        if radar is None:
            warn('No data within specified azimuth, elevation and'
                 ' range limits')
            return None
        ind_rng_start = np.where(rng_orig == radar.range['data'][0])[0]
        ind_rng_end = np.where(rng_orig == radar.range['data'][-1])[0]
        ind_rng = np.arange(ind_rng_start, ind_rng_end + 1, dtype=int)
    except OSError as ee:
        warn(str(ee))
        warn('Unable to read file ' + filename)
        return None

    # create metadata retrieval object
    if field_names is None:
        field_names = IQ_FIELD_NAMES.values()
    filemetadata = FileMetadata('IQ', field_names, additional_metadata,
                                file_field_names, exclude_fields,
                                include_fields)
    npulses = filemetadata('number_of_pulses')
    dBADU_to_dBm_hh = filemetadata('dBADU_to_dBm_hh')
    dBADU_to_dBm_vv = filemetadata('dBADU_to_dBm_vv')
    mfloss_h_dict = filemetadata('matched_filter_loss_h')
    mfloss_v_dict = filemetadata('matched_filter_loss_v')
    prt = filemetadata('prt')

    # Keep only IQ files with data from radar object
    filenames_iq_aux, npulses['data'] = get_valid_rays(
        filenames_iq, radar.azimuth['data'], radar.fixed_angle['data'][0],
        ang_tol=ang_tol)

    if filenames_iq_aux is None:
        return None

    # Select valid range gates
    if rng_min is None:
        rng_min = 0.
    if rng_max is None:
        rng_max = np.max(radar.range['data'])

    ind_rng = np.where(np.logical_and(
        radar.range['data'] >= rng_min,
        radar.range['data'] <= rng_max))[0]

    if ind_rng.size == 0:
        warn(
            'No range bins between ' +
            str(rng_min) +
            ' and ' +
            str(rng_max) +
            ' m')
        return None

    if ind_rng.size == radar.ngates:
        ind_rng = None

    # Read IQ data
    if 'IQ_hh_ADU' in field_names or 'IQ_vv_ADU' in field_names:
        npulses_max = np.max(npulses['data'])
        data_hh = np.ma.masked_all(
            (radar.nrays, radar.ngates, npulses_max), dtype=np.complex64)
        data_vv = np.ma.masked_all(
            (radar.nrays, radar.ngates, npulses_max), dtype=np.complex64)
        for i, npuls in enumerate(npulses['data']):
            data_ray_hh, data_ray_vv = read_iq_data(
                filenames_iq_aux[i], radar.ngates, npuls)
            if data_ray_hh is not None:
                data_hh[i, :, 0:npuls] = data_ray_hh
            if data_ray_vv is not None:
                data_vv[i, :, 0:npuls] = data_ray_vv

        if ind_rng is not None:
            data_hh = data_hh[:, ind_rng, :]
            data_vv = data_vv[:, ind_rng, :]

    # cut radar range
    if ind_rng is not None:
        radar.range['data'] = radar.range['data'][ind_rng]
        radar.init_gate_x_y_z()
        radar.init_gate_longitude_latitude()
        radar.init_gate_altitude()
        radar.ngates = ind_rng.size

    # Prepare output fields
    fields = {}
    for field_name in field_names:
        field_dict = filemetadata(field_name)
        if field_name == 'IQ_hh_ADU':
            field_dict['data'] = data_hh
        elif field_name == 'IQ_vv_ADU':
            field_dict['data'] = data_vv
        elif field_name == 'IQ_noise_power_hh_ADU' and noise_h is not None:
            npulses_max = np.max(npulses['data'])
            field_dict['data'] = np.ma.masked_all(
                (radar.nrays, radar.ngates, npulses_max), dtype=np.float64)
            for i, npuls in enumerate(npulses['data']):
                field_dict['data'][i, :, 0:npuls] = noise_h
        elif field_name == 'IQ_noise_power_vv_ADU' and noise_v is not None:
            npulses_max = np.max(npulses['data'])
            field_dict['data'] = np.ma.masked_all(
                (radar.nrays, radar.ngates, npulses_max), dtype=np.float64)
            for i, npuls in enumerate(npulses['data']):
                field_dict['data'][i, :, 0:npuls] = noise_v
        else:
            warn('Field name ' + field_name + ' not known')
            continue
        fields[field_name] = field_dict

    # get further metadata
    if rconst_h is not None and radconst_h is not None and mfloss_h is not None:
        dBADU_to_dBm_hh['data'] = np.array(
            [rconst_h - 40. - radconst_h - mfloss_h])
        if radar.radar_calibration is None:
            radar.radar_calibration = {'dBADU_to_dBm_hh': dBADU_to_dBm_hh}
        else:
            radar.radar_calibration.update(
                {'dBADU_to_dBm_hh': dBADU_to_dBm_hh})
    else:
        warn('Unable to compute dBADU_to_dBm_hh. Missing data')

    if rconst_v is not None and radconst_v is not None and mfloss_v is not None:
        dBADU_to_dBm_vv['data'] = np.array(
            [rconst_v - 40. - radconst_v - mfloss_v])
        if radar.radar_calibration is None:
            radar.radar_calibration = {'dBADU_to_dBm_vv': dBADU_to_dBm_vv}
        else:
            radar.radar_calibration.update(
                {'dBADU_to_dBm_vv': dBADU_to_dBm_vv})
    else:
        warn('Unable to compute dBADU_to_dBm_vv. Missing data')

    if mfloss_h is not None:
        mfloss_h_dict['data'] = np.array([mfloss_h])
        if radar.radar_calibration is None:
            radar.radar_calibration = {'matched_filter_loss_h': mfloss_h_dict}
        else:
            radar.radar_calibration.update(
                {'matched_filter_loss_h': mfloss_h_dict})
    else:
        warn('matched_filter_loss_h not known')

    if mfloss_v is not None:
        mfloss_v_dict['data'] = np.array([mfloss_v])
        if radar.radar_calibration is None:
            radar.radar_calibration = {'matched_filter_loss_v': mfloss_v_dict}
        else:
            radar.radar_calibration.update(
                {'matched_filter_loss_v': mfloss_v_dict})
    else:
        warn('matched_filter_loss_v not known')

    if prf is not None:
        prt['data'] = np.zeros(radar.nrays) + 1. / prf
        if radar.instrument_parameters is None:
            radar.instrument_parameters = {'prt': prt}
        else:
            radar.instrument_parameters.update({'prt': prt})
    else:
        warn('prt not known')

    Doppler_velocity = None
    Doppler_frequency = None
    if (prf is not None and radar.instrument_parameters is not None and
            'frequency' in radar.instrument_parameters):
        Doppler_velocity = filemetadata('Doppler_velocity')
        Doppler_frequency = filemetadata('Doppler_frequency')
        vel_data, freq_data = get_Doppler_info(
            np.zeros(radar.nrays) + prf, npulses['data'],
            speed_of_light / radar.instrument_parameters['frequency']['data'][0],
            fold=True)
        Doppler_velocity['data'] = vel_data
        Doppler_frequency['data'] = freq_data

    return RadarSpectra(
        radar.time, radar.range, fields, radar.metadata,
        radar.scan_type, radar.latitude, radar.longitude, radar.altitude,
        radar.sweep_number, radar.sweep_mode, radar.fixed_angle,
        radar.sweep_start_ray_index, radar.sweep_end_ray_index,
        radar.azimuth, radar.elevation, npulses,
        Doppler_velocity=Doppler_velocity,
        Doppler_frequency=Doppler_frequency,
        rays_are_indexed=radar.rays_are_indexed,
        ray_angle_res=radar.ray_angle_res,
        instrument_parameters=radar.instrument_parameters,
        radar_calibration=radar.radar_calibration)


def read_iq_data(filename, ngates, npulses, nchannels=2):
    """
    Reads the IQ data

    Parameters
    ----------
    filename : str
        Name of file containing the IQ data of a ray
    ngates : int
        Number of gates in ray
    npulses : int
        Number of pulses in ray
    nchannels : int
        Number of channels in file

    Returns
    -------
    data_hh, data_vv : 2D array
        arrays containing the HH and VV channels IQ data

    """
    try:
        with open(filename, 'rb') as file:
            # file.readline()
            data = np.fromfile(
                file, dtype=np.complex64,
                count=ngates * npulses * nchannels)
            if data.size != ngates * npulses * nchannels:
                warn('Data file containing ' + str(data.size) + ' elements. ' +
                     str(ngates * npulses * nchannels) + ' expected.')
                return None, None

            data = np.reshape(data, [ngates, nchannels, npulses], order='F')
            data_hh = data[:, 0, :]
            data_vv = data[:, 1, :]

            return data_hh, data_vv

    except OSError as ee:
        warn(str(ee))
        warn('Unable to read file ' + filename)
        return None, None


def get_valid_rays(filenames_iq, ref_azi, ref_ele, ang_tol=0.4):
    """
    Selects the IQ files corresponding to each ray azimuth and gets
    the number of pulses corresponding to each ray

    Parameters
    ----------
    filenames_iq : list of str
        List of files containing the IQ information of each ray
    ref_azi : float array
        The radar azimuths (deg)
    ref_ele : int
        The elevation of the current scan (deg)
    ang_tol : float
        The angle tolerance [deg] between the reference azimuth angle of the
        radar object and that of the IQ file

    Returns
    -------
    filenames_iq_out : array of strings
        List of files containing IQ information for valid rays
    npulses_vec : array of ints
        Number of pulses for each ray

    """
    # Get azimuth from file name
    azi_vec = []
    for filename_iq in filenames_iq:
        azi_vec.append(float(os.path.basename(filename_iq)[39:42]) + 0.5)
    azi_vec = np.array(azi_vec)
    filenames_aux = np.array(deepcopy(filenames_iq))

    # Keep only IQ files with data from radar object
    filenames_iq_out = []
    npulses_vec = []
    for azi in ref_azi:
        ind = np.where(np.logical_and(
            azi_vec >= azi - ang_tol, azi_vec <= azi + ang_tol))[0]

        if ind.size == 0:
            warn('No file found for azimuth angle ' + str(azi))
            return None, None
        if ind.size > 1:
            filenames_aux2 = filenames_aux[ind]
            ele_vec = []
            for fname in filenames_aux2:
                ele_vec.append(float(os.path.basename(fname)[34:38]) / 100.)

            delta_ele_vec = np.abs(ele_vec - ref_ele)
            ind = ind[np.argmin(delta_ele_vec)]
        else:
            ind = ind[0]

        filenames_iq_out.append(filenames_aux[ind])
        npulses_vec.append(
            int(int(os.path.basename(str(filenames_aux[ind]))[49:54]) / 2))

    return np.array(filenames_iq_out), np.array(npulses_vec)
