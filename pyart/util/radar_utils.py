"""
pyart.util.radar_utils
======================

Functions for working radar instances.

.. autosummary::
    :toctree: generated/

    compute_antenna_diagram
    is_vpt
    to_vpt
    compute_azimuthal_average
    join_radar
    join_spectra
    cut_radar
    cut_radar_spectra
    radar_from_spectra
    interpol_spectra
    find_neighbour_gates
    ma_broadcast_to

"""

import copy
from warnings import warn

import numpy as np
from scipy.interpolate import interp1d
from netCDF4 import date2num

from ..core import Radar
from ..config import get_metadata, get_fillvalue
from . import compute_directional_stats
from . import cross_section_rhi, get_target_elevations
from . import datetimes_from_radar

EPOCH_UNITS = "seconds since 1970-01-01T00:00:00Z"


def compute_antenna_diagram(npts_diagram=81, beam_factor=2., beam_width=1.):
    """
    Computes the antenna diagram. It is assumed a parabolic antenna diagram.
    The diagram is computed between -beam_factor*beam_width/2 and
    beam_factor*beam_width/2

    Parameters
    ----------
    npts_diagram : int
        The number of points that that the antenna diagram will have
    beam_factor : float
        the factor by which the antenna beam width is multiplied

    Returns
    -------
    ang_diag : array of floats
        The points where the antenna diagram is defined, with origin the
        center of the diagram (deg)
    weights_diag : array of floats
        the weights at each point

    """
    slope = 2.*beam_factor/(npts_diagram-1.)
    x = slope*np.arange(npts_diagram)-beam_factor
    weights_diag = np.power(2., -2*x*x)
    step = (beam_factor*beam_width)/(npts_diagram-1)
    ang_diag = np.arange(npts_diagram)*step-beam_factor*beam_width/2.
    return ang_diag, weights_diag


def is_vpt(radar, offset=0.5):
    """
    Determine if a Radar appears to be a vertical pointing scan.

    This function only verifies that the object is a vertical pointing scan,
    use the :py:func:`to_vpt` function to convert the radar to a vpt scan
    if this function returns True.

    Parameters
    ----------
    radar : Radar
        Radar object to determine if.
    offset : float, optional
        Maximum offset of the elevation from 90 degrees to still consider
        to be vertically pointing.

    Returns
    -------
    flag : bool
        True if the radar appear to be verticle pointing, False if not.

    """
    # check that the elevation is within offset of 90 degrees.
    elev = radar.elevation['data']
    return np.all((elev < 90.0 + offset) & (elev > 90.0 - offset))


def to_vpt(radar, single_scan=True):
    """
    Convert an existing Radar object to represent a vertical pointing scan.

    This function does not verify that the Radar object contains a vertical
    pointing scan. To perform such a check use :py:func:`is_vpt`.

    Parameters
    ----------
    radar : Radar
        Mislabeled vertical pointing scan Radar object to convert to be
        properly labeled. This object is converted in place, no copy of
        the existing data is made.
    single_scan : bool, optional
        True to convert the volume to a single scan, any azimuth angle data
        is lost. False will convert the scan to contain the same number of
        scans as rays, azimuth angles are retained.

    """
    if single_scan:
        nsweeps = 1
        radar.azimuth['data'][:] = 0.0
        seri = np.array([radar.nrays - 1], dtype='int32')
        radar.sweep_end_ray_index['data'] = seri
    else:
        nsweeps = radar.nrays
        # radar.azimuth not adjusted
        radar.sweep_end_ray_index['data'] = np.arange(nsweeps, dtype='int32')

    radar.scan_type = 'vpt'
    radar.nsweeps = nsweeps
    radar.target_scan_rate = None       # no scanning
    radar.elevation['data'][:] = 90.0

    radar.sweep_number['data'] = np.arange(nsweeps, dtype='int32')
    radar.sweep_mode['data'] = np.array(['vertical_pointing'] * nsweeps)
    radar.fixed_angle['data'] = np.ones(nsweeps, dtype='float32') * 90.0
    radar.sweep_start_ray_index['data'] = np.arange(nsweeps, dtype='int32')

    if radar.instrument_parameters is not None:
        for key in ['prt_mode', 'follow_mode', 'polarization_mode']:
            if key in radar.instrument_parameters:
                ip_dic = radar.instrument_parameters[key]
                ip_dic['data'] = np.array([ip_dic['data'][0]] * nsweeps)

    # Attributes that do not need any changes
    # radar.altitude
    # radar.altitude_agl
    # radar.latitude
    # radar.longitude

    # radar.range
    # radar.ngates
    # radar.nrays

    # radar.metadata
    # radar.radar_calibration

    # radar.time
    # radar.fields
    # radar.antenna_transition
    # radar.scan_rate


def compute_azimuthal_average(radar, field_names, angle=None, delta_azi=None,
                              avg_type='mean', nvalid_min=1, lin_trans=None):
    """
    Computes the azimuthal average

    Parameters
    ----------
    radar : Radar object
        input radar objects
    field_names : list of str
        name of the fields to include in the average
    angle : float or None
        The center angle to average. If not set or set to -1 all
        available azimuth angles will be used
    delta_azi : float
        The angle span to average. If not set or set to -1 all the
        available azimuth angles will be used
    avg_type : str
        Average type. Can be mean or median. Default mean
    nvalid_min : int
        the minimum number of valid points to consdier the average valid.
        Default 1
    lin_trans : dict or None
        If a dictionary, specifies which fields have to be averaged in linear
        units

    Returns
    -------
    radar_rhi : Radar object
        new radar object where azimuths have been averaged
    """
    # keep only fields present in radar object
    field_names_aux = []
    nfields_available = 0
    if avg_type == 'mean':
        lin_trans_aux = dict()
    for field_name in field_names:
        if field_name not in radar.fields:
            warn('Field name '+field_name+' not available in radar object')
            continue
        field_names_aux.append(field_name)
        nfields_available += 1
        if avg_type != 'mean':
            continue
        lin_trans_aux.update({field_name: False})
        if field_name in lin_trans:
            lin_trans_aux[field_name] = lin_trans[field_name]

    if nfields_available == 0:
        warn("Fields not available in radar data")
        return None

    # check input parameters
    if avg_type not in ('mean', 'median'):
        warn('Unsuported statistics '+avg_type)
        return None

    if delta_azi == -1:
        delta_azi = None
    if angle == -1:
        angle = None

    radar_aux = copy.deepcopy(radar)
    # transform radar into ppi over the required elevation
    if radar_aux.scan_type == 'rhi':
        target_elevations, el_tol = get_target_elevations(radar_aux)
        radar_ppi = cross_section_rhi(
            radar_aux, target_elevations, el_tol=el_tol)
    elif radar_aux.scan_type == 'ppi':
        radar_ppi = radar_aux
    else:
        warn('Error: unsupported scan type.')
        return None

    # range, metadata, radar position are the same as the original
    # time
    radar_rhi = copy.deepcopy(radar)
    radar_rhi.fields = dict()
    radar_rhi.scan_type = 'rhi'
    radar_rhi.sweep_number['data'] = np.array([0])
    radar_rhi.sweep_mode['data'] = np.array(['rhi'])
    radar_rhi.fixed_angle['data'] = np.array([0])
    radar_rhi.sweep_start_ray_index['data'] = np.array([0])
    radar_rhi.sweep_end_ray_index['data'] = np.array([radar_ppi.nsweeps-1])
    radar_rhi.rays_per_sweep['data'] = np.array([radar_ppi.nsweeps])
    radar_rhi.azimuth['data'] = np.ones(radar_ppi.nsweeps)
    radar_rhi.elevation['data'] = radar_ppi.fixed_angle['data']
    radar_rhi.time['data'] = np.zeros(radar_ppi.nsweeps)
    radar_rhi.nrays = radar_ppi.fixed_angle['data'].size
    radar_rhi.nsweeps = 1
    radar_rhi.rays_are_indexed = None
    radar_rhi.ray_angle_res = None

    # average radar data
    if angle is None:
        fixed_angle = np.zeros(radar_ppi.nsweeps)

    fields_dict = dict()
    for field_name in field_names_aux:
        fields_dict.update(
            {field_name: get_metadata(field_name)})
        fields_dict[field_name]['data'] = np.ma.masked_all(
            (radar_ppi.nsweeps, radar_ppi.ngates))

    for sweep in range(radar_ppi.nsweeps):
        radar_aux = copy.deepcopy(radar_ppi)
        radar_aux = radar_aux.extract_sweeps([sweep])

        # find neighbouring gates to be selected
        inds_ray, inds_rng = find_neighbour_gates(
            radar_aux, angle, None, delta_azi=delta_azi, delta_rng=None)

        if angle is None:
            fixed_angle[sweep] = np.median(
                radar_aux.azimuth['data'][inds_ray])

        # get time as the mean of the time of the selected rays
        radar_rhi.time['data'][sweep] = np.mean(
            radar_aux.time['data'][inds_ray])

        # keep only data we are interested in
        for field_name in field_names_aux:
            field_aux = radar_aux.fields[field_name]['data'][:, inds_rng]
            field_aux = field_aux[inds_ray, :]
            if avg_type == 'mean':
                if lin_trans_aux[field_name]:
                    field_aux = np.ma.power(10., 0.1*field_aux)

            vals, _ = compute_directional_stats(
                field_aux, avg_type=avg_type, nvalid_min=nvalid_min, axis=0)

            if avg_type == 'mean':
                if lin_trans_aux[field_name]:
                    vals = 10.*np.ma.log10(vals)

            fields_dict[field_name]['data'][sweep, :] = vals

    if angle is None:
        radar_rhi.fixed_angle['data'] = np.array([np.mean(fixed_angle)])
    else:
        radar_rhi.fixed_angle['data'] = np.array([angle])
    radar_rhi.azimuth['data'] *= radar_rhi.fixed_angle['data'][0]

    for field_name in field_names_aux:
        radar_rhi.add_field(field_name, fields_dict[field_name])

    return radar_rhi


def join_radar(radar1, radar2):
    """
    Combine two radar instances into one.

    Parameters
    ----------
    radar1 : Radar
        Radar object.
    radar2 : Radar
        Radar object.

    """
    # must have same gate spacing
    new_radar = copy.deepcopy(radar1)
    new_radar.azimuth['data'] = np.append(
        radar1.azimuth['data'], radar2.azimuth['data'])
    new_radar.elevation['data'] = np.append(
        radar1.elevation['data'], radar2.elevation['data'])
    new_radar.fixed_angle['data'] = np.append(
        radar1.fixed_angle['data'], radar2.fixed_angle['data'])
    new_radar.sweep_number['data'] = np.append(
        radar1.sweep_number['data'], radar2.sweep_number['data'])
    new_radar.sweep_start_ray_index['data'] = np.append(
        radar1.sweep_start_ray_index['data'],
        radar2.sweep_start_ray_index['data'] + radar1.nrays)
    new_radar.sweep_end_ray_index['data'] = np.append(
        radar1.sweep_end_ray_index['data'],
        radar2.sweep_end_ray_index['data'] + radar1.nrays)
    new_radar.nsweeps += radar2.nsweeps
    new_radar.sweep_mode['data'] = np.append(
        radar1.sweep_mode['data'], radar2.sweep_mode['data'])
    if ((radar1.rays_are_indexed is not None) and
            (radar2.rays_are_indexed is not None)):
        new_radar.rays_are_indexed['data'] = np.append(
            radar1.rays_are_indexed['data'],
            radar2.rays_are_indexed['data'])
    else:
        new_radar.rays_are_indexed = None

    if new_radar.instrument_parameters is not None:
        if 'nyquist_velocity' in new_radar.instrument_parameters:
            new_radar.instrument_parameters['nyquist_velocity']['data'] = (
                np.append(
                    radar1.instrument_parameters['nyquist_velocity']['data'],
                    radar2.instrument_parameters['nyquist_velocity']['data']))
        if 'pulse_width' in new_radar.instrument_parameters:
            new_radar.instrument_parameters['pulse_width']['data'] = (
                np.append(
                    radar1.instrument_parameters['pulse_width']['data'],
                    radar2.instrument_parameters['pulse_width']['data']))
        if 'number_of_pulses' in new_radar.instrument_parameters:
            new_radar.instrument_parameters['number_of_pulses']['data'] = (
                np.append(
                    radar1.instrument_parameters['number_of_pulses']['data'],
                    radar2.instrument_parameters['number_of_pulses']['data']))

    if ((radar1.ray_angle_res is not None) and
            (radar2.ray_angle_res is not None)):
        new_radar.ray_angle_res['data'] = np.append(
            radar1.ray_angle_res['data'],
            radar2.ray_angle_res['data'])
    else:
        new_radar.ray_angle_res = None

    if len(radar1.range['data']) >= len(radar2.range['data']):
        new_radar.range['data'] = radar1.range['data']
    else:
        new_radar.range['data'] = radar2.range['data']
    new_radar.ngates = len(new_radar.range['data'])

    # to combine times we need to reference them to a standard
    # for this we'll use epoch time
    r1num = datetimes_from_radar(radar1, epoch=True)
    r2num = datetimes_from_radar(radar2, epoch=True)
    new_radar.time['data'] = date2num(
        np.append(r1num, r2num), EPOCH_UNITS)
    new_radar.time['units'] = EPOCH_UNITS
    new_radar.nrays = len(new_radar.time['data'])

    fields_to_remove = []
    for var in new_radar.fields.keys():
        # if the field is present in both radars combine both fields
        # otherwise remove it from new radar
        if var in radar1.fields and var in radar2.fields:
            sh1 = radar1.fields[var]['data'].shape
            sh2 = radar2.fields[var]['data'].shape
            new_field_shape = (sh1[0] + sh2[0], max(sh1[1], sh2[1]))
            new_field = np.ma.masked_all(new_field_shape)
            new_field.set_fill_value(get_fillvalue())
            new_field[0:sh1[0], 0:sh1[1]] = radar1.fields[var]['data']
            new_field[sh1[0]:, 0:sh2[1]] = radar2.fields[var]['data']
            new_radar.fields[var]['data'] = new_field
        else:
            warn("Field "+var+" not present in both radars")
            fields_to_remove.append(var)

    if fields_to_remove:
        for field_name in fields_to_remove:
            new_radar.fields.pop(field_name, None)

    # radar locations
    # TODO moving platforms - any more?
    if (len(radar1.latitude['data']) == 1 &
            len(radar2.latitude['data']) == 1 &
            len(radar1.longitude['data']) == 1 &
            len(radar2.longitude['data']) == 1 &
            len(radar1.altitude['data']) == 1 &
            len(radar2.altitude['data']) == 1):

        lat1 = float(radar1.latitude['data'])
        lon1 = float(radar1.longitude['data'])
        alt1 = float(radar1.altitude['data'])
        lat2 = float(radar2.latitude['data'])
        lon2 = float(radar2.longitude['data'])
        alt2 = float(radar2.altitude['data'])

        if (lat1 != lat2) or (lon1 != lon2) or (alt1 != alt2):
            ones1 = np.ones(len(radar1.time['data']), dtype='float32')
            ones2 = np.ones(len(radar2.time['data']), dtype='float32')
            new_radar.latitude['data'] = np.append(ones1 * lat1, ones2 * lat2)
            new_radar.longitude['data'] = np.append(ones1 * lon1, ones2 * lon2)
            new_radar.latitude['data'] = np.append(ones1 * alt1, ones2 * alt2)
        else:
            new_radar.latitude['data'] = radar1.latitude['data']
            new_radar.longitude['data'] = radar1.longitude['data']
            new_radar.altitude['data'] = radar1.altitude['data']

    else:
        new_radar.latitude['data'] = np.append(radar1.latitude['data'],
                                               radar2.latitude['data'])
        new_radar.longitude['data'] = np.append(radar1.longitude['data'],
                                                radar2.longitude['data'])
        new_radar.altitude['data'] = np.append(radar1.altitude['data'],
                                               radar2.altitude['data'])

    return new_radar


def join_spectra(spectra1, spectra2):
    """
    Combine two spectra instances into one.

    Parameters
    ----------
    spectra1 : spectra
        spectra object.
    spectra2 : spectra
        spectra object.

    """
    # must have same gate spacing
    new_spectra = copy.deepcopy(spectra1)
    new_spectra.azimuth['data'] = np.append(
        spectra1.azimuth['data'], spectra2.azimuth['data'])
    new_spectra.elevation['data'] = np.append(
        spectra1.elevation['data'], spectra2.elevation['data'])
    new_spectra.fixed_angle['data'] = np.append(
        spectra1.fixed_angle['data'], spectra2.fixed_angle['data'])
    new_spectra.sweep_number['data'] = np.append(
        spectra1.sweep_number['data'], spectra2.sweep_number['data'])
    new_spectra.sweep_start_ray_index['data'] = np.append(
        spectra1.sweep_start_ray_index['data'],
        spectra2.sweep_start_ray_index['data'] + spectra1.nrays)
    new_spectra.sweep_end_ray_index['data'] = np.append(
        spectra1.sweep_end_ray_index['data'],
        spectra2.sweep_end_ray_index['data'] + spectra1.nrays)
    new_spectra.nsweeps += spectra2.nsweeps
    new_spectra.sweep_mode['data'] = np.append(
        spectra1.sweep_mode['data'], spectra2.sweep_mode['data'])
    if ((spectra1.rays_are_indexed is not None) and
            (spectra2.rays_are_indexed is not None)):
        new_spectra.rays_are_indexed['data'] = np.append(
            spectra1.rays_are_indexed['data'],
            spectra2.rays_are_indexed['data'])
    else:
        new_spectra.rays_are_indexed = None

    if new_spectra.instrument_parameters is not None:
        if 'nyquist_velocity' in new_spectra.instrument_parameters:
            new_spectra.instrument_parameters['nyquist_velocity']['data'] = (
                np.append(
                    spectra1.instrument_parameters['nyquist_velocity']['data'],
                    spectra2.instrument_parameters['nyquist_velocity']['data']))
        if 'pulse_width' in new_spectra.instrument_parameters:
            new_spectra.instrument_parameters['pulse_width']['data'] = (
                np.append(
                    spectra1.instrument_parameters['pulse_width']['data'],
                    spectra2.instrument_parameters['pulse_width']['data']))
        if 'number_of_pulses' in new_spectra.instrument_parameters:
            new_spectra.instrument_parameters['number_of_pulses']['data'] = (
                np.append(
                    spectra1.instrument_parameters['number_of_pulses']['data'],
                    spectra2.instrument_parameters['number_of_pulses']['data']))

    if ((spectra1.ray_angle_res is not None) and
            (spectra2.ray_angle_res is not None)):
        new_spectra.ray_angle_res['data'] = np.append(
            spectra1.ray_angle_res['data'],
            spectra2.ray_angle_res['data'])
    else:
        new_spectra.ray_angle_res = None

    if len(spectra1.range['data']) >= len(spectra2.range['data']):
        new_spectra.range['data'] = spectra1.range['data']
    else:
        new_spectra.range['data'] = spectra2.range['data']
    new_spectra.ngates = len(new_spectra.range['data'])

    # Combine Doppler parameters
    if spectra1.npulses_max != spectra2.npulses_max:
        new_spectra.npulses_max = max(
            spectra1.npulses_max, spectra2.npulses_max)

    new_spectra.npulses['data'] = np.append(
        spectra1.npulses['data'], spectra2.npulses['data'])

    if (spectra1.Doppler_velocity is not None and
            spectra2.Doppler_velocity is not None):
        sh1 = spectra1.Doppler_velocity['data'].shape
        sh2 = spectra2.Doppler_velocity['data'].shape
        new_field = np.ma.masked_all((sh1[0] + sh2[0], max(sh1[1], sh2[1])))
        new_field[0:sh1[0], 0:sh1[1]] = spectra1.Doppler_velocity['data']
        new_field[sh1[0]:, 0:sh2[1]] = spectra2.Doppler_velocity['data']
        new_spectra.Doppler_velocity['data'] = new_field
    else:
        new_spectra.Doppler_velocity = None

    if (spectra1.Doppler_frequency is not None and
            spectra2.Doppler_frequency is not None):
        sh1 = spectra1.Doppler_frequency['data'].shape
        sh2 = spectra2.Doppler_frequency['data'].shape
        new_field = np.ma.masked_all((sh1[0] + sh2[0], max(sh1[1], sh2[1])))
        new_field[0:sh1[0], 0:sh1[1]] = spectra1.Doppler_frequency['data']
        new_field[sh1[0]:, 0:sh2[1]] = spectra2.Doppler_frequency['data']
        new_spectra.Doppler_frequency['data'] = new_field
    else:
        new_spectra.Doppler_frequency = None

    # to combine times we need to reference them to a standard
    # for this we'll use epoch time
    r1num = datetimes_from_radar(spectra1, epoch=True)
    r2num = datetimes_from_radar(spectra2, epoch=True)
    new_spectra.time['data'] = date2num(
        np.append(r1num, r2num), EPOCH_UNITS)
    new_spectra.time['units'] = EPOCH_UNITS
    new_spectra.nrays = len(new_spectra.time['data'])

    fields_to_remove = []
    for var in new_spectra.fields.keys():
        # if the field is present in both spectras combine both fields
        # otherwise remove it from new spectra
        if var in spectra1.fields and var in spectra2.fields:
            sh1 = spectra1.fields[var]['data'].shape
            sh2 = spectra2.fields[var]['data'].shape
            new_field = np.ma.masked_all(
                (sh1[0] + sh2[0], max(sh1[1], sh2[1]), max(sh1[2], sh2[2])))
            new_field.set_fill_value(get_fillvalue())
            new_field[0:sh1[0], 0:sh1[1], 0:sh1[2]] = (
                spectra1.fields[var]['data'])
            new_field[sh1[0]:, 0:sh2[1], 0:sh2[2]] = (
                spectra2.fields[var]['data'])
            new_spectra.fields[var]['data'] = new_field
        else:
            warn("Field "+var+" not present in both spectras")
            fields_to_remove.append(var)

    if fields_to_remove:
        for field_name in fields_to_remove:
            new_spectra.fields.pop(field_name, None)

    # spectra locations
    # TODO moving platforms - any more?
    if (len(spectra1.latitude['data']) == 1 &
            len(spectra2.latitude['data']) == 1 &
            len(spectra1.longitude['data']) == 1 &
            len(spectra2.longitude['data']) == 1 &
            len(spectra1.altitude['data']) == 1 &
            len(spectra2.altitude['data']) == 1):

        lat1 = float(spectra1.latitude['data'])
        lon1 = float(spectra1.longitude['data'])
        alt1 = float(spectra1.altitude['data'])
        lat2 = float(spectra2.latitude['data'])
        lon2 = float(spectra2.longitude['data'])
        alt2 = float(spectra2.altitude['data'])

        if (lat1 != lat2) or (lon1 != lon2) or (alt1 != alt2):
            ones1 = np.ones(len(spectra1.time['data']), dtype='float32')
            ones2 = np.ones(len(spectra2.time['data']), dtype='float32')
            new_spectra.latitude['data'] = np.append(
                ones1 * lat1, ones2 * lat2)
            new_spectra.longitude['data'] = np.append(
                ones1 * lon1, ones2 * lon2)
            new_spectra.latitude['data'] = np.append(
                ones1 * alt1, ones2 * alt2)
        else:
            new_spectra.latitude['data'] = spectra1.latitude['data']
            new_spectra.longitude['data'] = spectra1.longitude['data']
            new_spectra.altitude['data'] = spectra1.altitude['data']

    else:
        new_spectra.latitude['data'] = np.append(
            spectra1.latitude['data'], spectra2.latitude['data'])
        new_spectra.longitude['data'] = np.append(
            spectra1.longitude['data'], spectra2.longitude['data'])
        new_spectra.altitude['data'] = np.append(
            spectra1.altitude['data'], spectra2.altitude['data'])

    return new_spectra


def cut_radar(radar, field_names, rng_min=None, rng_max=None, ele_min=None,
              ele_max=None, azi_min=None, azi_max=None):
    """
    Cuts the radar object into new dimensions

    Parameters
    ----------
    radar : radar object
        The radar object containing the data
    field_names : str or None
        The fields to keep in the new radar
    rng_min, rng_max : float
        The range limits [m]. If None the entire coverage of the radar is
        going to be used
    ele_min, ele_max, azi_min, azi_max : float or None
        The limits of the grid [deg]. If None the limits will be the limits
        of the radar volume

    Returns
    -------
    radar : radar object
        The radar object containing only the desired data

    """
    radar_aux = copy.deepcopy(radar)

    if (rng_min is None and rng_max is None and ele_min is None and
            ele_max is None and azi_min is None and azi_max is None):
        return radar_aux

    if rng_min is None:
        rng_min = 0.
    if rng_max is None:
        rng_max = np.max(radar_aux.range['data'])

    ind_rng = np.where(np.logical_and(
        radar_aux.range['data'] >= rng_min,
        radar_aux.range['data'] <= rng_max))[0]

    if ind_rng.size == 0:
        warn('No range bins between '+str(rng_min)+' and '+str(rng_max)+' m')
        return None

    # Determine angle limits
    if radar_aux.scan_type == 'ppi':
        if ele_min is None:
            ele_min = np.min(radar_aux.fixed_angle['data'])
        if ele_max is None:
            ele_max = np.max(radar_aux.fixed_angle['data'])
        if azi_min is None:
            azi_min = np.min(radar_aux.azimuth['data'])
        if azi_max is None:
            azi_max = np.max(radar_aux.azimuth['data'])
    else:
        if ele_min is None:
            ele_min = np.min(radar_aux.elevation['data'])
        if ele_max is None:
            ele_max = np.max(radar_aux.elevation['data'])
        if azi_min is None:
            azi_min = np.min(radar_aux.fixed_angle['data'])
        if azi_max is None:
            azi_max = np.max(radar_aux.fixed_angle['data'])

    if radar_aux.scan_type == 'ppi':
        # Get radar elevation angles within limits
        ele_vec = np.sort(radar_aux.fixed_angle['data'])
        ele_vec = ele_vec[
            np.logical_and(ele_vec >= ele_min, ele_vec <= ele_max)]
        if ele_vec.size == 0:
            warn('No elevation angles between '+str(ele_min)+' and ' +
                 str(ele_max))
            return None

        # get sweeps corresponding to the desired elevation angles
        ind_sweeps = []
        for ele in ele_vec:
            ind_sweeps.append(
                np.where(radar_aux.fixed_angle['data'] == ele)[0][0])
        radar_aux = radar_aux.extract_sweeps(ind_sweeps)

        # Get indices of rays within limits
        if azi_min < azi_max:
            ind_rays = np.where(np.logical_and(
                radar_aux.azimuth['data'] >= azi_min,
                radar_aux.azimuth['data'] <= azi_max))[0]
        else:
            ind_rays = np.where(np.logical_or(
                radar_aux.azimuth['data'] >= azi_min,
                radar_aux.azimuth['data'] <= azi_max))[0]

    else:
        # Get radar azimuth angles within limits
        azi_vec = radar_aux.fixed_angle['data']
        if azi_min < azi_max:
            azi_vec = np.sort(azi_vec[
                np.logical_and(azi_vec >= azi_min, azi_vec <= azi_max)])
        else:
            azi_vec = azi_vec[
                np.logical_or(azi_vec >= azi_min, azi_vec <= azi_max)]
        if azi_vec.size == 0:
            warn('No azimuth angles between '+str(azi_min)+' and ' +
                 str(azi_max))
            return None

        # get sweeps corresponding to the desired azimuth angles
        ind_sweeps = []
        for azi in azi_vec:
            ind_sweeps.append(
                np.where(radar_aux.fixed_angle['data'] == azi)[0][0])
        radar_aux = radar_aux.extract_sweeps(ind_sweeps)

        # Get indices of rays within limits
        ind_rays = np.where(np.logical_and(
            radar_aux.elevation['data'] >= ele_min,
            radar_aux.elevation['data'] <= ele_max))[0]

    # get new sweep start index and stop index
    sweep_start_inds = copy.deepcopy(radar_aux.sweep_start_ray_index['data'])
    sweep_end_inds = copy.deepcopy(radar_aux.sweep_end_ray_index['data'])

    nrays = 0
    ind_rays_aux = []
    for j in range(radar_aux.nsweeps):
        # get ray indices for this sweep
        ind_rays_sweep = ind_rays[np.logical_and(
            ind_rays >= sweep_start_inds[j], ind_rays <= sweep_end_inds[j])]
        # order rays
        if radar_aux.scan_type == 'ppi':
            ind = np.argsort(radar_aux.azimuth['data'][ind_rays_sweep])
            ind_rays_sweep = ind_rays_sweep[ind]
            # avoid large gaps in data
            azimuths = radar_aux.azimuth['data'][ind_rays_sweep]
            azi_steps = azimuths[1:]-azimuths[:-1]
            ind_gap = np.where(azi_steps > 2*np.median(azi_steps))[0]
            if ind_gap.size > 0:
                ind_rays_sweep = np.append(
                    ind_rays_sweep[ind_gap[0]+1:],
                    ind_rays_sweep[:ind_gap[0]+1])
            ind_rays_aux.extend(ind_rays_sweep)
        else:
            ind = np.argsort(radar_aux.elevation['data'][ind_rays_sweep])
            ind_rays_aux.extend(ind_rays_sweep[ind])

        rays_in_sweep = ind_rays_sweep.size
        radar_aux.rays_per_sweep['data'][j] = rays_in_sweep
        if j == 0:
            radar_aux.sweep_start_ray_index['data'][j] = 0
        else:
            radar_aux.sweep_start_ray_index['data'][j] = int(
                radar_aux.sweep_end_ray_index['data'][j-1]+1)
        radar_aux.sweep_end_ray_index['data'][j] = (
            radar_aux.sweep_start_ray_index['data'][j]+rays_in_sweep-1)
        nrays += rays_in_sweep

    ind_rays = np.array(ind_rays_aux)

    # Update metadata
    radar_aux.range['data'] = radar_aux.range['data'][ind_rng]
    radar_aux.time['data'] = radar_aux.time['data'][ind_rays]
    radar_aux.azimuth['data'] = radar_aux.azimuth['data'][ind_rays]
    radar_aux.elevation['data'] = radar_aux.elevation['data'][ind_rays]
    radar_aux.init_gate_x_y_z()
    radar_aux.init_gate_longitude_latitude()
    radar_aux.init_gate_altitude()
    radar_aux.nrays = nrays
    radar_aux.ngates = ind_rng.size

    if radar_aux.instrument_parameters is not None:
        if 'nyquist_velocity' in radar_aux.instrument_parameters:
            radar_aux.instrument_parameters['nyquist_velocity']['data'] = (
                radar_aux.instrument_parameters['nyquist_velocity']['data'][
                    ind_rays])
        if 'pulse_width' in radar_aux.instrument_parameters:
            radar_aux.instrument_parameters['pulse_width']['data'] = (
                radar_aux.instrument_parameters['pulse_width']['data'][
                    ind_rays])
        if 'number_of_pulses' in radar_aux.instrument_parameters:
            radar_aux.instrument_parameters['number_of_pulses']['data'] = (
                radar_aux.instrument_parameters['number_of_pulses']['data'][
                    ind_rays])

    # Get new fields
    if field_names is None:
        radar_aux.fields = dict()
    else:
        fields_aux = copy.deepcopy(radar_aux.fields)
        radar_aux.fields = dict()
        for field_name in field_names:
            if field_name not in fields_aux:
                warn('Field '+field_name+' not available')
                continue

            fields_aux[field_name]['data'] = (
                fields_aux[field_name]['data'][:, ind_rng])
            fields_aux[field_name]['data'] = (
                fields_aux[field_name]['data'][ind_rays, :])
            radar_aux.add_field(field_name, fields_aux[field_name])

    return radar_aux


def cut_radar_spectra(radar, field_names, rng_min=None, rng_max=None,
                      ele_min=None, ele_max=None, azi_min=None, azi_max=None):
    """
    Cuts the radar spectra object into new dimensions

    Parameters
    ----------
    radar : radar object
        The radar object containing the data
    field_names : str or None
        The fields to keep in the new radar
    rng_min, rng_max : float
        The range limits [m]. If None the entire coverage of the radar is
        going to be used
    ele_min, ele_max, azi_min, azi_max : float or None
        The limits of the grid [deg]. If None the limits will be the limits
        of the radar volume

    Returns
    -------
    radar : radar object
        The radar object containing only the desired data

    """
    radar_aux = copy.deepcopy(radar)

    if (rng_min is None and rng_max is None and ele_min is None and
            ele_max is None and azi_min is None and azi_max is None):
        return radar_aux

    if rng_min is None:
        rng_min = 0.
    if rng_max is None:
        rng_max = np.max(radar_aux.range['data'])

    ind_rng = np.where(np.logical_and(
        radar_aux.range['data'] >= rng_min,
        radar_aux.range['data'] <= rng_max))[0]

    if ind_rng.size == 0:
        warn('No range bins between '+str(rng_min)+' and '+str(rng_max)+' m')
        return None

    # Determine angle limits
    if radar_aux.scan_type == 'ppi':
        if ele_min is None:
            ele_min = np.min(radar_aux.fixed_angle['data'])
        if ele_max is None:
            ele_max = np.max(radar_aux.fixed_angle['data'])
        if azi_min is None:
            azi_min = np.min(radar_aux.azimuth['data'])
        if azi_max is None:
            azi_max = np.max(radar_aux.azimuth['data'])
    else:
        if ele_min is None:
            ele_min = np.min(radar_aux.elevation['data'])
        if ele_max is None:
            ele_max = np.max(radar_aux.elevation['data'])
        if azi_min is None:
            azi_min = np.min(radar_aux.fixed_angle['data'])
        if azi_max is None:
            azi_max = np.max(radar_aux.fixed_angle['data'])

    if radar_aux.scan_type == 'ppi':
        # Get radar elevation angles within limits
        ele_vec = np.sort(radar_aux.fixed_angle['data'])
        ele_vec = ele_vec[
            np.logical_and(ele_vec >= ele_min, ele_vec <= ele_max)]
        if ele_vec.size == 0:
            warn('No elevation angles between '+str(ele_min)+' and ' +
                 str(ele_max))
            return None

        # get sweeps corresponding to the desired elevation angles
        ind_sweeps = []
        for ele in ele_vec:
            ind_sweeps.append(
                np.where(radar_aux.fixed_angle['data'] == ele)[0][0])
        radar_aux = radar_aux.extract_sweeps(ind_sweeps)

        # Get indices of rays within limits
        if azi_min < azi_max:
            ind_rays = np.where(np.logical_and(
                radar_aux.azimuth['data'] >= azi_min,
                radar_aux.azimuth['data'] <= azi_max))[0]
        else:
            ind_rays = np.where(np.logical_or(
                radar_aux.azimuth['data'] >= azi_min,
                radar_aux.azimuth['data'] <= azi_max))[0]

    else:
        # Get radar azimuth angles within limits
        azi_vec = radar_aux.fixed_angle['data']
        if azi_min < azi_max:
            azi_vec = np.sort(azi_vec[
                np.logical_and(azi_vec >= azi_min, azi_vec <= azi_max)])
        else:
            azi_vec = azi_vec[
                np.logical_or(azi_vec >= azi_min, azi_vec <= azi_max)]
        if azi_vec.size == 0:
            warn('No azimuth angles between '+str(azi_min)+' and ' +
                 str(azi_max))
            return None

        # get sweeps corresponding to the desired azimuth angles
        ind_sweeps = []
        for azi in azi_vec:
            ind_sweeps.append(
                np.where(radar_aux.fixed_angle['data'] == azi)[0][0])
        radar_aux = radar_aux.extract_sweeps(ind_sweeps)

        # Get indices of rays within limits
        ind_rays = np.where(np.logical_and(
            radar_aux.elevation['data'] >= ele_min,
            radar_aux.elevation['data'] <= ele_max))[0]

    # get new sweep start index and stop index
    sweep_start_inds = copy.deepcopy(radar_aux.sweep_start_ray_index['data'])
    sweep_end_inds = copy.deepcopy(radar_aux.sweep_end_ray_index['data'])

    nrays = 0
    for j in range(radar_aux.nsweeps):
        # get azimuth indices for this elevation
        rays_in_sweep = np.size(
            ind_rays[np.logical_and(ind_rays >= sweep_start_inds[j],
                                    ind_rays <= sweep_end_inds[j])])
        radar_aux.rays_per_sweep['data'][j] = rays_in_sweep
        if j == 0:
            radar_aux.sweep_start_ray_index['data'][j] = 0
        else:
            radar_aux.sweep_start_ray_index['data'][j] = int(
                radar_aux.sweep_end_ray_index['data'][j-1]+1)
        radar_aux.sweep_end_ray_index['data'][j] = (
            radar_aux.sweep_start_ray_index['data'][j]+rays_in_sweep-1)
        nrays += rays_in_sweep

    # Update metadata
    radar_aux.range['data'] = radar_aux.range['data'][ind_rng]
    radar_aux.time['data'] = radar_aux.time['data'][ind_rays]
    radar_aux.azimuth['data'] = radar_aux.azimuth['data'][ind_rays]
    radar_aux.elevation['data'] = radar_aux.elevation['data'][ind_rays]
    radar_aux.init_gate_x_y_z()
    radar_aux.init_gate_longitude_latitude()
    radar_aux.init_gate_altitude()
    radar_aux.nrays = nrays
    radar_aux.ngates = ind_rng.size
    if radar_aux.Doppler_velocity is not None:
        radar_aux.Doppler_velocity['data'] = (
            radar_aux.Doppler_velocity['data'][ind_rays, :])
    if radar_aux.Doppler_frequency is not None:
        radar_aux.Doppler_frequency['data'] = (
            radar_aux.Doppler_frequency['data'][ind_rays, :])
    if radar_aux.instrument_parameters is not None:
        if 'nyquist_velocity' in radar_aux.instrument_parameters:
            radar_aux.instrument_parameters['nyquist_velocity']['data'] = (
                radar_aux.instrument_parameters['nyquist_velocity']['data'][
                    ind_rays])
        if 'pulse_width' in radar_aux.instrument_parameters:
            radar_aux.instrument_parameters['pulse_width']['data'] = (
                radar_aux.instrument_parameters['pulse_width']['data'][
                    ind_rays])
        if 'number_of_pulses' in radar_aux.instrument_parameters:
            radar_aux.instrument_parameters['number_of_pulses']['data'] = (
                radar_aux.instrument_parameters['number_of_pulses']['data'][
                    ind_rays])

    # Get new fields
    if field_names is None:
        radar_aux.fields = dict()
    else:
        fields_aux = copy.deepcopy(radar_aux.fields)
        radar_aux.fields = dict()
        for field_name in field_names:
            if field_name not in fields_aux:
                warn('Field '+field_name+' not available')
                continue

            fields_aux[field_name]['data'] = (
                fields_aux[field_name]['data'][:, ind_rng, :])
            fields_aux[field_name]['data'] = (
                fields_aux[field_name]['data'][ind_rays, :, :])
            radar_aux.add_field(field_name, fields_aux[field_name])

    return radar_aux


def radar_from_spectra(psr):
    """
    obtain a Radar object from a RadarSpectra object

    Parameters
    ----------
    psr : RadarSpectra object
        The reference object

    Returns
    -------
    radar : radar object
        The new radar object

    """
    return Radar(
        psr.time, psr.range, dict(), psr.metadata, psr.scan_type,
        psr.latitude, psr.longitude, psr.altitude, psr.sweep_number,
        psr.sweep_mode, psr.fixed_angle, psr.sweep_start_ray_index,
        psr.sweep_end_ray_index, psr.azimuth, psr.elevation,
        rays_are_indexed=psr.rays_are_indexed,
        ray_angle_res=psr.ray_angle_res,
        instrument_parameters=psr.instrument_parameters,
        radar_calibration=psr.radar_calibration)


def interpol_spectra(psr, kind='linear', fill_value=0.):
    """
    Interpolates the spectra so that it has a uniform grid

    Parameters
    ----------
    psr : RadarSpectra object
        The original spectra

    Returns
    -------
    psr_interp : RadarSpectra object
        The interpolated spectra

    """
    if psr.Doppler_velocity is not None:
        # check if all rays have the same Doppler velocity field
        if np.ma.all(np.ma.all(
                psr.Doppler_velocity['data'] == psr.Doppler_velocity['data'][0, :],
                axis=1)):
            return copy.deepcopy(psr)

        xaxis = psr.Doppler_velocity['data'][0, :].compressed()
        units = 'm/s'
    elif psr.Doppler_frequency is not None:
        # check if all rays have the same Doppler velocity field
        if np.ma.all(np.ma.all(
                psr.Doppler_frequency['data'] == psr.Doppler_frequency['data'][0, :],
                axis=1)):
            return copy.deepcopy(psr)

        xaxis = psr.Doppler_frequency['data'][0, :].compressed()
        units = 'Hz'
    else:
        warn('Unable to interpolate spectra. No Doppler velocity or Doppler'
             ' frequency available')
        return None
    npulses = xaxis.size

    psr_interp = copy.deepcopy(psr)

    # allocate memory for output fields
    for field_name in psr_interp.fields.keys():
        psr_interp.fields[field_name]['data'] = np.ma.masked_all(
            (psr_interp.nrays, psr_interp.ngates, npulses),
            dtype=psr.fields[field_name]['data'].dtype)

    for ray in range(psr.nrays):
        if units == 'm/s':
            xaxis_aux = psr.Doppler_velocity['data'][ray, :].compressed()
        else:
            xaxis_aux = psr.Doppler_frequency['data'][ray, :].compressed()

        for field_name in psr_interp.fields.keys():
            intp_func = interp1d(
                xaxis_aux, psr.fields[field_name]['data'][
                    ray, :, 0:xaxis_aux.size].filled(fill_value), kind=kind,
                fill_value='extrapolate', assume_sorted=True)

            psr_interp.fields[field_name]['data'][ray, :, 0:npulses] = (
                np.ma.masked_invalid(intp_func(xaxis)))

    psr_interp.npulses_max = npulses
    psr_interp.npulses['data'] = (
        np.zeros(psr_interp.nrays, dtype=np.int)+npulses)
    if psr_interp.Doppler_velocity is not None:
        xaxis = np.ma.expand_dims(
            psr.Doppler_velocity['data'][0, :].compressed(), axis=0)
        psr_interp.Doppler_velocity['data'] = ma_broadcast_to(
            xaxis, (psr_interp.nrays, psr_interp.npulses_max))
        psr_interp.Doppler_velocity['data'] = np.ma.array(
            psr_interp.Doppler_velocity['data'])

    if psr_interp.Doppler_frequency is not None:
        xaxis = np.ma.expand_dims(
            psr.Doppler_frequency['data'][0, :].compressed(), axis=0)
        psr_interp.Doppler_frequency['data'] = ma_broadcast_to(
            xaxis, (psr_interp.nrays, psr_interp.npulses_max))
        psr_interp.Doppler_frequency['data'] = np.ma.array(
            psr_interp.Doppler_frequency['data'])

    return psr_interp


def find_neighbour_gates(radar, azi, rng, delta_azi=None, delta_rng=None):
    """
    Find the neighbouring gates within +-delta_azi and +-delta_rng

    Parameters
    ----------
    radar : radar object
        the radar object
    azi, rng : float
        The azimuth [deg] and range [m] of the central gate
    delta_azi, delta_rng : float
        The extend where to look for

    Returns
    -------
    inds_ray_aux, ind_rng_aux : int
        The indices (ray, rng) of the neighbouring gates

    """
    # find gates close to lat lon point
    if delta_azi is None:
        inds_ray = np.ma.arange(radar.azimuth['data'].size)
    else:
        azi_max = azi+delta_azi
        azi_min = azi-delta_azi
        if azi_max > 360.:
            azi_max -= 360.
        if azi_min < 0.:
            azi_min += 360.
        if azi_max > azi_min:
            inds_ray = np.where(np.logical_and(
                radar.azimuth['data'] < azi_max,
                radar.azimuth['data'] > azi_min))[0]
        else:
            inds_ray = np.where(np.logical_or(
                radar.azimuth['data'] > azi_min,
                radar.azimuth['data'] < azi_max))[0]
    if delta_rng is None:
        inds_rng = np.ma.arange(radar.range['data'].size)
    else:
        inds_rng = np.where(np.logical_and(
            radar.range['data'] < rng+delta_rng,
            radar.range['data'] > rng-delta_rng))[0]

    return inds_ray, inds_rng


def ma_broadcast_to(array, tup):
    """
    Is used to guarantee that a masked array can be broadcasted without
    loosing the mask

    Parameters
    ----------
    array : Numpy masked array or normal array
    tup : shape as tuple

    Returns
    -------
    broadcasted_array
        The broadcasted numpy array including its mask if available
        otherwise only the broadcasted array is returned

    """
    broadcasted_array = np.broadcast_to(array, tup)

    if np.ma.is_masked(array):
        initial_mask = np.ma.getmask(array)
        initial_fill_value = array.fill_value
        broadcasted_mask = np.broadcast_to(initial_mask, tup)
        return np.ma.array(
            broadcasted_array, mask=broadcasted_mask,
            fill_value=initial_fill_value)

    return broadcasted_array
