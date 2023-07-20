"""
pyart.correct.bias_and_noise
===================

Corrects polarimetric variables for noise

.. autosummary::
    :toctree: generated/

    correct_noise_rhohv
    correct_bias
    correct_visibility
    get_sun_hits
    get_sun_hits_psr
    get_sun_hits_ivic
    sun_retrieval
    est_rhohv_rain
    est_zdr_precip
    est_zdr_snow
    selfconsistency_bias
    selfconsistency_bias2
    selfconsistency_kdp_phidp
    get_kdp_selfcons
    _est_sun_hit_pwr_hs
    _est_sun_hit_pwr_psr
    _est_sun_hit_pwr_ivic
    _est_sun_hit_zdr
    _selfconsistency_kdp_phidp


"""

import importlib
from copy import deepcopy
from warnings import warn

import numpy as np
from netCDF4 import num2date

# Check existence of required libraries
# Lint compatible version (wod, 20.07.2023)
if importlib.util.find_spec('pysolar'):
    _PYSOLAR_AVAILABLE = True
else:
    _PYSOLAR_AVAILABLE = False


from ..config import get_field_name, get_fillvalue, get_metadata
from ..filters import class_based_gate_filter, snr_based_gate_filter
from ..retrieve import get_coeff_attg, kdp_leastsquare_single_window
from ..util import (
    estimate_noise_hs74,
    estimate_noise_ivic13,
    ivic_flat_reg_var_max_table,
    ivic_flat_reg_wind_len_table,
    ivic_pct_table,
    ivic_snr_thr_table,
)
from .attenuation import get_mask_fzl
from .phase_proc import smooth_masked
from .sunlib import (
    gas_att_sun,
    gauss_fit,
    retrieval_result,
    sun_position_mfr,
    sun_position_pysolar,
)


def correct_noise_rhohv(radar, urhohv_field=None, snr_field=None,
                        zdr_field=None, nh_field=None, nv_field=None,
                        rhohv_field=None):
    """
    Corrects RhoHV for noise according to eq. 6 in Gourley et al. 2006.
    This correction should only be performed if noise has not been subtracted
    from the signal during the moments computation.

    Parameters
    ----------
    radar : Radar
        Radar object.
    urhohv_field : str, optional
        Name of the RhoHV uncorrected for noise field.
    snr_field, zdr_field, nh_field, nv_field : str, optional
        Names of the SNR, ZDR, horizontal channel noise in dBZ and vertical
        channel noise in dBZ used to correct RhoHV.
    rhohv_field : str, optional
        Name of the rhohv field to output.

    Returns
    -------
    rhohv : dict
        Noise corrected RhoHV field.

    References
    ----------
    Gourley et al. Data Quality of the Meteo-France C-Band Polarimetric
    Radar, JAOT, 23, 1340-1356

    """
    # parse the field parameters
    if urhohv_field is None:
        urhohv_field = get_field_name('uncorrected_cross_correlation_ratio')
    if snr_field is None:
        snr_field = get_field_name('signal_to_noise_ratio')
    if zdr_field is None:
        zdr_field = get_field_name('differential_reflectivity')
    if nh_field is None:
        nh_field = get_field_name('noisedBZ_hh')
    if nv_field is None:
        nv_field = get_field_name('noisedBZ_vv')
    if rhohv_field is None:
        rhohv_field = get_field_name('cross_correlation_ratio')

    # extract fields from radar
    radar.check_field_exists(urhohv_field)
    radar.check_field_exists(snr_field)
    radar.check_field_exists(zdr_field)
    radar.check_field_exists(nh_field)
    radar.check_field_exists(nv_field)

    urhohv = radar.fields[urhohv_field]['data']
    snrdB_h = radar.fields[snr_field]['data']
    zdrdB = radar.fields[zdr_field]['data']
    nh = radar.fields[nh_field]['data']
    nv = radar.fields[nv_field]['data']

    snr_h = np.ma.power(10., 0.1 * snrdB_h)
    zdr = np.ma.power(10., 0.1 * zdrdB)
    alpha = np.ma.power(10., 0.1 * (nh - nv))

    rhohv_data = urhohv * \
        np.ma.sqrt((1. + 1. / snr_h) * (1. + zdr / (alpha * snr_h)))
    rhohv_data[rhohv_data > 1.] = 1.

    rhohv = get_metadata(rhohv_field)
    rhohv['data'] = rhohv_data

    return rhohv


def correct_bias(radar, bias=0., field_name=None):
    """
    Corrects a radar data bias. If field name is none the correction is
    applied to horizontal reflectivity by default.

    Parameters
    ----------
    radar : Radar
        Radar object.
    bias : float, optional
        The bias magnitude.
    field_name: str, optional
        Names of the field to be corrected.

    Returns
    -------
    corrected_field : dict
        The corrected field

    """
    # parse the field parameters
    if field_name is None:
        field_name = get_field_name('reflectivity')

    # extract fields from radar
    radar.check_field_exists(field_name)
    field_data = radar.fields[field_name]['data']

    corr_field_data = field_data - bias

    if field_name.startswith('corrected_'):
        corr_field_name = field_name
    else:
        corr_field_name = 'corrected_' + field_name

    corr_field = get_metadata(corr_field_name)
    corr_field['data'] = corr_field_data

    return corr_field


def correct_visibility(radar, vis_field=None, field_name=None):
    """
    Corrects the reflectivity according to visibility.
    Applied to horizontal reflectivity by default

    Parameters
    ----------
    radar : Radar
        radar object

    vis_field : str
        the name of the visibility field

    field_name: str
        names of the field to be corrected

    Returns
    -------
    corrected_field : dict
        The corrected field

    """
    # parse the field parameters
    if vis_field is None:
        vis_field = get_field_name('visibility')
    if field_name is None:
        field_name = get_field_name('reflectivity')

    # extract fields from radar
    radar.check_field_exists(vis_field)
    vis_data = radar.fields[vis_field]['data']

    radar.check_field_exists(field_name)
    field_data = radar.fields[field_name]['data']

    corr_field_data = 10. * np.ma.log10(
        np.ma.power(10., 0.1 * field_data) * 100. / vis_data)

    if field_name.startswith('corrected_'):
        corr_field_name = field_name
    else:
        corr_field_name = 'corrected_' + field_name

    corr_field = get_metadata(corr_field_name)
    corr_field['data'] = corr_field_data

    return corr_field


def get_sun_hits(
        radar, delev_max=2., dazim_max=2., elmin=1., rmin=50000., hmin=10000.,
        nbins_min=20, attg=None, max_std_pwr=1., max_std_zdr=1.5,
        sun_position='MF', pwrh_field=None, pwrv_field=None, zdr_field=None):
    """
    get data from suspected sun hits. The sun hits are detected using the
    Hildebrand and Sekhon noise estimation

    Parameters
    ----------
    radar : Radar
        radar object
    delev_max, dazim_max : float
        maximum difference in elevation and azimuth between sun position and
        antenna pointing
    elmin : float
        minimum radar elevation angle
    rmin : float
        minimum range from which we can look for noise [m]
    hmin : float
        minimum altitude from which we can look for noise [m]. The actual
        range min will be the minimum between rmin and the range bin higher
        than hmin.
    nbins_min : int
        Minimum number of bins with valid data to consider a ray as
        potentially sun hit
    attg : float
        gas attenuation coefficient (1-way)
    max_std_pwr : float
        Maximum standard deviation of the estimated sun power to consider the
        sun signal valid [dB]
    max_std_zdr : float
        Maximum standard deviation of the estimated sun ZDR to consider the
        sun signal valid [dB]
    sun_position : str
        The function to use to compute the sun position. Can be 'MF' or
        'pysolar'
    pwrh_field, pwrv_field, zdr_field : str
        names of the signal power in dBm for the H and V polarizations and the
        differential reflectivity

    Returns
    -------
    sun_hits : dict
        a dictionary containing information of the sun hits
    new_radar : radar object
        radar object containing sweeps that contain sun hits

    """
    # get parameters
    if attg is None:
        # assign coefficients according to radar frequency
        if (radar.instrument_parameters is not None and
                'frequency' in radar.instrument_parameters):
            attg = get_coeff_attg(
                radar.instrument_parameters['frequency']['data'][0])
        else:
            attg = 0.
            warn('Unknown 1-way gas attenuation. It will be set to 0')

    ind_rmin = np.where(radar.range['data'] > rmin)[0]
    if ind_rmin.size > 0:
        ind_rmin = ind_rmin[0]
    else:
        warn('Maximum radar range below the minimum range for sun signal' + \
             ' estimation. The last ' + str(2 * nbins_min) + ' will be inspected')
        ind_rmin = int(radar.ngates - 2 * nbins_min)
        if ind_rmin < 0:
            warn('Radar range too short to retrieve sun signal')
            return None, None

    # parse the field parameters
    if pwrh_field is None:
        pwrh_field = get_field_name('signal_power_hh')
    if pwrv_field is None:
        pwrv_field = get_field_name('signal_power_vv')
    if zdr_field is None:
        zdr_field = get_field_name('differential_reflectivity')

    # extract fields from radar and prepare output
    try:
        radar.check_field_exists(pwrh_field)
        pwrh = radar.fields[pwrh_field]['data']
        mask_pwrh = np.ma.getmaskarray(pwrh)
        sun_hit_h = np.ma.zeros(np.shape(pwrh), dtype=np.uint8)
        sun_hit_h[mask_pwrh] = np.ma.masked
    except KeyError:
        pwrh = None
        sun_hit_h = None

    try:
        radar.check_field_exists(pwrv_field)
        pwrv = radar.fields[pwrv_field]['data']
        mask_pwrv = np.ma.getmaskarray(pwrv)
        sun_hit_v = np.ma.zeros(np.shape(pwrv), dtype=np.uint8)
        sun_hit_v[mask_pwrv] = np.ma.masked
    except KeyError:
        pwrv = None
        sun_hit_v = None

    try:
        radar.check_field_exists(zdr_field)
        zdr = radar.fields[zdr_field]['data']
        mask_zdr = np.ma.getmaskarray(zdr)
        if pwrh is not None:
            mask_zdr = np.logical_or(mask_zdr, mask_pwrh)
        if pwrv is not None:
            mask_zdr = np.logical_or(mask_zdr, mask_pwrv)
        zdr = np.ma.masked_where(mask_zdr, zdr)
        sun_hit_zdr = np.ma.zeros(np.shape(zdr), dtype=np.uint8)
        sun_hit_zdr[mask_zdr] = np.ma.masked
    except KeyError:
        zdr = None
        sun_hit_zdr = None

    if pwrh is None and pwrv is None and zdr is None:
        return None, None

    # get time at each ray
    time = num2date(radar.time['data'], radar.time['units'],
                    radar.time['calendar'])

    sun_hits = {
        'time': [], 'ray': [], 'NPrng': [],
        'rad_el': [], 'rad_az': [], 'sun_el': [], 'sun_az': [],
        'dBm_sun_hit': [], 'std(dBm_sun_hit)': [], 'NPh': [], 'NPhval': [],
        'dBmv_sun_hit': [], 'std(dBmv_sun_hit)': [], 'NPv': [], 'NPvval': [],
        'ZDR_sun_hit': [], 'std(ZDR_sun_hit)': [], 'NPzdr': [],
        'NPzdrval': []}

    for ray in range(radar.nrays):
        if radar.elevation['data'][ray] < elmin:
            continue

        if _PYSOLAR_AVAILABLE and sun_position == 'pysolar':
            elev_sun, azim_sun = sun_position_pysolar(
                time[ray], radar.latitude['data'][0],
                radar.longitude['data'][0])
        else:
            elev_sun, azim_sun = sun_position_mfr(
                time[ray], radar.latitude['data'][0],
                radar.longitude['data'][0], refraction=True)

        if elev_sun < 0:
            continue

        # get minimum range from where to compute the sun signal and
        # minimum number of gates to consider the signal valid
        ind_hmin = np.where(radar.gate_altitude['data'][ray, :] > hmin)[0]
        if ind_hmin.size > 0:
            ind_min = np.min([ind_rmin, ind_hmin[0]])
        else:
            ind_min = ind_rmin

        nrange = len(radar.range['data'])

        delev = np.ma.abs(radar.elevation['data'][ray] - elev_sun)
        dazim = np.ma.abs(
            (radar.azimuth['data'][ray] - azim_sun) *
            np.ma.cos(elev_sun * np.pi / 180.))
        if dazim > 360.:
            dazim -= 360.

        if delev > delev_max or dazim > dazim_max:
            continue

        # gas atmospheric attenuation from radar to TOA
        attg_sun = gas_att_sun(elev_sun, attg)

        sunpwrh_dBm = get_fillvalue()
        sunpwrh_std = get_fillvalue()
        sunpwrh_npoints = 0
        nvalidh = 0
        sun_hit_h_ray = None
        if pwrh is not None:
            (sunpwrh_dBm, sunpwrh_std, sunpwrh_npoints, nvalidh,
             sun_hit_h_ray) = _est_sun_hit_pwr_hs(
                 pwrh[ray, :], sun_hit_h[ray, :], attg_sun, max_std_pwr,
                 nbins_min, ind_min)
            sun_hit_h[ray, :] = sun_hit_h_ray

        sunpwrv_dBm = get_fillvalue()
        sunpwrv_std = get_fillvalue()
        sunpwrv_npoints = 0
        nvalidv = 0
        sun_hit_v_ray = None
        if pwrv is not None:
            (sunpwrv_dBm, sunpwrv_std, sunpwrv_npoints, nvalidv,
             sun_hit_v_ray) = _est_sun_hit_pwr_hs(
                 pwrv[ray, :], sun_hit_v[ray, :], attg_sun, max_std_pwr,
                 nbins_min, ind_min)
            sun_hit_v[ray, :] = sun_hit_v_ray

        sunzdr = get_fillvalue()
        sunzdr_std = get_fillvalue()
        sunzdr_npoints = 0
        nvalidzdr = 0
        if zdr is not None:
            (sunzdr, sunzdr_std, sunzdr_npoints, nvalidzdr,
             sun_hit_zdr_ray) = _est_sun_hit_zdr(
                 zdr[ray, :], sun_hit_zdr[ray, :], sun_hit_h_ray,
                 sun_hit_v_ray, max_std_zdr, nbins_min, ind_min)
            sun_hit_zdr[ray, :] = sun_hit_zdr_ray

        sun_hits['time'].append(time[ray])
        sun_hits['ray'].append(ray)
        sun_hits['NPrng'].append(nrange)
        sun_hits['rad_el'].append(radar.elevation['data'][ray])
        sun_hits['rad_az'].append(radar.azimuth['data'][ray])
        sun_hits['sun_el'].append(elev_sun)
        sun_hits['sun_az'].append(azim_sun)
        sun_hits['dBm_sun_hit'].append(sunpwrh_dBm)
        sun_hits['std(dBm_sun_hit)'].append(sunpwrh_std)
        sun_hits['NPh'].append(sunpwrh_npoints)
        sun_hits['NPhval'].append(nvalidh)
        sun_hits['dBmv_sun_hit'].append(sunpwrv_dBm)
        sun_hits['std(dBmv_sun_hit)'].append(sunpwrv_std)
        sun_hits['NPv'].append(sunpwrv_npoints)
        sun_hits['NPvval'].append(nvalidv)
        sun_hits['ZDR_sun_hit'].append(sunzdr)
        sun_hits['std(ZDR_sun_hit)'].append(sunzdr_std)
        sun_hits['NPzdr'].append(sunzdr_npoints)
        sun_hits['NPzdrval'].append(nvalidzdr)

    nhits = len(sun_hits['time'])
    if nhits == 0:
        return None, None

    # create output radar
    new_radar = deepcopy(radar)
    new_radar.fields = dict()

    if pwrh is not None:
        pwrh_dict = get_metadata(pwrh_field)
        pwrh_dict['data'] = pwrh

        sun_hit_h_dict = get_metadata('sun_hit_h')
        sun_hit_h_dict['data'] = sun_hit_h + 1
        sun_hit_h_dict.update({'_FillValue': 0})

        new_radar.add_field(pwrh_field, pwrh_dict)
        new_radar.add_field('sun_hit_h', sun_hit_h_dict)

    if pwrv is not None:
        pwrv_dict = get_metadata(pwrv_field)
        pwrv_dict['data'] = pwrv

        sun_hit_v_dict = get_metadata('sun_hit_v')
        sun_hit_v_dict['data'] = sun_hit_v + 1
        sun_hit_v_dict.update({'_FillValue': 0})

        new_radar.add_field(pwrv_field, pwrv_dict)
        new_radar.add_field('sun_hit_v', sun_hit_v_dict)

    if zdr is not None:
        zdr_dict = get_metadata(zdr_field)
        zdr_dict['data'] = zdr

        sun_hit_zdr_dict = get_metadata('sun_hit_zdr')
        sun_hit_zdr_dict['data'] = sun_hit_zdr + 1
        sun_hit_zdr_dict.update({'_FillValue': 0})

        new_radar.add_field(zdr_field, zdr_dict)
        new_radar.add_field('sun_hit_zdr', sun_hit_zdr_dict)

    sweeps = []
    for i in range(nhits):
        for sweep in range(new_radar.nsweeps):
            ray_start, ray_end = new_radar.get_start_end(sweep)
            if ((ray_start <= sun_hits['ray'][i]) and
                    (ray_end >= sun_hits['ray'][i])):
                sweeps.append(sweep)
                break

    new_radar = new_radar.extract_sweeps(sweeps)

    # write sun hit data as ndarray
    sun_hits['ray'] = np.asarray(sun_hits['ray'])
    sun_hits['NPrng'] = np.asarray(sun_hits['NPrng'])
    sun_hits['rad_el'] = np.asarray(sun_hits['rad_el'])
    sun_hits['rad_az'] = np.asarray(sun_hits['rad_az'])
    sun_hits['sun_el'] = np.asarray(sun_hits['sun_el'])
    sun_hits['sun_az'] = np.asarray(sun_hits['sun_az'])
    sun_hits['dBm_sun_hit'] = np.ma.masked_values(
        sun_hits['dBm_sun_hit'], get_fillvalue())
    sun_hits['std(dBm_sun_hit)'] = np.ma.masked_values(
        sun_hits['std(dBm_sun_hit)'], get_fillvalue())
    sun_hits['NPh'] = np.asarray(sun_hits['NPh'])
    sun_hits['NPhval'] = np.asarray(sun_hits['NPhval'])
    sun_hits['dBmv_sun_hit'] = np.ma.masked_values(
        sun_hits['dBmv_sun_hit'], get_fillvalue())
    sun_hits['std(dBmv_sun_hit)'] = np.ma.masked_values(
        sun_hits['std(dBm_sun_hit)'], get_fillvalue())
    sun_hits['NPv'] = np.asarray(sun_hits['NPv'])
    sun_hits['NPvval'] = np.asarray(sun_hits['NPvval'])
    sun_hits['ZDR_sun_hit'] = np.ma.masked_values(
        sun_hits['ZDR_sun_hit'], get_fillvalue())
    sun_hits['std(ZDR_sun_hit)'] = np.ma.masked_values(
        sun_hits['std(ZDR_sun_hit)'], get_fillvalue())
    sun_hits['NPzdr'] = np.asarray(sun_hits['NPzdr'])
    sun_hits['NPzdrval'] = np.asarray(sun_hits['NPzdrval'])

    return sun_hits, new_radar


def get_sun_hits_psr(
        radar, delev_max=2., dazim_max=2., elmin=5.,
        noise_thres=None, attg=None, sun_position='MF',
        pwrh_field=None, pwrv_field=None):
    """
    get data from suspected sun hits. The sun hits are detected by
    a simple noise threshold from the PSR data.

    Parameters
    ----------
    radar : Radar
        radar object
    delev_max, dazim_max : float
        maximum difference in elevation and azimuth between sun position and
        antenna pointing
    elmin : float
        minimum radar elevation angle
    noise_thres : float
        noise threshold to separate valid sun samples
    attg : float
        gas attenuation coefficient (1-way)
    sun_position : str
        The function to use to compute the sun position. Can be 'MF' or
        'pysolar'
    pwrh_field, pwrv_field : str
        names of the signal power in dBm for the H and V polarizations

    Returns
    -------
    sun_hits : dict
        a dictionary containing information of the sun hits
    """
    # get parameters
    if attg is None:
        # assign coefficients according to radar frequency
        if (radar.instrument_parameters is not None and
                'frequency' in radar.instrument_parameters):
            attg = get_coeff_attg(
                radar.instrument_parameters['frequency']['data'][0])
        else:
            attg = 0.
            warn('Unknown 1-way gas attenuation. It will be set to 0')

    # parse the field parameters
    if pwrh_field is None:
        pwrh_field = get_field_name('signal_power_hh')
    if pwrv_field is None:
        pwrv_field = get_field_name('signal_power_vv')

    # extract fields from radar and prepare output
    try:
        radar.check_field_exists(pwrh_field)
        pwrh = radar.fields[pwrh_field]['data'][:, 0]
    except KeyError:
        pwrh = None

    try:
        radar.check_field_exists(pwrv_field)
        pwrv = radar.fields[pwrv_field]['data'][:, 0]
    except KeyError:
        pwrv = None

    if pwrh is None and pwrv is None:
        return None, None

    # get time at each ray
    time = num2date(radar.time['data'], radar.time['units'],
                    radar.time['calendar'])

    sun_hits = {
        'time': [], 'ray': [], 'NPrng': [], 'nhits': [],
        'rad_el': [], 'rad_az': [], 'sun_el': [], 'sun_az': [],
        'dBm_sun_hit': [], 'std(dBm_sun_hit)': [],
        'dBmv_sun_hit': [], 'std(dBmv_sun_hit)': []}

    for ray in range(radar.nrays):
        if radar.elevation['data'][ray] < elmin:
            continue

        if _PYSOLAR_AVAILABLE and sun_position == 'pysolar':
            elev_sun, azim_sun = sun_position_pysolar(
                time[ray], radar.latitude['data'][0],
                radar.longitude['data'][0])
        else:
            elev_sun, azim_sun = sun_position_mfr(
                time[ray], radar.latitude['data'][0],
                radar.longitude['data'][0], refraction=True)

        if elev_sun < 0:
            continue

        if pwrh is not None:
            if np.ma.is_masked(pwrh[ray]):
                continue
            else:
                if pwrh[ray] <= noise_thres:
                    continue

        if pwrv is not None:
            if np.ma.is_masked(pwrv[ray]):
                continue
            else:
                if pwrv[ray] <= noise_thres:
                    continue

        nrange = len(radar.range['data'])

        delev = np.ma.abs(radar.elevation['data'][ray] - elev_sun)
        dazim = np.ma.abs(
            (radar.azimuth['data'][ray] - azim_sun) *
            np.ma.cos(elev_sun * np.pi / 180.))
        if dazim > 360.:
            dazim -= 360.

        if delev > delev_max or dazim > dazim_max:
            continue

        # gas atmospheric attenuation from radar to TOA
        attg_sun = gas_att_sun(elev_sun, attg)

        sunpwrh_dBm = get_fillvalue()
        if pwrh is not None:
            sunpwrh_dBm = _est_sun_hit_pwr_psr(pwrh[ray], attg_sun)

        sunpwrv_dBm = get_fillvalue()
        if pwrv is not None:
            sunpwrv_dBm = _est_sun_hit_pwr_psr(pwrv[ray], attg_sun)

        sun_hits['time'].append(time[ray])
        sun_hits['ray'].append(ray)
        sun_hits['NPrng'].append(nrange)
        sun_hits['rad_el'].append(radar.elevation['data'][ray])
        sun_hits['rad_az'].append(radar.azimuth['data'][ray])
        sun_hits['sun_el'].append(elev_sun)
        sun_hits['sun_az'].append(azim_sun)
        sun_hits['dBm_sun_hit'].append(sunpwrh_dBm)
        sun_hits['std(dBm_sun_hit)'].append(0.)
        sun_hits['dBmv_sun_hit'].append(sunpwrv_dBm)
        sun_hits['std(dBmv_sun_hit)'].append(0.)

    nhits = len(sun_hits['time'])
    if nhits == 0:
        return None, None

    # write sun hit data as ndarray
    sun_hits['ray'] = np.asarray(sun_hits['ray'])
    sun_hits['NPrng'] = np.asarray(sun_hits['NPrng'])
    sun_hits['nhits'] = np.asarray(nhits)
    sun_hits['rad_el'] = np.asarray(sun_hits['rad_el'])
    sun_hits['rad_az'] = np.asarray(sun_hits['rad_az'])
    sun_hits['sun_el'] = np.asarray(sun_hits['sun_el'])
    sun_hits['sun_az'] = np.asarray(sun_hits['sun_az'])
    sun_hits['dBm_sun_hit'] = np.ma.masked_values(
        sun_hits['dBm_sun_hit'], get_fillvalue())
    sun_hits['std(dBm_sun_hit)'] = np.ma.masked_values(
        sun_hits['std(dBm_sun_hit)'], get_fillvalue())
    sun_hits['dBmv_sun_hit'] = np.ma.masked_values(
        sun_hits['dBmv_sun_hit'], get_fillvalue())
    sun_hits['std(dBmv_sun_hit)'] = np.ma.masked_values(
        sun_hits['std(dBmv_sun_hit)'], get_fillvalue())

    return sun_hits


def get_sun_hits_ivic(
        radar, delev_max=2., dazim_max=2., elmin=1., npulses_ray=30,
        nbins_min=800, iterations=10, attg=None, sun_position='MF',
        max_std_zdr=1.5, pwrh_field=None, pwrv_field=None, zdr_field=None):
    """
    get data from suspected sun hits. The sun hits are detected using the
    Ivic et al. (2003) noise estimator

    Parameters
    ----------
    radar : Radar
        radar object
    delev_max, dazim_max : float
        maximum difference in elevation and azimuth between sun position and
        antenna pointing
    elmin : float
        minimum radar elevation angle
    npulses_ray : int
        Default number of pulses used in the computation of the ray. If the
        number of pulses is not in radar.instrument_parameters this will be
        used instead



    nbins_min: int
        minimum number of gates with noise to consider the retrieval valid
    iterations: int
        number of iterations in step 7 of Ivic algorithm
    attg : float
        gas attenuation coefficient (1-way)
    sun_position : str
        The function to use to compute the sun position. Can be 'MF' or
        'pysolar'
    max_std_zdr : float
        Maximum standard deviation of the estimated sun ZDR to consider the
        sun signal valid [dB]
    pwrh_field, pwrv_field, zdr_field : str
        names of the signal power in dBm for the H and V polarizations and the
        differential reflectivity

    Returns
    -------
    sun_hits : dict
        a dictionary containing information of the sun hits
    new_radar : radar object
        radar object containing sweeps that contain sun hits

    """
    # get parameters
    if attg is None:
        # assign coefficients according to radar frequency
        if (radar.instrument_parameters is not None and
                'frequency' in radar.instrument_parameters):
            attg = get_coeff_attg(
                radar.instrument_parameters['frequency']['data'][0])
        else:
            attg = 0.
            warn('Unknown 1-way gas attenuation. It will be set to 0')

    # parse the field parameters
    if pwrh_field is None:
        pwrh_field = get_field_name('signal_power_hh')
    if pwrv_field is None:
        pwrv_field = get_field_name('signal_power_vv')
    if zdr_field is None:
        zdr_field = get_field_name('differential_reflectivity')

    # extract fields from radar and prepare output
    try:
        radar.check_field_exists(pwrh_field)
        pwrh = radar.fields[pwrh_field]['data']
        mask_pwrh = np.ma.getmaskarray(pwrh)
        sun_hit_h = np.ma.zeros(np.shape(pwrh), dtype=np.uint8)
        sun_hit_h[mask_pwrh] = np.ma.masked
    except KeyError:
        pwrh = None
        sun_hit_h = None

    try:
        radar.check_field_exists(pwrv_field)
        pwrv = radar.fields[pwrv_field]['data']
        mask_pwrv = np.ma.getmaskarray(pwrv)
        sun_hit_v = np.ma.zeros(np.shape(pwrv), dtype=np.uint8)
        sun_hit_v[mask_pwrv] = np.ma.masked
    except KeyError:
        pwrv = None
        sun_hit_v = None

    try:
        radar.check_field_exists(zdr_field)
        zdr = radar.fields[zdr_field]['data']
        mask_zdr = np.ma.getmaskarray(zdr)
        if pwrh is not None:
            mask_zdr = np.logical_or(mask_zdr, mask_pwrh)
        if pwrv is not None:
            mask_zdr = np.logical_or(mask_zdr, mask_pwrv)
        zdr = np.ma.masked_where(mask_zdr, zdr)
        sun_hit_zdr = np.ma.zeros(np.shape(zdr), dtype=np.uint8)
        sun_hit_zdr[mask_zdr] = np.ma.masked
    except KeyError:
        zdr = None
        sun_hit_zdr = None

    if pwrh is None and pwrv is None and zdr is None:
        return None, None

    # get time at each ray
    time = num2date(radar.time['data'], radar.time['units'],
                    radar.time['calendar'])

    # get number of pulses per ray
    if radar.instrument_parameters is not None:
        if 'number_of_pulses' in radar.instrument_parameters:
            npulses = radar.instrument_parameters['number_of_pulses']['data']
        else:
            warn('Unknown number of pulses per ray. Default value ' +
                 str(npulses_ray) + ' will be used for all rays')
            npulses = np.zeros(radar.nrays, dtype=int) + npulses_ray
    else:
        warn('Unknown number of pulses per ray. Default value ' +
             str(npulses_ray) + ' will be used for all rays')
        npulses = np.zeros(radar.nrays, dtype=int) + npulses_ray

    if pwrh is not None or pwrv is not None:
        # threshold for step 1:
        pct = ivic_pct_table(npulses)

        # threshold for step 2:
        flat_reg_wlen = ivic_flat_reg_wind_len_table(npulses)
        # we want an odd window
        for i in range(radar.nrays):
            if flat_reg_wlen[i] % 2 == 0:
                flat_reg_wlen[i] += 1
        flat_reg_var_max = ivic_flat_reg_var_max_table(npulses)

        # threshold for step 3:
        snr_thr = ivic_snr_thr_table(npulses)

    sun_hits = {
        'time': [], 'ray': [], 'NPrng': [],
        'rad_el': [], 'rad_az': [], 'sun_el': [], 'sun_az': [],
        'dBm_sun_hit': [], 'std(dBm_sun_hit)': [], 'NPh': [], 'NPhval': [],
        'dBmv_sun_hit': [], 'std(dBmv_sun_hit)': [], 'NPv': [], 'NPvval': [],
        'ZDR_sun_hit': [], 'std(ZDR_sun_hit)': [], 'NPzdr': [],
        'NPzdrval': []}

    for ray, npuls in enumerate(npulses):
        if radar.elevation['data'][ray] < elmin:
            continue

        if _PYSOLAR_AVAILABLE and sun_position == 'pysolar':
            elev_sun, azim_sun = sun_position_pysolar(
                time[ray], radar.latitude['data'][0],
                radar.longitude['data'][0])
        else:
            elev_sun, azim_sun = sun_position_mfr(
                time[ray], radar.latitude['data'][0],
                radar.longitude['data'][0], refraction=True)

        if elev_sun < 0:
            continue

        delev = np.ma.abs(radar.elevation['data'][ray] - elev_sun)
        dazim = np.ma.abs(
            (radar.azimuth['data'][ray] - azim_sun) *
            np.ma.cos(elev_sun * np.pi / 180.))
        if dazim > 360.:
            dazim -= 360.

        if delev > delev_max or dazim > dazim_max:
            continue

        # gas atmospheric attenuation from radar to TOA
        attg_sun = gas_att_sun(elev_sun, attg)

        sunpwrh_dBm = get_fillvalue()
        sunpwrh_std = get_fillvalue()
        sunpwrh_npoints = 0
        nvalidh = 0
        sun_hit_h_ray = None
        if pwrh is not None:
            (sunpwrh_dBm, sunpwrh_std, sunpwrh_npoints, nvalidh,
             sun_hit_h_ray) = _est_sun_hit_pwr_ivic(
                 pwrh[ray, :], sun_hit_h[ray, :], attg_sun, pct[ray],
                 flat_reg_wlen[ray], flat_reg_var_max[ray], snr_thr[ray], npuls,
                 nbins_min, iterations)
            sun_hit_h[ray, :] = sun_hit_h_ray

        sunpwrv_dBm = get_fillvalue()
        sunpwrv_std = get_fillvalue()
        sunpwrv_npoints = 0
        nvalidv = 0
        sun_hit_v_ray = None
        if pwrv is not None:
            (sunpwrv_dBm, sunpwrv_std, sunpwrv_npoints, nvalidv,
             sun_hit_v_ray) = _est_sun_hit_pwr_ivic(
                 pwrv[ray, :], sun_hit_v[ray, :], attg_sun, pct[ray],
                 flat_reg_wlen, flat_reg_var_max[ray], snr_thr[ray], npuls,
                 nbins_min, iterations)
            sun_hit_v[ray, :] = sun_hit_v_ray

        sunzdr = get_fillvalue()
        sunzdr_std = get_fillvalue()
        sunzdr_npoints = 0
        nvalidzdr = 0
        if zdr is not None:
            (sunzdr, sunzdr_std, sunzdr_npoints, nvalidzdr,
             sun_hit_zdr_ray) = _est_sun_hit_zdr(
                 zdr[ray, :], sun_hit_zdr[ray, :], sun_hit_h_ray,
                 sun_hit_v_ray, max_std_zdr, int(nbins_min / npuls), 0)
            sun_hit_zdr[ray, :] = sun_hit_zdr_ray

        sun_hits['time'].append(time[ray])
        sun_hits['ray'].append(ray)
        sun_hits['NPrng'].append(radar.ngates)
        sun_hits['rad_el'].append(radar.elevation['data'][ray])
        sun_hits['rad_az'].append(radar.azimuth['data'][ray])
        sun_hits['sun_el'].append(elev_sun)
        sun_hits['sun_az'].append(azim_sun)
        sun_hits['dBm_sun_hit'].append(sunpwrh_dBm)
        sun_hits['std(dBm_sun_hit)'].append(sunpwrh_std)
        sun_hits['NPh'].append(sunpwrh_npoints)
        sun_hits['NPhval'].append(nvalidh)
        sun_hits['dBmv_sun_hit'].append(sunpwrv_dBm)
        sun_hits['std(dBmv_sun_hit)'].append(sunpwrv_std)
        sun_hits['NPv'].append(sunpwrv_npoints)
        sun_hits['NPvval'].append(nvalidv)
        sun_hits['ZDR_sun_hit'].append(sunzdr)
        sun_hits['std(ZDR_sun_hit)'].append(sunzdr_std)
        sun_hits['NPzdr'].append(sunzdr_npoints)
        sun_hits['NPzdrval'].append(nvalidzdr)

    nhits = len(sun_hits['time'])
    if nhits == 0:
        return None, None

    # create output radar
    new_radar = deepcopy(radar)
    new_radar.fields = dict()

    if pwrh is not None:
        pwrh_dict = get_metadata(pwrh_field)
        pwrh_dict['data'] = pwrh

        sun_hit_h_dict = get_metadata('sun_hit_h')
        sun_hit_h_dict['data'] = sun_hit_h + 1
        sun_hit_h_dict.update({'_FillValue': 0})

        new_radar.add_field(pwrh_field, pwrh_dict)
        new_radar.add_field('sun_hit_h', sun_hit_h_dict)

    if pwrv is not None:
        pwrv_dict = get_metadata(pwrv_field)
        pwrv_dict['data'] = pwrv

        sun_hit_v_dict = get_metadata('sun_hit_v')
        sun_hit_v_dict['data'] = sun_hit_v + 1
        sun_hit_v_dict.update({'_FillValue': 0})

        new_radar.add_field(pwrv_field, pwrv_dict)
        new_radar.add_field('sun_hit_v', sun_hit_v_dict)

    if zdr is not None:
        zdr_dict = get_metadata(zdr_field)
        zdr_dict['data'] = zdr

        sun_hit_zdr_dict = get_metadata('sun_hit_zdr')
        sun_hit_zdr_dict['data'] = sun_hit_zdr + 1
        sun_hit_zdr_dict.update({'_FillValue': 0})

        new_radar.add_field(zdr_field, zdr_dict)
        new_radar.add_field('sun_hit_zdr', sun_hit_zdr_dict)

    sweeps = []
    for i in range(nhits):
        for sweep in range(new_radar.nsweeps):
            ray_start, ray_end = new_radar.get_start_end(sweep)
            if ((ray_start <= sun_hits['ray'][i]) and
                    (ray_end >= sun_hits['ray'][i])):
                sweeps.append(sweep)
                break

    new_radar = new_radar.extract_sweeps(sweeps)

    # write sun hit data as ndarray
    sun_hits['ray'] = np.asarray(sun_hits['ray'])
    sun_hits['NPrng'] = np.asarray(sun_hits['NPrng'])
    sun_hits['rad_el'] = np.asarray(sun_hits['rad_el'])
    sun_hits['rad_az'] = np.asarray(sun_hits['rad_az'])
    sun_hits['sun_el'] = np.asarray(sun_hits['sun_el'])
    sun_hits['sun_az'] = np.asarray(sun_hits['sun_az'])
    sun_hits['dBm_sun_hit'] = np.ma.masked_values(
        sun_hits['dBm_sun_hit'], get_fillvalue())
    sun_hits['std(dBm_sun_hit)'] = np.ma.masked_values(
        sun_hits['std(dBm_sun_hit)'], get_fillvalue())
    sun_hits['NPh'] = np.asarray(sun_hits['NPh'])
    sun_hits['NPhval'] = np.asarray(sun_hits['NPhval'])
    sun_hits['dBmv_sun_hit'] = np.ma.masked_values(
        sun_hits['dBmv_sun_hit'], get_fillvalue())
    sun_hits['std(dBmv_sun_hit)'] = np.ma.masked_values(
        sun_hits['std(dBm_sun_hit)'], get_fillvalue())
    sun_hits['NPv'] = np.asarray(sun_hits['NPv'])
    sun_hits['NPvval'] = np.asarray(sun_hits['NPvval'])
    sun_hits['ZDR_sun_hit'] = np.ma.masked_values(
        sun_hits['ZDR_sun_hit'], get_fillvalue())
    sun_hits['std(ZDR_sun_hit)'] = np.ma.masked_values(
        sun_hits['std(ZDR_sun_hit)'], get_fillvalue())
    sun_hits['NPzdr'] = np.asarray(sun_hits['NPzdr'])
    sun_hits['NPzdrval'] = np.asarray(sun_hits['NPzdrval'])

    return sun_hits, new_radar


def sun_retrieval(
        az_rad, az_sun, el_rad, el_sun, sun_hit, sun_hit_std,
        az_width_co=None, el_width_co=None, az_width_cross=None,
        el_width_cross=None, is_zdr=False):
    """
    Estimates sun parameters from sun hits

    Parameters
    ----------
    az_rad, az_sun, el_rad, el_sun : float array
        azimuth and elevation values of the sun and the radar
    sun_hit : float array
        sun hit value. Either power in dBm or ZDR in dB
    sun_hit_std : float array
        standard deviation of the sun hit value in dB
    az_width_co, el_width_co, az_width_cross, el_width_cross : float
        azimuth and elevation antenna width for each channel
    is_zdr : boolean
        boolean to signal that is ZDR data

    Returns
    -------
    val, val_std : float
        retrieved value and its standard deviation
    az_bias, el_bias : float
        retrieved azimuth and elevation antenna bias respect to the sun
        position
    az_width, el_width : float
        retrieved azimuth and elevation antenna widths
    nhits : int
        number of sun hits used in the retrieval
    par : float array
        and array with the 5 parameters of the Gaussian fit

    """
    # mask non hit data
    mask = np.ma.getmaskarray(sun_hit)

    az_rad = np.ma.masked_where(mask, az_rad)
    az_sun = np.ma.masked_where(mask, az_sun)
    el_rad = np.ma.masked_where(mask, el_rad)
    el_sun = np.ma.masked_where(mask, el_sun)
    sun_hit = np.ma.masked_where(mask, sun_hit)

    az_rad = az_rad.compressed()
    az_sun = az_sun.compressed()
    el_rad = el_rad.compressed()
    el_sun = el_sun.compressed()
    sun_hit = sun_hit.compressed()

    nhits = len(sun_hit)
    if not is_zdr:
        npar = 3
        if (az_width_co is None) or (el_width_co is None):
            npar = 5

    else:
        npar = 3
        if ((az_width_co is None) or (el_width_co is None) or
                (az_width_cross is None) or (el_width_cross is None)):
            npar = 5

    if nhits < npar:
        warn('Unable to perform sun retrieval. Not enough sun hits')
        return None

    par_aux, alpha, beta = gauss_fit(
        az_rad, az_sun, el_rad, el_sun, sun_hit, npar)

    if npar == 3:
        par = np.ma.zeros(npar)
        par[0:npar - 1] = par_aux
        coeff = 40. * np.ma.log10(2.)
        if is_zdr:
            par[3] = (
                coeff * (1. / np.ma.power(az_width_cross, 2.) -
                         1. / np.ma.power(az_width_co, 2.)))
            par[4] = (
                coeff * (1. / np.ma.power(el_width_cross, 2.) -
                         1. / np.ma.power(el_width_co, 2.)))
        else:
            par[3] = -np.ma.power(az_width_co, 2.) / coeff
            par[4] = -np.ma.power(el_width_co, 2.) / coeff
    else:
        par = par_aux

    if (not is_zdr) and (par[3] > 0. or par[4] > 0.):
        warn('Parameter ax and/or ay of Psun equation positive. ' +
             'Retrieval not correct')

    val, val_std, az_bias, el_bias, az_width, el_width = retrieval_result(
        sun_hit, alpha, beta, par, npar)

    return val, val_std, az_bias, el_bias, az_width, el_width, nhits, par


def est_rhohv_rain(
        radar, ind_rmin=10, ind_rmax=500, zmin=20., zmax=40., thickness=700.,
        doc=None, fzl=None, rhohv_field=None, temp_field=None,
        iso0_field=None, refl_field=None, temp_ref='temperature'):
    """
    Estimates the quantiles of RhoHV in rain for each sweep

    Parameters
    ----------
    radar : Radar
        radar object
    ind_rmin, ind_rmax : int
        Min and max range index where to look for rain
    zmin, zmax : float
        The minimum and maximum reflectivity to consider the radar bin
        suitable rain
    thickness : float
        Assumed thickness of the melting layer
    doc : float
        Number of gates at the end of each ray to to remove from the
        calculation.
    fzl : float
        Freezing layer, gates above this point are not included in the
        correction.
    temp_field, iso0_field, rhohv_field, refl_field : str
        Field names within the radar object which represent the temperature,
        the height over the iso0, co-polar correlation and reflectivity
        fields. A value of None will use the default field name as defined in
        the Py-ART configuration file.
    temp_ref : str
        the field use as reference for temperature. Can be either temperature
        or height_over_iso0

    Returns
    -------
    rhohv_rain_dict : dict
        The estimated RhoHV in rain for each sweep and metadata

    """
    if (radar.instrument_parameters is not None and
            'radar_beam_width_h' in radar.instrument_parameters):
        beamwidth = (
            radar.instrument_parameters['radar_beam_width_h']['data'][0])
    else:
        warn('Unknown radar antenna beamwidth.')
        beamwidth = None

    # parse the field parameters
    if rhohv_field is None:
        rhohv_field = get_field_name('cross_correlation_ratio')
    if refl_field is None:
        refl_field = get_field_name('reflectivity')
    if temp_ref == 'temperature':
        if temp_field is None:
            temp_field = get_field_name('temperature')
    elif temp_ref == 'height_over_iso0':
        if iso0_field is None:
            iso0_field = get_field_name('height_over_iso0')

    # extract fields from radar
    radar.check_field_exists(rhohv_field)
    rhohv = deepcopy(radar.fields[rhohv_field]['data'])

    radar.check_field_exists(refl_field)
    refl = radar.fields[refl_field]['data']

    # determine the valid data (i.e. data below the melting layer)
    mask = np.ma.getmaskarray(rhohv)

    mask_fzl, _ = get_mask_fzl(
        radar, fzl=fzl, doc=doc, min_temp=0., max_h_iso0=0.,
        thickness=thickness, beamwidth=beamwidth, temp_field=temp_field,
        iso0_field=iso0_field, temp_ref=temp_ref)
    mask = np.logical_or(mask, mask_fzl)

    mask_refl = np.logical_or(np.ma.getmaskarray(refl),
                              np.logical_or(refl < zmin, refl > zmax))
    mask = np.logical_or(mask, mask_refl)

    rhohv_rain = np.ma.masked_where(mask, rhohv)
    rhohv_rain[:, 0:ind_rmin] = np.ma.masked
    rhohv_rain[:, ind_rmax:-1] = np.ma.masked

    rhohv_rain_dict = get_metadata('cross_correlation_ratio_in_rain')
    rhohv_rain_dict['data'] = rhohv_rain

    return rhohv_rain_dict


def est_zdr_precip(
        radar, ind_rmin=10, ind_rmax=500, zmin=20., zmax=22., rhohvmin=0.97,
        phidpmax=10., elmax=None, thickness=700., doc=None, fzl=None,
        zdr_field=None, rhohv_field=None, phidp_field=None, temp_field=None,
        iso0_field=None, refl_field=None, temp_ref='temperature'):
    """
    Filters out all undesired data to be able to estimate ZDR bias, either in
    moderate rain or from vertically pointing scans

    Parameters
    ----------
    radar : Radar
        radar object
    ind_rmin, ind_rmax : int
        Min and max range index where to look for rain
    zmin, zmax : float
        The minimum and maximum reflectivity to consider the radar bin
        suitable rain
    rhohvmin : float
        Minimum RhoHV to consider the radar bin suitable rain
    phidpmax : float
        Maximum PhiDP to consider the radar bin suitable rain
    elmax : float
        Maximum elevation
    thickness : float
        Assumed thickness of the melting layer
    doc : float
        Number of gates at the end of each ray to to remove from the
        calculation.
    fzl : float
        Freezing layer, gates above this point are not included in the
        correction.
    zdr_field, rhohv_field, refl_field, phidp_field, temp_field,
        iso0_field : str
        Field names within the radar object which represent the differential
        reflectivity, co-polar correlation, reflectivity, differential phase,
        temperature and height relative to the iso0 fields. A value of None
        will use the default field name as defined in the Py-ART configuration
        file.
    temp_ref : str
        the field use as reference for temperature. Can be either temperature,
        height_over_iso0, fixed_fzl or None

    Returns
    -------
    zdr_prec_dict : dict
        The ZDR data complying with specifications and metadata

    """
    if (radar.instrument_parameters is not None and
            'radar_beam_width_h' in radar.instrument_parameters):
        beamwidth = (
            radar.instrument_parameters['radar_beam_width_h']['data'][0])
    else:
        warn('Unknown radar antenna beamwidth.')
        beamwidth = None

    # parse the field parameters
    if zdr_field is None:
        zdr_field = get_field_name('differential_reflectivity')
    if rhohv_field is None:
        rhohv_field = get_field_name('cross_correlation_ratio')
    if refl_field is None:
        refl_field = get_field_name('reflectivity')
    if phidp_field is None:
        phidp_field = get_field_name('differential_phase')

    if temp_ref == 'temperature':
        if temp_field is None:
            temp_field = get_field_name('temperature')
    elif temp_ref == 'height_over_iso0':
        if iso0_field is None:
            iso0_field = get_field_name('height_over_iso0')

    # extract fields from radar
    radar.check_field_exists(zdr_field)
    zdr = deepcopy(radar.fields[zdr_field]['data'])

    radar.check_field_exists(refl_field)
    refl = radar.fields[refl_field]['data']

    radar.check_field_exists(rhohv_field)
    rhohv = radar.fields[rhohv_field]['data']

    radar.check_field_exists(phidp_field)
    phidp = radar.fields[phidp_field]['data']

    # determine the valid data (i.e. data below the melting layer)
    mask = np.ma.getmaskarray(zdr)

    if temp_ref is not None:
        mask_fzl, _ = get_mask_fzl(
            radar, fzl=fzl, doc=doc, min_temp=0., max_h_iso0=0.,
            thickness=thickness, beamwidth=beamwidth, temp_field=temp_field,
            iso0_field=iso0_field, temp_ref=temp_ref)
        mask = np.logical_or(mask, mask_fzl)

    mask_refl = np.logical_or(np.ma.getmaskarray(refl),
                              np.logical_or(refl < zmin, refl > zmax))
    mask = np.logical_or(mask, mask_refl)

    mask_rhohv = np.logical_or(np.ma.getmaskarray(rhohv), rhohv < rhohvmin)
    mask = np.logical_or(mask, mask_rhohv)

    mask_phidp = np.logical_or(np.ma.getmaskarray(phidp), phidp > phidpmax)
    mask = np.logical_or(mask, mask_phidp)

    zdr_prec = np.ma.masked_where(mask, zdr)
    zdr_prec[:, 0:ind_rmin] = np.ma.masked
    zdr_prec[:, ind_rmax:-1] = np.ma.masked

    if elmax is not None:
        ind_el = np.where(radar.elevation['data'] > elmax)
        zdr_prec[ind_el, :] = np.ma.masked

    zdr_prec_dict = get_metadata('differential_reflectivity_in_precipitation')
    zdr_prec_dict['data'] = zdr_prec

    return zdr_prec_dict


def est_zdr_snow(
        radar, ind_rmin=10, ind_rmax=500, zmin=0., zmax=30., snrmin=10.,
        snrmax=50., rhohvmin=0.97, kept_values=[2], phidpmax=10., kdpmax=None,
        tempmin=None, tempmax=None, elmax=None, zdr_field=None,
        rhohv_field=None, phidp_field=None, temp_field=None, snr_field=None,
        hydro_field=None, kdp_field=None, refl_field=None):
    """
    Filters out all undesired data to be able to estimate ZDR bias in snow

    Parameters
    ----------
    radar : Radar
        radar object
    ind_rmin, ind_rmax : int
        Min and max range index where to look for snow
    zmin, zmax : float
        The minimum and maximum reflectivity to consider the radar bin
        suitable snow
    snrmin, snrmax : float
        The minimum and maximum SNR to consider the radar bin suitable snow
    rhohvmin : float
        Minimum RhoHV to consider the radar bin suitable snow
    kept_values : list of int
        The hydrometeor classification values to keep
    phidpmax : float
        Maximum PhiDP to consider the radar bin suitable snow
    kdpmax : float or None
        Maximum KDP. If not none this is the maximum KDP value to consider
        the radar bin suitable snow
    tempmin, tempmax : float or None
        If not None, the minimum and maximum temperature to consider the
        radar bin suitable snow
    elmax : float
        Maximum elevation
    zdr_field, rhohv_field, refl_field, phidp_field, kdp_field, temp_field,
    snr_field, hydro_field : str
        Field names within the radar object which represent the differential
        reflectivity, co-polar correlation, reflectivity, differential phase,
        specific differnetial phase, signal to noise ratio, hydrometeor
        classification and temperature fields. A value of None will use the
        default field name as defined in the Py-ART configuration file.

    Returns
    -------
    zdr_snow_dict : dict
        The ZDR data complying with specifications and metadata

    """
    # parse the field parameters
    if zdr_field is None:
        zdr_field = get_field_name('differential_reflectivity')
    if rhohv_field is None:
        rhohv_field = get_field_name('cross_correlation_ratio')
    if refl_field is None:
        refl_field = get_field_name('reflectivity')
    if phidp_field is None:
        phidp_field = get_field_name('differential_phase')
    if temp_field is None:
        temp_field = get_field_name('temperature')
    if kdp_field is None:
        kdp_field = get_field_name('specific_differential_phase')
    if snr_field is None:
        snr_field = get_field_name('signal_to_noise_ratio')
    if hydro_field is None:
        hydro_field = get_field_name('radar_echo_classification')

    # extract fields from radar
    radar.check_field_exists(zdr_field)
    zdr = deepcopy(radar.fields[zdr_field]['data'])

    radar.check_field_exists(refl_field)
    refl = radar.fields[refl_field]['data']

    radar.check_field_exists(rhohv_field)
    rhohv = radar.fields[rhohv_field]['data']

    radar.check_field_exists(phidp_field)
    phidp = radar.fields[phidp_field]['data']

    radar.check_field_exists(hydro_field)

    # determine the valid data
    mask = np.ma.getmaskarray(zdr)

    hydro_gatefilter = class_based_gate_filter(
        radar, field=hydro_field, kept_values=kept_values)
    mask_hydro = hydro_gatefilter.gate_excluded == 1

    mask = np.logical_or(mask, mask_hydro)

    if snr_field in radar.fields:
        snr_gatefilter = snr_based_gate_filter(
            radar, snr_field=snr_field, min_snr=snrmin, max_snr=snrmax)
        mask_snr = snr_gatefilter.gate_excluded == 1

        mask = np.logical_or(mask, mask_snr)
    else:
        warn('No filtering according to SNR. SNR field not available')

    mask_refl = np.logical_or(np.ma.getmaskarray(refl),
                              np.logical_or(refl < zmin, refl > zmax))
    mask = np.logical_or(mask, mask_refl)

    mask_rhohv = np.logical_or(np.ma.getmaskarray(rhohv), rhohv < rhohvmin)
    mask = np.logical_or(mask, mask_rhohv)

    mask_phidp = np.logical_or(np.ma.getmaskarray(phidp), phidp > phidpmax)
    mask = np.logical_or(mask, mask_phidp)

    if (tempmin is not None and tempmax is not None and
            temp_field in radar.fields):
        temp = radar.fields[temp_field]['data']
        mask_temp = np.logical_or(temp < tempmin, temp > tempmax)
        mask = np.logical_or(mask, mask_temp)

    if kdpmax is not None and kdp_field in radar.fields:
        kdp = radar.fields[kdp_field]['data']
        mask = np.logical_or(mask, kdp > kdpmax)

    zdr_snow = np.ma.masked_where(mask, zdr)
    zdr_snow[:, 0:ind_rmin] = np.ma.masked
    zdr_snow[:, ind_rmax:-1] = np.ma.masked

    if elmax is not None:
        ind_el = np.where(radar.elevation['data'] > elmax)
        zdr_snow[ind_el, :] = np.ma.masked

    zdr_snow_dict = get_metadata('differential_reflectivity_in_snow')
    zdr_snow_dict['data'] = zdr_snow

    return zdr_snow_dict


def selfconsistency_bias(
        radar, zdr_kdpzh_dict, min_rhohv=0.92, filter_rain=True, max_phidp=20.,
        smooth_wind_len=5, doc=None, fzl=None, thickness=700., min_rcons=20,
        dphidp_min=2, dphidp_max=16, parametrization='None', refl_field=None,
        phidp_field=None, zdr_field=None, temp_field=None, iso0_field=None,
        hydro_field=None, rhohv_field=None, temp_ref='temperature',
        check_wet_radome=True, wet_radome_refl=25., wet_radome_len_min=4,
        wet_radome_len_max=8, wet_radome_ngates_min=180,
        valid_gates_only=False, keep_points=False, kdp_wind_len=12):
    """
    Estimates reflectivity bias at each ray using the self-consistency
    algorithm by Gourley

    Parameters
    ----------
    radar : Radar
        radar object
    zdr_kdpzh_dict : dict
        dictionary containing a look up table relating ZDR with KDP/Zh for
        different elevations
    min_rhohv : float
        minimum RhoHV value to consider the data valid
    filter_rain : bool
        If True the hydrometeor classification is going to be used to filter
        out all gates that are not rain
    max_phidp : float
        maximum PhiDP value to consider the data valid
    smooth_wind_len : int
        length of the smoothing window
    doc : float
        Number of gates at the end of each ray to to remove from the
        calculation.
    fzl : float
        Freezing layer, gates above this point are not included in the
        correction.
    thickness : float
        assumed melting layer thickness [m]
    min_rcons : int
        minimum number of consecutive gates to consider a valid segment of
        PhiDP
    dphidp_min : float
        minimum differential phase shift in a segment
    dphidp_max : float
        maximum differential phase shift in a segment
    parametrization : str
        The type of parametrization for the self-consistency curves. Can be
        'None', 'Gourley', 'Wolfensberger', 'Louf', 'Gorgucci' or 'Vaccarono'.
        'None' will use tables contained in zdr_kdpzh_dict.
    refl_field, phidp_field, zdr_field : str
        Field names within the radar object which represent the reflectivity,
        differential phase and differential reflectivity fields. A value of
        None will use the default field name as defined in the Py-ART
        configuration file.
    temp_field, iso0_field, hydro_field, rhohv_field : str
        Field names within the radar object which represent the temperature,
        the height relative to the iso0, the hydrometeor classification and
        the co-polar correlation fields. A value of None will use the default
        field name as defined in the Py-ART configuration file. They are going
        to be used only if available.
    kdpsim_field, phidpsim_field : str
        Field names which represent the estimated specific differential phase
        and differential phase. A value of None will use the default
        field name as defined in the Py-ART configuration file.
    temp_ref : str
        the field use as reference for temperature. Can be either temperature,
        height_over_iso0 or fixed_fzl
    check_wet_radome : Bool
        if True the average reflectivity of the closest gates to the radar
        is going to be check to find out whether there is rain over the
        radome. If there is rain no bias will be computed
    wet_radome_refl : Float
        Average reflectivity of the gates close to the radar to consider
        the radome as wet
    wet_radome_len_min, wet_radome_len_max : int
        Mim and max gate indices of the disk around the radome used to decide
        whether the radome is wet
    wet_radome_ngates_min : int
        Minimum number of valid gates to consider that the radome is wet
    valid_gates_only : Bool
        If True the reflectivity bias obtained for each valid ray is going to
        be assigned only to gates of the segment used. That will give more
        weight to longer segments when computing the total bias.
    keep_points : Bool
        If True the ZDR, ZH and KDP of the gates used in the self-
        consistency algorithm are going to be stored for further analysis
    kdp_wind_len : int
        The length of the window used to compute KDP with the single
        window least square method

    Returns
    -------
    refl_bias_dict : dict
        the bias at each ray field and metadata
    selfconsistency_dict : dict
        If keep_poinst set, a dictionary containing the measured valid values
        of ZDR, Zh and KDP. None otherwise

    """
    # parse the field parameters
    if refl_field is None:
        refl_field = get_field_name('reflectivity')
    if zdr_field is None:
        zdr_field = get_field_name('zdr')
    if phidp_field is None:
        # use corrrected_differential_phase or unfolded_differential_phase
        # fields if they are available, if not use differential_phase field
        phidp_field = get_field_name('corrected_differential_phase')
        if phidp_field not in radar.fields:
            phidp_field = get_field_name('unfolded_differential_phase')
        if phidp_field not in radar.fields:
            phidp_field = get_field_name('differential_phase')
    if rhohv_field is None:
        rhohv_field = get_field_name('cross_correlation_ratio')
    if hydro_field is None:
        hydro_field = get_field_name('radar_echo_classification')

    if temp_ref == 'temperature':
        if temp_field is None:
            temp_field = get_field_name('temperature')
    elif temp_ref == 'height_over_iso0':
        if iso0_field is None:
            iso0_field = get_field_name('height_over_iso0')

    # extract fields from radar, refl, zdr and phidp must exist
    radar.check_field_exists(refl_field)
    refl = radar.fields[refl_field]['data']

    radar.check_field_exists(zdr_field)
    zdr = radar.fields[zdr_field]['data']

    radar.check_field_exists(phidp_field)
    phidp = radar.fields[phidp_field]['data']

    rhohv = None
    if min_rhohv is not None:
        try:
            radar.check_field_exists(rhohv_field)
            rhohv = radar.fields[rhohv_field]['data']
        except KeyError:
            min_rhohv = None
            rhohv = None

    hydro = None
    if filter_rain:
        try:
            radar.check_field_exists(hydro_field)
            hydro = radar.fields[hydro_field]['data']
        except KeyError:
            filter_rain = False
            hydro = None

    _, phidp_sim = _selfconsistency_kdp_phidp(
        radar, refl, zdr, phidp, zdr_kdpzh_dict, max_phidp=max_phidp,
        smooth_wind_len=smooth_wind_len, rhohv=rhohv, min_rhohv=min_rhohv,
        hydro=hydro, filter_rain=filter_rain, doc=doc, fzl=fzl,
        thickness=thickness, parametrization=parametrization,
        temp_field=temp_field, iso0_field=iso0_field, temp_ref=temp_ref)

    refl_bias = np.ma.masked_all((radar.nrays, radar.ngates))
    # check if there is rain over the radome
    if check_wet_radome:
        # self_mask = np.ma.getmaskarray(phidp_sim)[:, 0:wet_radome_len]
        refl_radome = refl[:, wet_radome_len_min:wet_radome_len_max + 1]
        # refl_radome[self_mask] = np.ma.masked
        refl_avg = np.ma.mean(refl_radome)
        ngates_wet = (refl_radome.compressed()).size
        if refl_avg > wet_radome_refl and ngates_wet > wet_radome_ngates_min:
            warn('Rain over radome!!!\n Avg reflectivity between ' +
                 str(radar.range['data'][wet_radome_len_min]) + ' and ' +
                 str(radar.range['data'][wet_radome_len_max]) + ' km ' +
                 str(refl_avg) + '. Number of wet gates ' + str(ngates_wet))
            refl_bias_dict = get_metadata('reflectivity_bias')
            refl_bias_dict['data'] = refl_bias
            return refl_bias_dict, None

        warn('Avg reflectivity between ' +
             str(radar.range['data'][wet_radome_len_min]) + ' and ' +
             str(radar.range['data'][wet_radome_len_max]) + ' km ' +
             str(refl_avg) + '. Number of wet gates ' + str(ngates_wet))

    if keep_points:
        kdp = kdp_leastsquare_single_window(
            radar, wind_len=kdp_wind_len, min_valid=int(kdp_wind_len / 2.),
            phidp_field=phidp_field, kdp_field=None, vectorize=True)
        sm_refl = smooth_masked(refl, wind_len=smooth_wind_len, min_valid=1,
                                wind_type='mean')
        sm_refl_lin = np.ma.power(10., 0.1 * sm_refl)

        sm_zdr = smooth_masked(zdr, wind_len=smooth_wind_len, min_valid=1,
                               wind_type='mean')
        zdr_list = []
        zh_list = []
        kdp_list = []

    for ray in range(radar.nrays):
        # split ray in consecutive valid range bins
        isprec = np.ma.getmaskarray(phidp_sim[ray, :]) == 0
        ind_prec = np.where(isprec)[0]
        cons_list = np.split(ind_prec, np.where(np.diff(ind_prec) != 1)[0] + 1)

        # check if there is a cell long enough
        found_cell = False
        for ind_prec_cell in cons_list:
            if len(ind_prec_cell) >= min_rcons:
                found_cell = True
                break
        if not found_cell:
            continue

        # check if increase in phidp is within limits and compute reflectivity
        # bias
        dphidp_obs = (
            phidp[ray, ind_prec_cell[-1]] - phidp[ray, ind_prec_cell[0]])
        if dphidp_obs < dphidp_min:
            continue

        for i in range(len(ind_prec_cell)):
            dphidp_obs = (phidp[ray, ind_prec_cell[-1 - i]
                                ] - phidp[ray, ind_prec_cell[0]])
            if dphidp_obs > dphidp_max:
                continue

            dphidp_sim = (phidp_sim[ray, ind_prec_cell[-1 - i]] -
                          phidp_sim[ray, ind_prec_cell[0]])

            if valid_gates_only:
                refl_bias[ray, ind_prec_cell[0]:ind_prec_cell[-1 - i] + 1] = (
                    10. * np.ma.log10(dphidp_sim / dphidp_obs))
            else:
                refl_bias[ray, 0] = 10. * np.ma.log10(dphidp_sim / dphidp_obs)

            if keep_points:
                zdr_list.extend(
                    sm_zdr[ray, ind_prec_cell[0]:ind_prec_cell[-1 - i] + 1])
                kdp_list.extend(kdp['data'][ray,
                                            ind_prec_cell[0]:ind_prec_cell[-1 - i] + 1])
                zh_list.extend(
                    sm_refl_lin[ray, ind_prec_cell[0]:ind_prec_cell[-1 - i] + 1])
            break

    refl_bias_dict = get_metadata('reflectivity_bias')
    refl_bias_dict['data'] = refl_bias

    selfconsistency_dict = None
    if keep_points:
        selfconsistency_dict = {
            'zdr': zdr_list,
            'kdp': kdp_list,
            'zh': zh_list,
        }

    return refl_bias_dict, selfconsistency_dict


def selfconsistency_bias2(
        radar, zdr_kdpzh_dict, min_rhohv=0.92, min_zdr=0.2, filter_rain=True,
        max_phidp=20., smooth_wind_len=5, doc=None, fzl=None, thickness=700.,
        min_rcons=20, parametrization='None', refl_field=None,
        phidp_field=None, zdr_field=None, temp_field=None, iso0_field=None,
        hydro_field=None, rhohv_field=None, kdp_field=None,
        temp_ref='temperature', check_wet_radome=True, wet_radome_refl=25.,
        wet_radome_len_min=4, wet_radome_len_max=8, wet_radome_ngates_min=180,
        keep_points=False, bias_per_gate=False):
    """
    Estimates reflectivity bias at each ray using the self-consistency
    algorithm by Gourley

    Parameters
    ----------
    radar : Radar
        radar object
    zdr_kdpzh_dict : dict
        dictionary containing a look up table relating ZDR with KDP/Zh for
        different elevations
    min_rhohv : float
        minimum RhoHV value to consider the data valid
    min_zdr : float
        minimum ZDR value to consider the data valid
    filter_rain : bool
        If True the hydrometeor classification is going to be used to filter
        out all gates that are not rain
    max_phidp : float
        maximum PhiDP value to consider the data valid
    smooth_wind_len : int
        length of the smoothing window
    doc : float
        Number of gates at the end of each ray to to remove from the
        calculation.
    fzl : float
        Freezing layer, gates above this point are not included in the
        correction.
    thickness : float
        assumed melting layer thickness [m]
    min_rcons : int
        minimum number of consecutive gates to consider a valid segment of
        PhiDP
    parametrization : str
        The type of parametrization for the self-consistency curves. Can be
        'None', 'Gourley', 'Wolfensberger', 'Louf', 'Gorgucci' or 'Vaccarono'.
        'None' will use tables contained in zdr_kdpzh_dict.
    refl_field, kdp_field, zdr_field : str
        Field names within the radar object which represent the reflectivity,
        differential phase and differential reflectivity fields. A value of
        None will use the default field name as defined in the Py-ART
        configuration file.
    temp_field, iso0_field, hydro_field, rhohv_field, phidp_field : str
        Field names within the radar object which represent the temperature,
        the height relative to the iso0, the hydrometeor classification and
        the co-polar correlation fields. A value of None will use the default
        field name as defined in the Py-ART configuration file. They are going
        to be used only if available.
    kdpsim_field, phidpsim_field : str
        Field names which represent the estimated specific differential phase
        and differential phase. A value of None will use the default
        field name as defined in the Py-ART configuration file.
    temp_ref : str
        the field use as reference for temperature. Can be either temperature,
        height_over_iso0 or fixed_fzl
    check_wet_radome : Bool
        if True the average reflectivity of the closest gates to the radar
        is going to be check to find out whether there is rain over the
        radome. If there is rain no bias will be computed
    wet_radome_refl : Float
        Average reflectivity of the gates close to the radar to consider
        the radome as wet
    wet_radome_len_min, wet_radome_len_max : int
        Mim and max gate indices of the disk around the radome used to decide
        whether the radome is wet
    wet_radome_ngates_min : int
        Minimum number of valid gates to consider that the radome is wet
    keep_points : Bool
        If True the ZDR, ZH and KDP of the gates used in the self-
        consistency algorithm are going to be stored for further analysis
    bias_per_gate : Bool
        If True the bias per gate will be computed

    Returns
    -------
    kdp_data_dict : dict
        A dictionary containing valid observed and estimated using self-
        consistency values of KDP
    refl_bias_dict : dict
        If bias_per_gate is set, the bias at each gate field and metadata.
        None otherwise
    selfconsistency_dict : dict
        If keep_poinst set, a dictionary containing the measured valid values
        of ZDR, Zh and KDP. None otherwise

    """
    # parse the field parameters
    if refl_field is None:
        refl_field = get_field_name('reflectivity')
    if zdr_field is None:
        zdr_field = get_field_name('zdr')
    if phidp_field is None:
        # use corrrected_differential_phase or unfolded_differential_phase
        # fields if they are available, if not use differential_phase field
        phidp_field = get_field_name('corrected_differential_phase')
        if phidp_field not in radar.fields:
            phidp_field = get_field_name('unfolded_differential_phase')
        if phidp_field not in radar.fields:
            phidp_field = get_field_name('differential_phase')
    if kdp_field is None:
        kdp_field = get_field_name('specific_differential_phase')
    if rhohv_field is None:
        rhohv_field = get_field_name('cross_correlation_ratio')
    if hydro_field is None:
        hydro_field = get_field_name('radar_echo_classification')

    if temp_ref == 'temperature':
        if temp_field is None:
            temp_field = get_field_name('temperature')
    elif temp_ref == 'height_over_iso0':
        if iso0_field is None:
            iso0_field = get_field_name('height_over_iso0')

    # extract fields from radar, refl, zdr and kdp must exist
    radar.check_field_exists(refl_field)
    refl = radar.fields[refl_field]['data']

    radar.check_field_exists(zdr_field)
    zdr = radar.fields[zdr_field]['data']

    radar.check_field_exists(kdp_field)
    kdp = radar.fields[kdp_field]['data']

    phidp = None
    if max_phidp is not None:
        try:
            radar.check_field_exists(phidp_field)
            phidp = radar.fields[phidp_field]['data']
        except KeyError:
            max_phidp = None
            phidp = None

    rhohv = None
    if min_rhohv is not None:
        try:
            radar.check_field_exists(rhohv_field)
            rhohv = radar.fields[rhohv_field]['data']
        except KeyError:
            min_rhohv = None
            rhohv = None

    hydro = None
    if filter_rain:
        try:
            radar.check_field_exists(hydro_field)
            hydro = radar.fields[hydro_field]['data']
        except KeyError:
            filter_rain = False
            hydro = None

    kdp_sim_aux, _ = _selfconsistency_kdp_phidp(
        radar, refl, zdr, phidp, zdr_kdpzh_dict, max_phidp=max_phidp,
        smooth_wind_len=smooth_wind_len, rhohv=rhohv, min_rhohv=min_rhohv,
        min_zdr=min_zdr, hydro=hydro, filter_rain=filter_rain, doc=doc,
        fzl=fzl, thickness=thickness, parametrization=parametrization,
        temp_field=temp_field, iso0_field=iso0_field, temp_ref=temp_ref)

    if bias_per_gate:
        refl_bias = np.ma.masked_all((radar.nrays, radar.ngates))
    # check if there is rain over the radome
    if check_wet_radome:
        # self_mask = np.ma.getmaskarray(phidp_sim)[:, 0:wet_radome_len]
        refl_radome = refl[:, wet_radome_len_min:wet_radome_len_max + 1]
        # refl_radome[self_mask] = np.ma.masked
        refl_avg = np.ma.mean(refl_radome)
        ngates_wet = (refl_radome.compressed()).size
        if refl_avg > wet_radome_refl and ngates_wet > wet_radome_ngates_min:
            warn('Rain over radome!!!\n Avg reflectivity between ' +
                 str(radar.range['data'][wet_radome_len_min]) + ' and ' +
                 str(radar.range['data'][wet_radome_len_max]) + ' km ' +
                 str(refl_avg) + '. Number of wet gates ' + str(ngates_wet))

            if bias_per_gate:
                refl_bias_dict = get_metadata('reflectivity_bias')
                refl_bias_dict['data'] = refl_bias
                return None, refl_bias_dict, None

            return None, None, None

        warn('Avg reflectivity between ' +
             str(radar.range['data'][wet_radome_len_min]) + ' and ' +
             str(radar.range['data'][wet_radome_len_max]) + ' km ' +
             str(refl_avg) + '. Number of wet gates ' + str(ngates_wet))

    if keep_points:
        sm_refl = smooth_masked(refl, wind_len=smooth_wind_len, min_valid=1,
                                wind_type='mean')
        sm_refl_lin = np.ma.power(10., 0.1 * sm_refl)

        sm_zdr = smooth_masked(zdr, wind_len=smooth_wind_len, min_valid=1,
                               wind_type='mean')
        zdr_list = []
        zh_list = []
        kdp_list = []

    kdp_sim = []
    kdp_obs = []
    for ray in range(radar.nrays):
        # split ray in consecutive valid range bins
        isprec = np.ma.getmaskarray(kdp_sim_aux[ray, :]) == 0
        ind_prec = np.where(isprec)[0]
        cons_list = np.split(ind_prec, np.where(np.diff(ind_prec) != 1)[0] + 1)

        # check if there is a cell long enough
        for ind_prec_cell in cons_list:
            if len(ind_prec_cell) < min_rcons:
                continue

            kdp_sim.extend(
                kdp_sim_aux[ray, ind_prec_cell[0]:ind_prec_cell[-1] + 1])
            kdp_obs.extend(kdp[ray, ind_prec_cell[0]:ind_prec_cell[-1] + 1])

            if keep_points:
                zdr_list.extend(
                    sm_zdr[ray, ind_prec_cell[0]:ind_prec_cell[-1] + 1])
                kdp_list.extend(
                    kdp[ray, ind_prec_cell[0]:ind_prec_cell[-1] + 1])
                zh_list.extend(
                    sm_refl_lin[ray, ind_prec_cell[0]:ind_prec_cell[-1] + 1])

            if bias_per_gate:
                refl_bias[ray, ind_prec_cell[0]:ind_prec_cell[-1] + 1] = (
                    10. * np.ma.log10(
                        kdp_sim_aux[ray, ind_prec_cell[0]:ind_prec_cell[-1] + 1]
                        / kdp[ray, ind_prec_cell[0]:ind_prec_cell[-1] + 1]))

    kdp_data_dict = {
        'kdp_sim': kdp_sim,
        'kdp_obs': kdp_obs
    }
    refl_bias_dict = None
    if bias_per_gate:
        refl_bias_dict = get_metadata('reflectivity_bias')
        refl_bias_dict['data'] = refl_bias

    selfconsistency_dict = None
    if keep_points:
        selfconsistency_dict = {
            'zdr': zdr_list,
            'kdp': kdp_list,
            'zh': zh_list,
        }

    return kdp_data_dict, refl_bias_dict, selfconsistency_dict


def selfconsistency_kdp_phidp(
        radar, zdr_kdpzh_dict, min_rhohv=0.92, filter_rain=True,
        max_phidp=20., smooth_wind_len=5, doc=None, fzl=None, thickness=700.,
        parametrization='None', refl_field=None, phidp_field=None,
        zdr_field=None, temp_field=None, iso0_field=None, hydro_field=None,
        rhohv_field=None, kdpsim_field=None, phidpsim_field=None,
        temp_ref='temperature'):
    """
    Estimates KDP and PhiDP in rain from  Zh and ZDR using a selfconsistency
    relation between ZDR, Zh and KDP. Private method

    Parameters
    ----------
    radar : Radar
        radar object
    zdr_kdpzh_dict : dict
        dictionary containing a look up table relating ZDR with KDP/Zh for
        different elevations
    min_rhohv : float
        minimum RhoHV value to consider the data valid
    filter_rain : bool
        If True the hydrometeor classification is going to be used to filter
        out all gates that are not rain
    max_phidp : float
        maximum PhiDP value to consider the data valid
    smooth_wind_len : int
        length of the smoothing window
    doc : float
        Number of gates at the end of each ray to to remove from the
        calculation.
    fzl : float
        Freezing layer, gates above this point are not included in the
        correction.
    thickness : float
        assumed melting layer thickness [m]
    parametrization : str
        The type of parametrization for the self-consistency curves. Can be
        'None', 'Gourley', 'Wolfensberger', 'Louf', 'Gorgucci' or 'Vaccarono'.
        'None' will use tables contained in zdr_kdpzh_dict.
    refl_field, phidp_field, zdr_field : str
        Field names within the radar object which represent the reflectivity,
        differential phase and differential reflectivity fields. A value of
        None will use the default field name as defined in the Py-ART
        configuration file.
    temp_field, iso0_field, hydro_field, rhohv_field : str
        Field names within the radar object which represent the temperature,
        the height relative to the iso0, the hydrometeor classification and
        the co-polar correlation fields. A value of None will use the default
        field name as defined in the Py-ART configuration file. They are going
        to be used only if available.
    kdpsim_field, phidpsim_field : str
        Field names which represent the estimated specific differential phase
        and differential phase. A value of None will use the default
        field name as defined in the Py-ART configuration file.
    temp_ref : str
        the field use as reference for temperature. Can be either temperature,
        height_over_iso0 or fixed_fzl

    Returns
    -------
    kdp_sim_dict, phidp_sim_dict : dict
        the KDP and PhiDP estimated fields and metadata

    """
    # parse the field parameters
    if refl_field is None:
        refl_field = get_field_name('reflectivity')
    if zdr_field is None:
        zdr_field = get_field_name('zdr')
    if phidp_field is None:
        # use corrrected_differential_phase or unfolded_differential_phase
        # fields if they are available, if not use differential_phase field
        phidp_field = get_field_name('corrected_differential_phase')
        if phidp_field not in radar.fields:
            phidp_field = get_field_name('unfolded_differential_phase')
        if phidp_field not in radar.fields:
            phidp_field = get_field_name('differential_phase')
    if rhohv_field is None:
        rhohv_field = get_field_name('cross_correlation_ratio')
    if hydro_field is None:
        hydro_field = get_field_name('radar_echo_classification')
    if kdpsim_field is None:
        kdpsim_field = get_field_name('specific_differential_phase')
    if phidpsim_field is None:
        phidpsim_field = get_field_name('differential_phase')

    if temp_ref == 'temperature':
        if temp_field is None:
            temp_field = get_field_name('temperature')
    elif temp_ref == 'height_over_iso0':
        if iso0_field is None:
            iso0_field = get_field_name('height_over_iso0')
    elif temp_ref == 'hydroclass':
        if hydro_field is None:
            hydro_field = get_field_name('radar_echo_classification')

    # extract fields from radar, refl, zdr and phidp must exist
    radar.check_field_exists(refl_field)
    refl = radar.fields[refl_field]['data']

    radar.check_field_exists(zdr_field)
    zdr = radar.fields[zdr_field]['data']

    radar.check_field_exists(phidp_field)
    phidp = radar.fields[phidp_field]['data']

    if min_rhohv is not None:
        try:
            radar.check_field_exists(rhohv_field)
            rhohv = radar.fields[rhohv_field]['data']
        except KeyError:
            min_rhohv = None
            rhohv = None

    if filter_rain:
        try:
            radar.check_field_exists(hydro_field)
            hydro = radar.fields[hydro_field]['data']
        except KeyError:
            filter_rain = False
            hydro = None

    kdp_sim, phidp_sim = _selfconsistency_kdp_phidp(
        radar, refl, zdr, phidp, zdr_kdpzh_dict, max_phidp=max_phidp,
        smooth_wind_len=smooth_wind_len, rhohv=rhohv, min_rhohv=min_rhohv,
        hydro=hydro, filter_rain=filter_rain, doc=doc, fzl=fzl,
        thickness=thickness, parametrization=parametrization,
        temp_field=temp_field, iso0_field=iso0_field, temp_ref=temp_ref)

    kdp_sim_dict = get_metadata(kdpsim_field)
    kdp_sim_dict['data'] = kdp_sim

    phidp_sim_dict = get_metadata(phidpsim_field)
    phidp_sim_dict['data'] = phidp_sim

    return kdp_sim_dict, phidp_sim_dict


def get_kdp_selfcons(zdr, refl, ele_vec, zdr_kdpzh_dict,
                     parametrization='None'):
    """
    Estimates KDP and PhiDP in rain from  Zh and ZDR using a selfconsistency
    relation between ZDR, Zh and KDP

    Parameters
    ----------
    zdr, refl : ndarray 2D
        reflectivity and differential reflectivity fields
    ele_vec : ndarray 1D
        vector containing the elevation angles of each ray
    zdr_kdpzh_dict : dict
        dictionary containing a look up table relating ZDR with KDP/Zh for
        different elevations
    parametrization : str
        The type of parametrization for the self-consistency curves. Can be
        'None', 'Gourley', 'Wolfensberger', 'Louf', 'Gorgucci' or 'Vaccarono'.
        'None' will use tables contained in zdr_kdpzh_dict. The parametrized
        curves are obtained from literature except for Wolfensberger that was
        derived from disdrometer data obtained by MeteoSwiss and EPFL. All
        parametrizations are valid for C-band only except that of Gourley.

    Returns
    -------
    kdp_sim : ndarray 2D
        the KDP estimated from zdr and refl

    References
    ----------
    E. Gorgucci, G. Scarchilli, V. Chandrasekar, "Calibration of radars using
    polarimetric techniques", IEEE Transactions on Geoscience and Remote
    Sensing, 1992, 30

    J.J. Gourley, A.J. Illingworth, P. Tabary, "Absolute Calibration of Radar
    Reflectivity Using Redundancy of the Polarization Observations and Implied
    Constraints on Drop Shapes", J. of Atmospheric and Oceanic Technology,
    2009, 26

    V. Louf, A. Protat, R.A. Warren, S.M. Collis, D.B. Wolff, S. Raunyiar,
    C. Jakob, W. A. Petersen, "An Integrated Approach to Weather Radar
    Calibration and Monitoring Using Ground Clutter and Satellite
    Comparisons", J. of Atmospheric and Oceanic Technology, 2019, 36

    M. Vaccarono, R. Bechini, C. V. Chandrasekar, R. Cremonini, C. Cassardo,
    "An integrated approach to monitoring the calibration stability of
    operational dual-polarization radars", Atmos. Meas. Tech., 2016, 9

    """
    # prepare output
    kdpzh = np.ma.masked_all(np.shape(zdr))

    refl_lin = np.ma.power(10., refl / 10.)
    zdr_mask = np.ma.getmaskarray(zdr)

    if parametrization == 'None':
        # process each elevation in the look up table present in the field
        ele_rounded = ele_vec.astype(int)
        for i in range(len(zdr_kdpzh_dict['elev'])):
            # look for gates with valid elevation
            ind_ray = np.where(ele_rounded == zdr_kdpzh_dict['elev'][i])[0]
            if ind_ray.size == 0:
                continue

            # look for valid ZDR
            zdr_valid = zdr[ind_ray, :].compressed()
            if zdr_valid.size == 0:
                continue

            # auxiliary array with the size of valid rays
            kdpzh_aux = np.ma.masked_all(np.size(zdr[ind_ray, :]))

            mask_aux = zdr_mask[ind_ray, :].flatten()

            # sort ZDR
            zdr_sorted = np.sort(zdr_valid)
            ind_zdr_sorted = np.argsort(zdr_valid)

            # get the values of kdp/zh as linear interpolation of the table
            kdpzh_valid = np.interp(
                zdr_sorted, zdr_kdpzh_dict['zdr_kdpzh'][i][0, :],
                zdr_kdpzh_dict['zdr_kdpzh'][i][1, :])

            # reorder according to original order of the flat valid data array
            kdpzh_valid[ind_zdr_sorted] = kdpzh_valid

            # put it in the original matrix
            kdpzh_aux[np.logical_not(mask_aux)] = kdpzh_valid

            kdpzh[ind_ray, :] = np.reshape(
                kdpzh_aux, (len(ind_ray), np.shape(zdr)[1]))

        return refl_lin * kdpzh

    if parametrization == 'Gourley':
        zdr[zdr > 3.5] = np.ma.masked
        if zdr_kdpzh_dict['freq_band'] == 'S':
            kdpzh = 1e-5 * (3.696 - 1.963 * zdr + 0.504 *
                            zdr * zdr - 0.051 * zdr * zdr * zdr)
        elif zdr_kdpzh_dict['freq_band'] == 'C':
            kdpzh = 1e-5 * (6.746 - 2.970 * zdr + 0.711 *
                            zdr * zdr - 0.079 * zdr * zdr * zdr)
        elif zdr_kdpzh_dict['freq_band'] == 'X':
            kdpzh = 1e-5 * (11.74 - 4.020 * zdr - 0.140 *
                            zdr * zdr + 0.130 * zdr * zdr * zdr)
        else:
            raise ValueError(
                'Unable to use self-consistency curves. '
                'Unknown frequency band ' + zdr_kdpzh_dict['freq_band'])

        return refl_lin * kdpzh

    if parametrization == 'Wolfensberger':
        # Curve based on DSD from disdrometer data with temperature obtained
        # from disdrometer measurements. Assumed 1.0 deg elevation.
        if zdr_kdpzh_dict['freq_band'] == 'C':
            zdr[zdr > 3.5] = np.ma.masked
            kdpzh = 3.199e-5 * \
                np.ma.exp(-7.767e-1 * zdr) - 4.436e-6 * zdr + 3.464e-5
            return refl_lin * kdpzh

        raise ValueError(
            'Unable to use self-consistency curves. '
            'Unknown frequency band ' + zdr_kdpzh_dict['freq_band'])

    if parametrization == 'Louf':
        # Drop shape from Brandes et al. (2002)
        if zdr_kdpzh_dict['freq_band'] == 'C':
            zdr[zdr < 0.5] = np.ma.masked
            zdr[zdr > 3.5] = np.ma.masked
            kdpzh = 1e-5 * (6.607 - 4.577 * zdr + 1.577 *
                            zdr * zdr - 0.23 * zdr * zdr * zdr)
            return refl_lin * kdpzh

        raise ValueError(
            'Unable to use self-consistency curves. '
            'Unknown frequency band ' + zdr_kdpzh_dict['freq_band'])

    if parametrization == 'Gorgucci':
        if zdr_kdpzh_dict['freq_band'] == 'C':
            zdr[zdr > 3.5] = np.ma.masked
            zdr_lin = np.ma.power(10., 0.1 * zdr)
            kdpzh095 = 1.82e-4 * np.power(zdr_lin, -1.28)
            return np.ma.power(refl_lin, 0.95) * kdpzh095

        raise ValueError(
            'Unable to use self-consistency curves. '
            'Unknown frequency band ' + zdr_kdpzh_dict['freq_band'])

    if parametrization == 'Vaccarono':
        if zdr_kdpzh_dict['freq_band'] == 'C':
            zdr[zdr > 3.5] = np.ma.masked
            zdr_lin = np.ma.power(10., 0.1 * zdr)
            kdp085zh091 = 1.77e-4 * np.power(zdr_lin, -2.09)
            return np.ma.power(
                np.ma.power(
                    refl_lin,
                    0.91) * kdp085zh091,
                1. / 0.85)

        raise ValueError(
            'Unable to use self-consistency curves. '
            'Unknown frequency band ' + zdr_kdpzh_dict['freq_band'])

    raise ValueError(
        'Unable to use self-consistency curves. '
        'Unknown parametrization ' + parametrization)


def _est_sun_hit_pwr_hs(pwr, sun_hit, attg_sun, max_std, nbins_min, ind_rmin):
    """
    estimates sun hit power, standard deviation, and number and position of
    affected range bins in a ray.
    Uses Hildebrand and Sekhon 1974 to estimate the sun hit power.

    Parameters
    ----------
    pwr : 1D float array
        the power at each range bin in a ray
    sun_hit : 1D float array
        array used to flag sun hit range bins
    attg_sun : float
        attenuation suffered by the sun signal from the top of the atmosphere
        to the radar position
    max_std : float
        maximum standard deviation to consider the sun hit valid
    nbins_min : int
        minimum number of range gates with valid signal in the ray to consider
        the ray affected by a noise-like signal
    ind_rmin : int
        minimum range from which we can look for noise

    Returns
    -------
    sunpwr_dBm : float
        the estimated sun power
    sunpwr_std : float
        the standard deviation of the estimation in dB
    sunpwr_npoints : int
        the number of range gates affected by the sun hit
    nvalid : int
        the number of valid range gates
    sun_hit : 1D array
        array with flagged range bins

    """
    nvalid = len(pwr[ind_rmin:].compressed())

    if nvalid < nbins_min:
        return get_fillvalue(), get_fillvalue(), 0, nvalid, sun_hit

    pwr_toa_mw = np.ma.power(10., 0.1 * (pwr[ind_rmin:] + attg_sun))
    pwr_valid = pwr_toa_mw.compressed()
    sunpwr, _, _, sunpwr_npoints = estimate_noise_hs74(pwr_valid)

    ind_sun_hits = np.argsort(pwr_valid)[0:sunpwr_npoints]
    pwr_valid = np.sort(pwr_valid)[0:sunpwr_npoints]
    sunpwr_std = np.ma.std(10. * np.ma.log10(pwr_valid))

    if sunpwr_std > max_std:
        warn('Sun hit power not valid. Standard deviation ' + str(sunpwr_std) +
             ' larger than maximum expected')
        return get_fillvalue(), get_fillvalue(), 0, nvalid, sun_hit

    sunpwr_dBm = 10. * np.ma.log10(sunpwr)

    # set gates with valid sun hits to one
    is_valid = np.logical_not(np.ma.getmaskarray(sun_hit[ind_rmin:]))
    ind_valid = is_valid.nonzero()[0]
    sun_hit[ind_rmin + ind_valid[ind_sun_hits]] = 1

    return sunpwr_dBm, sunpwr_std, sunpwr_npoints, nvalid, sun_hit


def _est_sun_hit_pwr_psr(pwr, attg_sun):
    """
    estimates sun hit power

    Parameters
    ----------
    pwr : 1D float array
        the power at each range bin in a ray
    attg_sun : float
        attenuation suffered by the sun signal from the top of the atmosphere
        to the radar position

    Returns
    -------
    sunpwr_dBm : float
        the estimated sun power

    """

    pwr_toa_mw = np.ma.power(10., 0.1 * (pwr + attg_sun))
    sunpwr_dBm = 10. * np.ma.log10(pwr_toa_mw)

    return sunpwr_dBm


def _est_sun_hit_pwr_ivic(pwr, sun_hit, attg_sun, pct, flat_reg_wlen,
                          flat_reg_var_max, snr_thr, npulses, ngates_min,
                          iterations):
    """
    estimates sun hit power, standard deviation, and number and position of
    affected range bins in a ray.
    Uses Ivic et al 2013 to estimate the sun hit power.

    Parameters
    ----------
    pwr : 1D float array
        the power at each range bin in a ray
    sun_hit : 1D float array
        array used to flag sun hit range bins
    attg_sun : float
        attenuation suffered by the sun signal from the top of the atmosphere
        to the radar position
    pct : float
        Point Clutter Threshold
    flat_reg_wlen : int
        Minimum number of gates that should contain a valid region. Default
        gives a size of 8 km with 83.3 m resolution
    flat_reg_var_max : float
        Maximum local variance of powers in decibels to consider the region
        as flat.
    snr_thr : float
        Threshold applied in steps 3 and 6
    npulses : int
        Number of pulses used to compute the power of the array
    ngates_min: int
        minimum number of gates with noise to consider the retrieval valid
    iterations: int
        number of iterations in step 7

    Returns
    -------
    sunpwr_dBm : float
        the estimated sun power
    sunpwr_std : float
        the standard deviation of the estimation in dB
    sunpwr_npoints : int
        the number of range gates affected by the sun hit
    nvalid : int
        the number of valid range gates
    sun_hit : 1D array
        array with flagged range bins

    """
    nvalid = len(pwr.compressed())

    pwr_toa_w = 1e-3 * np.ma.power(10., 0.1 * (pwr + attg_sun))
    sunpwr, _, _, inds_noise = estimate_noise_ivic13(
        pwr_toa_w, pct=pct, delay=2, flat_reg_wlen=flat_reg_wlen,
        flat_reg_var_max=flat_reg_var_max, snr_thr=snr_thr,
        npulses=npulses, ngates_min=ngates_min, iterations=iterations,
        get_noise_pos=True)

    if sunpwr is None:
        warn('No sun hit found')
        return get_fillvalue(), get_fillvalue(), 0, nvalid, sun_hit

    sunpwr_dBm = 10. * np.ma.log10(sunpwr) + 30.
    pwr_toa_dBm = pwr + attg_sun
    sunpwr_std = np.ma.std(pwr_toa_dBm[inds_noise])

    # set gates with valid sun hits to one
    sun_hit[inds_noise] = 1

    return sunpwr_dBm, sunpwr_std, inds_noise.size, nvalid, sun_hit


def _est_sun_hit_zdr(zdr, sun_hit_zdr, sun_hit_h, sun_hit_v, max_std,
                     nbins_min, ind_rmin):
    """
    estimates sun hit ZDR, standard deviation, and number and position of
    affected range bins in a ray

    Parameters
    ----------
    zdr : 1D float array
        the ZDR at each range bin in a ray
    sun_hit_zdr : 1D float array
        array used to flag sun hit range bins
    sun_hit_h, sun_hit_v : 1D float array
        The position of sun hit range bins in each channel
    max_std : float
        maximum standard deviation
    nbins_min : int
        minimum number of range gates with valid signal in the ray to consider
        the ray affected by a noise-like signal
    ind_rmin : int
        minimum range from which we can look for noise

    Returns
    -------
    sunzdr : float
        the estimated sun power
    sunzdr_std : float
        the standard deviation of the estimation in dB
    sunzdr_npoints : int
        the number of range gates affected by the sun hit
    sun_hit_zdr : 1D array
        array with flagged range bins

    """
    nvalid = len(zdr[ind_rmin:].compressed())
    if nvalid < nbins_min:
        # warn('Sun hit ZDR not valid. Not enough gates with signal in ray')
        return get_fillvalue(), get_fillvalue(), 0, nvalid, sun_hit_zdr

    if sun_hit_h is None and sun_hit_v is None:
        # warn('Sun hit ZDR not valid. No sun power was detected')
        return get_fillvalue(), get_fillvalue(), 0, nvalid, sun_hit_zdr

    is_valid = np.logical_not(np.ma.getmaskarray(zdr))

    if sun_hit_h is not None:
        is_valid = np.logical_and(sun_hit_h.filled(fill_value=0), is_valid)
    if sun_hit_v is not None:
        is_valid = np.logical_and(sun_hit_v.filled(fill_value=0), is_valid)

    sunzdr_npoints = np.count_nonzero(is_valid)

    if sunzdr_npoints < 2:
        # warn('Sun hit ZDR not valid. ' +
        #      'Not enough gates with valid signal in ray')
        return get_fillvalue(), get_fillvalue(), 0, nvalid, sun_hit_zdr

    sunzdr = np.ma.mean(zdr[is_valid])
    sunzdr_std = np.ma.std(zdr[is_valid])

    if sunzdr_std > max_std:
        warn('Sun hit ZDR not valid. Standard deviation ' + str(sunzdr_std) +
             ' larger than maximum expected')
        return get_fillvalue(), get_fillvalue(), 0, nvalid, sun_hit_zdr

    sun_hit_zdr[is_valid] = 1

    return sunzdr, sunzdr_std, sunzdr_npoints, nvalid, sun_hit_zdr


def _selfconsistency_kdp_phidp(
        radar, refl, zdr, phidp, zdr_kdpzh_dict, max_phidp=20.,
        smooth_wind_len=5, rhohv=None, min_rhohv=None, min_zdr=0.,
        hydro=None, filter_rain=True, doc=None, fzl=None, thickness=700.,
        parametrization='None', temp_field=None, iso0_field=None,
        temp_ref='temperature'):
    """
    Estimates KDP and PhiDP in rain from  Zh and ZDR using a selfconsistency
    relation between ZDR, Zh and KDP. Private method

    Parameters
    ----------
    radar : Radar
        radar object
    refl, zdr, phidp : ndarray 2D
        reflectivity field, differential reflectivity field and differential
        phase field. They must exist
    zdr_kdpzh_dict : dict
        dictionary containing a look up table relating ZDR with KDP/Zh for
        different elevations
    max_phidp : float
        maximum PhiDP value to consider the data valid
    smooth_wind_len : int
        length of the smoothing window for Zh and ZDR data
    rhohv : ndarray 2D
        copolar correlation field used for masking data. Optional
    min_rhohv : float
        minimum RhoHV value to consider the data valid
    min_zdr : float
        minimum ZDR value to consider the data valid
    hydro : ndarray 2D
        hydrometer classification field used for masking data. Optional
    filter_rain : Bool
        If true gates not classified as rain are going to be removed from the
        data
    doc : float
        Number of gates at the end of each ray to to remove from the
        calculation.
    fzl : float
        Freezing layer, gates above this point are not included in the
        correction.
    thickness : float
        Assumed thickness of the melting layer [m]
    parametrization : str
        The type of parametrization for the self-consistency curves. Can be
        'None', 'Gourley', 'Wolfensberger', 'Louf', 'Gorgucci' or 'Vaccarono'.
        'None' will use tables contained in zdr_kdpzh_dict.
    temp_field, iso0_field, hydro_field : str
        Field name within the radar object which represent the temperature,
        the height relative to the iso0 and the hydrometeor classification
        fields. A value of None will use the default field name as defined in
        the Py-ART configuration file. It is going to be used only if
        available.
    temp_ref : str
        the field use as reference for temperature. Can be either temperature,
        height_over_iso0 or fixed_fzl

    Returns
    -------
    kdp_sim, phidp_sim : ndarray 2D
        the KDP and PhiDP estimated fields

    """
    if (radar.instrument_parameters is not None and
            'radar_beam_width_h' in radar.instrument_parameters):
        beamwidth = (
            radar.instrument_parameters['radar_beam_width_h']['data'][0])
    else:
        warn('Unknown radar antenna beamwidth.')
        beamwidth = None

    # smooth reflectivity and ZDR
    if smooth_wind_len > 0:
        sm_refl = smooth_masked(refl, wind_len=smooth_wind_len, min_valid=1,
                                wind_type='mean')
        sm_zdr = smooth_masked(zdr, wind_len=smooth_wind_len, min_valid=1,
                               wind_type='mean')
    else:
        sm_refl = refl
        sm_zdr = zdr

    # Remove data at melting layer or above
    mask = np.ma.getmaskarray(refl)
    mask_fzl, _ = get_mask_fzl(
        radar, fzl=fzl, doc=doc, min_temp=0., max_h_iso0=0.,
        thickness=thickness, beamwidth=beamwidth, temp_field=temp_field,
        iso0_field=iso0_field, temp_ref=temp_ref)
    mask = np.logical_or(mask, mask_fzl)

    # Remove data outside of valid range of ZDR
    mask_zdr = np.logical_or(sm_zdr < min_zdr, np.ma.getmaskarray(sm_zdr))
    if parametrization == 'None':
        # Remove data with ZDR out of table values
        ele_rounded = radar.elevation['data'].astype(int)
        mask_zdr_max = np.ones((radar.nrays, radar.ngates))
        for i in range(len(zdr_kdpzh_dict['elev'])):
            ind_ray = np.where(ele_rounded == zdr_kdpzh_dict['elev'][i])[0]
            if ind_ray.size == 0:
                continue
            mask_zdr_max[ind_ray, :] = (
                sm_zdr[ind_ray, :] > zdr_kdpzh_dict['zdr_kdpzh'][i][0, -1])

        mask_zdr = np.logical_or(mask_zdr, mask_zdr_max)
    mask = np.logical_or(mask, mask_zdr)

    if max_phidp is not None:
        mask_phidp = np.logical_or(
            phidp > max_phidp, np.ma.getmaskarray(phidp))
        mask = np.logical_or(mask, mask_phidp)

    if min_rhohv is not None:
        mask_rhohv = np.logical_or(rhohv < min_rhohv,
                                   np.ma.getmaskarray(rhohv))
        mask = np.logical_or(mask, mask_rhohv)

    # Remove gates that are not rain
    if filter_rain:
        # keep only data classified as light rain (4) or rain (6)
        mask_rain = np.logical_not(np.logical_or(hydro == 4, hydro == 6))
        mask = np.logical_or(mask, mask_rain)

    corr_refl = np.ma.masked_where(mask, sm_refl)
    corr_zdr = np.ma.masked_where(mask, sm_zdr)

    kdp_sim = get_kdp_selfcons(
        corr_zdr, corr_refl, radar.elevation['data'], zdr_kdpzh_dict,
        parametrization=parametrization)
    dr = (radar.range['data'][1] - radar.range['data'][0]) / 1000.0
    phidp_sim = np.ma.cumsum(2 * dr * kdp_sim, axis=1)

    return kdp_sim, phidp_sim
