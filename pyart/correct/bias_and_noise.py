"""
pyart.correct.bias_and_noise
===================

Corrects polarimetric variables for noise

.. autosummary::
    :toctree: generated/

    correct_noise_rhohv
    correct_bias
    get_sun_hits
    sun_retrieval
    est_rhohv_rain
    est_zdr_rain
    selfconsistency_bias
    selfconsistency_kdp_phidp
    _selfconsistency_kdp_phidp
    get_kdp_selfcons

"""

import numpy as np
from copy import deepcopy
from warnings import warn

from netCDF4 import num2date

from ..config import get_metadata, get_field_name, get_fillvalue
from ..util import estimate_noise_hs74
from .attenuation import get_mask_fzl
from .phase_proc import smooth_masked
from .sunlib import sun_position_mfr, gas_att_sun, gauss_fit, retrieval_result
from ..retrieve import get_coeff_attg


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
        radar object

    urhohv_field : str
        name of the RhoHV uncorrected for noise field

    snr_field, zdr_field, nh_field, nv_field: str
        names of the SNR, ZDR, horizontal channel noise in dBZ and vertical
        channel noise in dBZ used to correct RhoHV

    rhohv_field: str
        name of the rhohv field to output

    Returns
    -------
    rhohv : dict
        noise corrected RhoHV field

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

    snr_h = np.ma.power(10., 0.1*snrdB_h)
    zdr = np.ma.power(10., 0.1*zdrdB)
    alpha = np.ma.power(10., 0.1*(nh-nv))

    rhohv_data = urhohv*np.ma.sqrt((1.+1./snr_h)*(1.+zdr/(alpha*snr_h)))
    rhohv_data[rhohv_data > 1.] = 1.

    rhohv = get_metadata(rhohv_field)
    rhohv['data'] = rhohv_data

    return rhohv


def correct_bias(radar, bias=0., field_name=None):
    """
    Corrects a radar data bias. If field name is none the correction is
    applied to horizontal reflectivity by default

    Parameters
    ----------
    radar : Radar
        radar object

    bias : float
        the bias magnitude

    field_name: str
        names of the field to be corrected

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
        corr_field_name = 'corrected_'+field_name

    corr_field = get_metadata(corr_field_name)
    corr_field['data'] = corr_field_data

    return corr_field


def get_sun_hits(
        radar, delev_max=2., dazim_max=2., elmin=1., ind_rmin=100,
        percent_bins=10., attg=None, max_std=1., pwrh_field=None,
        pwrv_field=None, zdr_field=None):
    """
    get data from suspected sun hits

    Parameters
    ----------
    radar : Radar
        radar object
    delev_max, dazim_max : float
        maximum difference in elevation and azimuth between sun position and
        antenna pointing
    elmin : float
        minimum radar elevation angle
    ind_rmin : int
        minimum range from which we can look for noise
    percent_bins : float
        percentage of bins with valid data to consider a ray as potentially
        sun hit
    attg : float
        gas attenuation coefficient (1-way)
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
        if 'frequency' in radar.instrument_parameters:
            attg = get_coeff_attg(
                radar.instrument_parameters['frequency']['data'][0])
        else:
            attg = 0.
            warn('Unknown 1-way gas attenuation. It will be set to 0')

    nbins_min = int(len(radar.range['data'][ind_rmin:-1])*percent_bins/100.)
    nrange = len(radar.range['data'])

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
        sun_hit_h = np.ma.zeros(np.shape(pwrh))
        sun_hit_h[mask_pwrh] = np.ma.masked
    except KeyError:
        pwrh = None
        sun_hit_h = None

    try:
        radar.check_field_exists(pwrv_field)
        pwrv = radar.fields[pwrv_field]['data']
        mask_pwrv = np.ma.getmaskarray(pwrv)
        sun_hit_v = np.ma.zeros(np.shape(pwrv))
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
        sun_hit_zdr = np.ma.zeros(np.shape(zdr))
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
        if radar.elevation['data'][ray] >= elmin:
            elev_sun, azim_sun = sun_position_mfr(
                time[ray], radar.latitude['data'][0],
                radar.longitude['data'][0], refraction=True)

            if elev_sun >= 0:
                delev = np.ma.abs(radar.elevation['data'][ray]-elev_sun)
                dazim = np.ma.abs(
                    (radar.azimuth['data'][ray]-azim_sun) *
                    np.ma.cos(elev_sun*np.pi/180.))
                if dazim > 360.:
                    dazim -= 360.

                if (delev <= delev_max) and (dazim <= dazim_max):
                    # gas atmospheric attenuation from radar to TOA
                    attg_sun = gas_att_sun(elev_sun, attg)

                    sunpwrh_dBm = get_fillvalue()
                    sunpwrh_std = get_fillvalue()
                    sunpwrh_npoints = 0
                    nvalidh = 0
                    sun_hit_h_ray = None
                    if pwrh is not None:
                        (sunpwrh_dBm, sunpwrh_std, sunpwrh_npoints, nvalidh,
                         sun_hit_h_ray) = _est_sun_hit_pwr(
                            pwrh[ray, :], sun_hit_h[ray, :], attg_sun,
                            max_std, nbins_min, ind_rmin)
                        sun_hit_h[ray, :] = sun_hit_h_ray

                    sunpwrv_dBm = get_fillvalue()
                    sunpwrv_std = get_fillvalue()
                    sunpwrv_npoints = 0
                    nvalidv = 0
                    sun_hit_v_ray = None
                    if pwrv is not None:
                        (sunpwrv_dBm, sunpwrv_std, sunpwrv_npoints, nvalidv,
                         sun_hit_v_ray) = _est_sun_hit_pwr(
                            pwrv[ray, :], sun_hit_v[ray, :], attg_sun,
                            max_std, nbins_min, ind_rmin)
                        sun_hit_v[ray, :] = sun_hit_v_ray

                    sunzdr = get_fillvalue()
                    sunzdr_std = get_fillvalue()
                    sunzdr_npoints = 0
                    nvalidzdr = 0
                    if zdr is not None:
                        (sunzdr, sunzdr_std, sunzdr_npoints, nvalidzdr,
                         sun_hit_zdr_ray) = _est_sun_hit_zdr(
                            zdr[ray, :], sun_hit_zdr[ray, :], sun_hit_h_ray,
                            sun_hit_v_ray, max_std, nbins_min, ind_rmin)
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
        sun_hit_h_dict['data'] = sun_hit_h

        new_radar.add_field(pwrh_field, pwrh_dict)
        new_radar.add_field('sun_hit_h', sun_hit_h_dict)

    if pwrv is not None:
        pwrv_dict = get_metadata(pwrv_field)
        pwrv_dict['data'] = pwrv

        sun_hit_v_dict = get_metadata('sun_hit_v')
        sun_hit_v_dict['data'] = sun_hit_v

        new_radar.add_field(pwrv_field, pwrv_dict)
        new_radar.add_field('sun_hit_v', sun_hit_v_dict)

    if zdr is not None:
        zdr_dict = get_metadata(zdr_field)
        zdr_dict['data'] = zdr

        sun_hit_zdr_dict = get_metadata('sun_hit_zdr')
        sun_hit_zdr_dict['data'] = sun_hit_zdr

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


def _est_sun_hit_pwr(pwr, sun_hit, attg_sun, max_std, nbins_min, ind_rmin):
    """
    estimates sun hit power, standard deviation, and number and position of
    affected range bins in a ray

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
    sun_hit : 1D array
        array with flagged range bins

    """
    nvalid = len(pwr[ind_rmin:-1].compressed())

    if nvalid < nbins_min:
        return get_fillvalue(), get_fillvalue(), 0, nvalid, sun_hit

    pwr_toa_mw = np.ma.power(10., 0.1*(pwr[ind_rmin:-1]+attg_sun))
    pwr_valid = pwr_toa_mw.compressed()
    sunpwr, sunpwr_max, sunpwr_var, sunpwr_npoints = estimate_noise_hs74(
        pwr_valid)

    ind_sun_hits = np.argsort(pwr_valid)[0:sunpwr_npoints-1]
    sunpwr_std = np.ma.std(10.*np.ma.log10(pwr_toa_mw[ind_sun_hits]))

    if sunpwr_std > max_std:
        return get_fillvalue(), get_fillvalue(), 0, nvalid, sun_hit

    sunpwr_dBm = 10.*np.ma.log10(sunpwr)
    is_valid = np.logical_not(np.ma.getmaskarray(sun_hit[ind_rmin:-1]))
    ind_valid = is_valid.nonzero()
    sun_hit_valid = sun_hit[ind_rmin:-1].compressed()
    sun_hit_valid[ind_sun_hits] = 1
    sun_hit[ind_rmin+ind_valid] = sun_hit_valid

    return sunpwr_dBm, sunpwr_std, sunpwr_npoints, nvalid, sun_hit


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
        The position of sun hit range bins in eanch channel
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
    nvalid = len(zdr[ind_rmin:-1].compressed())
    if nvalid < nbins_min:
        return get_fillvalue(), get_fillvalue(), 0, nvalid, sun_hit_zdr

    if sun_hit_h is None and sun_hit_v is None:
        return get_fillvalue(), get_fillvalue(), 0, nvalid, sun_hit_zdr

    is_valid = np.logical_not(np.ma.getmaskarray(zdr))

    if sun_hit_h is not None:
        is_valid = np.logical_and(sun_hit_h.filled(fill_value=0), is_valid)
    if sun_hit_v is not None:
        is_valid = np.logical_and(sun_hit_v.filled(fill_value=0), is_valid)

    sunzdr_npoints = np.count_nonzero(is_valid)

    if sunzdr_npoints < 2:
        return get_fillvalue(), get_fillvalue(), 0, nvalid, sun_hit_zdr

    sunzdr = np.ma.mean(zdr[is_valid])
    sunzdr_std = np.ma.std(zdr[is_valid])

    if sunzdr_std > max_std:
        return get_fillvalue(), get_fillvalue(), 0, nvalid, sun_hit_zdr

    sun_hit_zdr[is_valid] = 1

    return sunzdr, sunzdr_std, sunzdr_npoints, nvalid, sun_hit_zdr


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
        par[0:npar-1] = par_aux
        coeff = 40.*np.ma.log10(2.)
        if is_zdr:
            par[3] = (
                coeff*(1./np.ma.power(az_width_cross, 2.) -
                       1./np.ma.power(az_width_co, 2.)))
            par[4] = (
                coeff*(1./np.ma.power(el_width_cross, 2.) -
                       1./np.ma.power(el_width_co, 2.)))
        else:
            par[3] = -np.ma.power(az_width_co, 2.)/coeff
            par[4] = -np.ma.power(el_width_co, 2.)/coeff
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
        refl_field=None):
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
    temp_field, rhohv_field, refl_field : str
        Field names within the radar object which represent the temperature,
        co-polar correlation and reflectivity fields. A value of None will use
        the default field name as defined in the Py-ART configuration file.

    Returns
    -------
    rhohv_rain_dict : dict
        The estimated RhoHV in rain for each sweep and metadata

    """
    if 'radar_beam_width_h' in radar.instrument_parameters:
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
    if temp_field is None:
        temp_field = get_field_name('temperature')

    # extract fields from radar
    radar.check_field_exists(rhohv_field)
    rhohv = deepcopy(radar.fields[rhohv_field]['data'])

    radar.check_field_exists(refl_field)
    refl = radar.fields[refl_field]['data']

    # determine the valid data (i.e. data below the melting layer)
    mask = np.ma.getmaskarray(rhohv)

    mask_fzl, end_gate_arr = get_mask_fzl(
        radar, fzl=fzl, doc=doc, min_temp=0., thickness=thickness,
        beamwidth=beamwidth, temp_field=temp_field)
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


def est_zdr_rain(
        radar, ind_rmin=10, ind_rmax=500, zmin=20., zmax=22., rhohvmin=0.97,
        phidpmax=10., elmax=30., thickness=700., doc=None, fzl=None,
        zdr_field=None, rhohv_field=None, phidp_field=None, temp_field=None,
        refl_field=None):
    """
    Estimates the average ZDR in moderate rain

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
    zdr_field, rhohv_field, refl_field, phidp_field, temp_field : str
        Field names within the radar object which represent the differential
        reflectivity, co-polar correlation, reflectivity, differential phase
        and temperature fields. A value of None will use the default field
        name as defined in the Py-ART configuration file.

    Returns
    -------
    zdr_rain_dict : dict
        The estimated RhoHV in rain for each sweep and metadata

    """
    if 'radar_beam_width_h' in radar.instrument_parameters:
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
    if temp_field is None:
        temp_field = get_field_name('temperature')

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
    mask = np.ma.getmaskarray(rhohv)

    mask_fzl, end_gate_arr = get_mask_fzl(
        radar, fzl=fzl, doc=doc, min_temp=0., thickness=thickness,
        beamwidth=beamwidth, temp_field=temp_field)
    mask = np.logical_or(mask, mask_fzl)

    mask_refl = np.logical_or(np.ma.getmaskarray(refl),
                              np.logical_or(refl < zmin, refl > zmax))
    mask = np.logical_or(mask, mask_refl)

    mask_rhohv = np.logical_or(np.ma.getmaskarray(rhohv), rhohv < rhohvmin)
    mask = np.logical_or(mask, mask_rhohv)

    mask_phidp = np.logical_or(np.ma.getmaskarray(phidp), phidp > phidpmax)
    mask = np.logical_or(mask, mask_phidp)

    zdr_rain = np.ma.masked_where(mask, zdr)
    zdr_rain[:, 0:ind_rmin] = np.ma.masked
    zdr_rain[:, ind_rmax:-1] = np.ma.masked

    ind_el = np.where(radar.elevation['data'] > elmax)
    zdr_rain[ind_el, :] = np.ma.masked

    zdr_rain_dict = get_metadata('differential_reflectivity_in_rain')
    zdr_rain_dict['data'] = zdr_rain

    return zdr_rain_dict


def selfconsistency_bias(
        radar, zdr_kdpzh_table, min_rhohv=0.92, max_phidp=20.,
        smooth_wind_len=5, doc=None, fzl=None, min_rcons=20, dphidp_min=2,
        dphidp_max=16, refl_field=None, phidp_field=None, zdr_field=None,
        temp_field=None, rhohv_field=None):
    """
    Estimates reflectivity bias at each ray using the self-consistency
    algorithm by Gourley

    Parameters
    ----------
    radar : Radar
        radar object
    zdr_kdpzh_table : ndarray 2D
        look up table relating ZDR with KDP/Zh
    min_rhohv : float
        minimum RhoHV value to consider the data valid
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
    min_rcons : int
        minimum number of consecutive gates to consider a valid segment of
        PhiDP
    dphidp_min : float
        minimum differential phase shift in a segment
    dphidp_max : float
        maximum differential phase shift in a segment
    refl_field, phidp_field, zdr_field : str
        Field names within the radar object which represent the reflectivity,
        differential phase and differential reflectivity fields. A value of
        None will use the default field name as defined in the Py-ART
        configuration file.
    temp_field, rhohv_field : str
        Field names within the radar object which represent the temperature,
        and co-polar correlation fields. A value of None will use the default
        field name as defined in the Py-ART configuration file. They are going
        to be used only if available.
    kdpsim_field, phidpsim_field : str
        Field names which represent the estimated specific differential phase
        and differential phase. A value of None will use the default
        field name as defined in the Py-ART configuration file.

    Returns
    -------
    refl_bias_dict : dict
        the bias at each ray field and metadata

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
    if temp_field is None:
        temp_field = get_field_name('temperature')
    if rhohv_field is None:
        rhohv_field = get_field_name('cross_correlation_ratio')

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

    kdp_sim, phidp_sim = _selfconsistency_kdp_phidp(
        radar, refl, zdr, phidp, zdr_kdpzh_table, max_phidp=max_phidp,
        smooth_wind_len=smooth_wind_len, rhohv=rhohv, min_rhohv=min_rhohv,
        doc=doc, fzl=fzl, temp_field=temp_field)

    refl_bias = np.ma.zeros((radar.nrays, 1))
    refl_bias[:] = np.ma.masked
    for ray in range(radar.nrays):
        # split ray in consecutive valid range bins
        isprec = np.ma.getmaskarray(phidp_sim[ray, :]) == 0
        ind_prec = np.where(isprec)[0]
        cons_list = np.split(ind_prec, np.where(np.diff(ind_prec) != 1)[0]+1)

        # check if there is a cell long enough
        found_cell = False
        for ind_prec_cell in cons_list:
            if len(ind_prec_cell) >= min_rcons:
                found_cell = True
                break
        # check if increase in phidp is within limits and compute reflectivity
        # bias
        if found_cell:
            dphidp_obs = (phidp[ray, ind_prec_cell[-1]] -
                          phidp[ray, ind_prec_cell[0]])
            if dphidp_obs >= dphidp_min:
                for i in range(len(ind_prec_cell)):
                    dphidp_obs = (phidp[ray, ind_prec_cell[-1-i]] -
                                  phidp[ray, ind_prec_cell[0]])
                    if dphidp_obs <= dphidp_max:
                        dphidp_sim = (phidp_sim[ray, ind_prec_cell[-1]] -
                                      phidp_sim[ray, ind_prec_cell[0]])
                        refl_bias[ray] = 10.*np.ma.log10(dphidp_sim/dphidp_obs)
                        break

    refl_bias_dict = get_metadata('reflectivity_bias')
    refl_bias_dict['data'] = refl_bias
    return refl_bias_dict


def selfconsistency_kdp_phidp(
        radar, zdr_kdpzh_table, min_rhohv=0.92, max_phidp=20.,
        smooth_wind_len=5, doc=None, fzl=None, refl_field=None,
        phidp_field=None, zdr_field=None, temp_field=None, rhohv_field=None,
        kdpsim_field=None, phidpsim_field=None):
    """
    Estimates KDP and PhiDP in rain from  Zh and ZDR using a selfconsistency
    relation between ZDR, Zh and KDP. Private method

    Parameters
    ----------
    radar : Radar
        radar object
    zdr_kdpzh_table : ndarray 2D
        look up table relating ZDR with KDP/Zh
    min_rhohv : float
        minimum RhoHV value to consider the data valid
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
    refl_field, phidp_field, zdr_field : str
        Field names within the radar object which represent the reflectivity,
        differential phase and differential reflectivity fields. A value of
        None will use the default field name as defined in the Py-ART
        configuration file.
    temp_field, rhohv_field : str
        Field names within the radar object which represent the temperature,
        and co-polar correlation fields. A value of None will use the default
        field name as defined in the Py-ART configuration file. They are going
        to be used only if available.
    kdpsim_field, phidpsim_field : str
        Field names which represent the estimated specific differential phase
        and differential phase. A value of None will use the default
        field name as defined in the Py-ART configuration file.

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
    if temp_field is None:
        temp_field = get_field_name('temperature')
    if rhohv_field is None:
        rhohv_field = get_field_name('cross_correlation_ratio')
    if kdpsim_field is None:
        kdpsim_field = get_field_name('specific_differential_phase')
    if phidpsim_field is None:
        phidpsim_field = get_field_name('differential_phase')

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

    kdp_sim, phidp_sim = _selfconsistency_kdp_phidp(
        radar, refl, zdr, phidp, zdr_kdpzh_table, max_phidp=max_phidp,
        smooth_wind_len=smooth_wind_len, rhohv=rhohv, min_rhohv=min_rhohv,
        doc=doc, fzl=fzl, temp_field=temp_field)

    kdp_sim_dict = get_metadata(kdpsim_field)
    kdp_sim_dict['data'] = kdp_sim

    phidp_sim_dict = get_metadata(phidpsim_field)
    phidp_sim_dict['data'] = phidp_sim

    return kdp_sim_dict, phidp_sim_dict


def _selfconsistency_kdp_phidp(
        radar, refl, zdr, phidp, zdr_kdpzh_table, max_phidp=20.,
        smooth_wind_len=5, rhohv=None, min_rhohv=None, doc=None, fzl=None,
        temp_field=None):
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
    zdr_kdpzh_table : ndarray 2D
        look up table relating ZDR with KDP/Zh
    rhohv : ndarray 2D
        copolar correlation field used for masking data. Optional
    max_phidp : float
        maximum PhiDP value to consider the data valid
    smooth_wind_len : int
        length of the smoothing window for Zh and ZDR data
    min_rhohv : float
        minimum RhoHV value to consider the data valid
    doc : float
        Number of gates at the end of each ray to to remove from the
        calculation.
    fzl : float
        Freezing layer, gates above this point are not included in the
        correction.
    temp_field : str
        Field name within the radar object which represent the temperature
        field. A value of None will use the default field name as defined in
        the Py-ART configuration file. It is going to be used only if
        available.

    Returns
    -------
    kdp_sim, phidp_sim : ndarray 2D
        the KDP and PhiDP estimated fields

    """
    # smooth reflectivity and ZDR
    if smooth_wind_len > 0:
        sm_refl = smooth_masked(refl, wind_len=smooth_wind_len, min_valid=1,
                                wind_type='mean')
        sm_zdr = smooth_masked(zdr, wind_len=smooth_wind_len, min_valid=1,
                               wind_type='mean')
    else:
        sm_refl = refl
        sm_zdr = zdr

    # determine the valid data (i.e. data below the melting layer)
    mask = np.ma.getmaskarray(refl)

    mask_fzl, end_gate_arr = get_mask_fzl(
        radar, fzl=fzl, doc=doc, min_temp=0., thickness=500.,
        temp_field=temp_field)
    mask = np.logical_or(mask, mask_fzl)

    mask_zdr = np.logical_or(sm_zdr < 0., np.ma.getmaskarray(sm_zdr))
    mask_zdr = np.logical_or(mask_zdr, sm_zdr > zdr_kdpzh_table[0, -1])
    mask = np.logical_or(mask, mask_zdr)

    mask_phidp = np.logical_or(phidp > max_phidp, np.ma.getmaskarray(phidp))
    mask = np.logical_or(mask, mask_phidp)

    if min_rhohv is not None:
        mask_rhohv = np.logical_or(rhohv < min_rhohv,
                                   np.ma.getmaskarray(rhohv))
        mask = np.logical_or(mask, mask_rhohv)

    corr_refl = np.ma.masked_where(mask, sm_refl)
    corr_zdr = np.ma.masked_where(mask, sm_zdr)

    kdp_sim = get_kdp_selfcons(corr_zdr, corr_refl, zdr_kdpzh_table)
    dr = (radar.range['data'][1] - radar.range['data'][0]) / 1000.0
    phidp_sim = np.ma.cumsum(2*dr*kdp_sim, axis=1)

    return kdp_sim, phidp_sim


def get_kdp_selfcons(zdr, refl, zdr_kdpzh_table):
    """
    Estimates KDP and PhiDP in rain from  Zh and ZDR using a selfconsistency
    relation between ZDR, Zh and KDP

    Parameters
    ----------
    zdr, refl : ndarray 2D
        reflectivity and differential reflectivity fields
    zdr_kdpzh_table : ndarray 2D
        look up table relating ZDR with KDP/Zh

    Returns
    -------
    kdp_sim : ndarray 2D
        the KDP estimated from zdr and refl

    """
    # prepare output
    kdpzh = np.ma.zeros(np.shape(zdr))
    kdpzh[:] = np.ma.masked

    refl_lin = np.ma.power(10., refl/10.)
    zdr_mask = np.ma.getmaskarray(zdr)

    zdr_valid = zdr.compressed()
    if np.size(zdr_valid) < 1:
        warn('No valid data for selfconsistency retrieval')
        return kdpzh

    # sort ZDR
    zdr_sorted = np.sort(zdr_valid)
    ind_zdr_sorted = np.argsort(zdr_valid)

    # get the values of kdp/zh as linear interpolation of the table
    kdpzh_valid = np.interp(
        zdr_sorted, zdr_kdpzh_table[0, :], zdr_kdpzh_table[1, :])

    # reorder according to original order of the flat valid data array
    kdpzh_valid[ind_zdr_sorted] = kdpzh_valid

    kdpzh[~zdr_mask] = kdpzh_valid

    return refl_lin * kdpzh
