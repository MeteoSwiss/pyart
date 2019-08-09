"""
pyart.retrieve.simple_moment_calculations
=========================================

Simple moment calculations.

.. autosummary::
    :toctree: generated/

    compute_ccor
    calculate_snr_from_reflectivity
    compute_noisedBZ
    compute_vol_refl
    compute_signal_power
    compute_rcs_from_pr
    compute_rcs
    compute_snr
    compute_l
    compute_cdr
    compute_bird_density
    calculate_velocity_texture
    atmospheric_gas_att
    get_coeff_attg
    _coeff_attg_table

"""
from warnings import warn
from copy import deepcopy

import numpy as np

from scipy import ndimage
from ..config import get_metadata, get_field_name
from ..core.transforms import antenna_to_cartesian
from .echo_class import get_freq_band
from ..util import angular_texture_2d


def compute_ccor(radar, filt_field=None, unfilt_field=None, ccor_field=None):
    """
    Computes the clutter correction ratio (CCOR), i.e. the ratio between the
    signal without Doppler filtering and the signal with Doppler filtering

    Parameters
    ----------
    radar : Radar
        Radar object
    filt_field, unfilt_field : str
        Name of Doppler filtered and unfiltered fields
    ccor_field : str
        Name of the CCOR field

    Returns
    -------
    ccor_dict : field dictionary
        Field dictionary containing the CCOR

    """
    # parse the field parameters
    if filt_field is None:
        filt_field = get_field_name('reflectivity')
    if unfilt_field is None:
        unfilt_field = get_field_name('unfiltered_reflectivity')
    if ccor_field is None:
        ccor_field = get_field_name('clutter_correction_ratio_hh')

    ccor_dict = get_metadata(ccor_field)
    ccor_dict['data'] = (
        radar.fields[unfilt_field]['data']-radar.fields[filt_field]['data'])

    return ccor_dict


def calculate_snr_from_reflectivity(
        radar, refl_field=None, snr_field=None, toa=25000.):
    """
    Calculate the signal to noise ratio, in dB, from the reflectivity field.

    Parameters
    ----------
    radar : Radar
        Radar object from which to retrieve reflectivity field.
    refl_field : str, optional
        Name of field in radar which contains the reflectivity.
        None will use the default field name in the Py-ART configuration file.
    snr_field : str, optional
        Name to use for snr metadata.
        None will use the default field name in the Py-ART configuration file.
    toa : float, optional
        Height above which to take noise floor measurements, in meters.

    Returns
    -------
    snr : field dictionary
        Field dictionary containing the signal to noise ratio.

    """
    if refl_field is None:
        refl_field = get_field_name('reflectivity')
    if snr_field is None:
        snr_field = get_field_name('signal_to_noise_ratio')

    range_grid = np.meshgrid(radar.range['data'],
                             np.ma.ones(radar.time['data'].shape))[0] + 1.0

    # remove range scale.. This is basically the radar constant scaled dBm
    pseudo_power = (radar.fields[refl_field]['data'] -
                    20.0*np.log10(range_grid / 1000.0))

    # Noise floor estimate
    # 25km.. should be no scatterers, not even planes, this high
    # we could get undone by AP though.. also sun
    rg, azg = np.meshgrid(radar.range['data'], radar.azimuth['data'])
    rg, eleg = np.meshgrid(radar.range['data'], radar.elevation['data'])
    _, _, z = antenna_to_cartesian(rg / 1000.0, azg, eleg)  # XXX: need to fix

    points_above = np.where(z > toa)
    noise_floor_estimate = pseudo_power[points_above].mean()

    snr_dict = get_metadata(snr_field)
    snr_dict['data'] = pseudo_power - noise_floor_estimate
    return snr_dict


def compute_noisedBZ(nrays, noisedBZ_val, rng, ref_dist,
                     noise_field=None):
    """
    Computes noise in dBZ from reference noise value.

    Parameters
    ----------
    nrays: int
        number of rays in the reflectivity field

    noisedBZ_val: float
        Estimated noise value in dBZ at reference distance

    rng: np array of floats
        range vector in m

    ref_dist: float
        reference distance in Km

    noise_field: str
        name of the noise field to use

    Returns
    -------
    noisedBZ : dict
        the noise field

    """
    # parse the field parameters
    if noise_field is None:
        noise_field = get_field_name('noisedBZ_hh')

    noisedBZ_vec = noisedBZ_val+20.*np.ma.log10(1e-3*rng/ref_dist)

    noisedBZ = get_metadata(noise_field)
    noisedBZ['data'] = np.tile(noisedBZ_vec, (nrays, 1))

    return noisedBZ


def compute_vol_refl(radar, kw=0.93, freq=None, refl_field=None,
                     vol_refl_field=None):
    """
    Computes the volumetric reflectivity from the effective reflectivity
    factor

    Parameters
    ----------
    radar : Radar
        radar object
    kw : float
        water constant
    freq : None or float
        radar frequency
    refl_field : str
        name of the reflectivity used for the calculations
    vol_refl_field : str
        name of the volumetric reflectivity

    Returns
    -------
    vol_refl_dict : dict
        volumetric reflectivity and metadata in 10log10(cm^2 km^-3)

    """
    # parse the field parameters
    if refl_field is None:
        refl_field = get_field_name('reflectivity')
    if vol_refl_field is None:
        vol_refl_field = get_field_name('volumetric_reflectivity')

    # extract fields from radar
    radar.check_field_exists(refl_field)
    refl = radar.fields[refl_field]['data']

    # determine the parameters
    if freq is None:
        # get frequency from radar metadata
        if 'frequency' in radar.instrument_parameters:
            freq = radar.instrument_parameters['frequency']['data'][0]
        else:
            warn('Unable to compute volumetric reflectivity. ' +
                 'Unknown radar frequency')
            return None

    wavelen = 3e8/freq*1e2  # [cm]
    vol_refl = (
        1e3*np.power(np.pi, 5.)*kw*np.ma.power(10., 0.1*refl) /
        np.power(wavelen, 4.))

    vol_refl_dict = get_metadata(vol_refl_field)
    vol_refl_dict['data'] = 10.*np.log10(vol_refl)

    return vol_refl_dict


def compute_signal_power(radar, lmf=None, attg=None, radconst=None,
                         lrx=0., lradome=0., refl_field=None, pwr_field=None):
    """
    Computes received signal power OUTSIDE THE RADOME in dBm from a
    reflectivity field.

    Parameters
    ----------
    radar : Radar
        radar object
    lmf : float
        matched filter losses
    attg : float
        1-way gas attenuation
    radconst : float
        radar constant
    lrx : float
        receiver losses from the antenna feed to the reference point
        (positive value) [dB]
    lradome : float
        1-way losses due to the radome (positive value) [dB]
    refl_field : str
        name of the reflectivity used for the calculations
    pwr_field : str
        name of the signal power field

    Returns
    -------
    s_pwr_dict : dict
        power field and metadata

    """
    # parse the field parameters
    if refl_field is None:
        refl_field = get_field_name('reflectivity')
    if pwr_field is None:
        pwr_field = get_field_name('signal_power_hh')
        if refl_field.endswith('_vv'):
            pwr_field = get_field_name('signal_power_vv')

    # extract fields from radar
    radar.check_field_exists(refl_field)
    refl = radar.fields[refl_field]['data']

    # determine the parameters
    if lmf is None:
        warn('Unknown matched filter losses. Assumed 1 dB')
        lmf = 1.
    if attg is None:
        # assign coefficients according to radar frequency
        if 'frequency' in radar.instrument_parameters:
            attg = get_coeff_attg(
                radar.instrument_parameters['frequency']['data'][0])
        else:
            attg = 0.0
            warn('Unknown 1-way gas attenuation. It will be set to 0')
    if radconst is None:
        # determine it from meta-data
        if refl_field.endswith('_vv'):
            if 'calibration_constant_vv' in radar.radar_calibration:
                radconst = (
                    radar.radar_calibration[
                        'calibration_constant_vv']['data'][0])
        elif 'calibration_constant_hh' in radar.radar_calibration:
            radconst = (
                radar.radar_calibration['calibration_constant_hh']['data'][0])

        if radconst is None:
            raise ValueError(
                'Radar constant unknown. ' +
                'Unable to determine the signal power')

    rng = radar.range['data']/1000.
    gas_att = 2.*attg*rng
    rangedB = 20.*np.ma.log10(rng)

    s_pwr = refl-rangedB-gas_att-radconst-lmf+lrx+lradome

    s_pwr_dict = get_metadata(pwr_field)
    s_pwr_dict['data'] = s_pwr

    return s_pwr_dict


def compute_rcs_from_pr(radar, lmf=None, attg=None, radconst=None,
                        tx_pwr=None, antenna_gain=None, lrx=0., ltx=0.,
                        lradome=0., freq=None, refl_field=None,
                        rcs_field=None, neglect_gas_att=False):
    """
    Computes the radar cross-section (assuming a point target) from radar
    reflectivity by first computing the received power and then the RCS from
    it.

    Parameters
    ----------
    radar : Radar
        radar object
    lmf : float
        matched filter losses
    attg : float
        1-way gas attenuation
    radconst : float
        radar constant
    tx_pwr : float
        radar transmitted power [dBm]
    antenna_gain : float
        antenna gain [dB]
    lrx : float
        receiver losses from the antenna feed to the reference point
        (positive value) [dB]
    lradome : float
        1-way losses due to the radome (positive value) [dB]
    freq : float
        radar frequency [Hz]. If none it will be obtained from the radar
        metadata
    refl_field : str
        name of the reflectivity used for the calculations
    rcs_field : str
        name of the RCS field
    neglect_gas_att : bool
        Whether to neglect or not gas attenuation in the estimation of the
        RCS

    Returns
    -------
    rcs_dict : dict
        RCS field and metadata

    """
    # parse the field parameters
    if refl_field is None:
        refl_field = get_field_name('reflectivity')
    if rcs_field is None:
        rcs_field = get_field_name('radar_cross_section_hh')
        if refl_field.endswith('_vv'):
            rcs_field = get_field_name('radar_cross_section_vv')

    # determine the parameters
    rng = radar.range['data']  # [m]
    if antenna_gain is None:
        warn('Unable to compute RCS. Unknown antenna gain')
        return None
    g_lin = np.power(10., 0.1*antenna_gain)
    if tx_pwr is None:
        # determine it from meta-data
        if refl_field.endswith('_vv'):
            if 'transmit_power_v' in radar.radar_calibration:
                tx_pwr = (
                    radar.radar_calibration['transmit_power_v']['data'][0])
        elif 'transmit_power_h' in radar.radar_calibration:
            tx_pwr = (
                radar.radar_calibration['transmit_power_h']['data'][0])

        if tx_pwr is None:
            raise ValueError(
                'Transmitted power unknown. ' +
                'Unable to compute RCS')
    if freq is None:
        # get frequency from radar metadata
        if 'frequency' in radar.instrument_parameters:
            freq = radar.instrument_parameters['frequency']['data'][0]
        else:
            raise ValueError(
                'Radar frequency unknown. ' +
                'Unable to compute RCS')

    # get received power OUTSIDE THE RADOME for reflectivity field [W]
    s_pwr = np.ma.power(10., 0.1*(compute_signal_power(
        radar, lmf=lmf, attg=attg, radconst=radconst, lrx=lrx,
        lradome=lradome, refl_field=refl_field, pwr_field=None)['data']-30.))

    wavelen = 3e8/freq  # [m]

    if neglect_gas_att:
        gas_att = 1.
    else:
        ELEV, RNG = np.meshgrid(
            radar.elevation['data'], rng/1000., indexing='ij')
        gas_att = np.power(10., 0.1*atmospheric_gas_att(freq, ELEV, RNG))

    tx_pwr_out = np.power(10., 0.1*(tx_pwr-ltx-lradome-30.))  # [W]

    rcs = 10*np.ma.log10(
        s_pwr*np.power(4*np.pi, 3.)*np.power(rng, 4)*np.power(gas_att, 2.) /
        (tx_pwr_out*np.power(g_lin, 2.)*np.power(wavelen, 2.)))

    rcs_dict = get_metadata(rcs_field)
    rcs_dict['data'] = rcs

    return rcs_dict


def compute_rcs(radar, kw2=0.93, pulse_width=None, beamwidth=None, freq=None,
                refl_field=None, rcs_field=None):
    """
    Computes the radar cross-section (assuming a point target) from radar
    reflectivity.

    Parameters
    ----------
    radar : Radar
        radar object
    kw2 : float
        water constant
    pulse_width : float
        pulse width [s]
    beamwidth : float
        beamwidth [degree]
    freq : float
        radar frequency [Hz]. If none it will be obtained from the radar
        metadata
    refl_field : str
        name of the reflectivity used for the calculations
    rcs_field : str
        name of the RCS field

    Returns
    -------
    rcs_dict : dict
        RCS field and metadata

    """
    # parse the field parameters
    if refl_field is None:
        refl_field = get_field_name('reflectivity')
    if rcs_field is None:
        rcs_field = get_field_name('radar_cross_section_hh')
        if refl_field.endswith('_vv'):
            rcs_field = get_field_name('radar_cross_section_vv')

    # extract fields from radar
    radar.check_field_exists(refl_field)
    refl_lin = np.ma.power(10., 0.1*radar.fields[refl_field]['data'])

    # determine the parameters
    rng = deepcopy(radar.range['data'])
    rng_mat = np.broadcast_to(
        rng.reshape(1, radar.ngates), (radar.nrays, radar.ngates))

    if freq is None:
        # get frequency from radar metadata
        if 'frequency' in radar.instrument_parameters:
            freq = radar.instrument_parameters['frequency']['data'][0]
        else:
            raise ValueError(
                'Radar frequency unknown. ' +
                'Unable to compute RCS')
    wavelen = 3e8/freq  # [m]
    if pulse_width is None:
        # get pulse width from radar metadata
        if 'pulse_width' in radar.instrument_parameters:
            pulse_width = (
                radar.instrument_parameters['pulse_width']['data'][0])
        else:
            raise ValueError(
                'Pulse width unknown. ' +
                'Unable to compute RCS')
    if beamwidth is None:
        # determine it from meta-data
        if refl_field.endswith('_vv'):
            if 'radar_beam_width_v' in radar.instrument_parameters:
                beamwidth = (
                    radar.instrument_parameters['radar_beam_width_v']['data'][
                        0])
        elif 'radar_beam_width_h' in radar.instrument_parameters:
            beamwidth = (
                radar.instrument_parameters['radar_beam_width_h']['data'][0])

        if beamwidth is None:
            raise ValueError(
                'Antenna beamwidth unknown. ' +
                'Unable to compute RCS')
    beamwidth_rad = beamwidth*np.pi/180.

    rcs = 10*np.ma.log10(
        np.power(np.pi, 6.)*kw2*3e8*pulse_width*np.power(beamwidth_rad, 2.) *
        np.power(rng_mat, 2.)*refl_lin*1e-18 /
        (16.*np.log(2.)*np.power(wavelen, 4.)))

    rcs_dict = get_metadata(rcs_field)
    rcs_dict['data'] = rcs

    return rcs_dict


def compute_snr(radar, refl_field=None, noise_field=None, snr_field=None):
    """
    Computes SNR from a reflectivity field and the noise in dBZ.

    Parameters
    ----------
    radar : Radar
        radar object

    refl_field, noise_field : str
        name of the reflectivity and noise field used for the calculations

    snr_field : str
        name of the SNR field

    Returns
    -------
    snr : dict
        the SNR field

    """
    # parse the field parameters
    if refl_field is None:
        refl_field = get_field_name('reflectivity')
    if noise_field is None:
        noise_field = get_field_name('noisedBZ_hh')
    if snr_field is None:
        snr_field = get_field_name('signal_to_noise_ratio')

    # extract fields from radar
    radar.check_field_exists(refl_field)
    radar.check_field_exists(noise_field)

    refl = radar.fields[refl_field]['data']
    noisedBZ = radar.fields[noise_field]['data']

    snr_data = refl-noisedBZ

    snr = get_metadata(snr_field)
    snr['data'] = snr_data

    return snr


def compute_l(radar, rhohv_field=None, l_field=None):
    """
    Computes Rhohv in logarithmic scale according to L=-log10(1-RhoHV)

    Parameters
    ----------
    radar : Radar
        radar object

    rhohv_field : str
        name of the RhoHV field used for the calculation

    l_field : str
        name of the L field

    Returns
    -------
    l : dict
        L field

    """
    # parse the field parameters
    if rhohv_field is None:
        rhohv_field = get_field_name('cross_correlation_ratio')
    if l_field is None:
        l_field = get_field_name('logarithmic_cross_correlation_ratio')

    # extract rhohv field from radar
    radar.check_field_exists(rhohv_field)
    rhohv = radar.fields[rhohv_field]['data']

    rhohv[rhohv >= 1.] = 0.9999
    l_data = -np.ma.log10(1.-rhohv)

    l = get_metadata(l_field)
    l['data'] = l_data

    return l


def compute_cdr(radar, rhohv_field=None, zdr_field=None, cdr_field=None):
    """
    Computes the Circular Depolarization Ratio

    Parameters
    ----------
    radar : Radar
        radar object

    rhohv_field, zdr_field : str
        name of the input RhoHV and ZDR fields

    cdr_field : str
        name of the CDR field

    Returns
    -------
    cdr : dict
        CDR field

    """
    # parse the field parameters
    if rhohv_field is None:
        rhohv_field = get_field_name('cross_correlation_ratio')
    if zdr_field is None:
        zdr_field = get_field_name('differential_reflectivity')
    if cdr_field is None:
        cdr_field = get_field_name('circular_depolarization_ratio')

    # extract fields from radar
    radar.check_field_exists(rhohv_field)
    radar.check_field_exists(zdr_field)

    rhohv = radar.fields[rhohv_field]['data']
    zdrdB = radar.fields[zdr_field]['data']

    zdr = np.ma.power(10., 0.1*zdrdB)

    cdr_data = (
        10.*np.ma.log10(
            (1.+1./zdr-2.*rhohv*np.ma.sqrt(1./zdr)) /
            (1.+1./zdr+2.*rhohv*np.ma.sqrt(1./zdr))))

    cdr = get_metadata(cdr_field)
    cdr['data'] = cdr_data

    return cdr


def compute_bird_density(radar, sigma_bird=11, vol_refl_field=None,
                         bird_density_field=None):
    """
    Computes the bird density from the volumetric reflectivity

    Parameters
    ----------
    radar : Radar
        radar object
    sigma_bird : float
        Estimated bird radar cross-section
    vol_refl_field : str
        name of the volumetric reflectivity used for the calculations
    bird_density_field : str
        name of the bird density field

    Returns
    -------
    bird_density_dict : dict
        bird density data and metadata [birds/km^3]

    """
    # parse the field parameters
    if vol_refl_field is None:
        vol_refl_field = get_field_name('volumetric_reflectivity')
    if bird_density_field is None:
        bird_density_field = get_field_name('bird_density')

    # extract fields from radar
    radar.check_field_exists(vol_refl_field)
    vol_refl = radar.fields[vol_refl_field]['data']

    bird_density = np.ma.power(10., 0.1*vol_refl)/sigma_bird

    bird_density_dict = get_metadata(bird_density_field)
    bird_density_dict['data'] = bird_density

    return bird_density_dict





def calculate_velocity_texture(radar, vel_field=None, wind_size=4, nyq=None,
                               check_nyq_uniform=True):
    """
    Derive the texture of the velocity field

    Parameters
    ----------
    radar: Radar
        Radar object from which velocity texture field will be made.
    vel_field_name : str
        Name of the velocity field. A value of None will force Py-ART to
        automatically determine the name of the velocity field.
    wind_size : int
        The size of the window to calculate texture from. The window is
        defined to be a square of size wind_size by wind_size.
    nyq : float
        The nyquist velocity of the radar. A value of None will force Py-ART
        to try and determine this automatically.
    check_nyquist_uniform : bool, optional
        True to check if the Nyquist velocities are uniform for all rays
        within a sweep, False will skip this check. This parameter is ignored
        when the nyq parameter is not None.

    Returns
    -------
    vel_dict: dict
        A dictionary containing the field entries for the radial velocity
        texture.

    """

    # Parse names of velocity field
    if vel_field is None:
        vel_field = get_field_name('velocity')

    # Allocate memory for texture field
    vel_texture = np.zeros(radar.fields[vel_field]['data'].shape)

    # If an array of nyquist velocities is derived, use different
    # nyquist velocites for each sweep in texture calculation according to
    # the nyquist velocity in each sweep.

    if nyq is None:
        # Find nyquist velocity if not specified
        nyq = [radar.get_nyquist_vel(i, check_nyq_uniform) for i in
               range(radar.nsweeps)]
        for i in range(0, radar.nsweeps):
            start_ray, end_ray = radar.get_start_end(i)
            inds = range(start_ray, end_ray)
            vel_sweep = radar.fields[vel_field]['data'][inds]
            vel_texture[inds] = angular_texture_2d(
                vel_sweep, wind_size, nyq[i])
    else:
        vel_texture = angular_texture_2d(
            radar.fields[vel_field]['data'], wind_size, nyq)
    vel_texture_field = get_metadata('velocity')
    vel_texture_field['long_name'] = 'Doppler velocity texture'
    vel_texture_field['standard_name'] = ('texture_of_radial_velocity' +
                                          '_of_scatters_away_from_instrument')
    vel_texture_field['data'] = ndimage.filters.median_filter(vel_texture,
                                                              size=(wind_size,
                                                                    wind_size))
    return vel_texture_field


def atmospheric_gas_att(freq, elev, rng):
    """
    Computes the one-way atmospheric gas attenuation [dB] according to the
    empirical formula in Doviak and Zrnic (1993) pp 44.
    This formula is valid for elev < 10 deg and rng < 200 km so values above
    these will be saturated to 10 deg and 200 km respectively

    Parameters
    ----------
    freq : float
        radar frequency [Hz]
    elev : float or array of floats
        elevation angle [deg]
    rng : float or array of floats. If array must have the same size as elev
        range [km]

    Returns
    -------
    latm : float or array of floats
        1-way gas attenuation [dB]

    """
    elev_aux = deepcopy(elev)
    rng_aux = deepcopy(rng)
    if np.isscalar(elev_aux):
        if elev_aux > 10.:
            elev_aux = 10.
    else:
        elev_aux[elev_aux > 10.] = 10.

    if np.isscalar(rng_aux):
        if rng_aux > 200.:
            rng_aux = 200.
    else:
        rng_aux[rng_aux > 200.] = 200.

    if not np.isscalar(elev_aux) and not np.isscalar(rng_aux):
        elev_size = np.size(elev_aux)
        rng_size = np.size(rng_aux)
        if elev_size != rng_size:
            raise ValueError(
                'Unable to compute gas attenuation field. ' +
                'radar elevation field size is '+str(elev_size) +
                ' and radar range field size is '+str(rng_size))

    # S-band atmospheric attenuation
    latm = (
        0.5*(0.4+3.45*np.exp(-elev_aux/1.8)) *
        (1-np.exp(-rng_aux/(27.8+154.*np.exp(-elev_aux/2.2)))))
    if freq > 12e9:
        # X-band
        latm *= 1.5
    elif 2e9 <= freq <= 12e9:
        # C-band
        latm *= 1.2

    return latm


def get_coeff_attg(freq):
    """
    get the 1-way gas attenuation for a particular frequency

    Parameters
    ----------
    freq : float
        radar frequency [Hz]

    Returns
    -------
    attg : float
        1-way gas attenuation

    """
    coeff_attg_dict = _coeff_attg_table()

    freq_band = get_freq_band(freq)
    if (freq_band is not None) and (freq_band in coeff_attg_dict):
        return coeff_attg_dict[freq_band]

    if freq < 2e9:
        freq_band_aux = 'S'
    elif freq > 12e9:
        freq_band_aux = 'X'

    warn('Radar frequency out of range. ' +
         'Coefficients only applied to S, C or X band. ' +
         freq_band + ' band coefficients will be used')

    return coeff_attg_dict[freq_band_aux]


def _coeff_attg_table():
    """
    defines the 1-way gas attenuation for each frequency band.

    Returns
    -------
    coeff_attg_dict : dict
        A dictionary with the coefficients at each band

    """
    coeff_attg_dict = dict()

    # S band
    coeff_attg_dict.update({'S': 0.0080})

    # C band
    coeff_attg_dict.update({'C': 0.0095})

    # X band
    coeff_attg_dict.update({'X': 0.0120})

    return coeff_attg_dict
