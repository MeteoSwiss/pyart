"""
pyart.retrieve.simple_moment_calculations
=========================================

Simple moment calculations.

.. autosummary::
    :toctree: generated/

    calculate_snr_from_reflectivity
    compute_noisedBZ
    compute_vol_refl
    compute_signal_power
    compute_snr
    compute_l
    compute_cdr
    compute_bird_density
    calculate_velocity_texture
    get_coeff_attg
    _coeff_attg_table

"""
from warnings import warn

import numpy as np

from scipy import ndimage
from ..config import get_metadata, get_field_name, get_fillvalue
from ..core.transforms import antenna_to_cartesian
from .echo_class import get_freq_band
from ..util import angular_texture_2d


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
    x, y, z = antenna_to_cartesian(rg / 1000.0, azg, eleg)  # XXX: need to fix

    points_above = np.where(z > toa)
    noise_floor_estimate = pseudo_power[points_above].mean()

    snr_dict = get_metadata(snr_field)
    snr_dict['data'] = pseudo_power - noise_floor_estimate
    return snr_dict


def compute_noisedBZ(nrays, noisedBZ_val, range, ref_dist,
                     noise_field=None):
    """
    Computes noise in dBZ from reference noise value.

    Parameters
    ----------
    nrays: int
        number of rays in the reflectivity field

    noisedBZ_val: float
        Estimated noise value in dBZ at reference distance

    range: np array of floats
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

    noisedBZ_vec = noisedBZ_val+20.*np.ma.log10(1e-3*range/ref_dist)

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
        volumetric reflectivity and metadata in cm^2 km^-3

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

    wavelen = 3e8/freq*1e2
    vol_refl = (
        1e3*np.power(np.pi, 5.)*kw*np.ma.power(10., 0.1*refl) /
        np.power(wavelen, 4.))

    vol_refl_dict = get_metadata(vol_refl_field)
    vol_refl_dict['data'] = vol_refl

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

    if(nyq is None):
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
