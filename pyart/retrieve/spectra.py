"""
pyart.retrieve.spectra
======================

Retrievals from spectral data.

.. autosummary::
    :toctree: generated/

    compute_spectral_power
    compute_spectral_reflectivity
    compute_spectral_differential_reflectivity
    compute_spectral_differential_phase
    compute_spectral_rhohv
    compute_spectral_phase
    compute_pol_variables
    compute_reflectivity
    compute_differential_reflectivity
    compute_differential_phase
    compute_rhohv
    compute_Doppler_velocity
    compute_Doppler_width
    _compute_power

"""
from copy import deepcopy

import numpy as np

from ..config import get_metadata, get_field_name
from ..util import radar_from_spectra


def compute_spectral_power(spectra, units='dBADU', subtract_noise=False,
                           signal_field=None, noise_field=None):
    """
    Computes the spectral power from the complex spectra in ADU

    Parameters
    ----------
    spectra : Radar spectra object
        Object containing the required fields
    units : str
        The units of the returned signal. Can be 'ADU', 'dBADU' or 'dBm'
    subtract_noise : Bool
            If True noise will be subtracted from the signal
    signal_field, noise_field : str, optional
        Name of the fields in radar which contains the signal and noise.
        None will use the default field name in the Py-ART configuration file.

    Returns
    -------
    pwr_dict : field dictionary
        Field dictionary containing the spectral power

    """
    if signal_field is None:
        signal_field = get_field_name('complex_spectra_hh_ADU')
    if noise_field is None:
        noise_field = get_field_name('noiseADU_hh')

    pol = 'hh'
    if 'vv' in signal_field:
        pol = 'vv'

    pwr = _compute_power(
        spectra.fields[signal_field]['data'],
        noise=spectra.fields[noise_field]['data'],
        subtract_noise=subtract_noise)

    if units in ('dBADU', 'dBm'):
        pwr = 10.*np.ma.log10(pwr)

        if units == 'dBm':
            dBADU2dBm = (
                spectra.radar_calibration['dBADU_to_dBm_hh']['data'][0])
            if pol == 'vv':
                dBADU2dBm = (
                    spectra.radar_calibration['dBADU_to_dBm_vv']['data'][0])
            # should it be divided by the number of pulses?
            pwr += dBADU2dBm

    power_field = 'spectral_power_'+pol+'_'+units

    pwr_dict = get_metadata(power_field)
    pwr_dict['data'] = pwr

    return pwr_dict


def compute_spectral_reflectivity(spectra, subtract_noise=False,
                                  signal_field=None, noise_field=None):
    """
    Computes the spectral reflectivity from the complex spectra in ADU

    Parameters
    ----------
    spectra : Radar spectra object
        Object containing the required fields
    subtract_noise : Bool
            If True noise will be subtracted from the signal
    signal_field, noise_field : str, optional
        Name of the fields in radar which contains the signal and noise.
        None will use the default field name in the Py-ART configuration file.

    Returns
    -------
    sdBZ_dict : field dictionary
        Field dictionary containing the spectral reflectivity

    """
    if signal_field is None:
        signal_field = get_field_name('complex_spectra_hh_ADU')
    if noise_field is None:
        noise_field = get_field_name('noiseADU_hh')

    pol = 'hh'
    dBADU2dBm = spectra.radar_calibration['dBADU_to_dBm_hh']['data'][0]
    radconst = spectra.radar_calibration['calibration_constant_hh']['data'][0]
    pathatt = spectra.radar_calibration['path_attenuation']['data'][0]
    mfloss = spectra.radar_calibration['matched_filter_loss']['data'][0]
    if 'vv' in signal_field:
        pol = 'vv'
        dBADU2dBm = spectra.radar_calibration['dBADU_to_dBm_vv']['data'][0]
        radconst = (
            spectra.radar_calibration['calibration_constant_vv']['data'][0])

    rangeKm = np.broadcast_to(
        np.atleast_3d(spectra.range['data']/1000.),
        (spectra.nrays, spectra.ngates, spectra.npulses_max))

    pwr = _compute_power(
        spectra.fields[signal_field]['data'],
        noise=spectra.fields[noise_field]['data'],
        subtract_noise=subtract_noise)

    sdBZ = (
        10.*np.ma.log10(pwr)+dBADU2dBm+radconst+mfloss+pathatt*rangeKm +
        20.*np.log10(rangeKm))

    sdBZ_field = 'spectral_reflectivity_'+pol

    sdBZ_dict = get_metadata(sdBZ_field)
    sdBZ_dict['data'] = sdBZ

    return sdBZ_dict


def compute_spectral_differential_reflectivity(spectra, subtract_noise=False,
                                               signal_h_field=None,
                                               signal_v_field=None,
                                               noise_h_field=None,
                                               noise_v_field=None):
    """
    Computes the spectral differential reflectivity from the complex spectras
    in ADU

    Parameters
    ----------
    spectra : Radar spectra object
        Object containing the required fields
    subtract_noise : Bool
            If True noise will be subtracted from the signals
    signal_h_field, signal_v_field, noise_h_field, noise_v_field : str
        Name of the fields in radar which contains the signal and noise.
        None will use the default field name in the Py-ART configuration file.

    Returns
    -------
    sZDR_dict : field dictionary
        Field dictionary containing the spectral differential reflectivity

    """
    if signal_h_field is None:
        signal_h_field = get_field_name('complex_spectra_hh_ADU')
    if signal_v_field is None:
        signal_v_field = get_field_name('complex_spectra_vv_ADU')
    if noise_h_field is None:
        noise_h_field = get_field_name('noiseADU_hh')
    if noise_v_field is None:
        noise_v_field = get_field_name('noiseADU_hh')

    dBADU2dBm_h = spectra.radar_calibration['dBADU_to_dBm_hh']['data'][0]
    dBADU2dBm_v = spectra.radar_calibration['dBADU_to_dBm_vv']['data'][0]
    radconst_h = (
        spectra.radar_calibration['calibration_constant_hh']['data'][0])
    radconst_v = (
        spectra.radar_calibration['calibration_constant_hh']['data'][0])

    pwr_h = _compute_power(
        spectra.fields[signal_h_field]['data'],
        noise=spectra.fields[noise_h_field]['data'],
        subtract_noise=subtract_noise)

    pwr_v = _compute_power(
        spectra.fields[signal_v_field]['data'],
        noise=spectra.fields[noise_v_field]['data'],
        subtract_noise=subtract_noise)

    sZDR = (
        (10.*np.ma.log10(pwr_h)+dBADU2dBm_h+radconst_h) -
        (10.*np.ma.log10(pwr_v)+dBADU2dBm_v+radconst_v))

    sZDR_dict = get_metadata('spectral_differential_reflectivity')
    sZDR_dict['data'] = sZDR

    return sZDR_dict


def compute_spectral_differential_phase(spectra, signal_h_field=None,
                                        signal_v_field=None,):
    """
    Computes the spectral differential reflectivity from the complex spectras
    in ADU

    Parameters
    ----------
    spectra : Radar spectra object
        Object containing the required fields
    signal_h_field, signal_v_field : str
        Name of the fields in radar which contains the signal.
        None will use the default field name in the Py-ART configuration file.

    Returns
    -------
    sPhiDP_dict : field dictionary
        Field dictionary containing the spectral differential phase

    """
    if signal_h_field is None:
        signal_h_field = get_field_name('complex_spectra_hh_ADU')
    if signal_v_field is None:
        signal_v_field = get_field_name('complex_spectra_vv_ADU')

    phase_h = np.ma.angle(spectra.fields[signal_h_field]['data'], deg=True)
    phase_v = np.ma.angle(spectra.fields[signal_v_field]['data'], deg=True)

    sPhiDP = phase_h-phase_v

    sPhiDP_dict = get_metadata('spectral_differential_phase')
    sPhiDP_dict['data'] = sPhiDP

    return sPhiDP_dict


def compute_spectral_rhohv(spectra, subtract_noise=False, signal_h_field=None,
                           signal_v_field=None, noise_h_field=None,
                           noise_v_field=None):
    """
    Computes the spectral RhoHV from the complex spectras in ADU

    Parameters
    ----------
    spectra : Radar spectra object
        Object containing the required fields
    subtract_noise : Bool
            If True noise will be subtracted from the signals
    signal_h_field, signal_v_field, noise_h_field, noise_v_field : str
        Name of the fields in radar which contains the signal and noise.
        None will use the default field name in the Py-ART configuration file.

    Returns
    -------
    sRhoHV_dict : field dictionary
        Field dictionary containing the spectral RhoHV

    """
    if signal_h_field is None:
        signal_h_field = get_field_name('complex_spectra_hh_ADU')
    if signal_v_field is None:
        signal_v_field = get_field_name('complex_spectra_vv_ADU')
    if noise_h_field is None:
        noise_h_field = get_field_name('noiseADU_hh')
    if noise_v_field is None:
        noise_v_field = get_field_name('noiseADU_hh')

    sRhoHV = (
        spectra.fields[signal_v_field]['data'] *
        np.ma.conjugate(spectra.fields[signal_h_field]['data']))

    pwr_h = _compute_power(
        spectra.fields[signal_h_field]['data'],
        noise=spectra.fields[noise_h_field]['data'],
        subtract_noise=subtract_noise)

    pwr_v = _compute_power(
        spectra.fields[signal_v_field]['data'],
        noise=spectra.fields[noise_v_field]['data'],
        subtract_noise=subtract_noise)

    sRhoHV /= np.ma.sqrt(pwr_h*pwr_v)

    sRhoHV_field = 'spectral_copolar_correlation_coefficient'

    sRhoHV_dict = get_metadata(sRhoHV_field)
    sRhoHV_dict['data'] = sRhoHV

    return sRhoHV_dict


def compute_spectral_phase(spectra, signal_field=None):
    """
    Computes the spectral phase from the complex spectra in ADU

    Parameters
    ----------
    spectra : Radar spectra object
        Object containing the required fields
    signal_field : str, optional
        Name of the field in radar which contains the signal.
        None will use the default field name in the Py-ART configuration file.

    Returns
    -------
    phase_dict : field dictionary
        Field dictionary containing the spectral phase

    """
    if signal_field is None:
        signal_field = get_field_name('complex_spectra_hh_ADU')

    pol = 'hh'
    if 'vv' in signal_field:
        pol = 'vv'

    phase = np.ma.angle(spectra.fields[signal_field]['data'], deg=True)

    phase_field = 'spectral_phase_'+pol

    phase_dict = get_metadata(phase_field)
    phase_dict['data'] = phase

    return phase_dict


def compute_pol_variables(spectra, fields_list, subtract_noise=False,
                          signal_h_field=None, signal_v_field=None,
                          noise_h_field=None, noise_v_field=None):
    """
    Computes the polarimetric variables from the complex spectras in ADU

    Parameters
    ----------
    spectra : Radar spectra object
        Object containing the required fields
    fields_list : list of str
        list of fields to compute
    subtract_noise : Bool
            If True noise will be subtracted from the signals
    signal_h_field, signal_v_field, noise_h_field, noise_v_field : str
        Name of the fields in radar which contains the signal and noise.
        None will use the default field name in the Py-ART configuration file.

    Returns
    -------
    sRhoHV_dict : field dictionary
        Field dictionary containing the spectral RhoHV

    """
    if signal_h_field is None:
        signal_h_field = get_field_name('complex_spectra_hh_ADU')
    if signal_v_field is None:
        signal_v_field = get_field_name('complex_spectra_vv_ADU')
    if noise_h_field is None:
        noise_h_field = get_field_name('noiseADU_hh')
    if noise_v_field is None:
        noise_v_field = get_field_name('noiseADU_hh')

    fields = {}
    if ('reflectivity' in fields_list or
            'differential_reflectivity' in fields_list or
            'differential_phase' in fields_list or
            'velocity' in fields_list or 'spectrum_width' in fields_list):
        sdBZ = compute_spectral_reflectivity(
            spectra, subtract_noise=subtract_noise, signal_field=signal_h_field,
            noise_field=noise_h_field)
        sdBZ_lin = np.ma.power(10., 0.1*sdBZ['data'])
        dBZ = 10.*np.ma.log10(np.ma.sum(sdBZ_lin, axis=-1))

        if 'reflectivity' in fields_list:
            dBZ_dict = get_metadata('reflectivity')
            dBZ_dict['data'] = dBZ
            fields.update({'reflectivity': dBZ_dict})

    if ('reflectivity_vv' in fields_list or
            'differential_reflectivity' in fields_list):
        sdBZv = compute_spectral_reflectivity(
            spectra, subtract_noise=subtract_noise, signal_field=signal_v_field,
            noise_field=noise_v_field)
        sdBZv_lin = np.ma.power(10., 0.1*sdBZv['data'])
        dBZv = 10.*np.ma.log10(np.ma.sum(sdBZv_lin, axis=-1))
        if 'reflectivity_vv' in fields_list:
            dBZv_dict = get_metadata('reflectivity_vv')
            dBZv_dict['data'] = dBZv
            fields.update({'reflectivity_vv': dBZv_dict})

        if 'differential_reflectivity' in fields_list:
            ZDR = dBZ-dBZv

            ZDR_dict = get_metadata('differential_reflectivity')
            ZDR_dict['data'] = ZDR
            fields.update({'differential_reflectivity': ZDR_dict})

    if 'differential_phase' in fields_list:
        sPhiDP = compute_spectral_differential_phase(
            spectra, signal_h_field=signal_h_field,
            signal_v_field=signal_v_field)
        sPhiDP['data'][sPhiDP['data'] < 0.] += 360.
        PhiDP = (np.ma.sum(sdBZ_lin*sPhiDP['data'], axis=-1) /
                 np.ma.sum(sdBZ_lin, axis=-1))
        PhiDP[PhiDP > 180.] -= 360.

        PhiDP_dict = get_metadata('differential_phase')
        PhiDP_dict['data'] = PhiDP
        fields.update({'differential_phase': PhiDP_dict})

    if 'cross_correlation_ratio' in fields_list:
        RhoHV_dict = compute_rhohv(
            spectra, subtract_noise=subtract_noise, signal_h_field=signal_h_field,
            signal_v_field=signal_v_field, noise_h_field=noise_h_field,
            noise_v_field=noise_v_field)
        fields.update({'cross_correlation_ratio': RhoHV_dict})

    if 'velocity' in fields_list or 'spectrum_width' in fields_list:
        vel = np.ma.expand_dims(spectra.Doppler_velocity['data'], axis=1)
        vel = np.broadcast_to(
            vel, (spectra.nrays, spectra.ngates, spectra.npulses_max))
        mean_vel = (
            np.ma.sum(sdBZ_lin*vel, axis=-1)/np.ma.sum(sdBZ_lin, axis=-1))

        if 'velocity' in fields_list:
            vel_dict = get_metadata('velocity')
            vel_dict['data'] = mean_vel
            fields.update({'velocity': vel_dict})

        if 'spectrum_width' in fields_list:
            mean_vel2 = np.ma.expand_dims(mean_vel, axis=2)
            mean_vel2 = np.broadcast_to(
                mean_vel2,
                (spectra.nrays, spectra.ngates, spectra.npulses_max))
            width = np.ma.sqrt(
                np.ma.sum(np.ma.power(vel-mean_vel2, 2.)*sdBZ_lin, axis=-1) /
                dBZ)
            width_dict = get_metadata('spectrum_width')
            width_dict['data'] = width
            fields.update({'spectrum_width': width_dict})

    radar = radar_from_spectra(spectra)

    for field_name in fields_list:
        radar.add_field(field_name, fields[field_name])

    return radar


def compute_reflectivity(spectra, sdBZ_field=None):
    """
    Computes the reflectivity from the spectral reflectivity

    Parameters
    ----------
    spectra : Radar spectra object
        Object containing the required fields
    sdBZ_field : str
        Name of the field that contains the spectral reflectivity.
        None will use the default field name in the Py-ART configuration file.

    Returns
    -------
    dBZ_dict : field dictionary
        Field dictionary containing the reflectivity

    """
    if sdBZ_field is None:
        sdBZ_field = get_field_name('spectral_reflectivity_hh')

    sdBZ_lin = np.ma.power(10., 0.1*spectra.fields[sdBZ_field]['data'])
    dBZ = 10.*np.ma.log10(np.ma.sum(sdBZ_lin, axis=-1))

    dBZ_field = 'reflectivity'
    if 'vv' in sdBZ_field:
        dBZ_field += '_vv'

    dBZ_dict = get_metadata(dBZ_field)
    dBZ_dict['data'] = dBZ

    return dBZ_dict


def compute_differential_reflectivity(spectra, sdBZ_field=None,
                                      sdBZv_field=None):
    """
    Computes the differential reflectivity from the horizontal and vertical
    spectral reflectivity

    Parameters
    ----------
    spectra : Radar spectra object
        Object containing the required fields
    sdBZ_field, sdBZv_field : str
        Name of the fields that contain the spectral reflectivity.
        None will use the default field name in the Py-ART configuration file.

    Returns
    -------
    ZDR_dict : field dictionary
        Field dictionary containing the differential reflectivity

    """
    if sdBZ_field is None:
        sdBZ_field = get_field_name('spectral_reflectivity_hh')
    if sdBZv_field is None:
        sdBZv_field = get_field_name('spectral_reflectivity_vv')

    sdBZ_lin = np.ma.power(10., 0.1*spectra.fields[sdBZ_field]['data'])
    dBZ = 10.*np.ma.log10(np.ma.sum(sdBZ_lin, axis=-1))

    sdBZv_lin = np.ma.power(10., 0.1*spectra.fields[sdBZv_field]['data'])
    dBZv = 10.*np.ma.log10(np.ma.sum(sdBZv_lin, axis=-1))

    ZDR_dict = get_metadata('differential_reflectivity')
    ZDR_dict['data'] = dBZ-dBZv

    return ZDR_dict


def compute_differential_phase(spectra, sdBZ_field=None, sPhiDP_field=None):
    """
    Computes the differential phase from the spectral differential phase and
    the spectral reflectivity

    Parameters
    ----------
    spectra : Radar spectra object
        Object containing the required fields
    sdBZ_field, sPhiDP_field : str
        Name of the fields that contain the spectral reflectivity and the
        spectral differential phase.
        None will use the default field name in the Py-ART configuration file.

    Returns
    -------
    PhiDP_dict : field dictionary
        Field dictionary containing the differential phase

    """
    if sdBZ_field is None:
        sdBZ_field = get_field_name('spectral_reflectivity_hh')
    if sPhiDP_field is None:
        sPhiDP_field = get_field_name('spectral_differential_phase')

    sdBZ_lin = np.ma.power(10., 0.1*spectra.fields[sdBZ_field]['data'])
    sPhiDP = deepcopy(spectra.fields[sPhiDP_field]['data'])
    sPhiDP[sPhiDP < 0.] += 360.
    PhiDP = np.ma.sum(sdBZ_lin*sPhiDP, axis=-1)/np.ma.sum(sdBZ_lin, axis=-1)
    PhiDP[PhiDP > 180.] -= 360.

    PhiDP_dict = get_metadata('differential_phase')
    PhiDP_dict['data'] = PhiDP

    return PhiDP_dict


def compute_rhohv(spectra, subtract_noise=False, signal_h_field=None,
                  signal_v_field=None, noise_h_field=None,
                  noise_v_field=None):
    """
    Computes RhoHV from the horizontal and vertical spectral reflectivity

    Parameters
    ----------
    spectra : Radar spectra object
        Object containing the required fields
    subtract_noise : Bool
            If True noise will be subtracted from the signals
    signal_h_field, signal_v_field, noise_h_field, noise_v_field : str
        Name of the fields in radar which contains the signal and noise.
        None will use the default field name in the Py-ART configuration file.

    Returns
    -------
    RhoHV_dict : field dictionary
        Field dictionary containing the RhoHV

    """
    if signal_h_field is None:
        signal_h_field = get_field_name('complex_spectra_hh_ADU')
    if signal_v_field is None:
        signal_v_field = get_field_name('complex_spectra_vv_ADU')
    if noise_h_field is None:
        noise_h_field = get_field_name('noiseADU_hh')
    if noise_v_field is None:
        noise_v_field = get_field_name('noiseADU_hh')

    sRhoHV = (
        spectra.fields[signal_v_field]['data'] *
        np.ma.conjugate(spectra.fields[signal_h_field]['data']))

    pwr_h = _compute_power(
        spectra.fields[signal_h_field]['data'],
        noise=spectra.fields[noise_h_field]['data'],
        subtract_noise=subtract_noise)

    pwr_v = _compute_power(
        spectra.fields[signal_v_field]['data'],
        noise=spectra.fields[noise_v_field]['data'],
        subtract_noise=subtract_noise)

    RhoHV = (
        np.ma.abs(np.ma.sum(sRhoHV, axis=-1)) /
        np.ma.sqrt(np.ma.sum(pwr_h, axis=-1)*np.ma.sum(pwr_v, axis=-1)))

    RhoHV_dict = get_metadata('cross_correlation_ratio')
    RhoHV_dict['data'] = RhoHV

    return RhoHV_dict


def compute_Doppler_velocity(spectra, sdBZ_field=None):
    """
    Computes the Doppler velocity from the spectral reflectivity

    Parameters
    ----------
    spectra : Radar spectra object
        Object containing the required fields
    sdBZ_field : str
        Name of the field that contains the spectral reflectivity.
        None will use the default field name in the Py-ART configuration file.

    Returns
    -------
    vel_dict : field dictionary
        Field dictionary containing the Doppler velocity

    """
    if sdBZ_field is None:
        sdBZ_field = get_field_name('spectral_reflectivity_hh')

    sdBZ_lin = np.ma.power(10., 0.1*spectra.fields[sdBZ_field]['data'])
    vel = np.ma.expand_dims(spectra.Doppler_velocity['data'], axis=1)
    vel = np.broadcast_to(
        vel, (spectra.nrays, spectra.ngates, spectra.npulses_max))
    mean_vel = np.ma.sum(sdBZ_lin*vel, axis=-1)/np.ma.sum(sdBZ_lin, axis=-1)

    vel_dict = get_metadata('velocity')
    vel_dict['data'] = mean_vel

    return vel_dict


def compute_Doppler_width(spectra, sdBZ_field=None):
    """
    Computes the Doppler width from the spectral reflectivity

    Parameters
    ----------
    spectra : Radar spectra object
        Object containing the required fields
    sdBZ_field : str
        Name of the field that contains the spectral reflectivity.
        None will use the default field name in the Py-ART configuration file.

    Returns
    -------
    width_dict : field dictionary
        Field dictionary containing the Doppler spectrum width

    """
    if sdBZ_field is None:
        sdBZ_field = get_field_name('spectral_reflectivity_hh')

    sdBZ_lin = np.ma.power(10., 0.1*spectra.fields[sdBZ_field]['data'])
    dBZ = np.ma.sum(sdBZ_lin, axis=-1)

    vel = np.ma.expand_dims(spectra.Doppler_velocity['data'], axis=1)
    vel = np.broadcast_to(
        vel, (spectra.nrays, spectra.ngates, spectra.npulses_max))

    mean_vel = np.ma.sum(sdBZ_lin*vel, axis=-1)/dBZ
    mean_vel = np.ma.expand_dims(mean_vel, axis=2)
    mean_vel = np.broadcast_to(
        mean_vel, (spectra.nrays, spectra.ngates, spectra.npulses_max))

    width = np.ma.sqrt(
        np.ma.sum(np.ma.power(vel-mean_vel, 2.)*sdBZ_lin, axis=-1)/dBZ)

    width_dict = get_metadata('spectrum_width')
    width_dict['data'] = width

    return width_dict


def _compute_power(signal, noise=None, subtract_noise=False):
    """
    Compute the signal power in linear units

    Parameters
    ----------
    signal : float array
        The complex spectra
    noise : float array
        The noise power per Doppler bin
    subtract_noise : Bool
        If True and noise not None the noise power will be subtracted from the
        signal power

    Returns
    -------
    pwr : float array
        The computed power

    """
    pwr = np.ma.power(np.ma.abs(signal), 2.)

    if subtract_noise and noise is not None:
        pwr -= noise
        pwr[pwr < 0.] = np.ma.masked

    return pwr
