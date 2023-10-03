"""
pyart.retrieve.spectra
======================

Retrievals from spectral data.

.. autosummary::
    :toctree: generated/

    compute_iq
    compute_spectral_power
    compute_spectral_noise
    compute_spectral_reflectivity
    compute_spectral_differential_reflectivity
    compute_spectral_differential_phase
    compute_spectral_rhohv
    compute_spectral_phase
    compute_pol_variables
    compute_noise_power
    compute_reflectivity
    compute_differential_reflectivity
    compute_differential_phase
    compute_rhohv
    compute_Doppler_velocity
    compute_Doppler_width
    dealias_spectra
    _compute_power
    _smooth_spectral_power

"""
from copy import deepcopy
from warnings import warn

import numpy as np
from scipy.signal.windows import gaussian, get_window

from ..config import get_field_name, get_metadata
from ..util import (
    estimate_noise_hs74,
    ma_broadcast_to,
    radar_from_spectra,
    rolling_window,
)


def compute_iq(spectra, fields_in_list, fields_out_list, window=None):
    """
    Computes the IQ data from the spectra through an inverse Fourier transform

    Parameters
    ----------
    spectra : Spectra radar object
        Object containing the spectra
    fields_in_list : list of str
        list of input spectra fields names
    fields_out_list : list of str
        list with the output IQ fields names obtained from the input fields
    window : string, tupple or None
        Parameters of the window used to obtain the spectra. The parameters
        are the ones corresponding to function
        scipy.signal.windows.get_window. If it is not None the inverse will be
        used to multiply the IQ data obtained by the IFFT

    Returns
    -------
    radar : IQ radar object
        radar object containing the IQ fields

    """
    radar = deepcopy(spectra)
    radar.fields = {}
    for field_name_in, field_name_out in zip(fields_in_list, fields_out_list):
        if field_name_out in ('IQ_hh_ADU', 'IQ_vv_ADU'):
            iq = np.ma.masked_all(
                (spectra.nrays, spectra.ngates, spectra.npulses_max),
                dtype=np.complex64)
            for ray, npuls in enumerate(spectra.npulses['data']):
                ray_data = spectra.fields[field_name_in]['data'][
                    ray, :, 0:npuls].filled(0.)
                iq[ray, :, 0:npuls] = np.fft.ifft(np.fft.ifftshift(
                    ray_data, axes=-1), axis=-1) * npuls

                if window is not None:
                    wind = get_window(window, npuls)
                    wind = wind / np.sqrt(np.sum(np.power(wind, 2.)) / npuls)
                    wind = np.broadcast_to(
                        np.atleast_2d(wind), (spectra.ngates, npuls))
                    iq[ray, :, 0:npuls] /= wind
        else:
            iq = np.ma.masked_all(
                (spectra.nrays, spectra.ngates, spectra.npulses_max),
                dtype=np.float32)
            for ray, npuls in enumerate(spectra.npulses['data']):
                iq[ray, :, 0:npuls] = spectra.fields[field_name_in]['data'][
                    ray, :, 0:npuls] * npuls

        field_dict = get_metadata(field_name_out)
        field_dict['data'] = iq
        radar.fields.update({field_name_out: field_dict})

    return radar


def compute_spectral_power(spectra, units='dBADU', subtract_noise=False,
                           smooth_window=None, signal_field=None,
                           noise_field=None):
    """
    Computes the spectral power from the complex spectra in ADU. Requires
    key dBADU_to_dBm_hh or dBADU_to_dBm_vv in radar_calibration if the
    units are to be dBm

    Parameters
    ----------
    spectra : Radar spectra object
        Object containing the required fields
    units : str
        The units of the returned signal. Can be 'ADU', 'dBADU' or 'dBm'
    subtract_noise : Bool
        If True noise will be subtracted from the signal
    smooth_window : int or None
        Size of the moving Gaussian smoothing window. If none no smoothing
        will be applied
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
        noise_field = get_field_name('spectral_noise_power_hh_ADU')

    pol = 'hh'
    if 'vv' in signal_field:
        pol = 'vv'

    noise = None
    if noise_field in spectra.fields:
        noise = spectra.fields[noise_field]['data']

    pwr = _compute_power(
        spectra.fields[signal_field]['data'], noise=noise,
        subtract_noise=subtract_noise, smooth_window=smooth_window)

    if units in ('dBADU', 'dBm'):
        pwr = 10. * np.ma.log10(pwr)

        if units == 'dBm':
            dBADU2dBm = None
            if spectra.radar_calibration is not None:
                if (pol == 'hh' and
                        'dBADU_to_dBm_hh' in spectra.radar_calibration):
                    dBADU2dBm = (
                        spectra.radar_calibration['dBADU_to_dBm_hh']['data'][
                            0])
                elif (pol == 'vv' and
                      'dBADU_to_dBm_vv' in spectra.radar_calibration):
                    dBADU2dBm = (
                        spectra.radar_calibration['dBADU_to_dBm_vv']['data'][0])

            if dBADU2dBm is None:
                raise ValueError(
                    'Unable to compute spectral power in dBm. ' +
                    'dBADU to dBm conversion factor unknown')

            # should it be divided by the number of pulses?
            pwr += dBADU2dBm

    power_field = 'spectral_power_' + pol + '_' + units
    if 'unfiltered' in signal_field:
        power_field = 'unfiltered_' + power_field

    pwr_dict = get_metadata(power_field)
    pwr_dict['data'] = pwr

    return pwr_dict


def compute_spectral_noise(spectra, units='dBADU', navg=1, rmin=0.,
                           nnoise_min=1, signal_field=None):
    """
    Computes the spectral noise power from the complex spectra in ADU.
    Requires key dBADU_to_dBm_hh or dBADU_to_dBm_vv in radar_calibration if
    the units are to be dBm. The noise is computed using the method described
    in Hildebrand and Sehkon, 1974.

    Parameters
    ----------
    spectra : Radar spectra object
        Object containing the required fields
    units : str
        The units of the returned signal. Can be 'ADU', 'dBADU' or 'dBm'
    navg : int
        Number of spectra averaged
    rmin : int
        Range from which the data is used to estimate the noise
    nnoise_min : int
        Minimum number of samples to consider the estimated noise power valid
    signal_field : str, optional
        Name of the field in radar which contains the signal.
        None will use the default field name in the Py-ART configuration file.

    Returns
    -------
    noise_dict : field dictionary
        Field dictionary containing the spectral noise power

    References
    ----------
    P. H. Hildebrand and R. S. Sekhon, Objective Determination of the Noise
    Level in Doppler Spectra. Journal of Applied Meteorology, 1974, 13,
    808-811.

    """
    if signal_field is None:
        signal_field = get_field_name('complex_spectra_hh_ADU')

    pol = 'hh'
    if 'vv' in signal_field:
        pol = 'vv'

    ind_rmin = np.where(spectra.range['data'] >= rmin)[0]
    if ind_rmin.size == 0:
        warn('Unable to compute spectral noise. ' +
             'Range at which start gathering data ' + str(rmin) +
             'km. larger than radar range')
        return None

    ind_rmin = ind_rmin[0]

    pwr = _compute_power(spectra.fields[signal_field]['data'])

    noise = np.ma.masked_all(
        (spectra.nrays, spectra.ngates, spectra.npulses_max))
    for ray, npuls in enumerate(spectra.npulses['data']):
        mean, _, _, _ = estimate_noise_hs74(
            pwr[ray, ind_rmin:, 0:npuls].compressed(), navg=navg,
            nnoise_min=nnoise_min)
        noise[ray, :, :] = mean

    if units in ('dBADU', 'dBm'):
        noise = 10. * np.ma.log10(noise)

        if units == 'dBm':
            dBADU2dBm = None
            if spectra.radar_calibration is not None:
                if (pol == 'hh' and
                        'dBADU_to_dBm_hh' in spectra.radar_calibration):
                    dBADU2dBm = (
                        spectra.radar_calibration['dBADU_to_dBm_hh']['data'][
                            0])
                elif (pol == 'vv' and
                      'dBADU_to_dBm_vv' in spectra.radar_calibration):
                    dBADU2dBm = (
                        spectra.radar_calibration['dBADU_to_dBm_vv']['data'][0])

            if dBADU2dBm is None:
                raise ValueError(
                    'Unable to compute spectral power in dBm. ' +
                    'dBADU to dBm conversion factor unknown')

            # should it be divided by the number of pulses?
            noise += dBADU2dBm

    noise_field = 'spectral_noise_power_' + pol + '_' + units
    noise_dict = get_metadata(noise_field)
    noise_dict['data'] = noise

    return noise_dict


def compute_spectral_reflectivity(spectra, compute_power=True,
                                  subtract_noise=False, smooth_window=None,
                                  pwr_field=None, signal_field=None,
                                  noise_field=None):
    """
    Computes the spectral reflectivity from the complex spectra in ADU or
    from the signal power in ADU. Requires
    keys dBADU_to_dBm_hh or dBADU_to_dBm_vv in radar_calibration if the
    to be computed

    Parameters
    ----------
    spectra : Radar spectra object
        Object containing the required fields
    compute_power : Bool
        If True the signal power will be computed. Otherwise the field given
        by the user will be used
    subtract_noise : Bool
        If True noise will be subtracted from the signal
    smooth_window : int or None
        Size of the moving Gaussian smoothing window. If none no smoothing
        will be applied
    pwr_field, signal_field, noise_field : str, optional
        Name of the fields in radar which contains the signal power, complex
        signal and noise. None will use the default field name in the Py-ART
        configuration file.

    Returns
    -------
    sdBZ_dict : field dictionary
        Field dictionary containing the spectral reflectivity

    """
    if compute_power:
        if signal_field is None:
            signal_field = get_field_name('complex_spectra_hh_ADU')
        if noise_field is None:
            noise_field = get_field_name('spectral_noise_power_hh_ADU')
    else:
        if pwr_field is None:
            pwr_field = get_field_name('spectral_power_hh_ADU')

    if spectra.radar_calibration is None:
        raise ValueError(
            'Unable to compute spectral reflectivity. ' +
            'Calibration parameters unknown')

    pol = 'hh'
    if ((signal_field is not None and 'vv' in signal_field) or
            (pwr_field is not None and 'vv' in pwr_field)):
        pol = 'vv'

    if pol == 'hh':
        if ('dBADU_to_dBm_hh' not in spectra.radar_calibration or
                'calibration_constant_hh' not in spectra.radar_calibration):
            raise ValueError(
                'Unable to compute spectral reflectivity. ' +
                'Calibration parameters unknown')
        dBADU2dBm = spectra.radar_calibration['dBADU_to_dBm_hh']['data'][0]
        radconst = (
            spectra.radar_calibration['calibration_constant_hh']['data'][0])
    else:
        if ('dBADU_to_dBm_vv' not in spectra.radar_calibration or
                'calibration_constant_vv' not in spectra.radar_calibration):
            raise ValueError(
                'Unable to compute spectral reflectivity. ' +
                'Calibration parameters unknown')
        dBADU2dBm = spectra.radar_calibration['dBADU_to_dBm_vv']['data'][0]
        radconst = (
            spectra.radar_calibration['calibration_constant_vv']['data'][0])

    if (pol == 'hh' and 'matched_filter_loss_h' in spectra.radar_calibration):
        mfloss = spectra.radar_calibration['matched_filter_loss_h']['data'][0]
    elif (pol == 'vv' and
          'matched_filter_loss_v' in spectra.radar_calibration):
        mfloss = spectra.radar_calibration['matched_filter_loss_v']['data'][0]
    else:
        warn('Unknown matched filter losses. Assumed 0 dB')
        mfloss = 0.

    if 'path_attenuation' in spectra.radar_calibration:
        pathatt = spectra.radar_calibration['path_attenuation']['data'][0]
    else:
        warn('Unknown gas path attenuation. Assumed 0 dB/km')
        pathatt = 0.

    rangeKm = np.broadcast_to(
        np.atleast_3d(spectra.range['data'] / 1000.),
        (spectra.nrays, spectra.ngates, spectra.npulses_max))

    if compute_power:
        noise = None
        if noise_field in spectra.fields:
            noise = spectra.fields[noise_field]['data']

        pwr = _compute_power(
            spectra.fields[signal_field]['data'], noise=noise,
            subtract_noise=subtract_noise, smooth_window=smooth_window)
    else:
        pwr = spectra.fields[pwr_field]['data']

    sdBZ = (
        10. *
        np.ma.log10(pwr) +
        dBADU2dBm +
        radconst +
        mfloss +
        pathatt *
        rangeKm +
        20. *
        np.log10(rangeKm))

    sdBZ_field = 'spectral_reflectivity_' + pol
    if ((signal_field is not None and 'unfiltered' in signal_field) or
            (pwr_field is not None and 'unfiltered' in pwr_field)):
        sdBZ_field = 'unfiltered_' + sdBZ_field

    sdBZ_dict = get_metadata(sdBZ_field)
    sdBZ_dict['data'] = sdBZ

    return sdBZ_dict


def compute_spectral_differential_reflectivity(spectra, compute_power=True,
                                               subtract_noise=False,
                                               smooth_window=None,
                                               pwr_h_field=None,
                                               pwr_v_field=None,
                                               signal_h_field=None,
                                               signal_v_field=None,
                                               noise_h_field=None,
                                               noise_v_field=None):
    """
    Computes the spectral differential reflectivity from the complex spectras
    or the power in ADU

    Parameters
    ----------
    spectra : Radar spectra object
        Object containing the required fields
    compute_power : Bool
        If True the signal power will be computed. Otherwise the field given
        by the user will be used
    subtract_noise : Bool
        If True noise will be subtracted from the signals
    smooth_window : int or None
        Size of the moving Gaussian smoothing window. If none no smoothing
        will be applied
    pwr_h_field, pwr_v_field, signal_h_field, signal_v_field, noise_h_field,
    noise_v_field : str
        Name of the fields in radar which contains the signal and noise.
        None will use the default field name in the Py-ART configuration file.

    Returns
    -------
    sZDR_dict : field dictionary
        Field dictionary containing the spectral differential reflectivity

    """
    if compute_power:
        if signal_h_field is None:
            signal_h_field = get_field_name('complex_spectra_hh_ADU')
        if signal_v_field is None:
            signal_v_field = get_field_name('complex_spectra_vv_ADU')
        if noise_h_field is None:
            noise_h_field = get_field_name('spectral_noise_power_hh_ADU')
        if noise_v_field is None:
            noise_v_field = get_field_name('spectral_noise_power_vv_ADU')
    else:
        if pwr_h_field is None:
            pwr_h_field = get_field_name('spectral_power_hh_ADU')
        if pwr_v_field is None:
            pwr_v_field = get_field_name('spectral_power_hh_ADU')

    if spectra.radar_calibration is None:
        raise ValueError(
            'Unable to compute spectral reflectivity. ' +
            'Calibration parameters unknown')

    if ('dBADU_to_dBm_hh' not in spectra.radar_calibration or
            'dBADU_to_dBm_vv' not in spectra.radar_calibration or
            'calibration_constant_hh' not in spectra.radar_calibration or
            'calibration_constant_vv' not in spectra.radar_calibration):
        raise ValueError(
            'Unable to compute spectral reflectivity. ' +
            'Calibration parameters unknown')

    dBADU2dBm_h = spectra.radar_calibration['dBADU_to_dBm_hh']['data'][0]
    dBADU2dBm_v = spectra.radar_calibration['dBADU_to_dBm_vv']['data'][0]
    radconst_h = (
        spectra.radar_calibration['calibration_constant_hh']['data'][0])
    radconst_v = (
        spectra.radar_calibration['calibration_constant_vv']['data'][0])

    if compute_power:
        noise = None
        if noise_h_field in spectra.fields:
            noise = spectra.fields[noise_h_field]['data']

        pwr_h = _compute_power(
            spectra.fields[signal_h_field]['data'], noise=noise,
            subtract_noise=subtract_noise, smooth_window=smooth_window)

        noise = None
        if noise_v_field in spectra.fields:
            noise = spectra.fields[noise_v_field]['data']

        pwr_v = _compute_power(
            spectra.fields[signal_v_field]['data'], noise=noise,
            subtract_noise=subtract_noise, smooth_window=smooth_window)
    else:
        pwr_h = spectra.fields[pwr_h_field]['data']
        pwr_v = spectra.fields[pwr_v_field]['data']

    sZDR = (
        (10. * np.ma.log10(pwr_h) + dBADU2dBm_h + radconst_h) -
        (10. * np.ma.log10(pwr_v) + dBADU2dBm_v + radconst_v))

    sZDR_field = 'spectral_differential_reflectivity'
    if ((signal_h_field is not None and 'unfiltered' in signal_h_field) or
            (pwr_h_field is not None and 'unfiltered' in pwr_h_field)):
        sZDR_field = 'unfiltered_' + sZDR_field

    sZDR_dict = get_metadata(sZDR_field)
    sZDR_dict['data'] = sZDR

    return sZDR_dict


def compute_spectral_differential_phase(spectra, use_rhohv=False,
                                        srhohv_field=None,
                                        signal_h_field=None,
                                        signal_v_field=None):
    """
    Computes the spectral differential reflectivity from the complex spectras
    in ADU or sRhoHV

    Parameters
    ----------
    spectra : Radar spectra object
        Object containing the required fields
    use_rhohv : Bool
        If true sRhoHV is going to be used to compute the differential phase.
        Otherwise the complex signals are used
    signal_h_field, signal_v_field : str
        Name of the fields in radar which contains the signal.
        None will use the default field name in the Py-ART configuration file.

    Returns
    -------
    sPhiDP_dict : field dictionary
        Field dictionary containing the spectral differential phase

    """
    if use_rhohv:
        if srhohv_field is None:
            srhohv_field = get_field_name(
                'spectral_copolar_correlation_coefficient')
    else:
        if signal_h_field is None:
            signal_h_field = get_field_name('complex_spectra_hh_ADU')
        if signal_v_field is None:
            signal_v_field = get_field_name('complex_spectra_vv_ADU')

    if not use_rhohv:
        phase_h = np.ma.angle(
            spectra.fields[signal_h_field]['data'], deg=True)
        phase_v = np.ma.angle(
            spectra.fields[signal_v_field]['data'], deg=True)

        sPhiDP = phase_h - phase_v
    else:
        sPhiDP = np.ma.angle(
            spectra.fields[srhohv_field]['data'], deg=True)

    sPhiDP_field = 'spectral_differential_phase'
    if ((signal_h_field is not None and 'unfiltered' in signal_h_field) or
            (srhohv_field is not None and 'unfiltered' in srhohv_field)):
        sPhiDP_field = 'unfiltered_' + sPhiDP_field

    sPhiDP_dict = get_metadata(sPhiDP_field)
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
        noise_h_field = get_field_name('spectral_noise_power_hh_ADU')
    if noise_v_field is None:
        noise_v_field = get_field_name('spectral_noise_power_vv_ADU')

    sRhoHV = (
        spectra.fields[signal_h_field]['data'] *
        np.ma.conjugate(spectra.fields[signal_v_field]['data']))

    noise = None
    if noise_h_field in spectra.fields:
        noise = spectra.fields[noise_h_field]['data']

    pwr_h = _compute_power(
        spectra.fields[signal_h_field]['data'], noise=noise,
        subtract_noise=subtract_noise)

    noise = None
    if noise_v_field in spectra.fields:
        noise = spectra.fields[noise_v_field]['data']

    pwr_v = _compute_power(
        spectra.fields[signal_v_field]['data'], noise=noise,
        subtract_noise=subtract_noise)

    sRhoHV /= np.ma.sqrt(pwr_h * pwr_v)

    sRhoHV_field = 'spectral_copolar_correlation_coefficient'
    if 'unfiltered' in signal_h_field:
        sRhoHV_field = 'unfiltered_' + sRhoHV_field

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

    phase_field = 'spectral_phase_' + pol
    if 'unfiltered' in signal_field:
        phase_field = 'unfiltered_' + phase_field

    phase_dict = get_metadata(phase_field)
    phase_dict['data'] = phase

    return phase_dict


def compute_pol_variables(spectra, fields_list, use_pwr=False,
                          subtract_noise=False, smooth_window=None,
                          srhohv_field=None, pwr_h_field=None,
                          pwr_v_field=None, signal_h_field=None,
                          signal_v_field=None, noise_h_field=None,
                          noise_v_field=None):
    """
    Computes the polarimetric variables from the complex spectra in ADU or
    the spectral powers and spectral RhoHV

    Parameters
    ----------
    spectra : Radar spectra object
        Object containing the required fields
    fields_list : list of str
        list of fields to compute
    use_pwr : Bool
        If True the polarimetric variables will be computed from the spectral
        power and the spectral RhoHV. Otherwise from the complex spectra
    subtract_noise : Bool
        If True noise will be subtracted from the signals
    smooth_window : int or None
        Size of the moving Gaussian smoothing window. If none no smoothing
        will be applied
    srhohv_field, pwr_h_field, pwr_v_field, signal_h_field, signal_v_field,
    noise_h_field, noise_v_field : str
        Name of the fields in radar which contains the signal and noise.
        None will use the default field name in the Py-ART configuration file.

    Returns
    -------
    radar : radar object
        Object containing the computed fields

    """
    if use_pwr:
        if srhohv_field is None:
            srhohv_field = get_field_name(
                'spectral_copolar_correlation_coefficient')
        if pwr_h_field is None:
            pwr_h_field = get_field_name('spectral_power_hh_ADU')
        if pwr_v_field is None:
            pwr_v_field = get_field_name('spectral_power_vv_ADU')

        compute_power = False
        use_rhohv = True
    else:
        if signal_h_field is None:
            signal_h_field = get_field_name('complex_spectra_hh_ADU')
        if signal_v_field is None:
            signal_v_field = get_field_name('complex_spectra_vv_ADU')
        if noise_h_field is None:
            noise_h_field = get_field_name('spectral_noise_power_hh_ADU')
        if noise_v_field is None:
            noise_v_field = get_field_name('spectral_noise_power_vv_ADU')

        compute_power = True
        use_rhohv = False

    fields = {}
    if ('reflectivity' in fields_list or
            'differential_reflectivity' in fields_list or
            'uncorrected_differential_phase' in fields_list or
            'velocity' in fields_list or 'spectrum_width' in fields_list or
            'unfiltered_reflectivity' in fields_list or
            'unfiltered_differential_reflectivity' in fields_list or
            'uncorrected_unfiltered_differential_phase' in fields_list or
            'unfiltered_velocity' in fields_list or
            'unfiltered_spectrum_width' in fields_list):
        sdBZ = compute_spectral_reflectivity(
            spectra, compute_power=compute_power,
            subtract_noise=subtract_noise, smooth_window=smooth_window,
            pwr_field=pwr_h_field, signal_field=signal_h_field,
            noise_field=noise_h_field)
        sdBZ_lin = np.ma.power(10., 0.1 * sdBZ['data'])
        dBZ = 10. * np.ma.log10(np.ma.sum(sdBZ_lin, axis=-1))

        if ('reflectivity' in fields_list or
                'unfiltered_reflectivity' in fields_list):
            dBZ_field = 'reflectivity'
            if 'unfiltered_reflectivity' in fields_list:
                dBZ_field = 'unfiltered_' + dBZ_field
            dBZ_dict = get_metadata(dBZ_field)
            dBZ_dict['data'] = dBZ
            fields.update({dBZ_field: dBZ_dict})

    if ('reflectivity_vv' in fields_list or
            'differential_reflectivity' in fields_list or
            'velocity_vv' in fields_list or
            'spectrum_width_vv' in fields_list or
            'unfiltered_reflectivity_vv' in fields_list or
            'unfiltered_differential_reflectivity' in fields_list or
            'unfiltered_velocity_vv' in fields_list or
            'unfiltered_spectrum_width_vv' in fields_list):
        sdBZv = compute_spectral_reflectivity(
            spectra, compute_power=compute_power,
            subtract_noise=subtract_noise, smooth_window=smooth_window,
            pwr_field=pwr_v_field, signal_field=signal_v_field,
            noise_field=noise_v_field)
        sdBZv_lin = np.ma.power(10., 0.1 * sdBZv['data'])
        dBZv = 10. * np.ma.log10(np.ma.sum(sdBZv_lin, axis=-1))
        if ('reflectivity_vv' in fields_list or
                'unfiltered_reflectivity_vv' in fields_list):
            dBZv_field = 'reflectivity_vv'
            if 'unfiltered_reflectivity_vv' in fields_list:
                dBZv_field = 'unfiltered_' + dBZv_field
            dBZv_dict = get_metadata(dBZv_field)
            dBZv_dict['data'] = dBZv
            fields.update({dBZv_field: dBZv_dict})

        if ('differential_reflectivity' in fields_list or
                'unfiltered_differential_reflectivity' in fields_list):
            ZDR = dBZ - dBZv

            ZDR_field = 'differential_reflectivity'
            if 'unfiltered_differential_reflectivity' in fields_list:
                ZDR_field = 'unfiltered_' + ZDR_field
            ZDR_dict = get_metadata(ZDR_field)
            ZDR_dict['data'] = ZDR
            fields.update({ZDR_field: ZDR_dict})

    if ('uncorrected_differential_phase' in fields_list or
            'uncorrected_unfiltered_differential_phase' in fields_list):
        sPhiDP = compute_spectral_differential_phase(
            spectra, use_rhohv=use_rhohv, srhohv_field=srhohv_field,
            signal_h_field=signal_h_field, signal_v_field=signal_v_field)
        sPhiDP['data'][sPhiDP['data'] < 0.] += 360.
        PhiDP = (np.ma.sum(sdBZ_lin * sPhiDP['data'], axis=-1) /
                 np.ma.sum(sdBZ_lin, axis=-1))
        PhiDP[PhiDP > 180.] -= 360.

        PhiDP_field = 'uncorrected_differential_phase'
        if 'uncorrected_unfiltered_differential_phase' in fields_list:
            PhiDP_field = 'uncorrected_unfiltered_differential_phase'
        PhiDP_dict = get_metadata(PhiDP_field)
        PhiDP_dict['data'] = PhiDP
        fields.update({PhiDP_field: PhiDP_dict})

    if ('cross_correlation_ratio' in fields_list or
            'unfiltered_cross_correlation_ratio' in fields_list):
        RhoHV_dict = compute_rhohv(
            spectra, use_rhohv=use_rhohv, subtract_noise=subtract_noise,
            srhohv_field=srhohv_field, pwr_h_field=pwr_h_field,
            pwr_v_field=pwr_v_field, signal_h_field=signal_h_field,
            signal_v_field=signal_v_field, noise_h_field=noise_h_field,
            noise_v_field=noise_v_field)
        RhoHV_field = 'cross_correlation_ratio'
        if 'unfiltered_cross_correlation_ratio' in fields_list:
            RhoHV_field = 'unfiltered_' + RhoHV_field
        fields.update({RhoHV_field: RhoHV_dict})

    if ('velocity' in fields_list or 'spectrum_width' in fields_list or
            'velocity_vv' in fields_list or
            'spectrum_width_vv' in fields_list or
            'unfiltered_velocity' in fields_list or
            'unfiltered_spectrum_width' in fields_list or
            'unfiltered_velocity_vv' in fields_list or
            'unfiltered_spectrum_width_vv' in fields_list):
        vel = np.ma.expand_dims(spectra.Doppler_velocity['data'], axis=1)
        vel = ma_broadcast_to(
            vel, (spectra.nrays, spectra.ngates, spectra.npulses_max))
        if ('velocity' in fields_list or
                'unfiltered_velocity' in fields_list or
                'spectrum_width' in fields_list or
                'unfiltered_spectrum_width' in fields_list):
            mean_vel = (
                np.ma.sum(
                    sdBZ_lin *
                    vel,
                    axis=-
                    1) /
                np.ma.sum(
                    sdBZ_lin,
                    axis=-
                    1))
        if ('velocity_vv' in fields_list or
                'unfiltered_velocity_vv' in fields_list or
                'spectrum_width_vv' in fields_list or
                'unfiltered_spectrum_width_vv' in fields_list):
            mean_vel_v = (
                np.ma.sum(sdBZv_lin * vel, axis=-1) /
                np.ma.sum(sdBZv_lin, axis=-1))

        if 'velocity' in fields_list or 'unfiltered_velocity' in fields_list:
            vel_field = 'velocity'
            if 'unfiltered_velocity' in fields_list:
                vel_field = 'unfiltered_' + vel_field
            vel_dict = get_metadata(vel_field)
            vel_dict['data'] = mean_vel
            fields.update({vel_field: vel_dict})

        if ('velocity_vv' in fields_list or
                'unfiltered_velocity_vv' in fields_list):
            vel_field = 'velocity_vv'
            if 'unfiltered_velocity_vv' in fields_list:
                vel_field = 'unfiltered_' + vel_field
            vel_dict = get_metadata(vel_field)
            vel_dict['data'] = mean_vel_v
            fields.update({vel_field: vel_dict})

        if ('spectrum_width' in fields_list or
                'unfiltered_spectrum_width' in fields_list):
            mean_vel2 = np.ma.expand_dims(mean_vel, axis=2)
            mean_vel2 = ma_broadcast_to(
                mean_vel2,
                (spectra.nrays, spectra.ngates, spectra.npulses_max))
            width = np.ma.sqrt(
                np.ma.sum(
                    np.ma.power(
                        vel - mean_vel2,
                        2.) * sdBZ_lin,
                    axis=-1) / dBZ)
            width_field = 'spectrum_width'
            if 'unfiltered_spectrum_width' in fields_list:
                width_field = 'unfiltered_' + width_field
            width_dict = get_metadata(width_field)
            width_dict['data'] = width
            fields.update({width_field: width_dict})

        if ('spectrum_width_vv' in fields_list or
                'unfiltered_spectrum_width_vv' in fields_list):
            mean_vel2 = np.ma.expand_dims(mean_vel_v, axis=2)
            mean_vel2 = ma_broadcast_to(
                mean_vel2,
                (spectra.nrays, spectra.ngates, spectra.npulses_max))
            width = np.ma.sqrt(
                np.ma.sum(
                    np.ma.power(
                        vel - mean_vel2,
                        2.) * sdBZv_lin,
                    axis=-1) / dBZv)
            width_field = 'spectrum_width_vv'
            if 'unfiltered_spectrum_width_vv' in fields_list:
                width_field = 'unfiltered_' + width_field
            width_dict = get_metadata(width_field)
            width_dict['data'] = width
            fields.update({width_field: width_dict})

    radar = radar_from_spectra(spectra)

    for field_name in fields_list:
        radar.add_field(field_name, fields[field_name])

    return radar


def compute_noise_power(spectra, units='dBADU', navg=1, rmin=0.,
                        nnoise_min=1, signal_field=None):
    """
    Computes the noise power from the complex spectra in ADU.
    Requires key dBADU_to_dBm_hh or dBADU_to_dBm_vv in radar_calibration if
    the units are to be dBm. The noise is computed using the method described
    in Hildebrand and Sehkon, 1974.

    Parameters
    ----------
    spectra : Radar spectra object
        Object containing the required fields
    units : str
        The units of the returned signal. Can be 'ADU', 'dBADU' or 'dBm'
    navg : int
        Number of spectra averaged
    rmin : int
        Range from which the data is used to estimate the noise
    nnoise_min : int
        Minimum number of samples to consider the estimated noise power valid
    signal_field : str, optional
        Name of the field in radar which contains the signal.
        None will use the default field name in the Py-ART configuration file.

    Returns
    -------
    noise_dict : field dictionary
        Field dictionary containing the noise power

    References
    ----------
    P. H. Hildebrand and R. S. Sekhon, Objective Determination of the Noise
    Level in Doppler Spectra. Journal of Applied Meteorology, 1974, 13,
    808-811.

    """
    if signal_field is None:
        signal_field = get_field_name('complex_spectra_hh_ADU')

    pol = 'hh'
    if 'vv' in signal_field:
        pol = 'vv'

    ind_rmin = np.where(spectra.range['data'] >= rmin)[0]
    if ind_rmin.size == 0:
        warn('Unable to compute spectral noise. ' +
             'Range at which start gathering data ' + str(rmin) +
             'km. larger than radar range')
        return None

    ind_rmin = ind_rmin[0]

    pwr = _compute_power(spectra.fields[signal_field]['data'])

    noise = np.ma.masked_all(
        (spectra.nrays, spectra.ngates))
    for ray, npuls in enumerate(spectra.npulses['data']):
        mean, _, _, _ = estimate_noise_hs74(
            pwr[ray, ind_rmin:, 0:npuls].compressed(), navg=navg,
            nnoise_min=nnoise_min)
        noise[ray, :] = mean * npuls

    if units in ('dBADU', 'dBm'):
        noise = 10. * np.ma.log10(noise)

        if units == 'dBm':
            dBADU2dBm = None
            if spectra.radar_calibration is not None:
                if (pol == 'hh' and
                        'dBADU_to_dBm_hh' in spectra.radar_calibration):
                    dBADU2dBm = (
                        spectra.radar_calibration['dBADU_to_dBm_hh']['data'][
                            0])
                elif (pol == 'vv' and
                      'dBADU_to_dBm_vv' in spectra.radar_calibration):
                    dBADU2dBm = (
                        spectra.radar_calibration['dBADU_to_dBm_vv']['data'][0])

            if dBADU2dBm is None:
                raise ValueError(
                    'Unable to compute spectral power in dBm. ' +
                    'dBADU to dBm conversion factor unknown')

            # should it be divided by the number of pulses?
            noise += dBADU2dBm

    noise_field = 'noise' + units + '_' + pol
    noise_dict = get_metadata(noise_field)
    noise_dict['data'] = noise

    return noise_dict


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

    sdBZ_lin = np.ma.power(10., 0.1 * spectra.fields[sdBZ_field]['data'])
    dBZ = 10. * np.ma.log10(np.ma.sum(sdBZ_lin, axis=-1))

    dBZ_field = 'reflectivity'
    if 'vv' in sdBZ_field:
        dBZ_field += '_vv'
    if 'unfiltered' in sdBZ_field:
        dBZ_field = 'unfiltered_' + dBZ_field

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

    sdBZ_lin = np.ma.power(10., 0.1 * spectra.fields[sdBZ_field]['data'])
    dBZ = 10. * np.ma.log10(np.ma.sum(sdBZ_lin, axis=-1))

    sdBZv_lin = np.ma.power(10., 0.1 * spectra.fields[sdBZv_field]['data'])
    dBZv = 10. * np.ma.log10(np.ma.sum(sdBZv_lin, axis=-1))

    zdr_field = 'differential_reflectivity'
    if 'unfiltered' in sdBZ_field:
        zdr_field = 'unfiltered_' + zdr_field

    ZDR_dict = get_metadata(zdr_field)
    ZDR_dict['data'] = dBZ - dBZv

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

    sdBZ_lin = np.ma.power(10., 0.1 * spectra.fields[sdBZ_field]['data'])
    sPhiDP = deepcopy(spectra.fields[sPhiDP_field]['data'])
    sPhiDP[sPhiDP < 0.] += 360.
    PhiDP = np.ma.sum(sdBZ_lin * sPhiDP, axis=-1) / \
        np.ma.sum(sdBZ_lin, axis=-1)
    PhiDP[PhiDP > 180.] -= 360.

    phidp_field = 'uncorrected_differential_phase'
    if 'unfiltered' in sPhiDP_field:
        phidp_field = 'uncorrected_unfiltered_differential_phase'
    PhiDP_dict = get_metadata(phidp_field)
    PhiDP_dict['data'] = PhiDP

    return PhiDP_dict


def compute_rhohv(spectra, use_rhohv=False, subtract_noise=False,
                  srhohv_field=None, pwr_h_field=None, pwr_v_field=None,
                  signal_h_field=None, signal_v_field=None,
                  noise_h_field=None, noise_v_field=None):
    """
    Computes RhoHV from the horizontal and vertical spectral reflectivity
    or from sRhoHV and the spectral powers

    Parameters
    ----------
    spectra : Radar spectra object
        Object containing the required fields
    use_rhohv : Bool
        If true the RhoHV will be computed from sRho_hv. Otherwise it will be
        computed using the complex spectra
    subtract_noise : Bool
            If True noise will be subtracted from the signals
    srhohv_field, pwr_h_field, pwr_v_field, signal_h_field, signal_v_field,
    noise_h_field, noise_v_field : str
        Name of the fields in radar which contains the signal and noise.
        None will use the default field name in the Py-ART configuration file.

    Returns
    -------
    RhoHV_dict : field dictionary
        Field dictionary containing the RhoHV

    """
    if use_rhohv:
        if srhohv_field is None:
            srhohv_field = get_field_name(
                'spectral_copolar_correlation_coefficient')
        if pwr_h_field is None:
            pwr_h_field = get_field_name('spectral_power_hh_ADU')
        if pwr_v_field is None:
            pwr_v_field = get_field_name('spectral_power_vv_ADU')
    else:
        if signal_h_field is None:
            signal_h_field = get_field_name('complex_spectra_hh_ADU')
        if signal_v_field is None:
            signal_v_field = get_field_name('complex_spectra_vv_ADU')
        if noise_h_field is None:
            noise_h_field = get_field_name('spectral_noise_power_hh_ADU')
        if noise_v_field is None:
            noise_v_field = get_field_name('spectral_noise_power_vv_ADU')

    if not use_rhohv:
        sRhoHV = (
            spectra.fields[signal_h_field]['data'] *
            np.ma.conjugate(spectra.fields[signal_v_field]['data']))

        noise = None
        if noise_h_field in spectra.fields:
            noise = spectra.fields[noise_h_field]['data']

        pwr_h = _compute_power(
            spectra.fields[signal_h_field]['data'], noise=noise,
            subtract_noise=subtract_noise)

        noise = None
        if noise_v_field in spectra.fields:
            noise = spectra.fields[noise_v_field]['data']

        pwr_v = _compute_power(
            spectra.fields[signal_v_field]['data'], noise=noise,
            subtract_noise=subtract_noise)
    else:
        pwr_h = spectra.fields[pwr_h_field]['data']
        pwr_v = spectra.fields[pwr_v_field]['data']
        sRhoHV = spectra.fields[srhohv_field]['data'] * \
            np.ma.sqrt(pwr_h * pwr_v)

    RhoHV = (
        np.ma.abs(np.ma.sum(sRhoHV, axis=-1)) /
        np.ma.sqrt(np.ma.sum(pwr_h, axis=-1) * np.ma.sum(pwr_v, axis=-1)))

    rhohv_field = 'cross_correlation_ratio'
    if (srhohv_field is not None and 'unfiltered' in srhohv_field or
            signal_h_field is not None and 'unfiltered' in signal_h_field):
        rhohv_field = 'unfiltered_cross_correlation_ratio'

    RhoHV_dict = get_metadata(rhohv_field)
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

    sdBZ_lin = np.ma.power(10., 0.1 * spectra.fields[sdBZ_field]['data'])
    vel = np.ma.expand_dims(spectra.Doppler_velocity['data'], axis=1)
    vel = ma_broadcast_to(
        vel, (spectra.nrays, spectra.ngates, spectra.npulses_max))
    mean_vel = np.ma.sum(sdBZ_lin * vel, axis=-1) / \
        np.ma.sum(sdBZ_lin, axis=-1)

    vel_field = 'velocity'
    if 'vv' in sdBZ_field:
        vel_field += '_vv'
    if 'unfiltered' in sdBZ_field:
        vel_field = 'unfiltered_' + vel_field
    vel_dict = get_metadata(vel_field)
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

    sdBZ_lin = np.ma.power(10., 0.1 * spectra.fields[sdBZ_field]['data'])
    dBZ = np.ma.sum(sdBZ_lin, axis=-1)

    vel = np.ma.expand_dims(spectra.Doppler_velocity['data'], axis=1)
    vel = ma_broadcast_to(
        vel, (spectra.nrays, spectra.ngates, spectra.npulses_max))

    mean_vel = np.ma.sum(sdBZ_lin * vel, axis=-1) / dBZ
    mean_vel = np.ma.expand_dims(mean_vel, axis=2)
    mean_vel = ma_broadcast_to(
        mean_vel, (spectra.nrays, spectra.ngates, spectra.npulses_max))

    width = np.ma.sqrt(
        np.ma.sum(np.ma.power(vel - mean_vel, 2.) * sdBZ_lin, axis=-1) / dBZ)

    width_field = 'spectrum_width'
    if 'vv' in sdBZ_field:
        width_field += '_vv'
    if 'unfiltered' in sdBZ_field:
        width_field = 'unfiltered_' + width_field
    width_dict = get_metadata(width_field)
    width_dict['data'] = width

    return width_dict


def dealias_spectra(spectra, pwr_field = None,  fields_out_list = None):
    """
    Performs a dealiasing of spectra data, assuming at most one fold

    The method is quite simple and works in the following way at every
    radar get

    - aliasing check
        check if there is no noise either on the left of the right of the spectrum
    - left/right tail computation
        identify left and right tail of the aliased spectrum
    - detect direction of aliasing
        compute Doppler mean velocity with right or left shift of the spectrum,
        perform the shift which minimizes the difference in Doppler velocity to
        previous range bin

    Parameters
    ----------
    spectra : Radar spectra object
        Object containing the required fields
    pwr_field : str
        Name of the field that contains the signal power.
        None will use the default field name in the Py-ART configuration file.
    fields_out_list : list of str
        list with the output dealiased fields names obtained, by default will
        dealias all spectral fields contained in the input spectra

    Returns
    -------
    dealias : Spectra
        Spectra object containing the dealiased fields contained in fields_out_list
        as well as updated Doppler_velocity bins

    """

    if pwr_field is None:
        pwr_field = get_field_name('spectral_power_hh_ADU')

    vel_bins = spectra.Doppler_velocity['data']
    new_bins = np.hstack(
        [vel_bins-2*vel_bins[:,-1][:,None], vel_bins, vel_bins+2*vel_bins[:,-1][:,None]])

    nrays = spectra.nrays
    ngates = spectra.ngates
    npulses = spectra.npulses_max
    spectra.range['data'][1] - spectra.range['data'][0]

    if fields_out_list is None:
        # Get all spectral fields
        fields_out_list = []
        for field in spectra.fields:
            if spectra.fields[field]['data'].shape == (nrays, ngates, npulses):
                fields_out_list.append(field)

    # Easier to work with nan and create mask later

    # Create dict of dealiased fields
    new_spectra_fields = {}
    for field in fields_out_list:
        if field in spectra.fields:
            new_spectra_fields[field] = np.nan*np.ones((nrays, ngates, npulses*3))

    dealiased_already = np.zeros((nrays, ngates))

    old_spectra_fields = {}
    for field in fields_out_list:
        old_spectra_fields[field] = spectra.fields[field]['data']
        old_spectra_fields[field] = np.ma.filled(old_spectra_fields[field], np.nan)

    vel = np.ma.expand_dims(vel_bins, axis=1)
    mean_vel_field = np.nansum(10**(0.1*old_spectra_fields[pwr_field]) * vel, axis=-1) / \
        np.nansum(10**(0.1*old_spectra_fields[pwr_field]), axis=-1)

    for i in range(ngates):
        for j in range(nrays):
            if not (np.isnan(old_spectra_fields[pwr_field][j, i, 0]) or
                    np.isnan(old_spectra_fields[pwr_field][j, i, -1])):
                noise_region = np.where(np.isnan(old_spectra_fields[pwr_field][j, i]))[0]
                if len(noise_region):
                    right_tail_len = npulses-noise_region[-1]
                    left_tail_len = int(noise_region[0])

                    pwr_1 = np.nan*np.ones(npulses*3)
                    pwr_2 = np.nan*np.ones(npulses*3)

                    pwr_1[npulses-right_tail_len:npulses] = \
                                old_spectra_fields[pwr_field][j, i, npulses-right_tail_len:npulses]
                    pwr_1[npulses:2*npulses-right_tail_len] = \
                            old_spectra_fields[pwr_field][j, i, 0:npulses-right_tail_len]

                    pwr_2[2*npulses:2*npulses+left_tail_len] = \
                                old_spectra_fields[field][j, i, 0:left_tail_len]
                    pwr_2[npulses+left_tail_len:2*npulses] = \
                        old_spectra_fields[field][j, i, left_tail_len:]

                    vel_1 = np.nansum(10**(0.1*pwr_1 * new_bins)) / np.nansum(10**(0.1*pwr_1))
                    vel_2 = np.nansum(10**(0.1*pwr_2 * new_bins)) / np.nansum(10**(0.1*pwr_2))

                    if np.abs(vel_1 - mean_vel_field[j, i]) >  np.abs(vel_2 - mean_vel_field[j, i]):
                        for field in new_spectra_fields:
                            new_spectra_fields[field][j, i, npulses-right_tail_len:npulses] = \
                                    old_spectra_fields[field][j, i, npulses-right_tail_len:npulses]
                            new_spectra_fields[field][j, i, npulses:2*npulses-right_tail_len] = \
                                    old_spectra_fields[field][j, i, 0:npulses-right_tail_len]
                    else:
                        for field in new_spectra_fields:
                            new_spectra_fields[field][j, i, 2*npulses:2*npulses+left_tail_len] = \
                                old_spectra_fields[field][j, i, 0:left_tail_len]
                            new_spectra_fields[field][j, i, npulses+left_tail_len:2*npulses] = \
                                old_spectra_fields[field][j, i, left_tail_len:]
                    dealiased_already[j, i] = 1
            else:
                for field in new_spectra_fields:
                    new_spectra_fields[field][j, i, npulses:2*npulses] = old_spectra_fields[field][j,i]


    dealias_spectra = deepcopy(spectra)

    dealias_spectra.Doppler_velocity['data'] = new_bins
    dealias_spectra.npulses['data'] = nrays * [len(new_bins)]
    dealias_spectra.npulses_max = len(new_bins[0])

    to_remove = []
    for field in dealias_spectra.fields:
        if field in fields_out_list:
            dealias_spectra.fields[field]['data'] = np.ma.array(new_spectra_fields[field],
                                                            mask = np.isnan(new_spectra_fields[field]))
        else:
            to_remove.append(field)


    for field in to_remove:
        dealias_spectra.fields.pop(field)

    return dealias_spectra

def _compute_power(signal, noise=None, subtract_noise=False,
                   smooth_window=None):
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
    smooth_window : int or None
        Size of the moving Gaussian smoothing window. If none no smoothing
        will be applied

    Returns
    -------
    pwr : float array
        The computed power

    """
    pwr = np.ma.power(np.ma.abs(signal), 2.)

    if subtract_noise and noise is not None:
        pwr -= noise
        pwr[pwr < 0.] = np.ma.masked

    if smooth_window is not None:
        pwr = _smooth_spectral_power(pwr, wind_len=smooth_window)

    return pwr


def _smooth_spectral_power(raw_data, wind_len=5):
    """
    smoothes the spectral power using a rolling Gaussian window.

    Parameters
    ----------
    raw_data : float masked array
        The data to smooth.
    wind_len : float
        Length of the moving window

    Returns
    -------
    data_smooth : float masked array
        smoothed data

    """
    # we want an odd window
    if wind_len % 2 == 0:
        wind_len += 1
    half_wind = int((wind_len - 1) / 2)

    # create window
    wind = gaussian(wind_len, std=1.)
    wind /= np.sum(wind)

    # initialize smoothed data
    nrays, ngates, nDoppler = np.shape(raw_data)
    data_smooth = np.ma.masked_all((nrays, ngates, nDoppler))

    # get rolling window and mask data
    data_wind = rolling_window(raw_data, wind_len)
    data_smooth[:, :, half_wind:-half_wind] = np.ma.dot(data_wind, wind)

    return data_smooth
