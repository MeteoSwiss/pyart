"""
pyart.correct.attenuation
=========================

Attenuation correction from polarimetric radars.

Code adapted from method in Gu et al, JAMC 2011, 50, 39.

Adapted by Scott Collis and Scott Giangrande, refactored by Jonathan Helmus.

.. autosummary::
    :toctree: generated/

    calculate_attenuation    
    _get_param_att
    _param_att_table


"""
import copy

import numpy as np
from scipy.integrate import cumtrapz

from ..config import get_metadata, get_field_name, get_fillvalue
from . import phase_proc
from ..filters import temp_based_gate_filter
from ..retrieve import get_freq_band


def calculate_attenuation(radar, doc=None, fzl=None, smooth_window_len=5,
                          a_coef=None, beta=None, refl_field=None,
                          phidp_field=None, zdr_field=None, temp_field=None,
                          spec_at_field=None, corr_refl_field=None,
                          spec_diff_at_field=None, corr_zdr_field=None):
    """
    Calculate the attenuation and the differential attenuation from a
    polarimetric radar using Z-PHI method. Optionally, perform clutter
    identification prior to the correction.
    The attenuation is computed up to a user defined freezing level height
    or up to where temperatures in a temperature field are positive.
    The coefficients are either user-defined or radar frequency dependent.

    Parameters
    ----------
    radar : Radar
        Radar object to use for attenuation calculations.  Must have
        copol_coeff, norm_coherent_power, phidp,
        refl fields.
    doc : float
        Number of gates at the end of each ray to to remove from the
        calculation.
    fzl : float
        Freezing layer, gates above this point are not included in the
        correction.
    smooth_window_len : int
        Size, in range bins, of the smoothing window
    a_coef : float
        A coefficient in attenuation calculation.
    beta : float
        Beta parameter in attenuation calculation.
    refl_field, phidp_field, zdr_field, temp_field : str
        Field names within the radar object which represent the horizonal
        reflectivity, the differential phase shift, the differential
        reflectivity and the temperature field. A value of None for any of
        these parameters will use the default field name as defined in the
        Py-ART configuration file. The ZDR field and temperature field are
        going to be used only if available.
    spec_at_field, corr_refl_field : str
        Names of the specific attenuation and the corrected
        reflectivity fields that will be used to fill in the metadata for
        the returned fields.  A value of None for any of these parameters
        will use the default field names as defined in the Py-ART
        configuration file.
    spec_diff_at_field, corr_zdr_field : str
        Names of the specific differential attenuation and the corrected
        differential reflectivity fields that will be used to fill in the
        metadata for the returned fields.  A value of None for any of these
        parameters will use the default field names as defined in the Py-ART
        configuration file. These fields will be computed only if the ZDR
        field is available.

    Returns
    -------
    spec_at : dict
        Field dictionary containing the specific attenuation.
    cor_z : dict
        Field dictionary containing the corrected reflectivity.
    spec_diff_at : dict
        Field dictionary containing the specific differential attenuation.
    cor_zdr : dict
        Field dictionary containing the corrected differential reflectivity.

    References
    ----------
    Gu et al. Polarimetric Attenuation Correction in Heavy Rain at C Band,
    JAMC, 2011, 50, 39-58.

    Ryzhkov et al. Potential Utilization of Specific Attenuation for Rainfall
    Estimation, Mitigation of Partial Beam Blockage, and Radar Networking,
    JAOT, 2014, 31, 599-619.

    """
    # select the coefficients as a function of frequency band
    if (a_coef is None) or (beta is None) or (c is None) or (d is None):
        if 'frequency' in radar.instrument_parameters:
            a_coef, beta, c, d = _get_param_att(
                radar.instrument_parameters['frequency']['data'][0])
        else:
            a_coef, beta, c, d = _param_att_table()['C']
            warn('Radar frequency unknown. ' +
                 'Default coefficients for C band will be applied')

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
    if spec_at_field is None:
        spec_at_field = get_field_name('specific_attenuation')
    if corr_refl_field is None:
        corr_refl_field = get_field_name('corrected_reflectivity')
    if spec_diff_at_field is None:
        spec_diff_at_field = get_field_name(
            'specific_differential_attenuation')
    if corr_zdr_field is None:
        corr_zdr_field = get_field_name(
            'corrected_differential_reflectivity')
    if temp_field is None:
        temp_field = get_field_name('temperature')

    # extract fields and parameters from radar if they exist
    # reflectivity and differential phase must exist
    radar.check_field_exists(refl_field)
    refl = radar.fields[refl_field]['data']

    radar.check_field_exists(phidp_field)
    phidp = radar.fields[phidp_field]['data']

    try:
        radar.check_field_exists(zdr_field)
        zdr = radar.fields[zdr_field]['data']
    except KeyError:
        zdr = None

    # determine the range up to which data is processed.
    # if fzl is specified use this range, otherwise determine it from
    # temperature field if available
    if fzl is None:
        if temp_field in radar.fields:
            gatefilter = temp_based_gate_filter(
                radar, temp_field=temp_field, min_temp=0., thickness=0.)
            end_gate_arr = np.zeros(radar.nrays, dtype='int32')
            for ray in range(radar.nrays):
                end_gate_arr[ray] = np.where(
                    np.ndarray.flatten(
                        gatefilter.gate_excluded[ray, :]) == 1)[0][0]
            mask_fzl = gatefilter.gate_excluded == 1
        else:
            fzl = 4000.
            doc = 15
            print('WARNING: Temperature field not available.' +
                  'Using default freezing level height ' +
                  str(fzl) + ' [m].')

    if fzl is not None:
        end_gate_arr = np.zeros(radar.nrays, dtype='int32')
        mask_fzl = np.zeros((radar.nrays, radar.ngates), dtype=np.bool)
        for sweep in range(radar.nsweeps):
            end_gate, start_ray, end_ray = (
                phase_proc.det_process_range(radar, sweep, fzl, doc=doc))
            end_gate_arr[start_ray:end_ray] = end_gate
            mask_fzl[start_ray:end_ray, end_gate+1:] = True

    # create array to hold specific attenuation and attenuation
    ah = np.ma.zeros(refl.shape, dtype='float64')
    pia = np.ma.zeros(refl.shape, dtype='float64')

    # if ZDR exists create array to hold specific differential attenuation
    # and path integrated differential attenuation
    if zdr is not None:
        adiff = np.ma.zeros(zdr.shape, dtype='float64')
        pida = np.ma.zeros(zdr.shape, dtype='float64')

    # prepare phidp: filter out values above freezing level and negative
    # makes sure phidp is monotonously increasing
    mask_phidp = np.ma.getmaskarray(phidp)
    mask_phidp = np.logical_or(mask_phidp, mask_fzl)
    mask_phidp = np.logical_or(mask_phidp, phidp.data < 0.)
    phidp = np.ma.masked_where(mask_phidp, phidp)
    phidp = np.maximum.accumulate(phidp, axis=1)

    mask = np.ma.getmaskarray(refl)

    # calculate initial reflectivity correction and gate spacing (in km)
    init_refl_correct = refl + phidp * a_coef
    dr = (radar.range['data'][1] - radar.range['data'][0]) / 1000.0

    for ray in range(radar.nrays):
        # perform attenuation calculation on a single ray
        # if number of valid range bins larger than smoothing window
        if end_gate_arr[ray] > smooth_window_len:
            # extract the ray's phase shift,
            # init. refl. correction and mask
            ray_phase_shift = phidp[ray, 0:end_gate_arr[ray]]
            ray_init_refl = init_refl_correct[ray, 0:end_gate_arr[ray]]
            ray_mask = mask[ray, 0:end_gate_arr[ray]]

            # perform calculation if there is valid data
            last_six_good = np.where(
                np.ndarray.flatten(ray_mask) == 0)[0][-6:]
            if(len(last_six_good)) == 6:
                if smooth_window_len > 0:
                    sm_refl_data = phase_proc.smooth_and_trim(
                        ray_init_refl, window_len=smooth_window_len)
                else:
                    sm_refl_data = ray_init_refl.data
                sm_refl = np.ma.masked_where(ray_mask, sm_refl_data)
                refl_linear = np.ma.power(10.0, 0.1 * beta * sm_refl)
                refl_linear[ray_mask] = 0.

                phidp_max = np.median(ray_phase_shift[last_six_good])
                self_cons_number = (
                    10.0 ** (0.1 * beta * a_coef * phidp_max) - 1.0)
                I_indef = cumtrapz(0.46 * beta * dr * refl_linear[::-1])
                I_indef = np.append(I_indef, I_indef[-1])[::-1]

                # set the specific attenutation and attenuation
                ah[ray, 0:end_gate_arr[ray]] = (
                    refl_linear * self_cons_number /
                    (I_indef[0] + self_cons_number * I_indef))

                pia[ray, :-1] = cumtrapz(ah[ray, :]) * dr * 2.0
                pia[ray, -1] = pia[ray, -2]

                # if ZDR exists, set the specific differential attenuation
                # and differential attenuation
                if zdr is not None:
                    adiff[ray, 0:end_gate_arr[ray]] = (
                        c * np.ma.power(ah[ray, 0:end_gate_arr[ray]], d))

                    pida[ray, :-1] = cumtrapz(adiff[ray, :]) * dr * 2.0
                    pida[ray, -1] = pida[ray, -2]

    # prepare output field dictionaries
    # for specific attenuation and corrected reflectivity
    specific_atten = np.ma.masked_where(mask, ah)
    corr_reflectivity = np.ma.masked_where(mask, pia + refl)

    spec_at = get_metadata(spec_at_field)
    spec_at['data'] = specific_atten
    spec_at['_FillValue'] = get_fillvalue()

    cor_z = get_metadata(corr_refl_field)
    cor_z['data'] = corr_reflectivity
    cor_z['_FillValue'] = get_fillvalue()

    # prepare output field dictionaries
    # for specific diff attenuation and ZDR
    if zdr is not None:
        specific_diff_atten = np.ma.masked_where(mask, adiff)
        corr_diff_reflectivity = np.ma.masked_where(mask, pida + zdr)

        spec_diff_at = get_metadata(spec_diff_at_field)
        spec_diff_at['data'] = specific_diff_atten
        spec_diff_at['_FillValue'] = get_fillvalue()

        cor_zdr = get_metadata(corr_zdr_field)
        cor_zdr['data'] = corr_diff_reflectivity
        cor_zdr['_FillValue'] = get_fillvalue()
    else:
        spec_diff_at = None
        cor_zdr = None

    return spec_at, cor_z, spec_diff_at, cor_zdr


def _get_param_att(freq):
    """
    get the parameters of Z-Phi attenuation estimation for a particular
    frequency

    Parameters
    ----------
    freq : float
        radar frequency [Hz]

    Returns
    -------
    a_coeff, beta, c, d : floats
        the coefficient and exponent of the power law

    """
    param_att_dict = _param_att_table()

    freq_band = get_freq_band(freq)
    if (freq_band is not None) and (freq_band in param_att_dict):
        return param_att_dict[freq_band]

    if freq < 2e9:
        freq_band_aux = 'S'
    elif freq > 12e9:
        freq_band_aux = 'X'

    warn('Radar frequency out of range. ' +
         'Coefficients only applied to S, C or X band. ' +
         freq_band + ' band coefficients will be used')

    return param_att_dict[freq_band_aux]


def _param_att_table():
    """
    defines the parameters of Z-Phi attenuation estimation at each frequency
    band.

    Returns
    -------
    param_att_dict : dict
        A dictionary with the coefficients at each band

    """
    param_att_dict = dict()

    # S band:
    param_att_dict.update({'S': (0.02, 0.64884, 0.15917, 1.0804)})

    # C band:
    param_att_dict.update({'C': (0.08, 0.64884, 0.3, 1.0804)})

    # X band:
    param_att_dict.update({'X': (0.31916, 0.64884, 0.15917, 1.0804)})

    return param_att_dict
