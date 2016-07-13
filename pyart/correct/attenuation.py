"""
pyart.correct.attenuation
=========================

Attenuation correction from polarimetric radars.

Code adapted from method in Gu et al, JAMC 2011, 50, 39.

Adapted by Scott Collis and Scott Giangrande, refactored by Jonathan Helmus.

.. autosummary::
    :toctree: generated/

    calculate_attenuation


"""
import copy

import numpy as np
from scipy.integrate import cumtrapz

from ..config import get_metadata, get_field_name, get_fillvalue
from . import phase_proc


def calculate_attenuation(radar, z_offset, debug=False, doc=None, fzl=None,
                          smooth_window_len=5, rhv_min=None, ncp_min=None, 
                          a_coef=None, beta=None, 
                          refl_field=None, phidp_field=None, zdr_field=None, 
                          temp_field=None, ncp_field=None, rhv_field=None,
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
        copol_coeff, norm_coherent_power, proc_dp_phase_shift,
        reflectivity_horizontal fields.
    z_offset : float
        Horizontal reflectivity offset in dBZ.
    debug : bool
        True to print debugging information, False supressed this printing.

    Returns
    -------
    spec_at : dict
        Field dictionary containing the specific attenuation.
    cor_z : dict
        Field dictionary containing the corrected reflectivity.

    Other Parameters
    ----------------
    doc : float
        Number of gates at the end of each ray to to remove from the
        calculation.
    fzl : float
        Freezing layer, gates above this point are not included in the
        correction.    
    smooth_window_len : int
        Size, in range bins, of the smoothing window        
    rhv_min : float
        Minimum copol_coeff value to consider valid.
    ncp_min : float
        Minimum norm_coherent_power to consider valid.
    a_coef : float
        A coefficient in attenuation calculation.
    beta : float
        Beta parameter in attenuation calculation.
    refl_field, ncp_field, rhv_field, phidp_field : str
        Field names within the radar object which represent the horizonal
        reflectivity, normal coherent power, the copolar coefficient, and the
        differential phase shift. A value of None for any of these parameters
        will use the default field name as defined in the Py-ART
        configuration file.
    spec_at_field, corr_refl_field : str
        Names of the specific attenuation and the corrected
        reflectivity fields that will be used to fill in the metadata for
        the returned fields.  A value of None for any of these parameters
        will use the default field names as defined in the Py-ART
        configuration file.
    spec_diff_at_field, corr_zdr_field : str
        Names of the specific differential attenuation and the corrected
        differential reflectivity fields that will be used to fill in the 
        metadata for the returned fields.  A value of None for any of these parameters
        will use the default field names as defined in the Py-ART
        configuration file
    temp_field : str
        Name of the temperature field that will be used to fill in the 
        metadata for the returned field.  A value of None will use the default
        field names as defined in the Py-ART configuration file


    References
    ----------
    Gu et al. Polarimetric Attenuation Correction in Heavy Rain at C Band,
    JAMC, 2011, 50, 39-58.
    
    Ryzhkov et al. Potential Utilization of Specific Attenuation for Rainfall
    Estimation, Mitigation of Partial Beam Blockage, and Radar Networking,
    JAOT, 2014, 31, 599-619.

    """
    # select the coefficients as a function of frequency band
    if a_coef == None or beta == None:
        # assign coefficients according to radar frequency
        if 'frequency' in radar.instrument_parameters:
            freq= radar.instrument_parameters['frequency']['data']
            # S band
            if freq >= 2e9 and freq < 4e9 :
                freq_band='S'
                a_coef=0.02
                beta=0.64884
                c=0.15917
                d=1.0804
            # C band
            elif freq >= 4e9 and freq < 8e9 :
                freq_band='C'
                a_coef=0.08
                beta=0.64884
                c=0.3
                d=1.0804
            # X band
            elif freq >= 8e9 and freq <= 12e9 :
                freq_band='X'
                a_coef=0.31916
                beta=0.64884
                c=0.15917
                d=1.0804
            else :
                if freq < 2e9 :
                    freq_band='S'
                    a_coef=0.02
                    beta=0.64884
                    c=0.15917
                    d=1.0804
                else :
                    freq_band='C'
                    a_coef=0.31916
                    beta=0.64884
                    c=0.15917
                    d=1.0804
                print('WARNING: Radar frequency out of range. \
                    Correction only applies to S, C or X band. '+
                    freq_band+' band coefficients will be applied')
        else :
            a_coef=0.06
            beta=0.8
            print('WARNING: radar frequency unknown. \
                Default coefficients for C band will be applied')
            
            
    # parse the field parameters
    if refl_field is None:
        refl_field = get_field_name('reflectivity')
    if zdr_field is None:
        zdr_field = get_field_name('differential_reflectivity')
    if ncp_field is None:
        ncp_field = get_field_name('normalized_coherent_power')
    if rhv_field is None:
        rhv_field = get_field_name('cross_correlation_ratio')
    if phidp_field is None:
        # use corrrected_differential_phae or unfolded_differential_phase
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
        spec_diff_at_field = get_field_name('specific_differential_attenuation')
    if corr_zdr_field is None:
        corr_zdr_field = get_field_name('corrected_differential_reflectivity')    
    if temp_field is None:
        temp_field = get_field_name('temperature')

    # extract fields and parameters from radar if they exist
    # reflectivity and differential phase must exist
    if ncp_field in radar.fields:
        norm_coherent_power = radar.fields[ncp_field]['data']
    else:
        print('WARNING: Normalized coherent power not available')
        norm_coherent_power = None
    if rhv_field in radar.fields:
        copol_coeff = radar.fields[rhv_field]['data']
    else:
        print('WARNING: Co-polar correlation coefficient not available')
        copol_coeff = None
    if zdr_field in radar.fields:
        differential_reflectivity = radar.fields[zdr_field]['data']        
    else:
        print('WARNING: Differential reflectivity not available')
        differential_reflectivity = None
    if refl_field in radar.fields:
        reflectivity_horizontal = radar.fields[refl_field]['data']
    else:
        raise KeyError('Field not available: ' + refl_field)
    if phidp_field in radar.fields:
        proc_dp_phase_shift = radar.fields[phidp_field]['data']
    else:
        raise KeyError('Field not available: ' + phidp_field)
    
    # if freezing level not specified use temperature field if available
    temp = None
    if fzl is None:
        if temp_field in radar.fields:
            temp = np.array(radar.fields[temp_field]['data'], dtype='int8')
        else:
            fzl=4000.
            doc=15
            print('WARNING: Temperature field not available. Using default freezing level height '+str(fzl)+' [m].')
        
    nsweeps = int(radar.nsweeps)

    # if thresholds input, determine where the reflectivity 
    # and ZDR are valid, mask out bad locations.
    # mask where there is no valid data
    is_phi = proc_dp_phase_shift != get_fillvalue()
    if (rhv_min is not None) and (copol_coeff is not None):
        is_cor = copol_coeff > rhv_min
    else:
        is_cor = None
    if (ncp_min is not None) and (norm_coherent_power is not None):
        is_coh = norm_coherent_power > ncp_min
    else:
        is_coh = None
    
    if (is_cor is not None) and (is_coh is not None):
        is_good = np.logical_and(is_cor, is_coh)
        is_good = np.logical_and(is_good, is_phi)
    elif (is_cor is not None):
        is_good = np.logical_and(is_cor, is_phi)
    elif (is_coh is not None):
        is_good = np.logical_and(is_coh, is_phi)
    else:
        is_good = is_phi
        
    mask = np.logical_not(is_good)
    refl = np.ma.masked_where(mask, reflectivity_horizontal + z_offset)
    if differential_reflectivity is not None:
        zdr = np.ma.masked_where(mask, differential_reflectivity)
    else:
        zdr = None
        
    # calculate initial reflectivity correction and gate spacing (in km)    
    init_refl_correct = refl + proc_dp_phase_shift * a_coef
    dr = (radar.range['data'][1] - radar.range['data'][0]) / 1000.0

    # create array to hold specific attenuation and attenuation
    specific_atten = np.zeros(reflectivity_horizontal.shape, dtype='float32')
    atten = np.zeros(reflectivity_horizontal.shape, dtype='float32')
    
    # if ZDR exists create array to hold specific differential attenuation 
    # and path integrated differential attenuation
    if zdr is not None:
        specific_diff_atten = np.zeros(differential_reflectivity.shape, dtype='float32')
        diff_atten = np.zeros(differential_reflectivity.shape, dtype='float32')
    
    for sweep in range(nsweeps):
        # loop over the sweeps
        if debug:
            print("Doing ", sweep)
        if fzl is None:            
            end_gate_arr, start_ray, end_ray=det_process_range_temp(radar, sweep, temp)
        else:
            end_gate, start_ray, end_ray = phase_proc.det_process_range(
                radar, sweep, fzl, doc=doc)
            end_gate_arr=np.zeros(end_ray+1-start_ray, dtype='int32')+end_gate
            
        for i in range(start_ray, end_ray):
            # perform attenuation calculation on a single ray
            # if number of valid range bins larger than smoothing window
            if end_gate_arr[i-start_ray] > smooth_window_len:
                # extract the ray's phase shift and init. refl. correction
                ray_phase_shift = proc_dp_phase_shift[i, 0:end_gate_arr[i-start_ray]]
                ray_init_refl = init_refl_correct[i, 0:end_gate_arr[i-start_ray]]
    
                # perform calculation
                last_six_good = np.where(is_good[i, 0:end_gate_arr[i-start_ray]])[0][-6:]
                phidp_max = np.median(ray_phase_shift[last_six_good])
                sm_refl = phase_proc.smooth_and_trim(ray_init_refl, window_len=smooth_window_len)
                reflectivity_linear = 10.0 ** (0.1 * beta * sm_refl)
                self_cons_number = 10.0 ** (0.1 * beta * a_coef * phidp_max) - 1.0
                I_indef = cumtrapz(0.46 * beta * dr * reflectivity_linear[::-1])
                I_indef = np.append(I_indef, I_indef[-1])[::-1]
    
                # set the specific attenutation and attenuation
                specific_atten[i, 0:end_gate_arr[i-start_ray]] = (
                    reflectivity_linear * self_cons_number /
                    (I_indef[0] + self_cons_number * I_indef))
    
                atten[i, :-1] = cumtrapz(specific_atten[i, :]) * dr * 2.0
                atten[i, -1] = atten[i, -2]
                
                # if ZDR exists, set the specific differential attenuation
                # and differential attenuation
                if zdr is not None:                    
                    specific_diff_atten[i, 0:end_gate_arr[i-start_ray]] = (
                        c * specific_atten[i, 0:end_gate_arr[i-start_ray]] ** d)                    
                    
                    diff_atten[i, :-1] = cumtrapz(specific_diff_atten[i, :]) * dr * 2.0
                    diff_atten[i, -1] = diff_atten[i, -2]
            

    # prepare output field dictionaries
    # for specific attenuation and corrected reflectivity 
    spec_at = get_metadata(spec_at_field)
    spec_at['data'] = specific_atten
    spec_at['_FillValue'] = get_fillvalue()

    cor_z = get_metadata(corr_refl_field)
    cor_z['data'] = atten + reflectivity_horizontal + z_offset
    cor_z['data'].mask = init_refl_correct.mask
    cor_z['_FillValue'] = get_fillvalue()
    
    # prepare output field dictionaries
    # for specific diff attenuation and ZDR
    if zdr is not None:
        spec_diff_at = get_metadata(spec_diff_at_field)
        spec_diff_at['data'] = specific_diff_atten
        spec_diff_at['_FillValue'] = get_fillvalue()
    
        cor_zdr = get_metadata(corr_zdr_field)
        cor_zdr['data'] = diff_atten + differential_reflectivity
        cor_zdr['data'].mask = init_refl_correct.mask
        cor_zdr['_FillValue'] = get_fillvalue()
    else :
        spec_diff_at = None
        cor_zdr = None

    return spec_at, cor_z, spec_diff_at, cor_zdr
    
    
def det_process_range_temp(radar, sweep, temp_field):
    """
    Determine the processing range for a given sweep.

    Queues the radar and returns the indices which can be used to slice
    the radar fields and select the desired sweep with gates which have
    positive temperature.

    Parameters
    ----------
    radar : Radar
        Radar object from which ranges will be determined.
    sweep : int
        Sweep (0 indexed) for which to determine processing ranges.
    temp_field : radar field
        temperature [degree Centigrade]

    Returns
    -------
    gate_end_arr : array
        Array of indexes of last gate with positive temperature
    ray_start : int
        Ray index which defines the start of the region.
    ray_end : int
        Ray index which defined the end of the region.

    """

    # determine the index of the last valid gate
    ray_start = radar.sweep_start_ray_index['data'][sweep]
    ray_end = radar.sweep_end_ray_index['data'][sweep] + 1
        
    gate_end_arr=np.zeros(ray_end+1-ray_start, dtype='int32')
    for i in range(ray_start, ray_end):        
        for j in range(radar.ngates):
            if temp_field[i, j] > 0:
                gate_end_arr[i-ray_start]=j
            else:
                break
                
    return gate_end_arr, ray_start, ray_end
