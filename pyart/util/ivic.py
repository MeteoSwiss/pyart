"""
pyart.util.ivic
===============

Estimation of noise in a ray using Ivic (2013) method

.. autosummary::
    :toctree: generated/

    estimate_noise_ivic13
    get_ivic_pct
    get_ivic_flat_reg_var_max
    get_ivic_snr_thr
    ivic_pct_table
    ivic_flat_reg_var_max_table
    ivic_snr_thr_table


"""

import numpy as np
from scipy.special import gamma, gammainc, gammaincc, factorial

from .sigmath import rolling_window


def estimate_noise_ivic13(pwr_w_ray, pct=2., delay=1, flat_reg_wlen=96,
                          flat_reg_var_max=1.5, snr_thr=1.6, npulses=30,
                          ngates_min=800, iterations=10, get_noise_pos=False):
    """
    Estimate noise parameters of a ray

    Use the method of estimating the noise level in a ray outlined
    by Ivic, 2013.

    Parameters
    ----------
    pwr_w_ray : array like
        Doppler spectrum in linear units.
    pct : float
        Point Clutter Threshold
    delay : int
        distance of the gate to which compare for point target threshold
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
    get_noise_pos : bool
        If True the indices of the gates identified as noise will be returned

    Returns
    -------
    mean : float-like
        Mean of the gates in the ray identified as noise.
    var : float-like
        Variance of the gates in the ray identified as noise.
    nnoise : int
        Number of noise gates in the ray.
    inds_ray : 1D-array
        The indices of the gates containing noise

    References
    ----------
    I.R. Ivic, C. Curtis and S.M. Torres, Radial-Based Noise Power Estimation
    for Weather Radars. Journal of Atmospheric and Oceanic Technology, 2013,
    30, 2737-2753.

    """
    half_flat_reg_wlen = int((flat_reg_wlen-1)/2)

    if get_noise_pos:
        inds_ray = np.ma.arange(pwr_w_ray.size)

    # step 1 remove gates with discontinuities in range
    pwr_w_ray_aux = pwr_w_ray[delay:-delay]
    pwr_w_ray_aux[np.logical_or(
        np.ma.greater(pwr_w_ray_aux, pct*pwr_w_ray[:-2*delay]),
        np.ma.greater(pwr_w_ray_aux, pct*pwr_w_ray[2*delay:]))] = np.ma.masked
    if get_noise_pos:
        inds_ray_aux = inds_ray[delay:-delay]
        inds_ray_aux[np.ma.getmaskarray(pwr_w_ray_aux)] = np.ma.masked
        inds_ray = np.ma.array(inds_ray_aux.compressed())
    pwr_w_ray = np.ma.array(pwr_w_ray_aux.compressed())

    if pwr_w_ray.size < flat_reg_wlen:
        return None, None, None, None

    # step 2 detect flat sections and estimate mean power of the grouped flat
    # sections. Take smallest power as intermediate power

    # compute moving variance (in dB)
    pwr_dB_ray = np.ma.log10(pwr_w_ray)
    pwr_dB_wind = rolling_window(pwr_dB_ray, flat_reg_wlen)
    pwr_dB_wind_mean = np.ma.mean(pwr_dB_wind, axis=-1)
    # add third dimension made of mean values of the window
    pwr_dB_wind_mean = np.broadcast_to(
        np.expand_dims(pwr_dB_wind_mean, axis=-1),
        (pwr_dB_wind_mean.size, flat_reg_wlen))
    pwr_dB_var = np.ma.sum(
        (pwr_dB_wind-pwr_dB_wind_mean)*(pwr_dB_wind-pwr_dB_wind_mean),
        axis=-1)

    # mask non-flat regions
    pwr_w_ray = pwr_w_ray[half_flat_reg_wlen:-half_flat_reg_wlen]
    inds_non_flat = np.ma.where(pwr_dB_var > flat_reg_var_max)[0]
    for ind in inds_non_flat:
        pwr_w_ray[ind-half_flat_reg_wlen:ind+half_flat_reg_wlen+1] = (
            np.ma.masked)

    # group consecutive gates and get the minimum mean power
    mask = np.ma.getmaskarray(pwr_w_ray)
    ind = np.ma.where(mask == False)[0]
    cons_list = np.split(ind, np.where(np.diff(ind) != 1)[0]+1)
    pwr_w_int = None
    for cons in cons_list:
        pwr_w_mean = np.ma.mean(pwr_w_ray[cons])
        if pwr_w_int is None:
            pwr_w_int = pwr_w_mean
        elif pwr_w_int > pwr_w_mean:
            pwr_w_int = pwr_w_mean

    if pwr_w_int is None:
        return None, None, None, None

    if get_noise_pos:
        inds_ray = inds_ray[half_flat_reg_wlen:-half_flat_reg_wlen]
        inds_ray[np.ma.getmaskarray(pwr_w_ray)] = np.ma.masked
        inds_ray = np.ma.array(inds_ray.compressed())
    pwr_w_ray = np.ma.array(pwr_w_ray.compressed())

    # step 3 remove gates exceeding a threshold based on pwr_int
    ind = np.ma.where(pwr_w_ray > pwr_w_int*snr_thr)[0]
    if ind.size > 0:
        pwr_w_ray[ind] = np.ma.masked
    if get_noise_pos:
        inds_ray[np.ma.getmaskarray(pwr_w_ray)] = np.ma.masked
        inds_ray = np.ma.array(inds_ray.compressed())
    pwr_w_ray = np.ma.array(pwr_w_ray.compressed())
    if pwr_w_ray.size == 0:
        return None, None, None, None

    # step 4 & 5: detect 10 or more consecutive gates with power larger than
    # the median. Compute the mean of the remaining data
    ind = np.ma.where(pwr_w_ray > np.ma.median(pwr_w_ray))[0]
    cons_list = np.split(ind, np.where(np.diff(ind) != 1)[0]+1)
    for cons in cons_list:
        if len(cons) > 10:
            pwr_w_ray[cons] = np.ma.masked
    if get_noise_pos:
        inds_ray[np.ma.getmaskarray(pwr_w_ray)] = np.ma.masked
        inds_ray = np.ma.array(inds_ray.compressed())
    pwr_w_ray = np.ma.array(pwr_w_ray.compressed())
    if pwr_w_ray.size == 0:
        return None, None, None, None

    pwr_w_mean = np.ma.mean(pwr_w_ray)

    # step 6: same as test 3 but with pwr_w_mean
    ind = np.ma.where(pwr_w_ray > pwr_w_mean*snr_thr)[0]
    if ind.size > 0:
        pwr_w_ray[ind] = np.ma.masked
    if get_noise_pos:
        inds_ray[np.ma.getmaskarray(pwr_w_ray)] = np.ma.masked
        inds_ray = np.ma.array(inds_ray.compressed())
    pwr_w_ray = np.ma.array(pwr_w_ray.compressed())
    if pwr_w_ray.size == 0:
        return None, None, None, None

    # step 7: running sum
    rs_wlen = int(round(500/npulses))
    if rs_wlen % 2 == 0:
        rs_wlen += 1
    half_rs_wlen = int((rs_wlen-1)/2)
    criterion = gammaincc(rs_wlen*npulses, 1.12*rs_wlen*npulses)

    valid_noise = False
    for _ in range(iterations):
        # check if there are sufficient noisy gates left
        if pwr_w_ray.compressed().size*npulses < ngates_min:
            return None, None, None, None

        # compute running sum
        pwr_mw_wind = rolling_window(pwr_w_ray, rs_wlen)
        rs = np.ma.sum(pwr_mw_wind, axis=-1)
        pwr_w_ray = pwr_w_ray[half_rs_wlen:-half_rs_wlen]

        # check if sums exceeding threshold are below the criteria
        inds_excess = np.ma.where(rs > 1.12*rs_wlen*np.ma.mean(pwr_w_ray))[0]
        if inds_excess.size/pwr_w_ray.size <= criterion:
            # The ratio of invalid running sums is below the criterion
            valid_noise = True
            break

        for ind in inds_excess:
            pwr_w_ray[ind-half_rs_wlen:ind+half_rs_wlen+1] = np.ma.masked
        if get_noise_pos:
            inds_ray = inds_ray[half_rs_wlen:-half_rs_wlen]
            inds_ray[np.ma.getmaskarray(pwr_w_ray)] = np.ma.masked
            inds_ray = np.ma.array(inds_ray.compressed())
        pwr_w_ray = np.ma.array(pwr_w_ray.compressed())
    if not valid_noise:
        return None, None, None, None

    if get_noise_pos:
        inds_ray = inds_ray.compressed()
    else:
        inds_ray = None

    return (
        np.ma.mean(pwr_w_ray), np.ma.var(pwr_w_ray), pwr_w_ray.size, inds_ray)


def get_ivic_pct(npulses, pct, z, prob_thr=1e-4):
    """
    Get the point clutter threshold (PCT)

    Parameters
    ----------
    npulses : 1D array
        array with the number of pulses in a ray
    pct : 1D array
        array with possible values of the PCT
    z : 1D array
        array of z values
    prob_thr : float
        Desired probability

    Returns
    -------
    pct_out : 1D array
        The PCT threshold corresponding to each number of pulses

    """
    delta_z = z[1]-z[0]
    pct_out = np.ma.masked_all(len(npulses))
    for i, npuls in enumerate(npulses):
        prob_func = np.ma.masked_all(len(pct))
        fact = 1./factorial(npuls-1)
        func_part = np.power(z, npuls-1)*np.exp(-z)
        for j, pct_val in enumerate(pct):
            gamma_inc = gammainc(npuls, z/pct_val)
            prob_func[j] = fact*np.sum(
                func_part*(2*gamma_inc-gamma_inc*gamma_inc))*delta_z
        delta_prob_func = np.abs(prob_thr-prob_func)
        ind = np.argmin(delta_prob_func)
        pct_out[i] = pct[ind]

    return pct_out


def get_ivic_flat_reg_var_max(npulses, y, x, flat_reg_wlen, ndBm=30.,
                              accuracy=1e-3):
    """
    Get the threshold for maximum local variance of noise [dB]

    Parameters
    ----------
    npulses : 1D array
        array with the number of pulses in a ray
    y : 1D array
        array of y values
    x : 1D array
        array of x values. These are the possible values of the threshold
    flat_reg_wlen : int
        the lenght of the flat region window in bins
    ndBm : float
        the mean noise power in dBm
    accuracy : float
        The desired accuracy of the threshold (difference with target
        probability)

    Returns
    -------
    var_thr : 1D array
        The thresholds corresponding to each number of pulses

    """
    delta_y = y[1]-y[0]
    delta_x = x[1]-x[0]

    wconst1 = flat_reg_wlen-1
    wconst2 = flat_reg_wlen-2+1/flat_reg_wlen
    wconst3 = flat_reg_wlen*flat_reg_wlen-3*flat_reg_wlen+5-3/flat_reg_wlen
    wconst4 = (
        (-2*flat_reg_wlen*flat_reg_wlen*flat_reg_wlen +
         12*flat_reg_wlen*flat_reg_wlen-22*flat_reg_wlen+12)/flat_reg_wlen)
    wconst5 = 4*(2-flat_reg_wlen-1/flat_reg_wlen)
    wconst6 = (
        (flat_reg_wlen-1)*(flat_reg_wlen-2)*(flat_reg_wlen-3)/flat_reg_wlen)

    var_thr = np.ma.masked_all(len(npulses))
    n = 1e-3*np.power(10., 0.1*ndBm)
    for i, npuls in enumerate(npulses):
        pdb_const = (
            np.power(npuls, npuls)*np.log(10.) /
            (10.*np.power(n, npuls)*factorial(npuls-1)))
        pdb_part2 = np.exp(-npuls/n*np.power(10., y/10.))
        pdb_part1 = np.power(10., y*npuls/10.)

        y2 = y*y
        y3 = y2*y
        y4 = y3*y

        pdb_1 = pdb_const*np.sum(y*pdb_part1*pdb_part2)*delta_y
        pdb_2 = pdb_const*np.sum(y2*pdb_part1*pdb_part2)*delta_y
        pdb_3 = pdb_const*np.sum(y3*pdb_part1*pdb_part2)*delta_y
        pdb_4 = pdb_const*np.sum(y4*pdb_part1*pdb_part2)*delta_y

        pdb_1_2 = pdb_1*pdb_1

        vardb_1 = wconst1*(pdb_2-pdb_1_2)
        vardb_2 = (
            pdb_4*wconst2+pdb_2*pdb_2*wconst3+pdb_2*pdb_1_2*wconst4 +
            pdb_1*pdb_3*wconst5+wconst6*pdb_1_2*pdb_1_2)

        vardb_1_2 = vardb_1*vardb_1

        alpha = vardb_1_2/(vardb_2-vardb_1_2)
        theta = (vardb_2-vardb_1_2)/vardb_1

        pdf = (
            np.power(x, alpha-1)*np.exp(-x/theta) /
            (np.power(theta, alpha)*gamma(alpha)))

        for ind in range(x.size):
            prob = np.sum(pdf[ind:])*delta_x
            delta_prob = np.abs(prob-0.01)
            # print('delta_prob: ', delta_prob, ' threshold: ', x[ind])
            if delta_prob < accuracy:
                var_thr[i] = x[ind]
                break

    return var_thr


def get_ivic_snr_thr(npulses, snr_thr, pfa_thr=1e-3):
    """
    Get the threshold for steps 3 and 6 of ivic

    Parameters
    ----------
    npulses : 1D array
        array with the number of pulses in a ray
    snr_thr : 1D array
        array with possible values of the snr threshold
    pfa_thr : float
        the desired probability of false alarm

    Returns
    -------
    snr_thr_out : 1D array
        The snr threshold corresponding to each number of pulses according to
        the desired probability of false alarm


    """
    snr_thr_out = np.ma.masked_all(len(npulses))
    for i, npuls in enumerate(npulses):
        pfa = gammaincc(npuls, snr_thr*npuls)
        delta_pfa = np.abs(pfa-pfa_thr)
        ind = np.argmin(delta_pfa)
        snr_thr_out[i] = snr_thr[ind]

    return snr_thr_out


def ivic_pct_table(npulses):
    """
    Get the point clutter threshold (PCT) of Ivic from a look up table.
    The thresholds are computed for between 3 and 78 pulses. If there
    number of pulses is beyond this range it throws an error

    Parameters
    ----------
    npulses : 1D array
        array with the number of pulses in a ray

    Returns
    -------
    pct_table : 1D array
        The PCT corresponding to each number of pulses

    """
    ind = np.where(np.logical_or(npulses > 78, npulses < 3))[0]
    if ind.size > 0:
        raise ValueError("Table valid for number of pulses between 3 and 78")

    ind_npulses = (npulses-3).astype(np.int)

    pct_table = np.array([
        56.97, 27.31, 17.37, 12.73, 10.12, 8.47, 7.34, 6.53, 5.91, 5.43, 5.04,
        4.72, 4.45, 4.23, 4.03, 3.87, 3.72, 3.59, 3.74, 3.37, 3.27, 3.19,
        3.11, 3.04, 2.97, 2.91, 2.85, 2.80, 2.75, 2.71, 2.66, 2.62, 2.59,
        2.55, 2.52, 2.49, 2.46, 2.43, 2.40, 2.37, 2.35, 2.33, 2.30, 2.28,
        2.26, 2.24, 2.22, 2.20, 2.19, 2.17, 2.15, 2.14, 2.12, 2.11, 2.09,
        2.08, 2.07, 2.05, 2.04, 2.03, 2.02, 2.01, 1.99, 1.98, 1.97, 1.96,
        1.95, 1.94, 1.93, 1.93, 1.92, 1.91, 1.90, 1.89, 1.88, 1.88])

    return pct_table[ind_npulses]


def ivic_flat_reg_var_max_table(npulses, flat_reg_wlen):
    """
    Get maximum variance of noise of Ivic from a look up table.

    Parameters
    ----------
    npulses : 1D array
        array with the number of pulses in a ray
    flat_reg_wlen : float
        The flat region window length in bins

    Returns
    -------
    flat_reg_var_max : 1D array
        The maximum variance threshold corresponding to each number of pulses

    """
    flat_reg_var_max = 0.99*np.log(flat_reg_wlen)-2.49
    flat_reg_var_max = npulses/15*flat_reg_var_max

    return flat_reg_var_max


def ivic_snr_thr_table(npulses):
    """
    Get the threshold for steps 3 and 6 of ivic from a look up table
    The thresholds are computed for between 3 and 200 pulses. If there
    number of pulses is beyond this range it throws an error

    Parameters
    ----------
    npulses : 1D array
        array with the number of pulses in a ray

    Returns
    -------
    snr_thr_out : 1D array
        The snr threshold corresponding to each number of pulses according to
        the desired probability of false alarm


    """
    ind = np.where(np.logical_or(npulses > 200, npulses < 3))[0]
    if ind.size > 0:
        raise ValueError("Table valid for number of pulses between 3 and 200")

    ind_npulses = (npulses-3).astype(np.int)

    snr_thr_table = np.array([
        3.74, 3.27, 2.96, 2.74, 2.58, 2.45, 2.35, 2.27, 2.19, 2.13, 2.08,
        2.03, 1.99, 1.95, 1.92, 1.89, 1.86, 1.84, 1.81, 1.79, 1.77, 1.75,
        1.73, 1.72, 1.70, 1.69, 1.67, 1.66, 1.65, 1.64, 1.63, 1.61, 1.60,
        1.60, 1.59, 1.58, 1.57, 1.56, 1.55, 1.55, 1.54, 1.53, 1.52, 1.52,
        1.51, 1.51, 1.50, 1.49, 1.49, 1.48, 1.48, 1.47, 1.47, 1.46, 1.46,
        1.46, 1.45, 1.45, 1.44, 1.44, 1.44, 1.43, 1.43, 1.42, 1.42, 1.42,
        1.41, 1.41, 1.41, 1.40, 1.40, 1.40, 1.40, 1.39, 1.39, 1.39, 1.38,
        1.38, 1.38, 1.38, 1.37, 1.37, 1.37, 1.37, 1.36, 1.36, 1.36, 1.36,
        1.36, 1.35, 1.35, 1.35, 1.35, 1.35, 1.34, 1.34, 1.34, 1.34, 1.34,
        1.33, 1.33, 1.33, 1.33, 1.33, 1.33, 1.32, 1.32, 1.32, 1.32, 1.32,
        1.32, 1.31, 1.31, 1.31, 1.31, 1.31, 1.31, 1.31, 1.30, 1.30, 1.30,
        1.30, 1.30, 1.30, 1.30, 1.30, 1.29, 1.29, 1.29, 1.29, 1.29, 1.29,
        1.29, 1.29, 1.29, 1.28, 1.28, 1.28, 1.28, 1.28, 1.28, 1.28, 1.28,
        1.28, 1.27, 1.27, 1.27, 1.27, 1.27, 1.27, 1.27, 1.27, 1.27, 1.27,
        1.27, 1.26, 1.26, 1.26, 1.26, 1.26, 1.26, 1.26, 1.26, 1.26, 1.26,
        1.26, 1.26, 1.25, 1.25, 1.25, 1.25, 1.25, 1.25, 1.25, 1.25, 1.25,
        1.25, 1.25, 1.25, 1.25, 1.24, 1.24, 1.24, 1.24, 1.24, 1.24, 1.24,
        1.24, 1.24, 1.24, 1.24, 1.24, 1.24, 1.24, 1.24, 1.23, 1.23, 1.23])

    return snr_thr_table[ind_npulses]
