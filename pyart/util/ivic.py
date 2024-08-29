"""
pyart.util.ivic
===============

Estimation of noise in a ray using Ivic (2013) method

.. autosummary::
    :toctree: generated/

    estimate_noise_ivic13
    get_ivic_pct
    _func_pct_int
    _func_pct
    get_ivic_flat_reg_var_max
    _func_flat_reg
    get_ivic_snr_thr
    ivic_pct_table
    ivic_flat_reg_var_max_table
    ivic_flat_reg_wind_len_table
    ivic_snr_thr_table


"""

from copy import deepcopy

import numpy as np
from scipy import integrate
from scipy.optimize import fsolve
from scipy.special import gammainc, gammaincc, gammainccinv, gammaln

from .radar_utils import ma_broadcast_to
from .sigmath import rolling_window


def estimate_noise_ivic13(
    pwr_w_ray,
    pct=3.270436,
    delay=2,
    flat_reg_wlen=32,
    flat_reg_var_max=0.439551,
    snr_thr=1.769572,
    npulses=30,
    ngates_min=800,
    ngates_final_min=200,
    ngates_median=10,
    iterations=10,
    get_noise_pos=False,
):
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
    ngates_final_min : int
        minimum number of gates that have to be left after the last step to
        consider the retrieval valid
    ngates_median : int
        number of consecutive gates above the median power that would result
        in them considered signal and removed
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
    half_flat_reg_wlen = int((flat_reg_wlen - 1) / 2)

    if get_noise_pos:
        inds_ray = np.ma.arange(pwr_w_ray.size)

    # step 1 remove gates with discontinuities in range
    pwr_w_ray_aux = pwr_w_ray[delay:-delay]
    pwr_w_ray_aux[
        np.logical_or(
            np.ma.greater(pwr_w_ray_aux, pct * pwr_w_ray[: -2 * delay]),
            np.ma.greater(pwr_w_ray_aux, pct * pwr_w_ray[2 * delay :]),
        )
    ] = np.ma.masked
    if get_noise_pos:
        inds_ray_aux = inds_ray[delay:-delay]
        inds_ray_aux[np.ma.getmaskarray(pwr_w_ray_aux)] = np.ma.masked
        inds_ray = np.ma.array(inds_ray_aux.compressed())
    pwr_w_ray = np.ma.array(pwr_w_ray_aux.compressed())

    if pwr_w_ray.size * npulses < ngates_min:
        return None, None, None, None

    # step 2 detect flat sections and estimate mean power of the grouped flat
    # sections. Take smallest power as intermediate power

    # compute moving variance (in dB)
    pwr_dB_ray = np.ma.log10(pwr_w_ray)
    pwr_dB_wind = rolling_window(pwr_dB_ray, flat_reg_wlen)
    pwr_dB_wind_mean = np.ma.mean(pwr_dB_wind, axis=-1)
    # add third dimension made of mean values of the window
    pwr_dB_wind_mean = ma_broadcast_to(
        np.expand_dims(pwr_dB_wind_mean, axis=-1),
        (pwr_dB_wind_mean.size, flat_reg_wlen),
    )
    pwr_dB_var = np.ma.sum(
        (pwr_dB_wind - pwr_dB_wind_mean) * (pwr_dB_wind - pwr_dB_wind_mean), axis=-1
    )

    # Compute intermediate power based on flat areas
    # mask non-flat regions
    pwr_w_ray_aux = deepcopy(pwr_w_ray[half_flat_reg_wlen:-half_flat_reg_wlen])
    inds_non_flat = np.ma.where(pwr_dB_var > flat_reg_var_max)[0]
    for ind in inds_non_flat:
        pwr_w_ray_aux[ind - half_flat_reg_wlen : ind + half_flat_reg_wlen + 1] = (
            np.ma.masked
        )

    # group consecutive gates and get the minimum mean power
    mask = np.ma.getmaskarray(pwr_w_ray_aux)
    ind = np.ma.where(mask is False)[0]
    cons_list = np.split(ind, np.where(np.diff(ind) != 1)[0] + 1)
    pwr_w_int = None
    for cons in cons_list:
        pwr_w_mean = np.ma.mean(pwr_w_ray_aux[cons])
        if pwr_w_int is None:
            pwr_w_int = pwr_w_mean
        elif pwr_w_int > pwr_w_mean:
            pwr_w_int = pwr_w_mean

    if pwr_w_int is None:
        return None, None, None, None

    # step 3 remove gates exceeding a threshold based on pwr_int
    ind = np.ma.where(pwr_w_ray > pwr_w_int * snr_thr)[0]
    if ind.size > 0:
        pwr_w_ray[ind] = np.ma.masked
    if get_noise_pos:
        inds_ray[np.ma.getmaskarray(pwr_w_ray)] = np.ma.masked
        inds_ray = np.ma.array(inds_ray.compressed())
    pwr_w_ray = np.ma.array(pwr_w_ray.compressed())
    if pwr_w_ray.size * npulses < ngates_min:
        return None, None, None, None

    # step 4 & 5: detect ngates_median or more consecutive gates with power
    # larger than the median. Compute the mean of the remaining data
    ind = np.ma.where(pwr_w_ray > np.ma.median(pwr_w_ray))[0]
    cons_list = np.split(ind, np.where(np.diff(ind) != 1)[0] + 1)
    for cons in cons_list:
        if len(cons) > ngates_median:
            pwr_w_ray[cons] = np.ma.masked
    if get_noise_pos:
        inds_ray[np.ma.getmaskarray(pwr_w_ray)] = np.ma.masked
        inds_ray = np.ma.array(inds_ray.compressed())
    pwr_w_ray = np.ma.array(pwr_w_ray.compressed())
    if pwr_w_ray.size * npulses < ngates_min:
        return None, None, None, None

    pwr_w_mean = np.ma.mean(pwr_w_ray)

    # step 6: same as test 3 but with pwr_w_mean
    ind = np.ma.where(pwr_w_ray > pwr_w_mean * snr_thr)[0]
    if ind.size > 0:
        pwr_w_ray[ind] = np.ma.masked
    if get_noise_pos:
        inds_ray[np.ma.getmaskarray(pwr_w_ray)] = np.ma.masked
        inds_ray = np.ma.array(inds_ray.compressed())
    pwr_w_ray = np.ma.array(pwr_w_ray.compressed())
    if pwr_w_ray.size * npulses < ngates_min:
        return None, None, None, None

    # step 7: running sum
    rs_wlen = int(round(500 / npulses))
    if rs_wlen % 2 == 0:
        rs_wlen += 1
    half_rs_wlen = int((rs_wlen - 1) / 2)
    criterion = gammaincc(rs_wlen * npulses, 1.12 * rs_wlen * npulses)

    for _ in range(iterations):
        # check if there are sufficient noisy gates left
        if pwr_w_ray.compressed().size * npulses < ngates_final_min:
            # exit with no estimate if after first iteration there are less
            # than ngates_final_min samples left
            return None, None, None, None

        # compute running sum
        pwr_mw_wind = rolling_window(pwr_w_ray, rs_wlen)
        rs = np.ma.sum(pwr_mw_wind, axis=-1)
        pwr_w_ray = pwr_w_ray[half_rs_wlen:-half_rs_wlen]

        # check if sums exceeding threshold are below the criteria
        inds_excess = np.ma.where(rs > 1.12 * rs_wlen * np.ma.mean(pwr_w_ray))[0]
        if inds_excess.size / pwr_w_ray.size <= criterion:
            # The ratio of invalid running sums is below the criterion
            break

        for ind in inds_excess:
            pwr_w_ray[ind - half_rs_wlen : ind + half_rs_wlen + 1] = np.ma.masked
        if get_noise_pos:
            inds_ray = inds_ray[half_rs_wlen:-half_rs_wlen]
            inds_ray[np.ma.getmaskarray(pwr_w_ray)] = np.ma.masked
            inds_ray = np.ma.array(inds_ray.compressed())
        pwr_w_ray = np.ma.array(pwr_w_ray.compressed())

    if get_noise_pos:
        inds_ray = inds_ray.compressed()
    else:
        inds_ray = None

    return (np.ma.mean(pwr_w_ray), np.ma.var(pwr_w_ray), pwr_w_ray.size, inds_ray)


def get_ivic_pct(npulses, pct=1, prob_thr=1e-4):
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
    x = np.arange(0.0001, 400.0, 0.01)
    pct_out = np.ma.masked_all(len(npulses))
    for i, npuls in enumerate(npulses):
        # default tolerance 1.49012e-8
        f = fsolve(_func_pct_int, pct, args=(x, npuls, prob_thr), xtol=1e-13)
        pct_out[i] = f

    return pct_out


def _func_pct_int(pct, x, npuls, prob_thr):
    """
    Function to solve for point clutter target

    Parameters
    ----------
    x : 1D array
        array with the variable in the integral
    pct : float
        The point clutter threshold
    npuls : float
        the number of pulses per ray

    Returns
    -------
    f : 1D array
        function f(x)

    """
    # default tolerance 1.49e-8
    f, _ = integrate.quad(
        _func_pct, 0.0, 400.0, args=(pct, npuls), epsabs=1e-10, epsrel=1e-06
    )

    return f - prob_thr


def _func_pct(x, pct, npuls):
    """
    Function to solve for point clutter target

    Parameters
    ----------
    x : 1D array
        array with the variable in the integral
    pct : float
        The point clutter threshold
    npuls : float
        the number of pulses per ray

    Returns
    -------
    f : 1D array
        function f(x)

    """
    log_e = np.log10(np.e)
    f = 2.0 * np.power(
        10.0,
        np.log10(x) * (npuls - 1)
        - log_e * (x + gammaln(npuls))
        + np.log10(gammainc(npuls, x / pct))
        - log_e * (gammaln(npuls + 1) + x / pct)
        + np.log10(x / pct) * npuls,
    ) - np.power(
        10.0,
        np.log10(x) * (npuls - 1)
        - log_e * (x + gammaln(npuls))
        + 2.0
        * (
            np.log10(gammainc(npuls, x / pct))
            - log_e * (gammaln(npuls + 1) + (x / pct))
            + np.log10(x / pct) * npuls
        ),
    )
    return f


def get_ivic_flat_reg_var_max(npulses, flat_reg_wlen, n=40.0, prob_thr=0.01):
    """
    Get the threshold for maximum local variance of noise [dB]

    Parameters
    ----------
    npulses : 1D array
        array with the number of pulses in a ray
    flat_reg_wlen : int
        the length of the flat region window in bins
    n : float
        the mean noise power
    prob_thr : float
        Probably of falsely detecting noise-only flat section as contaminated
        with signal

    Returns
    -------
    var_thr : 1D array
        The thresholds corresponding to each number of pulses

    """
    y = np.arange(0.0001, 1e4, 0.01)

    log_e = np.log10(np.e)
    log_log = np.log10(np.log(10.0))

    wconst1 = flat_reg_wlen - 1
    wconst2 = flat_reg_wlen - 2 + 1 / flat_reg_wlen
    wconst3 = flat_reg_wlen * flat_reg_wlen - 3 * flat_reg_wlen + 5 - 3 / flat_reg_wlen
    wconst4 = (
        -2 * flat_reg_wlen * flat_reg_wlen * flat_reg_wlen
        + 12 * flat_reg_wlen * flat_reg_wlen
        - 22 * flat_reg_wlen
        + 12
    ) / flat_reg_wlen
    wconst5 = 4 * (2 - flat_reg_wlen - 1 / flat_reg_wlen)
    wconst6 = (
        (flat_reg_wlen - 1) * (flat_reg_wlen - 2) * (flat_reg_wlen - 3) / flat_reg_wlen
    )

    var_thr = np.ma.masked_all(len(npulses))
    for i, npuls in enumerate(npulses):
        pdb_part1 = np.power(
            10.0, np.log10(npuls / n) * npuls + log_log - gammaln(npuls) * log_e
        )
        pdb = np.empty(4)
        for j in range(pdb.size):
            k = j + 1

            # find first 0 crossing
            f_y = _func_flat_reg(y, k, npuls, n)
            ind = np.where(f_y != 0.0)[0]
            cons_list = np.split(ind, np.where(np.diff(ind) != 1)[0] + 1)
            ind = cons_list[0][-1]

            # solve integral
            pdb[j] = (
                pdb_part1
                * integrate.quad(_func_flat_reg, 0.0, y[ind], args=(k, npuls, n))[0]
            )

        pdb_1_2 = pdb[0] * pdb[0]

        vardb_1 = wconst1 * (pdb[1] - pdb_1_2)
        vardb_2 = (
            pdb[3] * wconst2
            + pdb[1] * pdb[1] * wconst3
            + pdb[1] * pdb_1_2 * wconst4
            + pdb[0] * pdb[2] * wconst5
            + wconst6 * pdb_1_2 * pdb_1_2
        )

        vardb_1_2 = vardb_1 * vardb_1

        alpha = vardb_1_2 / (vardb_2 - vardb_1_2)
        theta = (vardb_2 - vardb_1_2) / vardb_1

        var_thr[i] = gammainccinv(alpha, prob_thr) * theta

    return var_thr


def _func_flat_reg(y, k, npuls, n):
    """
    Function to solve for flat region variance max. Derived from eq. B5 in
    paper

    Parameters
    ----------
    y : 1D array
        array with the variable in the integral
    k : int
        The order
    npuls : int
        the number of pulses per ray
    n : float
        noise power

    Returns
    -------
    f : 1D array
        the f(y) function

    """
    return np.power(
        10.0,
        k * np.log10(y) + y * npuls - np.power(10.0, y) * npuls / n * np.log10(np.e),
    )


def get_ivic_snr_thr(npulses, pfa_thr=1e-3):
    """
    Get the threshold for steps 3 and 6 of ivic

    Parameters
    ----------
    npulses : 1D array
        array with the number of pulses in a ray
    pfa_thr : float
        the desired probability of false alarm

    Returns
    -------
    snr_thr : 1D array
        The snr threshold corresponding to each number of pulses according to
        the desired probability of false alarm


    """
    return gammainccinv(npulses, pfa_thr) / npulses


def ivic_pct_table(npulses):
    """
    Get the point clutter threshold (PCT) of Ivic from a look up table.
    The thresholds are computed for between 1 and 200 pulses. If there
    number of pulses is beyond this range it throws an error. The table is
    computed for a PFA = 1e-4.
    WARNING: This thresholds were computed by I. Ivic in Matlab. the values
    differ from those computed by get_ivic_pct, particularly for small values
    of npulses.

    Parameters
    ----------
    npulses : 1D array
        array with the number of pulses in a ray

    Returns
    -------
    pct_table : 1D array
        The PCT corresponding to each number of pulses

    """
    ind = np.where(np.logical_or(npulses > 200, npulses < 1))[0]
    if ind.size > 0:
        raise ValueError("Table valid for number of pulses between 1 and 200")

    ind_npulses = (npulses - 1).astype(np.int)

    pct_table = np.array(
        [
            0.0,
            243.596101,
            56.972570,
            27.309237,
            17.372310,
            12.729620,
            10.119393,
            8.470525,
            7.342833,
            6.526113,
            5.908569,
            5.425731,
            5.038014,
            4.719832,
            4.453971,
            4.228437,
            4.034628,
            3.866220,
            3.718462,
            3.587719,
            3.471161,
            3.366555,
            3.272113,
            3.186389,
            3.108200,
            3.036567,
            2.970678,
            2.909848,
            2.853498,
            2.801135,
            2.752336,
            2.706737,
            2.664023,
            2.623917,
            2.586180,
            2.550599,
            2.516987,
            2.485179,
            2.455026,
            2.426399,
            2.399179,
            2.373260,
            2.348546,
            2.324953,
            2.302401,
            2.280820,
            2.260146,
            2.240321,
            2.221289,
            2.203003,
            2.185416,
            2.168489,
            2.152181,
            2.136459,
            2.121289,
            2.106642,
            2.092489,
            2.078804,
            2.065563,
            2.052745,
            2.040327,
            2.028291,
            2.016617,
            2.005289,
            1.994292,
            1.983609,
            1.973227,
            1.963132,
            1.953312,
            1.943755,
            1.934451,
            1.925388,
            1.916557,
            1.907949,
            1.899554,
            1.891365,
            1.883373,
            1.875571,
            1.867951,
            1.860508,
            1.853235,
            1.846125,
            1.839173,
            1.832373,
            1.825720,
            1.819209,
            1.812835,
            1.806593,
            1.800480,
            1.794491,
            1.788622,
            1.782869,
            1.777229,
            1.771698,
            1.766273,
            1.760950,
            1.755727,
            1.750600,
            1.745567,
            1.740624,
            1.735771,
            1.731003,
            1.726318,
            1.721715,
            1.717190,
            1.712742,
            1.708369,
            1.704069,
            1.699839,
            1.695678,
            1.691584,
            1.687556,
            1.683591,
            1.679689,
            1.675847,
            1.672064,
            1.668338,
            1.664669,
            1.661055,
            1.657494,
            1.653986,
            1.650528,
            1.647121,
            1.643762,
            1.640450,
            1.637186,
            1.633966,
            1.630792,
            1.627660,
            1.624572,
            1.621524,
            1.618518,
            1.615551,
            1.612624,
            1.609734,
            1.606882,
            1.604067,
            1.601287,
            1.598542,
            1.595832,
            1.593156,
            1.590512,
            1.587901,
            1.585322,
            1.582774,
            1.580256,
            1.577769,
            1.575311,
            1.572881,
            1.570480,
            1.568106,
            1.565760,
            1.563441,
            1.561148,
            1.558880,
            1.556638,
            1.554421,
            1.552228,
            1.550059,
            1.547913,
            1.545791,
            1.543691,
            1.541614,
            1.539559,
            1.537525,
            1.535513,
            1.533521,
            1.531550,
            1.529599,
            1.527668,
            1.525757,
            1.523864,
            1.521991,
            1.520136,
            1.518299,
            1.516481,
            1.514680,
            1.512897,
            1.511130,
            1.509381,
            1.507648,
            1.505932,
            1.504232,
            1.502548,
            1.500879,
            1.499226,
            1.497588,
            1.495965,
            1.494356,
            1.492763,
            1.491183,
            1.489618,
            1.488066,
            1.486528,
            1.485004,
            1.483493,
            1.481995,
            1.480511,
            1.479038,
            1.477579,
        ]
    )

    return pct_table[ind_npulses]


def ivic_flat_reg_var_max_table(npulses):
    """
    Get maximum variance of noise of Ivic from a look up table.
    These values are computed for a prescribed flat region window length that
    depends on the number of pulses. The window length as a function of number
    of pulses can be found using function ivic_flat_reg_wind_len

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
    ind = np.where(np.logical_or(npulses > 200, npulses < 1))[0]
    if ind.size > 0:
        raise ValueError("Table valid for number of pulses between 1 and 200")

    ind_npulses = (npulses - 1).astype(np.int)

    flat_reg_var_max = np.array(
        [
            0.0,
            12.501880,
            9.066873,
            6.978578,
            5.602964,
            3.184561,
            2.709687,
            2.352218,
            2.075309,
            1.855436,
            1.677084,
            1.529734,
            1.406049,
            1.300803,
            0.683347,
            0.625970,
            0.592077,
            0.560521,
            0.531522,
            0.505351,
            0.481602,
            0.459942,
            0.439551,
            0.420858,
            0.403674,
            0.387825,
            0.373164,
            0.359562,
            0.346910,
            0.335113,
            0.324087,
            0.313760,
            0.304067,
            0.292595,
            0.284680,
            0.277142,
            0.269806,
            0.262747,
            0.256040,
            0.249657,
            0.243574,
            0.237771,
            0.232228,
            0.226885,
            0.221765,
            0.216871,
            0.212188,
            0.207704,
            0.203405,
            0.199281,
            0.195321,
            0.191515,
            0.187856,
            0.184334,
            0.180942,
            0.177673,
            0.174515,
            0.171464,
            0.168519,
            0.165675,
            0.162927,
            0.160270,
            0.157700,
            0.155213,
            0.152805,
            0.150471,
            0.148210,
            0.146016,
            0.143889,
            0.141824,
            0.139819,
            0.137871,
            0.135979,
            0.134139,
            0.132349,
            0.130609,
            0.128915,
            0.127265,
            0.125659,
            0.124094,
            0.122569,
            0.121083,
            0.119633,
            0.118219,
            0.116839,
            0.115493,
            0.114178,
            0.112894,
            0.111640,
            0.110415,
            0.109218,
            0.108047,
            0.106903,
            0.105783,
            0.104688,
            0.103617,
            0.102568,
            0.101542,
            0.100537,
            0.099552,
            0.098588,
            0.097644,
            0.096718,
            0.095811,
            0.094921,
            0.094049,
            0.093194,
            0.092355,
            0.091059,
            0.090263,
            0.089482,
            0.088715,
            0.087962,
            0.087223,
            0.086497,
            0.085783,
            0.085083,
            0.084394,
            0.083718,
            0.083053,
            0.082399,
            0.081757,
            0.081125,
            0.080504,
            0.079893,
            0.079292,
            0.078700,
            0.078119,
            0.077546,
            0.076983,
            0.076428,
            0.075882,
            0.075345,
            0.074816,
            0.074295,
            0.073781,
            0.073276,
            0.072778,
            0.072287,
            0.071804,
            0.071327,
            0.070858,
            0.070395,
            0.069939,
            0.069489,
            0.069046,
            0.068609,
            0.068178,
            0.067753,
            0.067334,
            0.066920,
            0.066512,
            0.066110,
            0.065713,
            0.065321,
            0.064934,
            0.064553,
            0.064176,
            0.063804,
            0.063437,
            0.063075,
            0.062717,
            0.062364,
            0.062015,
            0.061671,
            0.061331,
            0.060995,
            0.060663,
            0.060335,
            0.060011,
            0.059691,
            0.059375,
            0.059063,
            0.058754,
            0.058449,
            0.058148,
            0.057850,
            0.057555,
            0.057262,
            0.056972,
            0.056686,
            0.056403,
            0.056123,
            0.055846,
            0.055573,
            0.055302,
            0.055034,
            0.054770,
            0.054508,
            0.054249,
            0.053993,
            0.053740,
            0.053489,
            0.053241,
            0.052996,
            0.052753,
            0.052513,
            0.052276,
            0.052040,
            0.051808,
        ]
    )

    return flat_reg_var_max[ind_npulses]


def ivic_flat_reg_wind_len_table(npulses):
    """
    Get the size of the flat region window length as a function of the number
    of pulses.

    Parameters
    ----------
    npulses : 1D array
        array with the number of pulses in a ray

    Returns
    -------
    flat_reg_var_max : 1D array
        The maximum variance threshold corresponding to each number of pulses

    """
    ind = np.where(np.logical_or(npulses > 200, npulses < 1))[0]
    if ind.size > 0:
        raise ValueError("Table valid for number of pulses between 1 and 200")

    ind_npulses = (npulses - 1).astype(np.int)

    flat_reg_wlen = np.zeros(200, dtype=np.int)
    flat_reg_wlen[0:5] += 100
    flat_reg_wlen[5:14] += 64
    flat_reg_wlen[14:] += 32

    return flat_reg_wlen[ind_npulses]


def ivic_snr_thr_table(npulses):
    """
    Get the threshold for steps 3 and 6 of ivic from a look up table
    The thresholds are computed for between 1 and 200 pulses. If there
    number of pulses is beyond this range it throws an error. The table is
    computed for a PFA = 1e-3.

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
    ind = np.where(np.logical_or(npulses > 200, npulses < 1))[0]
    if ind.size > 0:
        raise ValueError("Table valid for number of pulses between 1 and 200")

    ind_npulses = (npulses - 1).astype(np.int)

    snr_thr_table = np.array(
        [
            0.0,
            4.616707,
            3.742957,
            3.265560,
            2.958830,
            2.742458,
            2.580234,
            2.453272,
            2.350689,
            2.265737,
            2.193997,
            2.132442,
            2.078922,
            2.031867,
            1.990102,
            1.952726,
            1.919036,
            1.888477,
            1.860602,
            1.835049,
            1.811518,
            1.789762,
            1.769572,
            1.750774,
            1.733216,
            1.716772,
            1.701330,
            1.686795,
            1.673083,
            1.660121,
            1.647843,
            1.636193,
            1.625119,
            1.614578,
            1.604528,
            1.594932,
            1.585759,
            1.576978,
            1.568564,
            1.560490,
            1.552737,
            1.545282,
            1.538108,
            1.531199,
            1.524537,
            1.518110,
            1.511904,
            1.505906,
            1.500106,
            1.494493,
            1.489057,
            1.483789,
            1.478682,
            1.473726,
            1.468916,
            1.464243,
            1.459703,
            1.455287,
            1.450992,
            1.446812,
            1.442742,
            1.438776,
            1.434912,
            1.431144,
            1.427469,
            1.423883,
            1.420383,
            1.416964,
            1.413625,
            1.410363,
            1.407173,
            1.404054,
            1.401004,
            1.398019,
            1.395097,
            1.392237,
            1.389436,
            1.386692,
            1.384004,
            1.381369,
            1.378785,
            1.376252,
            1.373767,
            1.371330,
            1.368937,
            1.366589,
            1.364284,
            1.362021,
            1.359798,
            1.357614,
            1.355468,
            1.353359,
            1.351286,
            1.349248,
            1.347244,
            1.345273,
            1.343334,
            1.341427,
            1.339550,
            1.337703,
            1.335884,
            1.334094,
            1.332332,
            1.330596,
            1.328886,
            1.327202,
            1.325543,
            1.323908,
            1.322296,
            1.320708,
            1.319142,
            1.317599,
            1.316077,
            1.314576,
            1.313096,
            1.311635,
            1.310195,
            1.308774,
            1.307371,
            1.305987,
            1.304621,
            1.303273,
            1.301942,
            1.300627,
            1.299330,
            1.298048,
            1.296783,
            1.295533,
            1.294298,
            1.293078,
            1.291873,
            1.290682,
            1.289506,
            1.288343,
            1.287193,
            1.286057,
            1.284934,
            1.283824,
            1.282726,
            1.281641,
            1.280568,
            1.279506,
            1.278456,
            1.277418,
            1.276391,
            1.275375,
            1.274370,
            1.273375,
            1.272391,
            1.271417,
            1.270454,
            1.269500,
            1.268556,
            1.267622,
            1.266697,
            1.265781,
            1.264875,
            1.263978,
            1.263089,
            1.262209,
            1.261338,
            1.260475,
            1.259620,
            1.258774,
            1.257935,
            1.257105,
            1.256282,
            1.255467,
            1.254659,
            1.253859,
            1.253066,
            1.252281,
            1.251502,
            1.250731,
            1.249966,
            1.249209,
            1.248458,
            1.247713,
            1.246975,
            1.246244,
            1.245518,
            1.244799,
            1.244086,
            1.243380,
            1.242679,
            1.241984,
            1.241295,
            1.240611,
            1.239933,
            1.239261,
            1.238594,
            1.237933,
            1.237277,
            1.236626,
            1.235981,
            1.235341,
            1.234705,
            1.234075,
            1.233450,
            1.232829,
        ]
    )

    return snr_thr_table[ind_npulses]
