"""
pyart.util.sigmath
==================

Function for mathematical, signal processing and numerical routines.

.. autosummary::
    :toctree: generated/

    angular_texture_2d
    grid_texture_2d
    rolling_window
    texture
    texture_along_ray
    compute_nse
    compute_corr
    compute_mse

"""

from warnings import warn

import numpy as np
from scipy import ndimage, signal

from ..config import get_fillvalue
from .radar_utils import ma_broadcast_to


def angular_texture_2d(image, N, interval):
    """
    Compute the angular texture of an image. Uses convolutions
    in order to speed up texture calculation by a factor of ~50
    compared to using ndimage.generic_filter.

    Parameters
    ----------
    image : 2D array of floats
        The array containing the velocities in which to calculate
        texture from.
    N : int
        This is the window size for calculating texture. The texture will be
        calculated from an N by N window centered around the gate.
    interval : float
        The absolute value of the maximum velocity. In conversion to
        radial coordinates, pi will be defined to be interval
        and -pi will be -interval. It is recommended that interval be
        set to the Nyquist velocity.

    Returns
    -------
    std_dev : float array
        Texture of the radial velocity field.

    """
    # transform distribution from original interval to [-pi, pi]
    interval_max = interval
    interval_min = -interval
    half_width = (interval_max - interval_min) / 2.
    center = interval_min + half_width

    # Calculate parameters needed for angular std. dev
    im = (np.asarray(image) - center) / (half_width) * np.pi
    x = np.cos(im)
    y = np.sin(im)

    # Calculate convolution
    kernel = np.ones((N, N))
    xs = signal.convolve2d(x, kernel, mode="same", boundary="symm")
    ys = signal.convolve2d(y, kernel, mode="same", boundary="symm")
    ns = N**2

    # Calculate norm over specified window
    xmean = xs / ns
    ymean = ys / ns
    norm = np.sqrt(xmean**2 + ymean**2)
    std_dev = np.sqrt(-2 * np.log(norm)) * (half_width) / np.pi
    return std_dev


def grid_texture_2d(field, xwind=7, ywind=7):
    """
    Compute the local standard deviation of a gridded field

    Parameters
    ----------
    field : ndarray 2D
        The field over which the grid is computed in (y, x) coordinates
    xwind, ywind : int
        This is the window size in each dimension for calculating the texture.
        The texture will be calculated from a xwind by ywind window centered
        around the gate.

    Returns
    -------
    win_std : float array
        Texture of the field.

    """
    win_mean = ndimage.uniform_filter(field, (ywind, xwind))
    win_sqr_mean = ndimage.uniform_filter(field**2, (ywind, xwind))
    win_var = win_sqr_mean - win_mean**2
    ind = np.where(win_var < 0.)
    if ind[0].size > 0:
        warn('field variance contains ' + str(ind[0].size) +
             ' pixels with negative variance')
    return np.sqrt(win_var)


def rolling_window(a, window):
    """ create a rolling window object for application of functions
    eg: result=np.ma.std(array, 11), 1)"""
    # create a rolling window with the data
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1], )
    data_wind = np.lib.stride_tricks.as_strided(
        a, shape=shape, strides=strides)

    # create a rolling window with the mask
    mask = np.ma.getmaskarray(a)
    shape = mask.shape[:-1] + (mask.shape[-1] - window + 1, window)
    strides = mask.strides + (mask.strides[-1], )
    mask_wind = np.lib.stride_tricks.as_strided(
        mask, shape=shape, strides=strides)

    # masked rolled data
    data_wind = np.ma.masked_where(mask_wind, data_wind)

    return data_wind


def texture(radar, var):
    """ Determine a texture field using an 11pt stdev
    texarray=texture(pyradarobj, field). """
    fld = radar.fields[var]['data']
    print(fld.shape)
    tex = np.ma.zeros(fld.shape)
    for timestep in range(tex.shape[0]):
        ray = np.ma.std(rolling_window(fld[timestep, :], 11), 1)
        tex[timestep, 5:-5] = ray
        tex[timestep, 0:4] = np.ones(4) * ray[0]
        tex[timestep, -5:] = np.ones(5) * ray[-1]
    return tex


def texture_along_ray(radar, var, wind_size=7):
    """
    Compute field texture along ray using a user specified
    window size.

    Parameters
    ----------
    radar : radar object
        The radar object where the field is.
    var : str
        Name of the field which texture has to be computed.
    wind_size : int, optional
        Optional. Size of the rolling window used.

    Returns
    -------
    tex : radar field
        The texture of the specified field.

    """
    half_wind = int((wind_size - 1) / 2)
    fld = radar.fields[var]['data']
    tex = np.ma.zeros(fld.shape)
    tex[:] = np.ma.masked
    tex.set_fill_value(get_fillvalue())

    tex_aux = np.ma.std(rolling_window(fld, wind_size), -1)
    tex[:, half_wind:-half_wind] = tex_aux
    tex[:, 0:half_wind] = ma_broadcast_to(
        tex_aux[:, 0].reshape(tex.shape[0], 1), (tex.shape[0], half_wind))
    tex[:, -half_wind:] = ma_broadcast_to(
        tex_aux[:, -1].reshape(tex.shape[0], 1), (tex.shape[0], half_wind))

    return tex


def compute_nse(obs_data, mod_data):
    """
    Computes the Nash-Sutcliff model efficiency coefficient between observed
    and modeled profiles

    Parameters
    ----------
    obs_data, mod_data : array of floats
        The vectors on which to compute the nash

    Returns
    -------
    nash_coeff : float or None
        The Nash coefficient if it could be computed
        None otherwise

    """
    mask_obs_data = np.ma.getmaskarray(obs_data)
    mask_mod_data = np.ma.getmaskarray(mod_data)
    mask = np.logical_or(mask_obs_data, mask_mod_data)
    nvalid = np.sum(np.logical_not(mask), dtype=int)
    if nvalid == 0:
        return None
    obs_data_ma = np.ma.masked_where(mask, obs_data)
    mod_data_ma = np.ma.masked_where(mask, mod_data)

    sum_obs = np.ma.sum(obs_data_ma)
    sum_obs2 = np.ma.sum(obs_data_ma * obs_data_ma)
    diff_data = obs_data_ma - mod_data_ma
    sum_diff2 = np.ma.sum(diff_data * diff_data)

    denominator = sum_obs2 - sum_obs * sum_obs / nvalid
    if denominator <= 0.:
        return None
    return 1 - sum_diff2 / denominator


def compute_corr(vec1, vec2):
    """
    Computes the Pearson correlation coefficient between two arrays

    Parameters
    ----------
    vec1, vec2 : array of floats
        The vectors on which to compute the correlation coefficient

    Returns
    -------
    corr : float or None
        The correlation coefficient

    """
    mask_vec1 = np.ma.getmaskarray(vec1)
    mask_vec2 = np.ma.getmaskarray(vec2)
    mask = np.logical_or(mask_vec1, mask_vec2)
    nvalid = np.sum(np.logical_not(mask), dtype=int)
    if nvalid == 0:
        return None
    vec1_ma = np.ma.masked_where(mask, vec1)
    vec2_ma = np.ma.masked_where(mask, vec2)

    corr = np.ma.sum(vec1_ma * vec2_ma)
    sum_vec1 = np.ma.sum(vec1_ma)
    sum_vec2 = np.ma.sum(vec2_ma)
    sum2_vec1 = np.ma.sum(vec1_ma * vec1_ma)
    sum2_vec2 = np.ma.sum(vec2_ma * vec2_ma)

    corr = (
        (corr - sum_vec1 * sum_vec2 / nvalid)
        / np.sqrt(sum2_vec1 - sum_vec1 * sum_vec1 / nvalid)
        / np.sqrt(sum2_vec2 - sum_vec2 * sum_vec2 / nvalid))
    return corr


def compute_mse(arr1, arr2):
    """
    Computes the mean square error between two arrays

    Parameters
    ----------
    vec1, vec2 : array of floats
        The vectors on which to compute the correlation coefficient

    Returns
    -------
    corr : float or None
        The correlation coefficient

    """
    mask_arr1 = np.ma.getmaskarray(arr1)
    mask_arr2 = np.ma.getmaskarray(arr2)
    mask = np.logical_or(mask_arr1, mask_arr2)
    nvalid = np.sum(np.logical_not(mask), dtype=int)
    if nvalid == 0:
        return None
    arr1_ma = np.ma.masked_where(mask, arr1)
    arr2_ma = np.ma.masked_where(mask, arr2)

    diff = arr2_ma - arr1_ma
    return np.ma.sum(diff * diff) / nvalid
