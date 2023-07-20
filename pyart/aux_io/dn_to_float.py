"""
pyart.aux_io.metranet_reader
============================

Set of scales that convert DN to float values

.. autosummary::
    :toctree: generated/

    float_mapping_p
    float_mapping_m
    nyquist_vel

"""
import sys

import numpy as np


# fix for python3
if sys.version_info[0] == 3:
    def xrange(i):
        return range(i)

SCALETYPE_LINEAR = 1
SCALETYPE_LOG = 2


def float_mapping_p(moment, time, radar, nyquist_vel=None):
    """
    Converts DN to their float equivalent for old p files

    Parameters
    ----------
    moment : numpy array or numpy masked array
        array that contains the DN for a given moment
    time: timestamp in UNIX format
        timestamp at which the data was recorded
    radar : char
        the radar which recorded the data
    nyquist_vel : float
        the nyquist velocity for this particular ray, only needed if moment
        is radial velocity or spectral width
    Returns
    -------
    ret_data : numpy array or numpy masked array
        Array containing the moment data in float format (physical units)
    """
    if moment in ('ZH', 'ZV', 'ZHC', 'ZVC'):
        prd_data_level = np.fromiter(xrange(256), dtype=np.float32) / 2. - 32.
        prd_data_level[0] = np.nan
    elif moment == 'ZDR':
        prd_data_level = (
            (np.fromiter(xrange(256), dtype=np.float32) + 1) /
            16.1259842 - 7.9375)
        prd_data_level[0] = np.nan
    elif moment == 'RHO':
        if ((time > 1341619200) or
                (time > 1335484800 and
                 (radar == ord('D') or
                  radar == ord('L')))):
            # logaritmic scale
            prd_data_level = (
                1.003 - 10.**(-np.fromiter(xrange(256), dtype=np.float32) * 0.01))
        else:
            # linear scale (old data)
            prd_data_level = (
                np.fromiter(xrange(256), dtype=np.float32) / 255.)

        prd_data_level[0] = np.nan
    elif moment == 'PHI':
        prd_data_level = ((np.fromiter(
            xrange(256 * 256), dtype=np.float32) - 32768) / 32767. * 180.)
        prd_data_level[0] = np.nan
    elif moment == 'VEL':
        prd_data_level = (
            (np.fromiter(xrange(256), dtype=np.float32) - 128) / 127. *
            nyquist_vel)
        prd_data_level[0] = np.nan
    elif moment == 'WID':
        prd_data_level = (np.fromiter(
            xrange(256), dtype=np.float32) / 255. * 2. * nyquist_vel)
        prd_data_level[0] = np.nan
    elif moment == 'MPH':
        prd_data_level = ((np.fromiter(
            xrange(256), dtype=np.float32) - 128) / 127. * 180.)
    elif moment in ('ST1', 'ST2', 'WBN'):
        prd_data_level = (np.fromiter(
            xrange(256), dtype=np.float32) / 10.)
    elif moment == "CLT":
        prd_data_level = np.fromiter(xrange(256), dtype=np.float32)
    return prd_data_level


def float_mapping_m(moment, momhead, time, radar, nyquist_vel=None,
                    upperlimit=False):
    """
    Returns a mapping that converts DN to their float equivalent for new m
    files

    Parameters
    ----------
    moment : string
        name of the moment
    momhead : dict
        moment header that contains the conversion rules
    time: timestamp in UNIX format
        timestamp at which the data was recorded
    radar : char
        the radar which recorded the data
    nyquist_vel : float
        the nyquist velocity for this particular ray, only needed if moment
        is radial velocity or spectral width

    Returns
    -------
    prd_data_level : numpy array or numpy masked array
        Array containing the moment data in float format (physical units)
        for every DN value

    """
    if moment == 'PHI':
        prd_data_level = np.arange(256 * 256, dtype=np.float32)  # 2 bytes
    else:
        prd_data_level = np.arange(256, dtype=np.float32)  # 1 byte

    scale_type = momhead['scale_type']
    factor_a = momhead['a']
    factor_b = momhead['b']
    factor_c = momhead['c']

    if upperlimit:
        prd_data_level += 1
    else:
        prd_data_level += 0.5

    if scale_type == SCALETYPE_LINEAR:
        prd_data_level = factor_a * prd_data_level + factor_b
    elif scale_type == SCALETYPE_LOG:
        prd_data_level = factor_a + factor_c * 10**(
            (1 - prd_data_level) / factor_b)
    else:
        # not defined
        prd_data_level *= np.nan

    if moment in ('ZH', 'ZV', 'ZHC', 'ZVC'):
        prd_data_level[0] = np.nan
    elif moment == 'ZDR':
        prd_data_level[128] = 0
        prd_data_level[0] = np.nan
    elif moment == 'RHO':
        if not ((time > 1341619200) or
                (time > 1335484800 and
                 (radar == ord('D') or
                  radar == ord('L')))):
            # revert to linear scale (old data)
            prd_data_level = np.arange(256)  # 1 byte
            prd_data_level = 1 / 255. * prd_data_level
        prd_data_level[0] = np.nan
    elif moment == 'PHI':
        prd_data_level *= 180.
        prd_data_level[0] = np.nan
    elif moment == 'VEL':
        prd_data_level *= nyquist_vel
        prd_data_level[0] = np.nan
    elif moment == 'WID':
        prd_data_level *= 2 * nyquist_vel
        prd_data_level[0] = np.nan
    elif moment == 'MPH':
        prd_data_level *= 180.
    elif moment == "CLT":
        prd_data_level = np.fromiter(range(256), dtype=np.float32)
    return prd_data_level


def nyquist_vel(sweep_number):
    """
    Returns the nyquist velocity for a given sweep-number

    Parameters
    ----------
    sweep_number : int
        sweep number (starting from zero), 1 = -0.2°, 20 = 40°


    Returns
    -------
    nv_value : float
        Nyquist velocity (in m/s)

    """
    nv_value = 20.625
    if sweep_number in (9, 10, 11):
        nv_value = 16.50
    elif sweep_number in (6, 8):
        nv_value = 13.75
    elif sweep_number in (3, 5, 7):
        nv_value = 12.375
    elif sweep_number == 4:
        nv_value = 11.
    elif sweep_number == 1:
        nv_value = 9.625
    elif sweep_number in (0, 2):
        nv_value = 8.25
    return nv_value
