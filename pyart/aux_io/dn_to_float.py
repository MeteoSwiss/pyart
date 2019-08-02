"""
pyart.aux_io.metranet_reader
============================

Set of scales that convert DN to float values

.. autosummary::
    :toctree: generated/

    float_mapping
    nyquist_vel

"""
import numpy as np
import sys

# fix for python3
if sys.version_info[0] == 3:
    def xrange(i):
        return range(i)

def float_mapping(moment, time, radar, nyquist_vel=None):
    """
    Converts DN to their float equivalent

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
        prd_data_level = np.fromiter(xrange(256), dtype=np.float32)/2.-32.
        prd_data_level[0] = np.nan
    elif moment == 'ZDR':
        prd_data_level = (
            (np.fromiter(xrange(256), dtype=np.float32)+1) /
            16.1259842-7.9375)
        prd_data_level[0] = np.nan
    elif moment == 'RHO':
        if ((time > 1341619200) or
                (time > 1335484800 and
                 (radar == ord('D') or
                  radar == ord('L')))):
            # logaritmic scale
            prd_data_level = (
                1.003-10.**(-np.fromiter(xrange(256), dtype=np.float32)*0.01))
        else:
            # linear scale (old data)
            prd_data_level = (
                np.fromiter(xrange(256), dtype=np.float32)/255.)

        prd_data_level[0] = np.nan
    elif moment == 'PHI':
        prd_data_level = ((np.fromiter(
            xrange(256*256), dtype=np.float32)-32768)/32767.*180.)
        prd_data_level[0] = np.nan
    elif moment == 'VEL':
        prd_data_level = (
            (np.fromiter(xrange(256), dtype=np.float32)-128)/127. *
            nyquist_vel)
        prd_data_level[0] = np.nan
    elif moment == 'WID':
        prd_data_level = (np.fromiter(
            xrange(256), dtype=np.float32)/255.*2.*nyquist_vel)
        prd_data_level[0] = np.nan
    elif moment == 'MPH':
        prd_data_level = ((np.fromiter(
            xrange(256), dtype=np.float32)-128)/127.*180.)
    elif moment in ('ST1', 'ST2', 'WBN'):
        prd_data_level = (np.fromiter(
            xrange(256), dtype=np.float32)/10.)
    elif moment == "CLT":
        prd_data_level = np.fromiter(xrange(256), dtype=np.float32)
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

