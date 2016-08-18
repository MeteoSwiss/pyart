"""
pyart.correct.noise
===================

Corrects polarimetric variables for noise

.. autosummary::
    :toctree: generated/

    correct_noise_rhohv

"""

import numpy as np

from ..config import get_metadata, get_field_name, get_fillvalue


def correct_noise_rhohv(urhohv, snrdB_h, zdrdB, alpha, rhohv_field=None):
    """
    Corrects RhoHV for noise

    Parameters
    ----------
    urhohv : dictionary of dictionaries
        the Rhohv field without noise correction

    snrdB_h : dictionary of dictionaries
        the SNR of the horizontal channel in dB

    zdrdB: dictionary of dictionaries
        the differential reflectivity

    alpha: float
        power imbalance between horizontal and vertical channel

    rhohv_field: str
        name of the rhohv field to output

    Returns
    -------
    rhohv : dictionary of dictionaries
        noise corrected RhoHV field

    """
    # parse the field parameters
    if rhohv_field is None:
        rhohv_field = get_field_name('cross_correlation_ratio')

    snr_h = np.ma.power(10., 0.1*snrdB_h['data'])
    zdr = np.ma.power(10., 0.1*zdrdB['data'])

    mask = np.ma.getmaskarray(urhohv['data'])
    fill_value = urhohv['data'].get_fill_value()

    rhohv_data = np.ma.masked_where(
        mask, urhohv['data']*np.ma.sqrt((1.+1./snr_h)*(1.+zdr/(alpha*snr_h))))
    rhohv_data.set_fill_value(fill_value)
    rhohv_data.data[mask.nonzero()] = fill_value
    is_above1 = rhohv_data > 1.
    rhohv_data[is_above1.nonzero()] = 1.

    rhohv = get_metadata(rhohv_field)
    rhohv['data'] = rhohv_data

    return rhohv
