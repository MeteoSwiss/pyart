"""
pyart.retrieve.simple_moment_calculations
=========================================

Simple moment calculations.

.. autosummary::
    :toctree: generated/

    calculate_snr_from_reflectivity
    compute_snr
    compute_l
    compute_cdr

"""

import numpy as np

from ..config import get_metadata, get_field_name
from ..core.transforms import antenna_to_cartesian


def calculate_snr_from_reflectivity(
        radar, refl_field=None, snr_field=None, toa=25000.):
    """
    Calculate the signal to noise ratio, in dB, from the reflectivity field.

    Parameters
    ----------
    radar : Radar
        Radar object from which to retrieve reflectivity field.
    refl_field : str, optional
        Name of field in radar which contains the reflectivity.
        None will use the default field name in the Py-ART configuration file.
    snr_field : str, optional
        Name to use for snr metadata.
        None will use the default field name in the Py-ART configuration file.
    toa : float, optional
        Height above which to take noise floor measurements, in meters.

    Returns
    -------
    snr : field dictionary
        Field dictionary containing the signal to noise ratio.

    """
    if refl_field is None:
        refl_field = get_field_name('reflectivity')
    if snr_field is None:
        snr_field = get_field_name('signal_to_noise_ratio')

    range_grid = np.meshgrid(radar.range['data'],
                             np.ma.ones(radar.time['data'].shape))[0] + 1.0

    # remove range scale.. This is basically the radar constant scaled dBm
    pseudo_power = (radar.fields[refl_field]['data'] -
                    20.0*np.log10(range_grid / 1000.0))

    # Noise floor estimate
    # 25km.. should be no scatterers, not even planes, this high
    # we could get undone by AP though.. also sun
    rg, azg = np.meshgrid(radar.range['data'], radar.azimuth['data'])
    rg, eleg = np.meshgrid(radar.range['data'], radar.elevation['data'])
    x, y, z = antenna_to_cartesian(rg / 1000.0, azg, eleg)  # XXX: need to fix

    points_above = np.where(z > toa)
    noise_floor_estimate = pseudo_power[points_above].mean()

    snr_dict = get_metadata(snr_field)
    snr_dict['data'] = pseudo_power - noise_floor_estimate
    return snr_dict


def compute_snr(refl, noisedBZ, snr_field=None):
    """
    Computes SNR from a reflectivity field and the noise in dBZ.

    Parameters
    ----------
    refl : dictionary of dictionaries
        the reflectivity field

    noisedBZ : dictionary of dictionaries
        the noise field

    snr_field: str
        name of the SNR field to use

    Returns
    -------
    snr : dictionary of dictionaries
        the SNR field

    """
    # parse the field parameters
    if snr_field is None:
        snr_field = get_field_name('signal_to_noise_ratio_hh')

    mask = np.ma.getmaskarray(refl['data'])
    fill_value = refl['data'].get_fill_value()

    snr_data = np.ma.masked_where(mask, refl['data']-noisedBZ['data'])
    snr_data.set_fill_value(fill_value)
    snr_data.data[mask.nonzero()] = fill_value

    snr = get_metadata(snr_field)
    snr['data'] = snr_data

    return snr


def compute_l(rhohv, l_field=None):
    """
    Computes Rhohv in logarithmic scale according to L=-log10(1-RhoHV)

    Parameters
    ----------
    rhohv : dictionary of dictionaries
        the Rhohv field

    l_field : str
        name of the L field

    Returns
    -------
    l : dictionary of dictionaries
        L field

    """
    # parse the field parameters
    if l_field is None:
        l_field = get_field_name('logarithmic_cross_correlation_ratio')

    mask = np.ma.getmaskarray(rhohv['data'])
    fill_value = rhohv['data'].get_fill_value()
    is_one = rhohv['data'] >= 1.
    rhohv['data'][is_one.nonzero()] = 0.9999

    l_data = np.ma.masked_where(mask, -np.ma.log10(1.-rhohv['data']))
    l_data.set_fill_value(fill_value)
    l_data.data[mask.nonzero()] = fill_value

    l = get_metadata(l_field)
    l['data'] = l_data

    return l


def compute_cdr(rhohv, zdrdB, cdr_field=None):
    """
    Computes the Circular Depolarization Ratio

    Parameters
    ----------
    rhohv : dictionary of dictionaries
        the Rhohv field

    zdrdB : dictionary of dictionaries
        the ZDR field

    cdr_field : str
        name of the CDR field

    Returns
    -------
    cdr : dictionary of dictionaries
        CDR field

    """
    zdr = np.power(10., 0.1*zdrdB['data'])

    # parse the field parameters
    if cdr_field is None:
        cdr_field = get_field_name('circular_depolarization_ratio')

    mask = np.ma.getmaskarray(rhohv['data'])
    fill_value = rhohv['data'].get_fill_value()

    cdr_data = np.ma.masked_where(
        mask, 10.*np.ma.log10(
            (1.+1./zdr-2.*rhohv['data']*np.ma.sqrt(1./zdr)) /
            (1.+1./zdr+2*rhohv['data']*np.ma.sqrt(1./zdr))))
    cdr_data.set_fill_value(fill_value)
    cdr_data.data[mask.nonzero()] = fill_value

    cdr = get_metadata(cdr_field)
    cdr['data'] = cdr_data

    return cdr
