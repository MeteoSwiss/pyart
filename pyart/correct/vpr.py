"""
pyart.correct.vpr
=================

Computes and corrects the vertical profile of reflectivity

.. autosummary::
    :toctree: generated/

    correct_vpr
    compute_theoretical_vpr
    compute_apparent_vpr


"""
from copy import deepcopy

import numpy as np

from ..config import get_metadata, get_field_name
from ..retrieve import get_ml_rng_limits, get_iso0_val
from ..util.radar_utils import compute_azimuthal_average
from ..core.transforms import antenna_to_cartesian


def correct_vpr(radar, refl_field=None):
    """
    Correct VPR

    Parameters
    ----------
    radar : Radar
        Radar object.
    refl_field : str
        Name of the reflectivity field to correct

    Returns
    -------
    corr_refl : dict
        The corrected reflectivity

    """
    # parse the field parameters
    if refl_field_obs is None:
        refl_field_obs = get_field_name('reflectivity')
    if temp_ref == 'temperature':
        if temp_field is None:
            temp_field = get_field_name('temperature')
        temp_ref_field = temp_field
    elif temp_ref == 'height_over_iso0':
        if iso0_field is None:
            iso0_field = get_field_name('height_over_iso0')
        temp_ref_field = iso0_field

    if refl_field_theo is None:
        refl_field_theo = get_field_name(
            'theoretical_cross_correlation_ratio')

    # compute linear reflectivity

    # average data in azimuth
    radar_rhi = compute_azimuthal_average(
        radar, [refl_field_obs, temp_ref_field], nvalid_min=nvalid_min)

    iso0 = get_iso0_val(
        radar_rhi, temp_ref_field=temp_ref_field, temp_ref=temp_ref)
    print('iso0:', iso0)

    # compute ratios of reflectivity


    # find best profile


    # correct the reflectivity


    refl_corr = get_metadata('corrected_reflectivity')
    refl_corr['data'] = refl

    return refl_corr


def compute_theoretical_vpr(ml_top=3000., ml_thickness=200., val_ml=3.,
                            val_dr=-3., h_max=6000., h_res=1.):
    """
    Computes an idealized vertical profile of reflectivity

    Parameters
    ----------
    ml_top : float
        melting layer top [m asl]
    ml_thickness : float
        melting layer thickness [m]
    val_ml : float
        value in the peak of the melting layer
    val_dr : float
        decreasing ratio above the melting layer
    h_max : float
        maximum altitude at which to compute the profile [m asl]
    h_res : float
        profile resolution [m]

    Returns
    -------
    val_theo_dict : dict
        A dictionary containg the value at each altitude, the reference
        altitude and the top and bottom of the melting layer

    """
    h = np.arange(0, h_max, h_res)
    val_theo = np.ma.masked_all(h.size)

    ml_bottom = ml_top - ml_thickness
    ml_peak = ml_top - ml_thickness/2.

    val_theo[h < ml_bottom] = 1.
    val_theo[(h >= ml_bottom) & (h < ml_peak)] = (
        1. + 2.*(val_ml-1.)/ml_thickness
        * (h[(h >= ml_bottom) & (h < ml_peak)]-ml_bottom))
    val_theo[(h >= ml_peak) & (h <= ml_top)] = (
        val_ml + 2.*(1-val_ml)/ml_thickness
        * (h[(h >= ml_peak) & (h <= ml_top)]-ml_peak))
    val_theo[h > ml_top] = np.exp(
        np.log(10.)/10000.*val_dr*(h[(h > ml_top)]-ml_top))

    val_theo_dict = {
        'value': val_theo,
        'altitude': h,
        'ml_top': ml_top,
        'ml_bottom': ml_bottom,
        'val_ml_peak': val_ml,
        'val_dr': val_dr
    }
    return val_theo_dict


def compute_apparent_vpr(radar, ml_top=3000., ml_thickness=200., val_ml=3.,
                         val_dr=-3., h_max=6000., h_res=1., max_weight=9.,
                         refl_field='normalized_reflectivity'):
    """
    Computes the apparent VPR

    Parameters
    ----------
    radar : radar object
        the reference radar object
    ml_top, ml_thickness : float
        melting layer top [m asl] and thickness [m]
    val_ml : float
        value in the peak of the melting layer
    val_dr : float
        decreasing ratio above the melting layer
    h_max : float
        maximum altitude at which to compute the theoretical profiles [m asl]
    h_res : float
        profile resolution [m]
    max_weight : float
        the factor by which the antenna beam width is multiplied
    npts_diagram : int
        The number of points that that the antenna diagram will have
    rng_bottom_max: float
        maximum range at which the bottom of the melting layer can be placed
    rhohv_field: str
        Name of the apparent RhoHV profile obtained

    Returns
    -------
    radar_out : radar object
        A radar object containing the apparent normalized reflectivity profile
    vpr_theo_dict : dict
        A dictionary containg the theoretical normalized reflectivity profile

    """
    ml_bottom = ml_top - ml_thickness
    radar_out = deepcopy(radar)
    radar_out.fields = dict()
    refl_dict = get_metadata(refl_field)
    refl_dict['data'] = np.ma.masked_all((radar_out.nrays, radar_out.ngates))
    radar_out.add_field(refl_field, refl_dict)
    if ml_bottom < radar_out.altitude['data']:
        return radar_out

    # get theoretical profile as a function of altitude
    vpr_theo_dict = compute_theoretical_vpr(
        ml_top=ml_top, ml_thickness=ml_thickness, val_ml=val_ml,
        val_dr=val_dr, h_max=h_max, h_res=h_res)

    rng = radar_out.range['data']
    # range resolution of the radar resolution volume
    rng_res = rng[1] - rng[0]
    rng_left_km = (rng-rng_res/2.)/1000.
    rng_right_km = (rng+rng_res/2.)/1000.
    beam_width = (
        radar_out.instrument_parameters['radar_beam_width_h']['data'][0])
    half_bw = beam_width/2.
    half_bw_rad = half_bw*np.pi/180.

    for ind_ray, ang in enumerate(radar_out.elevation['data']):
        rng_bottom, _ = get_ml_rng_limits(
            rng_left_km, rng_right_km, rng, ang, beam_width,
            radar_out.altitude['data'][0], ml_bottom, ml_top)

        i_rng_btm = np.where(rng >= rng_bottom)[0][0]

        # values below the melting layer affected area
        radar_out.fields[refl_field]['data'][ind_ray, 0:i_rng_btm] = 1.

        rng_ml_vals = rng[i_rng_btm:] / 1000.  # km
        for i_rng, rng_val in enumerate(rng_ml_vals):
            # altitudes affected by the antenna diagram
            _, _, z_diag_min = antenna_to_cartesian(rng_val, 0., ang-half_bw)
            z_diag_min += radar_out.altitude['data']

            _, _, z_diag_max = antenna_to_cartesian(rng_val, 0., ang+half_bw)
            z_diag_max += radar_out.altitude['data']

            _, _, z_diag_center = antenna_to_cartesian(rng_val, 0., ang+half_bw)
            z_diag_center += radar_out.altitude['data']

            h = vpr_theo_dict['altitude'][
                (vpr_theo_dict['altitude'] >= z_diag_min)
                & (vpr_theo_dict['altitude'] <= z_diag_max)]
            vpr_vals = vpr_theo_dict['value'][
                (vpr_theo_dict['altitude'] >= z_diag_min)
                & (vpr_theo_dict['altitude'] <= z_diag_max)]

            weights = (h-z_diag_center)/(1000.*rng_val*half_bw_rad)
            weights = weights*weights
            weights = np.ma.masked_greater(weights, max_weight)
            weights = np.ma.exp(-2.*np.log(2.)*weights)

            radar_out.fields[refl_field]['data'][
                ind_ray, i_rng_btm+i_rng] = (
                    np.ma.sum(vpr_vals*weights)/np.ma.sum(weights))

    return radar_out, vpr_theo_dict


def find_best_profile(radar_obs, ml_thickness_min=200., ml_thickness_max=1400.,
                      ml_thickness_step=100., iso0=3000., iso0_max=4500.,
                      ml_top_diff_max=700., ml_top_step=100., h_max=6000.,
                      h_res=1., max_weight=9.):
    """
    gets the theoretical profile that best matches the observations for each
    elevation angle

    Parameters
    ----------
    ml_thickness_min, ml_thickness_max, ml_thickness_step : float
        Minimum, maximum and step of the melting layer thickness of the models
        to explore [m]
    iso0 : float
        iso0 [masl]
    iso0_max : float
        maximum iso0 altitude of the profile
    ml_top_diff_max, ml_top_step : float
        maximum difference +- between iso-0 and top of the melting layer [m]
        of the models to explore. Step
    rhohv_snow, rhohv_rain, rhohv_ml : float
        values of RhoHV above, below and at the peak of the melting layer used
        to construct the model
    zh_snow, zh_rain, zh_ml : float
        values of horizontal reflectivity above, below and at the peak of the
        melting layer used to construct the model
    zv_snow, zv_rain, zv_ml : float
        values of vertical reflectivity above, below and at the peak of the
        melting layer used to construct the model
    h_max : float
        maximum altitude [masl] where to compute the model RhoHV profile
    h_res : float
        resolution of the model RhoHV profile
    beam_factor : float
        factor by which to multiply the antenna beamwidth. Used to select the
        range of angles where the antenna pattern is going to be computed
    rng_bottom_max : float
        Maximum range up to which the bottom of the melting layer can be
        placed in order to try to find a suitable model
    ns_factor : float
        multiplicative factor applied to the number of valid model gates when
        comparing with the valid observations to decide whether the
        observations and the model can be compared
    rhohv_corr_min, rhohv_nash_min : float
        minimum correlation and NSE to consider the comparison between model
        and observations valid
    rhohv_field_obs : str
        name of the RhoHV observed field
    rhohv_field_theo: str
        name of the RhoHV modelled field

    Returns
    -------
    best_ml_thickness, best_ml_bottom : array of floats
        The ML thickness and bottom of the best model for each elevation angle
    best_rhohv_nash, best_rhohv_nash_bottom,  : array of floats
        The NSE coefficient resulting from comparing the best model

    """
    # RhoHV model possible parameters
    ml_thickness_vals = np.arange(
        ml_thickness_min, ml_thickness_max+ml_thickness_step,
        ml_thickness_step)
    ml_top_max = iso0+ml_top_diff_max
    if ml_top_max > iso0_max:
        ml_top_max = iso0_max
    ml_top_min = iso0-ml_top_diff_max
    if ml_top_min < radar_obs.altitude['data']:
        ml_top_min = radar_obs.altitude['data']
    ml_top_vals = np.arange(
        ml_top_min, ml_top_max+ml_top_step, ml_top_step)

    ml_peak_vals = np.arange(1., 7.)
    dr_vals = np.arange(-1.5, 7.5, 1.5)

    best_error = 1e6
    best_val_ml = -999.
    best_ml_thickness = -999.
    best_ml_top = -999.
    best_val_dr = -999.
    best_radar_theo = None
    for val_ml in ml_peak_vals:
        for ml_thickness in ml_thickness_vals:
            for ml_top in ml_top_vals:
                for val_dr in dr_vals:
                    print('\nChecking model with ml top'
                          ' {} [masl] and ml thickness {} m'.format(
                          ml_top, ml_thickness))
                    radar_theo, vpr_theo_dict = compute_apparent_vpr(
                        radar_obs, ml_top=ml_top, ml_thickness=ml_thickness,
                        val_ml=val_ml, val_dr=val_dr, h_max=h_max, h_res=h_res,
                        max_weight=max_weight)
                    ratios_theo = get_refl_ratios(radar_theo)
                    error = compute_error(ratios_theo, ratios_obs)
                    if error < best_error:
                        best_error = error
                        best_val_ml = val_ml
                        best_ml_thickness = ml_thickness
                        best_ml_top = ml_top
                        best_val_dr = val_dr
                        best_radar_theo = deepcopy(radar_theo)

    return (vpr_theo_dict, best_ml_top, best_ml_thickness,
            best_val_ml, best_val_dr, best_error)