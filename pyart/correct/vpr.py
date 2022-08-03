"""
pyart.correct.vpr
=================

Computes and corrects the vertical profile of reflectivity

.. autosummary::
    :toctree: generated/

    correct_vpr
    compute_theoretical_vpr
    compute_apparent_vpr
    compute_avg
    compute_refl_time_avg
    compute_refl_ratios
    find_best_profile
    compute_vpr_correction
    filter_vpr

"""
from copy import deepcopy
from warnings import warn

import numpy as np

from ..config import get_metadata, get_field_name
from ..retrieve import get_ml_rng_limits, get_iso0_val
from ..util.radar_utils import compute_azimuthal_average
from ..util import compute_mse, subset_radar
from ..core.transforms import antenna_to_cartesian
from ..core import Radar


def correct_vpr(radar, nvalid_min=20, angle_min=0., angle_max=4.,
                ml_thickness_min=200., ml_thickness_max=800.,
                ml_thickness_step=200., iso0_max=5000.,
                ml_top_diff_max=200., ml_top_step=200., ml_peak_min=1.,
                ml_peak_max=6., ml_peak_step=1., dr_min=-6., dr_max=-1.5,
                dr_step=1.5, dr_default=-4.5, dr_alt=800., h_max=6000.,
                h_res=1., max_weight=9., rmin_obs=5000., rmax_obs=150000.,
                iso0=None, weight_mem=0.75, vpr_theo_dict_mem=None,
                radar_mem_list=None, refl_field=None, lin_refl_field=None,
                norm_refl_field=None, corr_refl_field=None, corr_field=None,
                temp_field=None, iso0_field=None, temp_ref=None):
    """
    Correct VPR using the Meteo-France operational algorithm

    Parameters
    ----------
    radar : Radar
        Radar object
    nvalid_min : int
        Minimum number of rays with data to consider the azimuthal average
        valid
    angle_min, angle_max : float
        Minimum and maximum elevation angles used to compute the ratios of
        reflectivity
    ml_thickness_min, ml_thickness_max, ml_thickness_step : float
        Minimum, maximum and step of the melting layer thickness of the models
        to explore [m]
    iso0_max : float
        maximum iso0 altitude of the profile
    ml_top_diff_max, ml_top_step : float
        maximum difference +- between iso-0 and top of the melting layer [m]
        of the models to explore. Step
    ml_peak_min, ml_peak_max, ml_peak_step: float
        min, max and step of the value at the peak of the melting layer of the
        models to explore
    dr_min, dr_max, dr_step : float
        min, max and step of the decreasing ratio above the melting layer
    dr_default : float
        default decreasing ratio to use if a proper model could not be found
    dr_alt : float
        altitude above the melting layer top where theoretical profile needs
        to be defined to be able to compute DR. If the theoretical profile is
        not defined up to the resulting altitude a default DR is used
    h_max : float
        maximum altitude [masl] where to compute the model profile
    h_res : float
        resolution of the model profile (m)
    max_weight : float
        Maximum weight of the antenna pattern
    rmin_obs, rmax_obs : float
        minimum and maximum range (m) of the observations that are compared
        with the model
    iso0 : float
        reference iso0 value
    weight_mem : float
        Weight given to the previous VPR retrieval when filtering the current
        VPR retrieval by the previous one
    vpr_theo_dict_mem : dict or None
        Dictionary containing the theoretical VPR computed in the previous
        time step
    radar_mem_list : list of radar objects or None
        list of radar objects that contain the azimuthally averaged
        reflectivity computed in the past
    refl_field : str
        Name of the reflectivity field to correct
    lin_refl_field : str
        Name of the linear reflectivity field
    norm_refl_field : str
        Name of the normalized linear reflectivity field
    corr_refl_field : str
        Name of the VPR-corrected reflectivity field
    corr_field : str
        Name of the VPR correction field
    temp_field, iso0_field : str
        Name of the temperature and height over the iso-0 fields
    temp_ref : str
        the field use as reference for temperature. Can be temperature
        or height_over_iso0.

    Returns
    -------
    refl_corr_dict : dict
        The corrected reflectivity
    corr_field_dict : dict
        The correction applied
    vpr_theo_dict_filtered : dict
        The theoretical VPR profile used for the correction
    radar_rhi : radar object
        A radar object containing the azimuthally averaged reflectivity in
        linear units

    """
    # parse the field parameters
    if refl_field is None:
        refl_field = get_field_name('reflectivity')
    if lin_refl_field is None:
        lin_refl_field = get_field_name('linear_reflectivity')
    if norm_refl_field is None:
        norm_refl_field = get_field_name('normalized_reflectivity')
    if corr_refl_field is None:
        corr_refl_field = get_field_name('corrected_reflectivity')
    if corr_field is None:
        corr_field = get_field_name('vpr_correction')
    if temp_ref == 'temperature':
        if temp_field is None:
            temp_field = get_field_name('temperature')
        temp_ref_field = temp_field
    elif temp_ref == 'height_over_iso0':
        if iso0_field is None:
            iso0_field = get_field_name('height_over_iso0')
        temp_ref_field = iso0_field

    if iso0 is None:
        # filter out temperature reference where there is no valid data
        radar_aux = deepcopy(radar)
        mask = np.ma.getmaskarray(radar.fields[refl_field]['data'])
        radar_aux.fields[temp_ref_field]['data'] = np.ma.masked_where(
            mask, radar_aux.fields[temp_ref_field]['data'])

        # get iso-0 reference (to use when data is insuficient)
        if temp_ref == 'heigh_over_iso0':
            iso0_ref = (
                radar.gate_altitude['data'][0, 0]
                - radar.fields[temp_ref_field]['data'][0, 0])
        else:
            ind = np.ma.where(
                radar.fields[temp_ref_field]['data'][0, :] <= 0.)[0]
            if ind.size == 0:
                # all gates below the iso-0
                iso0_ref = radar.gate_altitude['data'][0, -1]
            else:
                iso0_ref = radar.gate_altitude['data'][0, ind[0]]

        # get azimuthally averaged data within the region of interest
        radar_azi_avg = compute_avg(
            radar_aux, rng_min=rmin_obs, rng_max=rmax_obs, ele_min=angle_min,
            ele_max=angle_max, h_max=h_max, refl_field=refl_field,
            temp_ref_field=temp_ref_field, lin_refl_field=lin_refl_field)

        iso0 = get_iso0_val(
                radar_azi_avg, temp_ref_field=temp_ref_field,
                temp_ref=temp_ref, iso0_ref=iso0_ref)
    else:
        radar_azi_avg = compute_avg(
            radar, rng_min=rmin_obs, rng_max=rmax_obs, ele_min=angle_min,
            ele_max=angle_max, h_max=h_max, refl_field=refl_field,
            temp_ref_field=temp_ref_field, lin_refl_field=lin_refl_field)
    print('iso0:', iso0)

    # compute the temporal average
    radar_time_avg = compute_refl_time_avg(
        radar_azi_avg, refl_field=lin_refl_field,
        radar_mem_list=radar_mem_list, nvalid_min=nvalid_min)

    # compute ratios of reflectivity
    ele_ratios, refl_ratios = compute_refl_ratios(
        radar_time_avg, refl_field=lin_refl_field)
    print(ele_ratios)

    # find best profile
    (_, best_ml_top, best_ml_thickness, best_val_ml, best_val_dr,
     best_error) = find_best_profile(
        radar_time_avg, refl_ratios, ml_thickness_min=ml_thickness_min,
        ml_thickness_max=ml_thickness_max,
        ml_thickness_step=ml_thickness_step, iso0=iso0, iso0_max=iso0_max,
        ml_top_diff_max=ml_top_diff_max, ml_top_step=ml_top_step,
        ml_peak_min=ml_peak_min, ml_peak_max=ml_peak_max,
        ml_peak_step=ml_peak_step, dr_min=dr_min, dr_max=dr_max,
        dr_step=dr_step, dr_default=dr_default, dr_alt=dr_alt, h_max=h_max,
        h_res=h_res, max_weight=max_weight)
    print('best_ml_top', best_ml_top)
    print('best_ml_thickness', best_ml_thickness)
    print('best_val_ml', best_val_ml)
    print('best_val_dr', best_val_dr)
    print('best_error', best_error)

    # get theoretical profile as a function of altitude
    vpr_theo_dict = compute_theoretical_vpr(
        ml_top=best_ml_top, ml_thickness=best_ml_thickness,
        val_ml=best_val_ml, val_dr=best_val_dr, h_max=15000., h_res=h_res)

    # filter profile with previously found profile
    vpr_theo_dict_filtered = filter_vpr(
        vpr_theo_dict, vpr_theo_dict_mem=vpr_theo_dict_mem,
        weight_mem=weight_mem)

    # correct the reflectivity
    refl_corr_dict, corr_field_dict = compute_vpr_correction(
        radar, vpr_theo_dict_filtered, max_weight=max_weight,
        corr_field=corr_field, norm_refl_field=norm_refl_field,
        corr_refl_field=corr_refl_field, refl_field=refl_field)

    return (refl_corr_dict, corr_field_dict, vpr_theo_dict_filtered,
            radar_azi_avg)


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
    if ml_bottom < 0.:
        ml_bottom = 0.
    ml_peak = ml_top - ml_thickness/2.

    val_theo[h < ml_bottom] = 1.
    if ml_thickness > 0:
        val_theo[(h >= ml_bottom) & (h < ml_peak)] = (
            1. + 2.*(val_ml-1.)/ml_thickness
            * (h[(h >= ml_bottom) & (h < ml_peak)]-ml_bottom))
        val_theo[(h >= ml_peak) & (h <= ml_top)] = (
            val_ml + 2.*(1-val_ml)/ml_thickness
            * (h[(h >= ml_peak) & (h <= ml_top)]-ml_peak))
        val_theo[h > ml_top] = np.exp(
            np.log(10.)/10000.*val_dr*(h[(h > ml_top)]-ml_top))
    else:
        val_theo[h >= ml_top] = np.exp(
            np.log(10.)/10000.*val_dr*(h[(h >= ml_top)]-ml_top))

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
                         vpr_theo_dict=None,
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
    vpr_theo_dict : dict or None
        A dictionary containing the theoretical VPR profile. If it is None the
        theoretical VPR profile is computed using the parameters describing it
    refl_field: str
        Name of the apparent reflectivity profile obtained

    Returns
    -------
    radar_out : radar object
        A radar object containing the apparent normalized reflectivity profile
    vpr_theo_dict : dict
        A dictionary containg the theoretical normalized reflectivity profile

    """
    radar_out = deepcopy(radar)
    radar_out.fields = dict()
    refl_dict = get_metadata(refl_field)
    refl_dict['data'] = np.ma.masked_all((radar_out.nrays, radar_out.ngates))
    radar_out.add_field(refl_field, refl_dict)

    rng = radar_out.range['data']
    known_ml_btm = False
    if vpr_theo_dict is None:
        # get theoretical profile as a function of altitude
        ml_bottom = ml_top - ml_thickness
        vpr_theo_dict = compute_theoretical_vpr(
            ml_top=ml_top, ml_thickness=ml_thickness, val_ml=val_ml,
            val_dr=val_dr, h_max=h_max, h_res=h_res)

        # range resolution of the radar resolution volume
        rng_res = rng[1] - rng[0]
        rng_left_km = (rng-rng_res/2.)/1000.
        rng_right_km = (rng+rng_res/2.)/1000.
        known_ml_btm = True

    beam_width = (
        radar_out.instrument_parameters['radar_beam_width_h']['data'][0])
    half_bw = beam_width/2.
    half_bw_rad = half_bw*np.pi/180.

    for ind_ray, ang in enumerate(radar_out.elevation['data']):
        if known_ml_btm:
            rng_bottom, _ = get_ml_rng_limits(
                rng_left_km, rng_right_km, rng, ang, beam_width,
                radar_out.altitude['data'][0], ml_bottom, ml_top)

            i_rng_btm = np.where(rng >= rng_bottom)[0][0]

            # values below the melting layer affected area
            radar_out.fields[refl_field]['data'][ind_ray, 0:i_rng_btm] = 1.
        else:
            i_rng_btm = 0

        rng_ml_vals = rng[i_rng_btm:] / 1000.  # km
        for i_rng, rng_val in enumerate(rng_ml_vals):
            # altitudes affected by the antenna diagram
            _, _, z_diag_min = antenna_to_cartesian(rng_val, 0., ang-half_bw)
            z_diag_min += radar_out.altitude['data']

            _, _, z_diag_max = antenna_to_cartesian(rng_val, 0., ang+half_bw)
            z_diag_max += radar_out.altitude['data']

            _, _, z_diag_center = antenna_to_cartesian(
                rng_val, 0., ang+half_bw)
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


def compute_avg(radar, rng_min=5000., rng_max=150000., ele_min=0., ele_max=4.,
                h_max=6000., refl_field='reflectivity',
                temp_ref_field='height_over_iso0',
                lin_refl_field='linear_reflectivity'):
    """
    Prepares the data for VPR correction. Cuts the original radar to keep
    only the region of interest for VPR correction, transforms the
    reflectivity in linear units and makes an azimuthal average of the
    reflectivity and the reference temperature field

    Parameters
    ----------
    radar : Radar
        Radar object
    rng_min, rng_max : float
        Minimum and maximum range (m) where the data is used for VPR
        estimation
    ele_min, ele_max : float
        Minimum and maximum elevation angles where the data is used for VPR
        estimation
    h_max : float
        maximum altitude [masl] where to compute the model profile
    refl_field, tem_ref_field : str
        name of the reflectivity and temperature reference fields
    lin_refl_field : str
        name of the reflectivity in linear units field

    Returns
    -------
    radar_rhi : radar object
        a radar object with the data of interest

    """
    radar_aux = subset_radar(
        radar, [refl_field, temp_ref_field], rng_min=rng_min, rng_max=rng_max,
        ele_min=ele_min, ele_max=ele_max, azi_min=None, azi_max=None)
    if radar_aux is None:
        warn('No data within limits')
        return None

    # order sweeps by increased elevation angle
    radar_aux = radar_aux.extract_sweeps(
        np.argsort(radar_aux.fixed_angle['data']))

    # compute linear reflectivity
    refl_lin = get_metadata(lin_refl_field)
    refl_lin['data'] = np.ma.power(
        10., 0.1*radar_aux.fields[refl_field]['data'])
    radar_aux.fields.pop(refl_field)
    radar_aux.add_field(lin_refl_field, refl_lin)

    # average data in azimuth
    radar_rhi = compute_azimuthal_average(
        radar_aux, [lin_refl_field, temp_ref_field], nvalid_min=1)

    # only data below iso0_max is valid
    radar_rhi.fields[lin_refl_field]['data'][
        radar_rhi.gate_altitude['data'] >= h_max] = np.ma.masked

    return radar_rhi


def compute_refl_time_avg(radar, refl_field=None, radar_mem_list=None,
                          nvalid_min=20):
    """
    Computes the time average of the reflectivity

    Parameters
    ----------
    radar : radar object
        the radar object
    refl_field : str
        name of the reflectivity field used to compute the accumulation
    radar_mem_list : list of radar objects or None
        list of radar objects that contain the azimuthally averaged
        reflectivity computed in the past
    nvalid_min : float
        Minimum number of samples to consider the average valid

    Returns
    -------
    radar_time_avg : radar object
        A radar object containing the time-averaged reflectivity field and
        metadata

    """
    radar_arr = [radar]
    if radar_mem_list is not None:
        radar_arr = radar_arr + radar_mem_list

    # find unique elevations
    eles = np.array([])
    for radar_aux in radar_arr:
        eles = np.append(eles, np.round(radar_aux.elevation['data'], 1))
    eles = np.sort(np.unique(eles))
    nrays = eles.size

    # dummy radar
    radar_out = deepcopy(radar)
    radar_out.sweep_end_ray_index['data'] = nrays-1
    radar_out.azimuth['data'] = radar.azimuth['data'][0]+np.zeros(nrays)
    radar_out.elevation['data'] = eles

    # compute temporal average at each elevation
    refl_data = np.ma.zeros((nrays, radar.ngates))
    ns_data = np.ma.zeros((nrays, radar.ngates))
    for ind_ele, ele in enumerate(eles):
        for radar_aux in radar_arr:
            eles_aux = np.round(radar_aux.elevation['data'], 1)
            if ele not in eles_aux:
                continue
            ind_ele_aux = np.where(eles_aux == ele)[0]

            ns_data[ind_ele, :] = (
                ns_data[ind_ele, :] +
                radar_aux.fields['number_of_samples']['data'][ind_ele_aux, :])
            refl_data[ind_ele, :] = (
                refl_data[ind_ele, :]
                + radar_aux.fields[refl_field]['data'][ind_ele_aux, :]
                * ns_data[ind_ele, :])
            radar_out.time['data'][ind_ele] = radar_aux.time['data'][
                ind_ele_aux]
    refl_data /= ns_data
    refl_data[ns_data < nvalid_min] = np.ma.masked
    radar_out.fields[refl_field]['data'] = refl_data
    fields = {
        refl_field: radar_out.fields[refl_field]}

    return Radar(
        radar_out.time, radar_out.range, fields, radar_out.metadata,
        radar_out.scan_type, radar_out.latitude, radar_out.longitude,
        radar_out.altitude, radar_out.sweep_number, radar_out.sweep_mode,
        radar_out.fixed_angle, radar_out.sweep_start_ray_index,
        radar_out.sweep_end_ray_index, radar_out.azimuth, radar_out.elevation,
        instrument_parameters=radar_out.instrument_parameters)


def compute_refl_ratios(radar, refl_field=None):
    """
    Computes reflectivity ratios

    Parameters
    ----------
    radar : radar object
        the radar object
    refl_field : str
        name of the reflectivity field used to compute the ratios

    Returns
    -------
    ele_ratios : dict
        dict containing the top and bottom elevation angles of the ratios
    refl_ratios : 2D array of floats
        A 2D array which contains the ratios

    """
    ele_ratios = {
        'top': np.array([], dtype=float),
        'btm': np.array([], dtype=float),
    }
    first_ratio = True
    for ind_btm, ele_btm in enumerate(radar.elevation['data'][:-1]):
        for ele_top in radar.elevation['data'][ind_btm+1:]:
            ele_ratios['top'] = np.append(ele_ratios['top'], ele_top)
            ele_ratios['btm'] = np.append(ele_ratios['btm'], ele_btm)
            ratio_aux = np.expand_dims(
                radar.fields[refl_field]['data'][ind_btm+1, :]
                / radar.fields[refl_field]['data'][ind_btm, :], axis=0)
            if first_ratio:
                refl_ratios = ratio_aux
                first_ratio = False
            else:
                refl_ratios = np.ma.append(refl_ratios, ratio_aux, axis=0)

    return ele_ratios, refl_ratios


def find_best_profile(radar_obs, ratios_obs, ml_thickness_min=200.,
                      ml_thickness_max=800., ml_thickness_step=200.,
                      iso0=3000., iso0_max=5000., ml_top_diff_max=200.,
                      ml_top_step=200., ml_peak_min=1., ml_peak_max=6.,
                      ml_peak_step=1., dr_min=-6., dr_max=-1.5, dr_step=1.5,
                      dr_default=-4.5, dr_alt=800., h_max=6000., h_res=1.,
                      max_weight=9.,
                      norm_refl_field='normalized_reflectivity'):
    """
    gets the theoretical profile that best matches the observations for each
    elevation angle

    Parameters
    ----------
    radar_obs : radar object
        radar object containing the observed data
    ratios_obs : 2D array
        2D array field containing the ratios between higher elevation and
        lower elevations of the observed linear reflectivity
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
    ml_peak_min, ml_peak_max, ml_peak_step: float
        min, max and step of the value at the peak of the melting layer of the
        models to explore
    dr_min, dr_max, dr_step : float
        min, max and step of the decreasing ratio above the melting layer
    dr_default : float
        default decreasing ratio to set if VPR model could not be found
    dr_alt : float
        altitude above the melting layer top where theoretical profile needs
        to be defined to be able to compute DR. If the theoretical profile is
        not defined up to the resulting altitude a default DR is used
    h_max : float
        maximum altitude [masl] where to compute the model profile
    h_res : float
        resolution of the model profile (m)
    max_weight : float
        Maximum weight of the antenna pattern

    Returns
    -------
    best_vpr_theo_dict : dict
        Dictionary containing the values of the best VPR algorithm
    best_ml_top, best_ml_thickness : float
        The value of the melting layer top [masl] and thickness [m] of the
        model that closest ressembles the observations
    best_val_ml : float
        The value at the peak of the melting layer of the best model
    best_val_dr : float
        The decreasing ratio at the top of the melting layer of the best model
    best_error : the quadratic error of the best model

    """
    best_error = 1e8
    best_val_ml = -999.
    best_ml_thickness = -999.
    best_ml_top = -999.
    best_val_dr = -999.
    best_vpr_theo_dict = None

    if iso0 >= iso0_max or iso0 <= radar_obs.altitude['data']:
        warn('iso0 {} masl higher than {} m or lower than the radar altitude'.format(
            iso0, iso0_max))
        return (best_vpr_theo_dict, iso0, 0., 1., dr_default, best_error)

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

    ml_peak_vals = np.arange(
        ml_peak_min, ml_peak_max+ml_peak_step, ml_peak_step)
    if iso0+dr_alt >= h_max:
        warn('theoretical profile not defined high enough above the melting'
             ' layer. A default DR will be used')
        dr_vals = np.array([dr_default])
    else:
        dr_vals = np.arange(dr_min, dr_max+dr_step, dr_step)

    for ml_top in ml_top_vals:
        for ml_thickness in ml_thickness_vals:
            for val_ml in ml_peak_vals:
                for val_dr in dr_vals:
                    print('\nml top {} masl, ml thickness {}, ml peak {}, dr {}, '.format(
                              ml_top, ml_thickness, val_ml, val_dr))
                    radar_theo, vpr_theo_dict = compute_apparent_vpr(
                        radar_obs, ml_top=ml_top, ml_thickness=ml_thickness,
                        val_ml=val_ml, val_dr=val_dr, h_max=h_max, h_res=h_res,
                        max_weight=max_weight, refl_field=norm_refl_field)
                    _, ratios_theo = compute_refl_ratios(
                        radar_theo, refl_field=norm_refl_field)
                    error = compute_mse(ratios_theo, ratios_obs)
                    if error is None:
                        warn('Not enough observations to compute VPR')
                        return (best_vpr_theo_dict, iso0, 0., 1., dr_default,
                                best_error)
                    if error < best_error:
                        print('\nVALID MODEL. Error: {}\n'.format(error))
                        best_error = error
                        best_val_ml = val_ml
                        best_ml_thickness = ml_thickness
                        best_ml_top = ml_top
                        best_val_dr = val_dr
                        best_vpr_theo_dict = vpr_theo_dict

    return (best_vpr_theo_dict, best_ml_top, best_ml_thickness,
            best_val_ml, best_val_dr, best_error)


def compute_vpr_correction(radar, vpr_theo_dict, max_weight=9.,
                           corr_field='vpr_correction',
                           norm_refl_field='normalized_reflectivity',
                           corr_refl_field='corrected_reflectivity',
                           refl_field='reflectivity'):
    """
    Computes the VPR correction

    Parameters
    ----------
    radar : radar object
        reference radar object
    vpr_theo_dict : dict
        dictionary containing the theoretical VPR
    max_weight : float
        Maximum weight of the antenna pattern

    Returns
    -------
    corr_field_dict : dict
        Dictionary containing the values of the VPR correction and metadata

    """
    radar_rhi = deepcopy(radar)
    radar_rhi.fields = dict()
    radar_rhi.scan_type = 'rhi'
    radar_rhi.sweep_number['data'] = np.array([0])
    radar_rhi.sweep_mode['data'] = np.array(['rhi'])
    radar_rhi.fixed_angle['data'] = np.array([0])
    radar_rhi.sweep_start_ray_index['data'] = np.array([0])
    radar_rhi.sweep_end_ray_index['data'] = np.array([radar.nsweeps-1])
    radar_rhi.rays_per_sweep['data'] = np.array([radar.nsweeps])
    radar_rhi.azimuth['data'] = np.ones(radar.nsweeps)
    radar_rhi.elevation['data'] = radar.fixed_angle['data']
    radar_rhi.time['data'] = np.zeros(radar.nsweeps)
    radar_rhi.nrays = radar.fixed_angle['data'].size
    radar_rhi.nsweeps = 1
    radar_rhi.rays_are_indexed = None
    radar_rhi.ray_angle_res = None

    radar_corr, _ = compute_apparent_vpr(
        radar_rhi, max_weight=max_weight, vpr_theo_dict=vpr_theo_dict)

    corr_field_dict = get_metadata(corr_field)
    corr_field_dict['data'] = np.ma.zeros((radar.nrays, radar.ngates))
    for ind_ray, (ind_start, ind_end) in enumerate(zip(
                                radar.sweep_start_ray_index['data'],
                                radar.sweep_end_ray_index['data'])):
        corr_field_dict['data'][ind_start:ind_end+1, :] = 10.*np.ma.log10(
            1./radar_corr.fields[norm_refl_field]['data'][ind_ray, :])

    refl_corr_dict = get_metadata(corr_refl_field)
    refl_corr_dict['data'] = (
        radar.fields[refl_field]['data']+corr_field_dict['data'])

    return refl_corr_dict, corr_field_dict


def filter_vpr(vpr_theo_dict, vpr_theo_dict_mem=None, weight_mem=0.75):
    """
    Filters the current retrieved VPR with past retrievals

    Parameters
    ----------
    vpr_theo_dict : dict
        the current VPR retrieval
    vpr_theo_dict_mem : dict
        the past retrieval
    weight_mem : float
        weight to give to the past retrieval

    Returns
    -------
    vpr_filt_dict : dict
        The filtered retrieval
    """
    if vpr_theo_dict_mem is None:
        return deepcopy(vpr_theo_dict)
    if np.array_equal(vpr_theo_dict['altitude'],
                      vpr_theo_dict_mem['altitude']):
        vals = (
            (1.-weight_mem)*vpr_theo_dict['value']
            + weight_mem*vpr_theo_dict_mem['value'])
        vpr_filt_dict = {
            'value': vals,
            'altitude': vpr_theo_dict['altitude'],
        }
        return vpr_filt_dict
    return deepcopy(vpr_theo_dict)
