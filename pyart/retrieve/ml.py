"""
pyart.retrieve.ml
=========================================

Routines to detect the ML from polarimetric RHI scans.

.. autosummary::
    :toctree: generated/

    melting_layer_mf
    detect_ml
    melting_layer_giangrande
    melting_layer_hydroclass
    compute_theoretical_profile
    compute_apparent_profile
    get_ml_rng_limits
    find_best_profile
    filter_ml
    compare_rhohv
    enough_data
    mask_ml_top
    get_iso0_val
    get_flag_ml
    get_pos_ml
    compute_iso0
    find_ml_field
    interpol_field
    _create_ml_obj
    _prepare_radar
    _get_ml_global
    _get_target_azimuths
    _find_ml_gates
    _insert_ml_points
    _find_ml_limits
    _interpol_ml_limits
    _get_res_vol_sides
    _detect_ml_sweep
    _process_map_ml
    _process_map_ml_only_zh
    _r_to_h
    _remap_to_polar
    _normalize_image
    _gradient_2D
    _convolve_with_nan
    _mean_filter
    _calc_sub_ind

"""

from warnings import warn
from copy import deepcopy
import datetime

import numpy as np
from scipy.ndimage.filters import convolve
from scipy.interpolate import InterpolatedUnivariateSpline, pchip
from scipy.interpolate import RegularGridInterpolator, interp1d

from ..config import get_field_name, get_metadata, get_fillvalue
from ..map.polar_to_cartesian import get_earth_radius, polar_to_cartesian
from ..util.datetime_utils import datetime_from_radar
from ..util.xsect import cross_section_ppi, cross_section_rhi
from ..util.sigmath import compute_nse, compute_corr
from ..util.radar_utils import compute_azimuthal_average
from ..util.radar_utils import compute_antenna_diagram
from ..core.transforms import antenna_vectors_to_cartesian
from ..core.transforms import antenna_to_cartesian

# Parameters
# They shouldn not be changed ideally
MAXTHICKNESS_ML = 1000
MAXHEIGHT_ML = 6000.
MINHEIGHT_ML = 1000.
LOWMLBOUND = 0.7
UPMLBOUND = 1.3
SIZEFILT_M = 75
ZH_IM_BOUNDS = (10, 60)
RHOHV_IM_BOUNDS = (0.75, 1)
RHOHV_VALID_BOUNDS = (0.6, 1)
KE = 4 / 3.  # Constant in the 4/3 earth radius model


def melting_layer_mf(radar, nvalid_min=180, ml_thickness_min=200.,
                     ml_thickness_max=1400., ml_thickness_step=100.,
                     iso0_max=4500., ml_top_diff_max=700., ml_top_step=100.,
                     rhohv_snow=0.99, rhohv_rain=0.99, rhohv_ml=0.93,
                     zh_snow=20., zh_rain=20., zh_ml=27., zv_snow=20.,
                     zv_rain=20., zv_ml=26., h_max=6000., h_res=1.,
                     beam_factor=2., npts_diagram=81, rng_bottom_max=200000.,
                     ns_factor=0.6, rhohv_corr_min=0.9, rhohv_nash_min=0.5,
                     ang_iso0=10., age_iso0=3., ml_thickness_iso0=700.,
                     rhohv_field_obs=None, temp_field=None, iso0_field=None,
                     rhohv_field_theo=None, ml_field=None, ml_pos_field=None,
                     temp_ref=None, get_iso0=True):
    """
    Detects the melting layer following the approach implemented at
    Météo-France

    Parameters
    ----------
    radar : radar
        radar object

    Other Parameters
    ----------------
    nvalid_min : int
        Number of volume scans to aggregate
    ml_thickness_min, ml_thickness_max, ml_thickness_step : float
        Minimum, maximum and step of the melting layer thickness of the models
        to explore [m]
    iso0_max : maximum iso0 altitude [masl]
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
    ang_iso0 : float
        the equivalent iso0 angle; Used for the computation of the weights
    age_iso0 : float
        the equivalent age of the iso0 (hours)
    ml_thickness_iso0 : float
        Default iso-0 thickness
    rhohv_field_obs, temp_field, iso0_field : str
        name of the RhoHV observed field, temperature field and height over
        iso0 field
    rhohv_field_theo: str
        name of the RhoHV modelled field
    ml_field : str
        Output. Field name which represents the melting layer field.
        A value of None will use the default field name as defined in the
        Py-ART configuration file.
    ml_pos_field : str
        Output. Field name which represents the melting layer top and bottom
        height field. A value of None will use the default field name as
        defined in the Py-ART configuration file.
    temp_ref : str
        the field use as reference for temperature. Can be temperature
        or height_over_iso0.
    get_iso0 : bool
        returns height w.r.t. freezing level top for each gate in the radar
        volume.

    Returns
    -------
    ml_obj : radar-like object
        A radar-like object containing the field melting layer height with
        the bottom (at range position 0) and top (at range position one) of
        the melting layer at each ray
    ml_dict : dict
        A dictionary containg the position of the range gate respect to the
        melting layer and metadata
    iso0_dict : dict or None
        A dictionary containing the distance respect to the melting layer
        and metadata
    ml_global : dict or None
        stack of previous volume data to introduce some time dependency. Its
        max size is controlled by the nVol parameter. It is always in
        (pseudo-)RHI mode.

    """
    # parse the field parameters
    if rhohv_field_obs is None:
        rhohv_field_obs = get_field_name('cross_correlation_ratio')
    if temp_ref == 'temperature':
        if temp_field is None:
            temp_field = get_field_name('temperature')
        temp_ref_field = temp_field
    elif temp_ref == 'height_over_iso0':
        if iso0_field is None:
            iso0_field = get_field_name('height_over_iso0')
        temp_ref_field = iso0_field

    if rhohv_field_theo is None:
        rhohv_field_theo = get_field_name(
            'theoretical_cross_correlation_ratio')
    if ml_field is None:
        ml_field = get_field_name('melting_layer')
    if ml_pos_field is None:
        ml_pos_field = get_field_name('melting_layer_height')

    # average RhoHV
    radar_rhi = compute_azimuthal_average(
        radar, [rhohv_field_obs, temp_ref_field], nvalid_min=nvalid_min)

    iso0 = get_iso0_val(
        radar_rhi, temp_ref_field=temp_ref_field, temp_ref=temp_ref)
    print('iso0:', iso0)

    # get best instantaneous model by elevation angle
    (best_ml_thickness, best_ml_bottom, best_rhohv_nash,
     best_rhohv_nash_bottom) = find_best_profile(
        radar_rhi, ml_thickness_min=ml_thickness_min,
        ml_thickness_max=ml_thickness_max,
        ml_thickness_step=ml_thickness_step, iso0=iso0, iso0_max=iso0_max,
        ml_top_diff_max=ml_top_diff_max, ml_top_step=ml_top_step,
        rhohv_snow=rhohv_snow, rhohv_rain=rhohv_rain, rhohv_ml=rhohv_ml,
        zh_snow=zh_snow, zh_rain=zh_rain, zh_ml=zh_ml, zv_snow=zv_snow,
        zv_rain=zv_rain, zv_ml=zv_ml, h_max=h_max, h_res=h_res,
        beam_factor=beam_factor, npts_diagram=npts_diagram,
        rng_bottom_max=rng_bottom_max, ns_factor=ns_factor,
        rhohv_corr_min=rhohv_corr_min, rhohv_nash_min=rhohv_nash_min,
        rhohv_field_obs=rhohv_field_obs, rhohv_field_theo=rhohv_field_theo)

    print('best_ml_thickness', best_ml_thickness)
    print('best_ml_bottom', best_ml_bottom)
    print('best_rhohv_nash', best_rhohv_nash)
    print('best_rhohv_nash_bottom', best_rhohv_nash_bottom)

    ml_bottom, ml_thickness = filter_ml(
        best_ml_thickness, best_ml_bottom, iso0, radar_rhi.elevation['data'],
        ang_iso0=ang_iso0, age_iso0=age_iso0,
        ml_thickness_iso0=ml_thickness_iso0)
    print('best_ml_thickness', ml_thickness)
    print('best_ml_bottom', ml_bottom)

    # Create melting layer object containing top and bottom and metadata
    # and melting layer field
    ml_dict = get_metadata(ml_field)
    ml_dict.update({'_FillValue': 0})
    ml_obj = _create_ml_obj(radar, ml_pos_field)
    ml_obj.fields[ml_pos_field]['data'][:, 0] = ml_bottom
    ml_obj.fields[ml_pos_field]['data'][:, 1] = ml_bottom+ml_thickness

    # Find position of range gates respect to melting layer top and bottom
    ml_dict = find_ml_field(
        radar, ml_obj, ml_pos_field=ml_pos_field, ml_field=ml_field)

    # get the iso0
    iso0_dict = None
    if get_iso0:
        iso0_dict = compute_iso0(
            radar, ml_obj.fields[ml_pos_field]['data'][:, 1],
            iso0_field=iso0_field)

    return ml_obj, ml_dict, iso0_dict, None


def detect_ml(radar, gatefilter=None, fill_value=None, refl_field=None,
              rhohv_field=None, ml_field=None, ml_pos_field=None,
              iso0_field=None, max_range=20000,
              detect_threshold=0.02, interp_holes=False, max_length_holes=250,
              check_min_length=True, get_iso0=False):
    '''
    Detects the melting layer (ML) using the reflectivity and copolar
    correlation coefficient. Internally it uses RHIs

    Paremeters
    ----------
        radar : Radar
            A Radar class instance
        gatefilter : GateFilter, optional
            A GateFilter indicating radar gates that should be excluded
        fill_value : float, optional
            Value indicating missing or bad data in differential phase
            field, if  not specified, the default in the Py-ART
            configuration file will be used
        refl_field : str, optional
            Reflectivity field. If None, the default field name must be
            specified in the Py-ART configuration file.
        rhohv_field : str, optional
            Copolar correlation coefficient field. If None, the default
            field name must be specified in the Py-ART configuration file.
        ml_field : str, optional
            Melting layer field. If None, the default field name must
            be specified in the Py-ART configuration file.
        ml_pos_field : str, optional
            Melting layer height field. If None, the default field name must
            be specified in the Py-ART configuration file.
        iso0_field : str, optional
            height respect to the iso0 field.
        max_range : float, optional
            the max. range from the radar to be used in the ML determination
        detect_threshold : float, optional
            the detection threshold (see paper), you can play around and
            see how it affects the output. Lowering the value makes the
            algorithm more sensitive but increases the number of
            erroneous detections.
        interp_holes : bool, optional
            Flag to allow for interpolation of small holes in the detected ML
        max_length_holes : float, optional
            The maximum size of holes in the ML for them to be interpolated
        check_min_length : bool, optional
            If true, the length of the detected ML will
            be compared with the length of the valid data and the
            ML will be kept only if sufficiently long
        get_iso0 : bool
            returns height w.r.t. freezing level top for each gate in the
            radar volume.

    Returns
    -------
    ml_obj : radar-like object
        A radar-like object containing the field melting layer height with
        the bottom (at range position 0) and top (at range position one) of
        the melting layer at each ray
    ml_dict : dict
        A dictionary containg the position of the range gate respect to the
        melting layer and metadata
    iso0_dict : dict or None
        A dictionary containing the distance respect to the melting layer
        and metadata
    all_ml : dict
        Dictionary containing internal parameters in polar and cartesian
        coordinates

    Reference:
    ----------
    Wolfensberger, D. , Scipion, D. and Berne, A. (2016), Detection and
    characterization of the melting layer based on polarimetric radar scans.
    Q.J.R. Meteorol. Soc., 142: 108-124. doi:10.1002/qj.2672
    '''
    # parse fill value
    if fill_value is None:
        fill_value = get_fillvalue()

    # parse field names
    if refl_field is None:
        refl_field = get_field_name('reflectivity')
    if rhohv_field is None:
        rhohv_field = get_field_name('copolar_correlation_coefficient')
    if ml_field is None:
        ml_field = get_field_name('melting_layer')
    if ml_pos_field is None:
        ml_pos_field = get_field_name('melting_layer_height')

    # get radar with only relevant fields
    radar_in = _prepare_radar(
        radar, [refl_field, rhohv_field], temp_ref='no_field')
    if radar_in is None:
        warn('Unable to obtain melting layer information for this radar scan')
        return None, None, None, None

    # mask radar gates indicated by the gate filter
    if gatefilter is not None:
        radar_in.fields[refl_field]['data'] = np.ma.masked_where(
            gatefilter.gate_excluded, radar_in.fields[refl_field]['data'])
        radar_in.fields[rhohv_field]['data'] = np.ma.masked_where(
            gatefilter.gate_excluded, radar_in.fields[rhohv_field]['data'])

    # transform radar into rhi
    if radar_in.scan_type == 'ppi':
        target_azimuths, az_tol = _get_target_azimuths(radar_in)
        radar_rhi = cross_section_ppi(
            radar_in, target_azimuths, az_tol=az_tol)
    elif radar_in.scan_type == 'rhi':
        radar_rhi = radar_in
    else:
        warn('Error: unsupported scan type.')
        return None, None, None, None

    # get melting layer data
    all_ml = []
    ml_exists = []
    for sweep in range(radar_rhi.nsweeps):
        radar_sweep = radar_rhi.extract_sweeps([sweep])

        out = _detect_ml_sweep(
            radar_sweep, fill_value, refl_field, rhohv_field,
            ml_field, max_range, detect_threshold, interp_holes,
            max_length_holes, check_min_length)

        all_ml.append(out)
        ml_exists.append(out['ml_exists'])

    # Check if melting layer has been found
    if not any(ml_exists):
        warn('Unable to obtain melting layer information for this radar scan')
        return None, None, None, None

    # Create melting layer object containing top and bottom and metadata
    # and melting layer field
    ml_dict = get_metadata(ml_field)
    ml_dict.update({'_FillValue': 0})
    ml_obj = _create_ml_obj(radar_rhi, ml_pos_field)

    ml_data = np.ma.masked_all(
        (radar_rhi.nrays, radar_rhi.ngates), dtype=np.uint8)
    for sweep in range(radar_rhi.nsweeps):
        sweep_start = radar_rhi.sweep_start_ray_index['data'][sweep]
        sweep_end = radar_rhi.sweep_end_ray_index['data'][sweep]
        ml_obj.fields[ml_pos_field]['data'][sweep_start:sweep_end+1, 0] = (
            all_ml[sweep]['ml_pol']['bottom_ml'])
        ml_obj.fields[ml_pos_field]['data'][sweep_start:sweep_end+1, 1] = (
            all_ml[sweep]['ml_pol']['top_ml'])
        ml_data[sweep_start:sweep_end+1, :] = all_ml[sweep]['ml_pol']['data']
    ml_dict['data'] = ml_data

    valid_values = ml_obj.fields[ml_pos_field]['data'][:, 1].compressed()
    print('Before PPI transformation ', valid_values[valid_values > 6000.])

    # transform back into PPI volume
    if radar_in.scan_type == 'ppi':
        radar_rhi.add_field(ml_field, ml_dict)
        radar_out = cross_section_rhi(radar_rhi, radar_in.fixed_angle['data'])
        ml_dict['data'] = radar_out.fields[ml_field]['data']
        ml_obj = cross_section_rhi(ml_obj, radar_in.fixed_angle['data'])

    valid_values = ml_obj.fields[ml_pos_field]['data'][:, 1].compressed()
    print('After PPI transformation ', valid_values[valid_values > 6000.])

    # get the iso0
    iso0_dict = None
    if get_iso0:
        iso0_dict = compute_iso0(
            radar_in, ml_obj.fields[ml_pos_field]['data'][:, 1],
            iso0_field=iso0_field)

    return ml_obj, ml_dict, iso0_dict, all_ml


def melting_layer_giangrande(radar, nVol=3, maxh=6000., hres=50.,
                             rmin=1000., elmin=4., elmax=10., rhomin=0.75,
                             rhomax=0.94, zhmin=20., hwindow=500.,
                             mlzhmin=30., mlzhmax=50., mlzdrmin=1.,
                             mlzdrmax=5., htol=500., ml_bottom_diff_max=1000.,
                             time_accu_max=1800., nml_points_min=None,
                             wlength=20., percentile_bottom=0.3,
                             percentile_top=0.9, interpol=True,
                             time_nodata_allowed=3600., refl_field=None,
                             zdr_field=None, rhv_field=None, temp_field=None,
                             iso0_field=None, ml_field=None,
                             ml_pos_field=None, temp_ref=None,
                             get_iso0=False, ml_global=None):
    """
    Detects the melting layer following the approach by Giangrande et al
    (2008)

    Parameters
    ----------
    radar : radar
        radar object

    Other Parameters
    ----------------
    nVol : int
        Number of volume scans to aggregate
    maxh : float
        Maximum possible height of the melting layer [m MSL]
    hres : float
        Step of the height of the melting layer [m]
    rmin : float
        Minimum range from radar where to look for melting layer contaminated
        range gates [m]
    elmin, elmax : float
        Minimum and maximum elevation angles where to look for melting layer
        contaminated range gates [degree]
    rhomin, rhomax : float
        min and max rhohv to consider pixel potential melting layer pixel
    zhmin : float
        Minimum reflectivity level of a range gate to consider it a potential
        melting layer gate [dBZ]
    hwindow : float
        Maximum distance (in range) from potential melting layer gate where to
        look for a maximum [m]
    mlzhmin, mlzhmax : float
        Minimum and maximum values that a peak in reflectivity within the
        melting layer may have to consider the range gate melting layer
        contaminated [dBZ]
    mlzdrmin, mlzdrmax : float
        Minimum and maximum values that a peak in differential reflectivity
        within the melting layer may have to consider the range gate melting
        layer contaminated [dB]
    htol : float
        maximum distance from the iso0 coming from model allowed to consider
        the range gate melting layer contaminated [m]
    ml_bottom_dif_max : float
        Maximum distance from the bottom of the melting layer computed in the
        previous time step to consider a range gate melting layer contaminated
        [m]
    time_accu_max : float
        Maximum time allowed to accumulate data from consecutive scans [s]
    nml_points_min : int
        minimum number of melting layer points to consider valid melting layer
        detection
    wlength : float
        length of the window to select the azimuth angles used to compute the
        melting layer limits at a particular azimuth [degree]
    percentile_bottom, percentile_top : float [0,1]
        percentile of ml points above which is considered that the bottom of
        the melting layer starts and the top ends
    interpol : bool
        Whether to interpolate the obtained results in order to get a value
        for each azimuth
    time_nodata_allowed : float
        The maximum time allowed for no data before considering the melting
        layer not valid [s]
    refl_field, zdr_field, rhv_field, temp_field, iso0_field : str
        Inputs. Field names within the radar object which represent the
        horizonal reflectivity, the differential reflectivity, the copolar
        correlation coefficient, the temperature and the height respect to the
        iso0 fields. A value of None for any of these parameters will use the
        default field name as defined in the Py-ART configuration file.
    ml_field : str
        Output. Field name which represents the melting layer field.
        A value of None will use the default field name as defined in the
        Py-ART configuration file.
    ml_pos_field : str
        Output. Field name which represents the melting layer top and bottom
        height field. A value of None will use the default field name as
        defined in the Py-ART configuration file.
    temp_ref : str
        the field use as reference for temperature. Can be temperature
        or height_over_iso0.
        If None, it excludes model data from the algorithm.
    get_iso0 : bool
        returns height w.r.t. freezing level top for each gate in the radar
        volume.
    ml_global :
        stack of previous volume data to introduce some time dependency. Its
        max size is controlled by the nVol parameter. It is always in
        (pseudo-)RHI mode.

    Returns
    -------
    ml_obj : radar-like object
        A radar-like object containing the field melting layer height with
        the bottom (at range position 0) and top (at range position one) of
        the melting layer at each ray
    ml_dict : dict
        A dictionary containg the position of the range gate respect to the
        melting layer and metadata
    iso0_dict : dict or None
        A dictionary containing the distance respect to the melting layer
        and metadata
    ml_global : dict or None
        stack of previous volume data to introduce some time dependency. Its
        max size is controlled by the nVol parameter. It is always in
        (pseudo-)RHI mode.

    References
    ----------
    Giangrande, S.E., Krause, J.M., Ryzhkov, A.V.: Automatic Designation of
    the Melting Layer with a Polarimetric Prototype of the WSR-88D Radar,
    J. of Applied Meteo. and Clim., 47, 1354-1364, doi:10.1175/2007JAMC1634.1,
    2008

    """
    # parse the field parameters
    if refl_field is None:
        refl_field = get_field_name('reflectivity')
    if zdr_field is None:
        zdr_field = get_field_name('differential_reflectivity')
    if rhv_field is None:
        rhv_field = get_field_name('cross_correlation_ratio')
    if temp_ref == 'temperature':
        if temp_field is None:
            temp_field = get_field_name('temperature')
    elif temp_ref == 'height_over_iso0':
        if iso0_field is None:
            iso0_field = get_field_name('height_over_iso0')

    if ml_field is None:
        ml_field = get_field_name('melting_layer')
    if ml_pos_field is None:
        ml_pos_field = get_field_name('melting_layer_height')

    # prepare radar input (select relevant radar fields)
    field_list = [refl_field, zdr_field, rhv_field]
    if temp_ref == 'temperature':
        field_list.append(temp_field)
    elif temp_ref == 'height_over_iso0':
        field_list.append(iso0_field)
    radar_in = _prepare_radar(
        radar, field_list, temp_ref=temp_ref, iso0_field=iso0_field,
        temp_field=temp_field, lapse_rate=-6.5)
    if radar_in is None:
        warn('Unable to obtain melting layer information for this radar scan')
        return None, None, None, ml_global

    ml_global, is_valid = _get_ml_global(
        radar_in, ml_global=ml_global, nVol=nVol, maxh=maxh, hres=hres)

    if not is_valid:
        warn('Unable to obtain melting layer information for this radar scan')
        return None, None, None, ml_global

    # Find gates suspected to belong to the melting layer
    ml_points, nml_total = _find_ml_gates(
        ml_global, refl_field=refl_field, zdr_field=zdr_field,
        rhv_field=rhv_field, iso0_field=iso0_field, rmin=rmin, elmin=elmin,
        elmax=elmax, rhomin=rhomin, rhomax=rhomax, zhmin=zhmin,
        hwindow=hwindow, htol=htol, mlzhmin=mlzhmin, mlzhmax=mlzhmax,
        mlzdrmin=mlzdrmin, mlzdrmax=mlzdrmax,
        ml_bottom_diff_max=ml_bottom_diff_max)

    now_time = datetime_from_radar(radar_in)
    if nml_total > 0:
        ml_global = _insert_ml_points(
            ml_global, ml_points, now_time, time_accu_max=time_accu_max)
        # Find melting layer limits using accumulated global data
        ml_top, ml_bottom = _find_ml_limits(
            ml_global, nml_points_min=nml_points_min, wlength=wlength,
            percentile_top=percentile_top,
            percentile_bottom=percentile_bottom, interpol=interpol)
        if ml_top.all() is np.ma.masked:
            if ml_global['time_nodata_start'] is None:
                ml_global['time_nodata_start'] = deepcopy(now_time)
            elif ((now_time - ml_global['time_nodata_start']).total_seconds() >
                  time_nodata_allowed):
                warn('Invalid melting layer data')
                return None, None, None, None
        else:
            ml_global['ml_top'] = ml_top
            ml_global['ml_bottom'] = ml_bottom
            ml_global['time_nodata_start'] = None
    else:
        if ml_global['time_nodata_start'] is None:
            ml_global['time_nodata_start'] = deepcopy(now_time)
        elif ((now_time - ml_global['time_nodata_start']).total_seconds() >
              time_nodata_allowed):
            warn('Invalid melting layer data')
            return None, None, None, None

    # check if valid melting layer limits are available
    if ml_global['ml_top'].all() is np.ma.masked:
        warn('Invalid melting layer data')
        return None, None, None, ml_global

    # Find melting layer top and bottom height of each ray in current radar
    ml_obj = _interpol_ml_limits(
        radar_in, ml_global['ml_top'], ml_global['ml_bottom'],
        ml_global['azi_vec'], ml_pos_field=ml_pos_field)

    # Find position of range gates respect to melting layer top and bottom
    ml_dict = find_ml_field(
        radar_in, ml_obj, ml_pos_field=ml_pos_field, ml_field=ml_field)

    # get the iso0
    iso0_dict = None
    if get_iso0:
        iso0_dict = compute_iso0(
            radar_in, ml_obj.fields[ml_pos_field]['data'][:, 1],
            iso0_field=iso0_field)

    return ml_obj, ml_dict, iso0_dict, ml_global


def melting_layer_hydroclass(radar, hydro_field=None, ml_field=None,
                             ml_pos_field=None, iso0_field=None,
                             force_continuity=True, dist_max=350.,
                             get_iso0=False):
    """
    Using the results of the hydrometeor classification by Besic et al.
    estimates the position of the range gates respect to the melting layer,
    the melting layer top and bottom height and the distance of the range
    gate with respect to the freezing level.

    Parameters
    ----------
    radar : Radar
        Radar object.  Must have and hydrometeor classification field
    hydro_field : str
        Name of the hydrometeor classification field. A value
        of None will use the default field name as defined in the Py-ART
        configuration file.
    ml_field, ml_pos_field, iso0_field : str
        Name of the melting layer, melting layer heightand iso0 field.
        A value of None for any of these parameters will use the default field
        names as defined in the Py-ART configuration file.
    force_continuity : Bool
        If True, the melting layer is forced to be continuous in range
    dist_max : float
        The maximum distance between range gates flagged as inside the melting
        layer to consider them as gates in the melting layer.

    Returns
    -------
    ml_obj : radar-like object
        A radar-like object containing the field melting layer height with
        the bottom (at range position 0) and top (at range position one) of
        the melting layer at each ray
    ml_dict : dict
        A dictionary containg the position of the range gate respect to the
        melting layer and metadata
    iso0_dict : dict or None
        A dictionary containing the distance respect to the melting layer
        and metadata

    """
    # parse the field parameters
    if hydro_field is None:
        hydro_field = get_field_name('radar_echo_classification')
    if ml_field is None:
        ml_field = get_field_name('melting_layer')
    if ml_pos_field is None:
        ml_pos_field = get_field_name('melting_layer_height')
    if iso0_field is None:
        iso0_field = get_field_name('height_over_iso0')

    # check if fieldnames exists
    radar.check_field_exists(hydro_field)

    # get the position of the range gates respect to the melting layer
    ml_dict = get_flag_ml(
        radar, hydro_field=hydro_field, ml_field=ml_field,
        force_continuity=force_continuity, dist_max=dist_max)

    # get the melting layer top and bottom
    ml_obj = get_pos_ml(radar, ml_dict['data'], ml_pos_field=ml_pos_field)

    # get the iso0
    iso0_dict = None
    if get_iso0:
        iso0_dict = compute_iso0(
            radar, ml_obj.fields[ml_pos_field]['data'][:, 1],
            iso0_field=iso0_field)

    return ml_obj, ml_dict, iso0_dict


def compute_theoretical_profile(ml_top=3000., ml_thickness=200.,
                                val_snow=0.99, val_rain=0.99, val_ml=0.93,
                                h_max=6000., h_res=1.):
    """
    Computes an idealized vertical profile. The default values are those of
    RhoHV

    Parameters
    ----------
    ml_top : float
        melting layer top [m asl]
    ml_thickness : float
        melting layer thickness [m]
    val_snow, val_rain, val_ml : float
        values in snow, rain and in the peak of the melting layer
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

    val_theo[h < ml_bottom] = val_rain
    val_theo[(h >= ml_bottom) & (h < ml_peak)] = (
        val_rain - 2.*(val_rain - val_ml)/ml_thickness
        * (h[(h >= ml_bottom) & (h < ml_peak)]-ml_bottom))
    val_theo[(h >= ml_peak) & (h <= ml_top)] = (
        val_ml + 2.*(val_snow - val_ml)/ml_thickness
        * (h[(h >= ml_peak) & (h <= ml_top)]-ml_peak))
    val_theo[h > ml_top] = val_snow

    val_theo_dict = {
        'value': val_theo,
        'altitude': h,
        'ml_top': ml_top,
        'ml_bottom': ml_bottom
    }
    return val_theo_dict


def compute_apparent_profile(radar, ml_top=3000., ml_thickness=200.,
                             rhohv_snow=0.99, rhohv_rain=0.99, rhohv_ml=0.93,
                             zh_snow=20., zh_rain=20., zh_ml=27.,
                             zv_snow=20., zv_rain=20., zv_ml=26.,
                             h_max=6000., h_res=1., beam_factor=2.,
                             npts_diagram=81, rng_bottom_max=200000.,
                             rhohv_field='theoretical_cross_correlation_ratio'):
    """
    Computes the apparent profile of RhoHV

    Parameters
    ----------
    radar : radar object
        the reference radar object
    ml_top, ml_thickness : float
        melting layer top [m asl] and thickness [m]
    rhohv_snow, rhohv_rain, rhohv_ml : float
        values of RhoHV in snow, rain and in the peak of the melting layer
    zh_snow, zh_rain, zh_ml : float
        values of horizontal reflectivity [dBZ] in snow, rain and in the peak
        of the melting layer
    zv_snow, zv_rain, zv_ml : float
        values of vertical reflectivity [dBZ] in snow, rain and in the peak
        of the melting layer
    h_max : float
        maximum altitude at which to compute the theoretical profiles [m asl]
    h_res : float
        profile resolution [m]
    beam_factor : float
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
        A radar object containing the apparent RhoHV profile
    rhohv_theo_dict : dict
        A dictionary containg the theoretical RhoHV profile

    """
    ml_bottom = ml_top - ml_thickness
    radar_out = deepcopy(radar)
    radar_out.fields = dict()
    rhohv_dict = get_metadata(rhohv_field)
    rhohv_dict['data'] = np.ma.masked_all((radar_out.nrays, radar_out.ngates))
    radar_out.add_field(rhohv_field, rhohv_dict)
    if ml_bottom < radar_out.altitude['data']:
        return radar_out

    # get theoretical profiles as a function of altitude
    rhohv_theo_dict = compute_theoretical_profile(
        ml_top=ml_top, ml_thickness=ml_thickness, val_snow=rhohv_snow,
        val_rain=rhohv_rain, val_ml=rhohv_ml, h_max=h_max, h_res=h_res)
    zh_theo_dict = compute_theoretical_profile(
        ml_top=ml_top, ml_thickness=ml_thickness, val_snow=zh_snow,
        val_rain=zh_rain, val_ml=zh_ml, h_max=h_max, h_res=h_res)
    zv_theo_dict = compute_theoretical_profile(
        ml_top=ml_top, ml_thickness=ml_thickness, val_snow=zv_snow,
        val_rain=zv_rain, val_ml=zv_ml, h_max=h_max, h_res=h_res)
    alt_theo = rhohv_theo_dict['altitude']
    rhohv_theo = rhohv_theo_dict['value']
    zh_theo_lin = np.power(10., 0.1*zh_theo_dict['value'])
    zv_theo_lin = np.power(10., 0.1*zv_theo_dict['value'])

    rng = radar_out.range['data']
    # range resolution of the radar resolution volume
    rng_res = rng[1] - rng[0]
    rng_left_km = (rng-rng_res/2.)/1000.
    rng_right_km = (rng+rng_res/2.)/1000.
    # angular resolution of the radar resolution volume defined as a factor
    # of the antenna beam width
    beam_width = (
        radar_out.instrument_parameters['radar_beam_width_h']['data'][0])
    ang_res = beam_factor*beam_width

    ang_diag, weights_diag = compute_antenna_diagram(
        npts_diagram=npts_diagram, beam_factor=beam_factor,
        beam_width=beam_width)

    f_rhohv = interp1d(
        alt_theo, rhohv_theo, kind='nearest', bounds_error=False,
        fill_value=np.nan)
    f_zh = interp1d(
        alt_theo, zh_theo_lin, kind='nearest', bounds_error=False,
        fill_value=np.nan)
    f_zv = interp1d(
        alt_theo, zv_theo_lin, kind='nearest', bounds_error=False,
        fill_value=np.nan)
    for ind_ray, ang in enumerate(radar_out.elevation['data']):
        rng_bottom, rng_top = get_ml_rng_limits(
            rng_left_km, rng_right_km, rng, ang, ang_res,
            radar_out.altitude['data'][0], ml_bottom, ml_top)
        if rng_bottom > rng_bottom_max:
            # the bottom of the area affected by the melting layer is too far
            continue

        i_rng_btm = np.where(rng >= rng_bottom)[0][0]
        i_rng_top = np.where(rng >= rng_top)[0][0]

        # maximum range where to define the apparent RhoHV profile
        rng_max = rng_top + (rng_top - rng_bottom) / 2.
        i_rng_max = np.where(rng >= rng_max)[0]
        if i_rng_max.size == 0:
            i_rng_max = rng.size - 1
        else:
            i_rng_max = i_rng_max[0]

        # values above and below the melting layer affected area
        radar_out.fields[rhohv_field]['data'][ind_ray, 0:i_rng_btm] = (
            rhohv_rain)
        radar_out.fields[rhohv_field]['data'][
            ind_ray, i_rng_top+1:i_rng_max+1] = rhohv_snow

        # values in the area affected by the melting layer
        rng_ml_vals = rng[i_rng_btm:i_rng_top+1] / 1000.  # km
        for i_rng, rng_ml in enumerate(rng_ml_vals):
            # altitudes affected by the antenna diagram
            _, _, z_diag = antenna_to_cartesian(rng_ml, 0., ang+ang_diag)
            z_diag += radar_out.altitude['data']

            rhohv_vals = f_rhohv(z_diag)
            rhohv_vals = np.ma.masked_invalid(rhohv_vals)

            zh_vals = f_zh(z_diag)
            zh_vals = np.ma.masked_invalid(zh_vals)

            zv_vals = f_zv(z_diag)
            zv_vals = np.ma.masked_invalid(zh_vals)

            radar_out.fields[rhohv_field]['data'][
                ind_ray, i_rng_btm+i_rng] = (
                    np.ma.sum(
                        rhohv_vals*np.ma.sqrt(zh_vals*zv_vals)*weights_diag)
                    / np.ma.sqrt(np.ma.sum(zh_vals*weights_diag)
                                 * np.ma.sum(zv_vals*weights_diag)))

    return radar_out, rhohv_theo_dict


def get_ml_rng_limits(rng_left_km, rng_right_km, rng, ang, ang_res,
                      radar_altitude, ml_bottom, ml_top):
    """
    get the minimum and maximum range affected by the melting layer

    Parameters
    ----------
    rng_left_km, rng_right_km : array of floats
        the left and right limits of each range resolution volume [km]
    rng : array of floats
        the radar range (center of the bin) [m]
    ang : float
        the elevation angle
    ang_res : float
        the angle resolution considered
    radar_altitude : float
        the radar altitude [masl]
    ml_bottom, ml_top : float
        the top and bottom of the melting layer [m msl]

    Returns
    -------
    rng_min, rng_max : radar object
        the minimum and maximum ranges affected by the melting layer

    """
    # get altitude of the corners of the radar resolution volume
    _, _, z_top_left = antenna_to_cartesian(rng_left_km, 0., ang+ang_res/2.)
    _, _, z_top_right = antenna_to_cartesian(rng_right_km, 0., ang+ang_res/2.)
    _, _, z_btm_left = antenna_to_cartesian(rng_left_km, 0., ang-ang_res/2.)
    _, _, z_btm_right = antenna_to_cartesian(rng_right_km, 0., ang-ang_res/2.)
    z_top_left += radar_altitude
    z_top_right += radar_altitude
    z_btm_left += radar_altitude
    z_btm_right += radar_altitude

    # check when the corners are crossing the top and bottom of the
    # melting layer
    rng_top_left_min = rng[z_top_left > ml_bottom]
    if rng_top_left_min.size == 0:
        rng_top_left_min = rng[-1]
    else:
        rng_top_left_min = rng_top_left_min[0]

    rng_top_left_max = rng[z_top_left > ml_top]
    if rng_top_left_max.size == 0:
        rng_top_left_max = rng[-1]
    else:
        rng_top_left_max = rng_top_left_max[0]

    rng_top_right_min = rng[z_top_right > ml_bottom]
    if rng_top_right_min.size == 0:
        rng_top_right_min = rng[-1]
    else:
        rng_top_right_min = rng_top_right_min[0]

    rng_top_right_max = rng[z_top_right > ml_top]
    if rng_top_right_max.size == 0:
        rng_top_right_max = rng[-1]
    else:
        rng_top_right_max = rng_top_right_max[0]

    rng_btm_left_min = rng[z_btm_left > ml_bottom]
    if rng_btm_left_min.size == 0:
        rng_btm_left_min = rng[-1]
    else:
        rng_btm_left_min = rng_btm_left_min[0]

    rng_btm_left_max = rng[z_btm_left > ml_top]
    if rng_btm_left_max.size == 0:
        rng_btm_left_max = rng[-1]
    else:
        rng_btm_left_max = rng_btm_left_max[0]

    rng_btm_right_min = rng[z_btm_right > ml_bottom]
    if rng_btm_right_min.size == 0:
        rng_btm_right_min = rng[-1]
    else:
        rng_btm_right_min = rng_btm_right_min[0]

    rng_btm_right_max = rng[z_btm_right > ml_top]
    if rng_btm_right_max.size == 0:
        rng_btm_right_max = rng[-1]
    else:
        rng_btm_right_max = rng_btm_right_max[0]

    rng_ml = np.array([
        rng_top_left_min, rng_top_left_max, rng_top_right_min,
        rng_top_right_max, rng_btm_left_min, rng_btm_left_max,
        rng_btm_right_min, rng_btm_right_max])

    # minimum and maximumrange affected by the melting layer
    return np.min(rng_ml), np.max(rng_ml)


def find_best_profile(radar_obs, ml_thickness_min=200., ml_thickness_max=1400.,
                      ml_thickness_step=100., iso0=3000., iso0_max=4500.,
                      ml_top_diff_max=700., ml_top_step=100., rhohv_snow=0.99,
                      rhohv_rain=0.99, rhohv_ml=0.93, zh_snow=20.,
                      zh_rain=20., zh_ml=27., zv_snow=20., zv_rain=20.,
                      zv_ml=26., h_max=6000., h_res=1., beam_factor=2.,
                      npts_diagram=81, rng_bottom_max=200000., ns_factor=0.6,
                      rhohv_corr_min=0.9, rhohv_nash_min=0.5,
                      rhohv_field_obs='cross_correlation_ratio',
                      rhohv_field_theo='theoretical_cross_correlation_ratio'):
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

    best_rhohv_nash = np.ma.zeros(radar_obs.nrays)-999.
    best_rhohv_nash_bottom = np.ma.zeros(radar_obs.nrays)-999.
    best_ml_thickness = np.ma.zeros(radar_obs.nrays)-999.
    best_ml_bottom = np.ma.zeros(radar_obs.nrays)-999.
    for ml_thickness in ml_thickness_vals:
        for ml_top in ml_top_vals:
            print('\nChecking model with ml top'
                  ' {} [masl] and ml thickness {} m'.format(
                      ml_top, ml_thickness))
            radar_theo, _ = compute_apparent_profile(
                radar_obs, ml_top=ml_top, ml_thickness=ml_thickness,
                rhohv_snow=rhohv_snow, rhohv_rain=rhohv_rain,
                rhohv_ml=rhohv_ml, zh_snow=zh_snow, zh_rain=zh_rain,
                zh_ml=zh_ml, zv_snow=zv_snow, zv_rain=zv_rain,
                zv_ml=zv_ml, h_max=h_max, h_res=h_res,
                beam_factor=beam_factor, npts_diagram=npts_diagram,
                rng_bottom_max=rng_bottom_max, rhohv_field=rhohv_field_theo)
            for i_ang, ang in enumerate(radar_obs.elevation['data']):
                # print('Angle: {}'.format(ang))
                rhohv_nash = compare_rhohv(
                    radar_obs.fields[rhohv_field_obs]['data'][i_ang, :],
                    radar_theo.fields[rhohv_field_theo]['data'][i_ang, :],
                    ns_factor=ns_factor, rhohv_corr_min=rhohv_corr_min,
                    rhohv_nash_min=rhohv_nash_min,
                    best_rhohv_nash=best_rhohv_nash[i_ang])
                if rhohv_nash is not None:
                    best_rhohv_nash[i_ang] = rhohv_nash
                    best_ml_thickness[i_ang] = ml_thickness
                    best_ml_bottom[i_ang] = ml_top - ml_thickness
                    print('\nVALID MODEL for top and bottom ML at angle'
                          ' {}. Nash: {}\n'.format(ang, rhohv_nash))
                if best_ml_thickness[i_ang] > 0:
                    continue
                # print('No valid model for top and bottom ML found')
                rhohv_theo_ma = mask_ml_top(
                    radar_theo.fields[rhohv_field_theo]['data'][i_ang, :])
                rhohv_nash = compare_rhohv(
                    radar_obs.fields[rhohv_field_obs]['data'][i_ang, :],
                    rhohv_theo_ma, ns_factor=ns_factor,
                    rhohv_corr_min=rhohv_corr_min,
                    rhohv_nash_min=rhohv_nash_min,
                    best_rhohv_nash=best_rhohv_nash_bottom[i_ang])
                if rhohv_nash is not None:
                    best_rhohv_nash_bottom[i_ang] = rhohv_nash
                    best_ml_bottom[i_ang] = ml_top - ml_thickness
                    print('\nVALID MODEL for bottom ML at angle'
                          ' {}. Nash: {}\n'.format(ang, rhohv_nash))

    best_ml_thickness = np.ma.masked_values(best_ml_thickness, -999.)
    best_ml_bottom = np.ma.masked_values(best_ml_bottom, -999.)
    best_rhohv_nash = np.ma.masked_values(best_rhohv_nash, -999.)
    best_rhohv_nash_bottom = np.ma.masked_values(best_rhohv_nash_bottom, -999.)

    return (best_ml_thickness, best_ml_bottom, best_rhohv_nash,
            best_rhohv_nash_bottom)


def filter_ml(best_ml_thickness, best_ml_bottom, iso0, ang, ang_iso0=10.,
              age_iso0=3., ml_thickness_iso0=700.):
    """
    Get the best estimate of the melting layer with the information available

    Parameters
    ----------
    best_ml_thickness, best_ml_bottom : array of floats
        The estimated melting layer thickness [m] and bottom [masl] at each
        elevation
    iso0 : float
        the iso0 altitude [masl]
    ang : array of floats
        The elevation angles
    ang_iso0 : float
        the equivalent iso0 angle; Used for the computation of the weights
    age_iso0 : float
        the equivalent age of the iso0 (hours)
    ml_thickness_iso0 : float
        Default iso-0 thickness

    Returns
    -------
    ml_bottom, ml_thickness : float
        The melting layer bottom and thickness

    """
    ml_thickness_arr = np.ma.append(best_ml_thickness, ml_thickness_iso0)
    ml_bottom_arr = np.ma.append(best_ml_bottom, iso0-ml_thickness_iso0)
    ang_arr = np.ma.append(ang, ang_iso0)
    age_arr = np.ma.zeros(ang.size+1)
    age_arr[-1] = age_iso0

    weight = np.sqrt(ang_arr)*np.power(2., -age_arr)
    weight_ml_thickness = np.ma.masked_where(
        np.ma.getmaskarray(ml_thickness_arr), weight)
    weight_ml_bottom = np.ma.masked_where(
        np.ma.getmaskarray(ml_bottom_arr), weight)
    ml_thickness = (
        np.ma.sum(weight_ml_thickness*ml_thickness_arr)
        / np.ma.sum(weight_ml_thickness))
    ml_bottom = (
        np.ma.sum(weight_ml_bottom*ml_bottom_arr)
        / np.ma.sum(weight_ml_bottom))

    return ml_bottom, ml_thickness


def compare_rhohv(rhohv_obs, rhohv_theo, ns_factor=0.6, rhohv_corr_min=0.9,
                  rhohv_nash_min=0.5, best_rhohv_nash=-999.):
    """
    Compares the observed and the modelled RhoHV profiles

    Parameters
    ----------
    rhohv_obs : array of floats
        The observed RhoHV profile
    rhohv_theo : array of floats
        The modelled RhoHV profile
    ns_factor : float
        multiplicative factor to the number of valid modelled RhoHV gates.
        Used when comparing with the observations
    rhohv_corr_min : float
        Minimum correlation coefficient to consider the comparison valid
    rhohv_nash_min : float
        Minimum Nash to consider the comparison valid
    best_rhohv_nash : float
        Best RhoHV Nash from previous comparisons

    Returns
    -------
    rhohv_nash : Float or None
        If the Nash is better than in previous ones returns its value.
        Otherwise returns None

    """
    if not enough_data(rhohv_obs, rhohv_theo, ns_factor=ns_factor):
        # warn('Not enough valid data in profile')
        return None
    rhohv_corr = compute_corr(rhohv_obs, rhohv_theo)
    if rhohv_corr is None:
        # warn('Unable to compute corr')
        return None
    if rhohv_corr <= rhohv_corr_min:
        # warn('Correlation {} below {}'.format(rhohv_corr, rhohv_corr_min))
        return None
    # print('Correlation {}'.format(rhohv_corr))
    rhohv_nash = compute_nse(rhohv_obs, rhohv_theo)
    if rhohv_nash is None:
        # warn('Unable to compute NSE')
        return None
    if rhohv_nash <= rhohv_nash_min or rhohv_nash <= best_rhohv_nash:
        # warn('NSE {} below min NSE {} or best NSE {}'.format(
        #      rhohv_nash, rhohv_nash_min, best_rhohv_nash))
        return None
    return rhohv_nash


def enough_data(rhohv_obs, rhohv_theo, ns_factor=0.6):
    """
    Check whether there is enough valid data to compare the observed and the
    modelled RhoHV profiles

    Parameters
    ----------
    rhohv_obs : array of floats
        The observed RhoHV profile
    rhohv_theo : array of floats
        The modelled RhoHV profile
    ns_factor : float
        multiplicative factor to the number of valid modelled RhoHV gates.
        Used when comparing with the observations

    Returns
    -------
    enough_data : bool
        True if there is enough data. False otherwise

    """
    nvalid_theo = np.sum(np.ma.getmaskarray(rhohv_obs))
    nvalid_obs = np.sum(np.ma.getmaskarray(rhohv_theo))
    if nvalid_obs > nvalid_theo * ns_factor:
        return True
    return False


def mask_ml_top(rhohv):
    """
    Masks the RhoHV profile above the bright band peak

    Parameters
    ----------
    rhohv : array of floats
        The RhoHV profile

    Returns
    -------
    rhohv_masked : array of floats
        The masked RhoHV profile

    """
    rhohv_masked = deepcopy(rhohv)
    ind_min = np.argmin(rhohv_masked)
    rhohv_masked[ind_min+1:] = np.ma.masked
    return rhohv_masked


def get_iso0_val(radar, temp_ref_field='heigh_over_iso0',
                 temp_ref='heigh_over_iso0'):
    """
    Computes the altitude of the iso-0°

    Parameters
    ----------
    radar : Radar
        Radar object.
    iso0_field : str
        Name of the field, can be height over the iso0 field or temperature
    temp_ref : str
        temperature reference field to use

    Returns
    -------
    iso0 : float
        The altitude of the iso-0

    """
    iso0_min = 20000
    for i_ang in range(radar.elevation['data'].size):
        if temp_ref == 'height_over_iso0':
            ind = np.ma.where(
                radar.fields[temp_ref_field]['data'][i_ang, :] >= 0.)[0]
        else:
            ind = np.ma.where(
                radar.fields[temp_ref_field]['data'][i_ang, :] <= 0.)[0]
        if ind.size == 0:
            # all gates below the iso-0
            iso0 = radar.gate_altitude['data'][i_ang, -1]
        else:
            iso0 = radar.gate_altitude['data'][i_ang, ind[0]]
        if iso0 < iso0_min:
            iso0_min = iso0
    return iso0_min


def get_flag_ml(radar, hydro_field='radar_echo_classification',
                ml_field='melting_layer', force_continuity=False,
                dist_max=350.):
    """
    Using the results of the hydrometeor classification by Besic et al.
    estimates the position of the range gates respect to the melting layer.

    Parameters
    ----------
    radar : Radar
        Radar object.  Must have and hydrometeor classification field
    hydro_field : str
        Name of the hydrometeor classification field.
    ml_field : str
        Name of the melting layer field.
    force_continuity : Bool
        If True, the melting layer is forced to be continuous in range
    dist_max : float
        The maximum distance between range gates flagged as inside the melting
        layer to consider them as gates in the melting layer.

    Returns
    -------
    ml_dict : dict
        A dictionary containg the position of the range gate respect to the
        melting layer and metadata

    """
    hydro_data = radar.fields[hydro_field]['data']
    is_ds = hydro_data == 2
    is_cr = hydro_data == 3
    is_lr = hydro_data == 4
    is_gr = hydro_data == 5
    is_rn = hydro_data == 6
    is_vi = hydro_data == 7
    is_ws = hydro_data == 8
    # is_mh = hydro_data == 9
    is_ih = hydro_data == 10

    # Assign melting layer flag
    ml_data = np.ma.masked_all(hydro_data.shape, dtype=np.uint8)
    ml_data[is_ds] = 5
    ml_data[is_cr] = 5
    ml_data[is_lr] = 1
    ml_data[is_gr] = 5
    ml_data[is_rn] = 1
    ml_data[is_vi] = 5
    ml_data[is_ws] = 3
    ml_data[is_ih] = 5

    if not force_continuity:
        ml_dict = get_metadata(ml_field)
        ml_dict['data'] = ml_data
        return ml_dict

    # keep original mask
    mask = deepcopy(np.ma.getmaskarray(ml_data))
    for ind_ray in range(radar.nrays):
        inds_rng_ml = np.ma.where(ml_data[ind_ray, :] == 3)[0]

        # No melting layer identified in ray. Mask the entire ray
        if inds_rng_ml.size == 0:
            ml_data[ind_ray, :] = np.ma.masked
            continue

        # There is just one gate. Do nothing
        if inds_rng_ml.size == 1:
            continue

        # identify continuos regions
        rng_ml = radar.range['data'][inds_rng_ml]
        dist_ml = np.append(rng_ml[1:]-rng_ml[0:-1], rng_ml[-1]-rng_ml[-2])
        ind_valid = np.where(dist_ml < dist_max)[0]
        inds_rng_ml = inds_rng_ml[ind_valid]

        # melting layer too fractioned. Remove ray
        if inds_rng_ml.size == 0:
            ml_data[ind_ray, :] = np.ma.masked
            continue

        # Fill in gaps
        ml_data[ind_ray, :] = 3
        if inds_rng_ml[0] > 0:
            ml_data[ind_ray, 0:inds_rng_ml[0]] = 1
        if inds_rng_ml[-1] < radar.ngates-1:
            ml_data[ind_ray, inds_rng_ml[-1]+1:] = 5

    ml_data = np.ma.masked_where(mask, ml_data)

    # prepare output
    ml_dict = get_metadata(ml_field)
    ml_dict['data'] = ml_data

    return ml_dict


def get_pos_ml(radar, ml_data, ml_pos_field='melting_layer_height'):
    """
    Estimates the height of the top and bottom of the melting layer from a
    field containing the position of each range gate respect to the melting
    layer.

    Parameters
    ----------
    radar : Radar
        Radar object
    ml_data : 2D array
        field containing the position of each range gate respect to the
        melting layer.
    ml_pos_field : str
        Name of the melting layer height field.

    Returns
    -------
    ml_obj : radar-like object
        A radar-like object containing the field melting layer height with
        the bottom (at range position 0) and top (at range position one) of
        the melting layer at each ray

    """
    # Create melting layer object containing top and bottom and metadata
    ml_obj = _create_ml_obj(radar, ml_pos_field)

    hlowerleft, hupperright = _get_res_vol_sides(radar)
    for ind_ray in range(radar.nrays):
        inds_rng_ml = np.ma.where(ml_data[ind_ray, :] == 3)[0]

        # No melting layer
        if inds_rng_ml.size == 0:
            continue

        # get top and bottom
        ml_obj.fields[ml_pos_field]['data'][ind_ray, 0] = (
            hlowerleft[ind_ray, inds_rng_ml[0]])
        ml_obj.fields[ml_pos_field]['data'][ind_ray, 1] = (
            hupperright[ind_ray, inds_rng_ml[-1]])

    return ml_obj


def compute_iso0(radar, ml_top, iso0_field='height_over_iso0'):
    """
    Estimates the distance respect to the freezing level of each range gate
    using the melting layer top as a proxy

    Parameters
    ----------
    radar : Radar
        Radar object
    ml_top : 1D array
        The height of the melting layer at each ray
    iso0_field : str
        Name of the iso0 field.

    Returns
    -------
    iso0_dict : dict
        A dictionary containing the distance respect to the melting layer
        and metadata

    """
    iso0_data = np.ma.masked_all((radar.nrays, radar.ngates))
    for ind_ray in range(radar.nrays):
        iso0_data[ind_ray, :] = (
            radar.gate_altitude['data'][ind_ray, :]-ml_top[ind_ray])

    iso0_dict = get_metadata(iso0_field)
    iso0_dict['data'] = iso0_data

    return iso0_dict


def find_ml_field(radar_in, ml_obj, ml_pos_field='melting_layer_height',
                  ml_field='melting_layer'):
    """
    Obtains the field of position respect to the melting layer from the top and
    bottom height of the melting layer

    Parameters
    ----------
    radar_in : radar object
        The current radar
    ml_obj : radar-like object
        A radar-like object containing the field melting layer height with
        the bottom (at range position 0) and top (at range position one) of
        the melting layer at each ray
    ml_pos_field : 1D-array
        The reference azimuth angle
    ml_pos_field : str
        The name of the melting layer height field
    ml_field : str
        The name of the melting layer field

    Returns
    -------
    ml_dict : dict
        A dictionary containg the position of the range gate respect to the
        melting layer and metadata
    """
    hlowerleft, hupperright = _get_res_vol_sides(radar_in)

    ml_data = np.ma.masked_all(
        (radar_in.nrays, radar_in.ngates), dtype=np.uint8)
    for ind_ray in range(radar_in.nrays):
        hlowerleft_ray = hlowerleft[ind_ray, :]
        hupperright_ray = hupperright[ind_ray, :]
        ml_bottom_ray = ml_obj.fields[ml_pos_field]['data'][ind_ray, 0]
        ml_top_ray = ml_obj.fields[ml_pos_field]['data'][ind_ray, 1]

        # flag gates
        gates_below = hupperright_ray < ml_bottom_ray
        gates_entering = np.logical_and(
            hupperright_ray >= ml_bottom_ray, hlowerleft_ray < ml_bottom_ray)
        gates_within = np.logical_and(
            hupperright_ray <= ml_top_ray, hlowerleft_ray >= ml_bottom_ray)
        gates_exiting = np.logical_and(
            hupperright_ray > ml_top_ray, hlowerleft_ray >= ml_bottom_ray)
        gates_above = hlowerleft_ray > ml_top_ray

        ml_data[ind_ray, gates_below] = 1
        ml_data[ind_ray, gates_entering] = 2
        ml_data[ind_ray, gates_within] = 3
        ml_data[ind_ray, gates_exiting] = 4
        ml_data[ind_ray, gates_above] = 5

    ml_dict = get_metadata(ml_field)
    ml_dict['data'] = ml_data

    return ml_dict


def interpol_field(radar_dest, radar_orig, field_name, fill_value=None):
    """
    interpolates field field_name contained in radar_orig to the grid in
    radar_dest

    Parameters
    ----------
    radar_dest : radar object
        the destination radar
    radar_orig : radar object
        the radar object containing the original field
    field_name: str
        name of the field to interpolate

    Returns
    -------
    field_dest : dict
        interpolated field and metadata

    """
    if fill_value is None:
        fill_value = get_fillvalue()

    field_orig_data = radar_orig.fields[field_name]['data'].filled(
        fill_value=fill_value)
    field_dest = deepcopy(radar_orig.fields[field_name])
    field_dest['data'] = np.ma.masked_all(
        (radar_dest.nrays, radar_dest.ngates))

    for sweep in range(radar_dest.nsweeps):
        sweep_start_orig = radar_orig.sweep_start_ray_index['data'][sweep]
        sweep_end_orig = radar_orig.sweep_end_ray_index['data'][sweep]

        sweep_start_dest = radar_dest.sweep_start_ray_index['data'][sweep]
        sweep_end_dest = radar_dest.sweep_end_ray_index['data'][sweep]

        if radar_dest.scan_type == 'ppi':
            angle_old = np.sort(radar_orig.azimuth['data'][
                sweep_start_orig:sweep_end_orig+1])
            ind_ang = np.argsort(radar_orig.azimuth['data'][
                sweep_start_orig:sweep_end_orig+1])
            angle_new = radar_dest.azimuth['data'][
                sweep_start_dest:sweep_end_dest+1]
        elif radar_dest.scan_type == 'rhi':
            angle_old = np.sort(radar_orig.elevation['data'][
                sweep_start_orig:sweep_end_orig+1])
            ind_ang = np.argsort(radar_orig.azimuth['data'][
                sweep_start_orig:sweep_end_orig+1])
            angle_new = radar_dest.elevation['data'][
                sweep_start_dest:sweep_end_dest+1]

        field_orig_sweep_data = field_orig_data[
            sweep_start_orig:sweep_end_orig+1, :]
        interpol_func = RegularGridInterpolator(
            (angle_old, radar_orig.range['data']),
            field_orig_sweep_data[ind_ang], method='nearest',
            bounds_error=False, fill_value=fill_value)

        # interpolate data to radar_dest grid
        angv, rngv = np.meshgrid(
            angle_new, radar_dest.range['data'], indexing='ij')

        field_dest_sweep = interpol_func((angv, rngv))
        field_dest_sweep = np.ma.masked_where(
            field_dest_sweep == fill_value, field_dest_sweep)

        field_dest['data'][sweep_start_dest:sweep_end_dest+1, :] = (
            field_dest_sweep)

    return field_dest


def _create_ml_obj(radar, ml_pos_field='melting_layer_height'):
    """
    Creates a radar-like object that will be used to contain the melting layer
    top and bottom

    Parameters
    ----------
    radar : Radar
        Radar object
    ml_pos_field : str
        Name of the melting layer height field

    Returns
    -------
    ml_obj : radar-like object
        A radar-like object containing the field melting layer height with
        the bottom (at range position 0) and top (at range position one) of
        the melting layer at each ray

    """
    ml_obj = deepcopy(radar)

    # modify original metadata
    ml_obj.range['data'] = np.array([0, 1], dtype='float64')
    ml_obj.ngates = 2

    ml_obj.gate_x = np.zeros((ml_obj.nrays, ml_obj.ngates), dtype=float)
    ml_obj.gate_y = np.zeros((ml_obj.nrays, ml_obj.ngates), dtype=float)
    ml_obj.gate_z = np.zeros((ml_obj.nrays, ml_obj.ngates), dtype=float)

    ml_obj.gate_longitude = np.zeros(
        (ml_obj.nrays, ml_obj.ngates), dtype=float)
    ml_obj.gate_latitude = np.zeros(
        (ml_obj.nrays, ml_obj.ngates), dtype=float)
    ml_obj.gate_altitude = np.zeros(
        (ml_obj.nrays, ml_obj.ngates), dtype=float)

    # Create field
    ml_obj.fields = dict()
    ml_dict = get_metadata(ml_pos_field)
    ml_dict['data'] = np.ma.masked_all((ml_obj.nrays, ml_obj.ngates))
    ml_obj.add_field(ml_pos_field, ml_dict)

    return ml_obj


def _prepare_radar(radar, field_list, temp_ref='temperature',
                   iso0_field='height_over_iso0', temp_field='temperature',
                   lapse_rate=-6.5):
    """
    Select radar fields to use. Prepare the field height over the iso0 as
    according to the temperature reference

    Parameters
    ----------
    radar : Radar object
        current radar object
    field_list : str
        List of fields that will be used to get the melting layer
    temp_ref : str
        the field used as reference for temperature. Can be temperature
        height_over_iso0, no_field.
        If None, Outputs a dummy height_over_iso0 field
    iso0_field, temp_field : str
        Name of the fields height over iso0 and temperature
    lapse_rate : float
        lapse rate to convert temperature into height with respect to the iso0

    Returns
    -------
    radar_in : Radar object or None
        The radar object containing only relevant fields

    """
    radar_in = deepcopy(radar)
    radar_in.fields = dict()
    for field in field_list:
        try:
            radar.check_field_exists(field)
            radar_in.add_field(field, radar.fields[field])
        except KeyError:
            return None

    if temp_ref == 'temperature':
        # convert temp in relative height respect to iso0
        temp = radar_in.fields[temp_field]['data']
        relh = temp*(1000./lapse_rate)
        iso0_dict = get_metadata(iso0_field)
        iso0_dict['data'] = relh
        radar_in.add_field(iso0_field, iso0_dict)
    elif temp_ref is None:
        relh = np.ma.zeros((radar.nrays, radar.ngates))
        iso0_dict = get_metadata(iso0_field)
        iso0_dict['data'] = relh
        radar_in.add_field(iso0_field, iso0_dict)

    return radar_in


def _get_ml_global(radar_in, ml_global=None, nVol=3, maxh=6000., hres=50.):
    """
    Gets global data to be used in melting layer detection

    Parameters
    ----------
    radar_in : Radar object
        current radar object
    nVol : int
        Number of consecutive volumes to use to obtain the melting layer
    maxh : float
        Maximum possible height of the melting layer [m MSL]
    hres : float
        Resolution of the height vector

    Returns
    -------
    ml_global : dict or None
        A dictionary with the data necessary to estimate the melting
        layer. It has the following keys::
            iVol: int
                index of current radar volume (Maximum nVol)
            time_nodata_start : datetime object
                The date and time where the first radar volume with no data
                was found
            ml_points : 3D-array
                an array (nAzimuth, nHeight, nVol) to store the number of
                suspected melting layer points
            ml_top, ml_bottom: 1D-array
                an array (nAzimuth) to store the top and bottom height of the
                melting layer
            azi_vec : 1D-array
                The azimuth values
            alt_vec : 1D-array
                The altitude values
            radar_ref : radar object
                The rhi volume radar used as reference
    is_valid : Bool
        Indicates whether the current radar volume can be processed
    """
    if ml_global is None:
        if radar_in.scan_type == 'ppi':
            target_azimuths, az_tol = _get_target_azimuths(radar_in)
            radar_rhi = cross_section_ppi(
                radar_in, target_azimuths, az_tol=az_tol)
        elif radar_in.scan_type == 'rhi':
            radar_rhi = radar_in
        else:
            warn('Error: unsupported scan type.')
            return None, False

        nHeight = int(maxh/hres)+1
        alt_vec = np.arange(nHeight)*hres
        nAzimuth = radar_rhi.nsweeps
        ml_global = {
            'time_nodata_start': None,
            'ml_points': np.ma.masked_all(
                (nAzimuth, nHeight, nVol), dtype=int),
            'ml_top': np.ma.masked_all(nAzimuth),
            'ml_bottom': np.ma.masked_all(nAzimuth),
            'azi_vec': np.sort(radar_rhi.fixed_angle['data']),
            'alt_vec': alt_vec,
            'time_accu_vec': np.ma.masked_all(nVol, dtype=datetime.datetime),
            'radar_ref': radar_rhi}
    else:
        if radar_in.scan_type == 'ppi':
            target_azimuths, az_tol = _get_target_azimuths(radar_in)
            radar_rhi = cross_section_ppi(
                radar_in, ml_global['azi_vec'], az_tol=az_tol)
        elif radar_in.scan_type == 'rhi':
            # put radar fields in reference radar format
            radar_rhi = deepcopy(ml_global['radar_ref'])
            radar_rhi.fields = dict()

            for field in radar_in.fields:
                field_dict = interpol_field(
                    ml_global['radar_ref'], radar_in, field)
                radar_rhi.add_field(field, field_dict)
        else:
            warn('Error: unsupported scan type.')
            return ml_global, False

        ml_global['radar_ref'] = radar_rhi

    return ml_global, True


def _get_target_azimuths(radar_in):
    """
    Gets RHI taget azimuths

    Parameters
    ----------
    radar_in : Radar object
        current radar object

    Returns
    -------
    target_azimuths : 1D-array
        Azimuth angles
    az_tol : float
        azimuth tolerance
    """
    sweep_start = radar_in.sweep_start_ray_index['data'][0]
    sweep_end = radar_in.sweep_end_ray_index['data'][0]
    target_azimuths = np.sort(
        radar_in.azimuth['data'][sweep_start:sweep_end+1])
    az_tol = np.median(target_azimuths[1:]-target_azimuths[:-1])

    return target_azimuths, az_tol


def _find_ml_gates(ml_global, refl_field='reflectivity',
                   zdr_field='differential_reflectivity',
                   rhv_field='cross_correlation_ratio',
                   iso0_field='height_over_iso0', rmin=1000., elmin=4.,
                   elmax=10., rhomin=0.75, rhomax=0.94, zhmin=20.,
                   htol=500., hwindow=500., mlzhmin=30., mlzhmax=50.,
                   mlzdrmin=1.0, mlzdrmax=5.0, ml_bottom_diff_max=1000.):
    """
    Find gates suspected to be melting layer contaminated

    Parameters
    ----------
    ml_global : dict
        Dictionary containing global melting layer data
    refl_field, zdr_field, rhv_field, iso0_field : str
        Name of fields used to find melting layer
    rmin : float
        Minimum range from radar where to start looking for melting layer
        gates
    elmin, elmax : float
        Minimum and maximum elevation angles to use
    rhomin, rhomax : float
        Minimum and maximum values of RhoHV to consider a range gate
        potentially melting layer contaminated
    zhmin : float
        Minimum value of reflectivity to consider a range gate potentially
        melting layer contaminated
    htol : float
        Maximum height above the iso-0 where to look for melting layer
        contaminated gates
    hwindow : float
        Maximum range above the suspected melting layer contaminated gate to
        look for a peak
    mlzhmin, mlzhmax : float
        Minimum and maximum values of the peak reflectivity above the melting
        layer contaminated range gate to consider it valid
    mlzdrmin, mlzdrmax : float
        Minimum and maximum values of the peak differential reflectivity above
        the melting layer contaminated range gate to consider it valid
    ml_bottom_diff_max : float
        The maximum difference in altitude between the current suspected
        melting layer gate and the previously retrieved melting layer [m]

    Returns
    -------
    ml_points : 2D-array
        A 2D-array (nAzimuth, nHeight) with the number of points found
    nml_total : int
        Number of range gates identified as suspected melting layer
        contamination
    """
    maxh = ml_global['alt_vec'][-1]
    hres = ml_global['alt_vec'][1]-ml_global['alt_vec'][0]

    radar_rhi = ml_global['radar_ref']
    ind_rmin = np.where(radar_rhi.range['data'] >= rmin)[0][0]

    # loop azimuths
    ml_points = np.zeros(ml_global['ml_points'].shape[:2], dtype=int)
    nml_total = 0
    for ind_azi, sweep_slice in enumerate(radar_rhi.iter_slice()):
        # extract sweep (fixed azimuth, excluding close ranges)
        elevation_sweep = radar_rhi.elevation['data'][sweep_slice]
        rhohv_sweep = (
            radar_rhi.fields[rhv_field]['data'][sweep_slice, ind_rmin:])
        refl_sweep = (
            radar_rhi.fields[refl_field]['data'][sweep_slice, ind_rmin:])
        zdr_sweep = (
            radar_rhi.fields[zdr_field]['data'][sweep_slice, ind_rmin:])
        hcenter_sweep = (
            radar_rhi.gate_altitude['data'][sweep_slice, ind_rmin:])
        iso0_height_sweep = (
            radar_rhi.fields[iso0_field]['data'][sweep_slice, ind_rmin:])

        # gates in sweep that fulfill conditions on elevation angle, rhohv,
        # refl and gate altitude
        ind_ml = np.logical_and.reduce((
            np.repeat(
                np.logical_or(
                    np.logical_and(
                        elevation_sweep >= elmin, elevation_sweep <= elmax),
                    np.logical_and(
                        elevation_sweep <= 180-elmin,
                        elevation_sweep >= 180-elmax)
                )[:, None],
                radar_rhi.ngates - ind_rmin, axis=1),
            rhohv_sweep <= rhomax,
            rhohv_sweep >= rhomin,
            refl_sweep >= zhmin,
            hcenter_sweep < maxh,
            ), axis=0)

        # number of valid gates
        nml = ind_ml.sum()
        if nml == 0:
            continue

        nml_total += nml

        # all valid gates in n-th sweep
        ind_rays, ind_range = np.where(ind_ml)
        # loop over valid gates in n-th sweep
        for ind in range(nml):
            # Check if point is well below the limits of the previous melting
            # layer bottom
            if ml_global['ml_bottom'][ind_azi] is not np.ma.masked:
                if (hcenter_sweep[ind_rays[ind], ind_range[ind]] <
                        ml_global['ml_bottom'][ind_azi]-ml_bottom_diff_max):
                    continue

            # Check if point is within the tolerance of the freezing level
            if iso0_height_sweep[ind_rays[ind], ind_range[ind]] > htol:
                continue

            # and find all valid gates that are within hwindow meters
            # above (but along the same ray!)
            ind_gates_above = np.logical_and(
                (hcenter_sweep[ind_rays[ind], :] -
                 hcenter_sweep[ind_rays[ind], ind_range[ind]] < hwindow),
                (hcenter_sweep[ind_rays[ind], :] -
                 hcenter_sweep[ind_rays[ind], ind_range[ind]] >= 0))

            # compute peaks of zh and zdr
            zhmax = refl_sweep[ind_rays[ind], ind_gates_above].max()
            zdrmax = zdr_sweep[ind_rays[ind], ind_gates_above].max()

            # check whether peaks respect given criteria
            if (mlzhmin <= zhmax <= mlzhmax
                    and mlzdrmin <= zdrmax <= mlzdrmax):
                # add point to given azimuth and height
                ind_alt = int(
                    hcenter_sweep[ind_rays[ind], ind_range[ind]]/hres)
                ml_points[ind_azi, ind_alt] += 1

    return ml_points, nml_total


def _insert_ml_points(ml_global, ml_points, time_current, time_accu_max=1800.):
    """
    Insert the current suspected melting layer points in the memory array

    Parameters
    ----------
    ml_global : dict
        Dictionary containing global melting layer data
    ml_points : 2D-array
        A 2D-array (nAzimuth, nHeight) with the current number of points found
    time_current : datetime object
        The current time
    time_accu_max : float
        Maximum accumulation time [s]

    Returns
    -------
    ml_global : dict
        The global melting layer data with the points updated
    """
    time_accu_vec = ml_global['time_accu_vec']
    time_diff = np.ma.masked_all(time_accu_vec.size)
    for ind_vol, time_accu in enumerate(time_accu_vec):
        if time_accu is np.ma.masked:
            continue
        time_diff[ind_vol] = (time_current-time_accu).total_seconds()

    # remove data that is too old
    ind_vol = np.ma.where(time_diff > time_accu_max)[0]
    ml_global['ml_points'][:, :, ind_vol] = np.ma.masked
    time_accu_vec[ind_vol] = np.ma.masked

    # place the data in a free spot or in the place of the oldest data
    ind_vol_free = np.where(np.ma.getmaskarray(time_accu_vec))[0]
    if ind_vol_free.size > 0:
        ind_vol = ind_vol_free[0]
    else:
        ind_vol = np.ma.argmax(time_diff)
    ml_global['ml_points'][:, :, ind_vol] = ml_points
    ml_global['time_accu_vec'][ind_vol] = time_current

    return ml_global


def _find_ml_limits(ml_global, nml_points_min=None, wlength=20.,
                    percentile_bottom=0.3, percentile_top=0.9, interpol=True):
    """
    Find melting layer limits

    Parameters
    ----------
    ml_global : dict
        Dictionary containing global melting layer data
    nml_points_min : int or None
        Minimum number of suspected melting layer contaminated range gates
        to obtain limits. If none it will be defined as a function of the
        number of azimuths
    wlength : int or None
        Size of the window in azimuth to use when identifying melting layer
        points [degree]
    percentile_bottom, percentile_top : float
        Percentile of range gates suspected of being melting layer
        contaminated that has to be reached to estimate the bottom and top of
        the melting layer
    interpol : bool
        Whether to interpolate or not accross the azimuths

    Returns
    -------
    ml_top, ml_bottom : 1D-array
        The top and bottom melting layer height at each azimuth
    """
    # Find minimum number of points according to azimuths at disposal
    if nml_points_min is None:
        nazi = int(
            wlength /
            np.median(ml_global['azi_vec'][1:]-ml_global['azi_vec'][:-1]))
        if nazi == 0:
            nml_points_min = 10
        else:
            nml_points_min = int(10*nazi)
        warn('Minimum number of suspected melting layer range gates ' +
             'for a valid retrieval: '+str(nml_points_min))

    azi_angles = ml_global['azi_vec']
    ml_points = ml_global['ml_points']
    alt_vec = ml_global['alt_vec']

    # loop azimuths
    whalflength = 0.5*wlength
    ml_bottom = np.ma.masked_all(azi_angles.size, dtype='float32')
    ml_top = np.ma.masked_all(azi_angles.size, dtype='float32')
    for ind_azi, azi_angl in enumerate(azi_angles):
        # identify neighbouring azimuths as defined by the wlength parameter
        angl_diff = 180.-np.abs(np.abs(azi_angles - azi_angl) - 180.)
        ind_azi_wind = angl_diff <= whalflength

        # ml points for given azimuth, all volumes,
        # considering neighbouring azimuths
        ml_points_ext = ml_points[ind_azi_wind, :, :].sum(axis=(0, 2))

        if ml_points_ext.sum() < nml_points_min:
            # print('Not enough suspected melting layer gates. ' +
            #       'Min points: '+str(nml_points_min)+' Points: ' +
            #       str(ml_points_ext.sum()))
            continue

        # get the indices of the top and bottom percentiles for ml at a
        # given azimuth
        ml_points_cumsum = np.cumsum(ml_points_ext)
        ind_ml_top = np.argmax(
            ml_points_cumsum >= percentile_top*ml_points_ext.sum())
        ind_ml_bottom = np.argmax(
            ml_points_cumsum >= percentile_bottom*ml_points_ext.sum())

        # get the corresponding ml top and bottom heights [m]
        if ind_ml_top > ind_ml_bottom:
            ml_top[ind_azi] = alt_vec[ind_ml_top]
            ml_bottom[ind_azi] = alt_vec[ind_ml_bottom]

    if not interpol:
        return ml_top, ml_bottom

    # if no points where found do not do interpolation
    if ml_top.all() is np.ma.masked:
        return ml_top, ml_bottom

    # fill the gaps with angular interpolation
    ml_top[ml_top.mask] = np.interp(
        azi_angles[ml_top.mask], azi_angles[~ml_top.mask],
        ml_top[~ml_top.mask], period=360)
    ml_bottom[ml_bottom.mask] = np.interp(
        azi_angles[ml_bottom.mask], azi_angles[~ml_bottom.mask],
        ml_bottom[~ml_bottom.mask], period=360)

    return ml_top, ml_bottom


def _interpol_ml_limits(radar_in, ml_top, ml_bottom, ml_azi_angl,
                        ml_pos_field='melting_layer_height'):
    """
    Interpolate melting layer limits to obtain a value at each ray of the
    current radar object

    Parameters
    ----------
    radar_in : radar object
        The current radar
    ml_top, ml_bottom : 1D-array
        The top and bottom of the melting layer at each reference azimuth
        angle
    ml_azi_angl : 1D-array
        The reference azimuth angle
    ml_pos_field : str
        The name of the melting layer height field

    Returns
    -------
    ml_obj : radar-like object
        A radar-like object containing the field melting layer height with
        the bottom (at range position 0) and top (at range position one) of
        the melting layer at each ray
    """
    # Create melting layer object containing top and bottom and metadata
    ml_obj = _create_ml_obj(radar_in, ml_pos_field)

    # interpolate melting layer to output
    for ind_azi, azi_angl in enumerate(ml_obj.azimuth['data']):
        abs_dist = np.abs(ml_azi_angl - azi_angl)
        ind_azi_ml = np.argmin(abs_dist)
        ml_obj.fields[ml_pos_field]['data'][ind_azi, 0] = ml_bottom[ind_azi_ml]
        ml_obj.fields[ml_pos_field]['data'][ind_azi, 1] = ml_top[ind_azi_ml]

    return ml_obj


def _get_res_vol_sides(radar):
    """
    Computes the height of the lower left and upper right points of the
    range resolution volume.

    Parameters
    ----------
    radar : radar object
        The current radar

    Returns
    -------
    hlowerleft, hupperright : 2D-arrays
        The matrix (rays, range) with the lower left and upper right height
        of the resolution volume
    """
    deltar = radar.range['data'][1]-radar.range['data'][0]
    if (radar.instrument_parameters is not None and
            'radar_beam_width_h' in radar.instrument_parameters):
        beamwidth = (
            radar.instrument_parameters['radar_beam_width_h']['data'][0])
    else:
        warn('Unknown radar antenna beamwidth. Assumed 1 degree')
        beamwidth = 1.

    _, _, hlowerleft = (
        antenna_vectors_to_cartesian(
            radar.range['data'] - deltar/2, radar.azimuth['data'],
            radar.elevation['data'] - beamwidth/2) +
        radar.altitude['data'][0])

    _, _, hupperright = (
        antenna_vectors_to_cartesian(
            radar.range['data'] + deltar/2, radar.azimuth['data'],
            radar.elevation['data'] + beamwidth/2) +
        radar.altitude['data'][0])

    return hlowerleft, hupperright


def _detect_ml_sweep(radar_sweep, fill_value, refl_field, rhohv_field,
                     melting_layer_field, max_range, detect_threshold,
                     interp_holes, max_length_holes, check_min_length):

    '''
    Detects the melting layer (ML) on an RHI scan of reflectivity and copolar
    correlation coefficient and returns its properties both in the original
    polar radar coordinates and in projected Cartesian coordinates

    Parameters
    ----------
        radar_sweep : Radar
            A Radar class instance of a single sweep
        fill_value : float
            Value indicating missing or bad data in differential phase
        refl_field : str
            Reflectivity field. If None, the default field name must be
            specified in the Py-ART configuration file.
        rhohv_field : str
            Copolar correlation coefficient field.
        melting_layer_field : str
            Melting layer field.
        max_range : float
            the max. range from the radar to be used in the ML determination
        detect_threshold : float
            the detection threshold (see paper), you can play around and
            see how it affects the output. Lowering the value makes the
            algorithm more sensitive but increases the number of
            erroneous detections.
        interp_holes : bool
            Flag to allow for interpolation of small holes in the detected ML
        max_length_holes : float
            The maximum size of holes in the ML for them to be interpolated
        check_min_length : bool
            If true, the length of the detected ML will
            be compared with the length of the valid data and the
            ML will be kept only if sufficiently long

    Returns
    -------
    ml : dict
        ml is a dictionnary with the following fields::
        ​ml_pol a dict with the following keys:
            theta (list of elevation angles)
            range (list of ranges)
            data (2D map with 1 where detected ML and 0 otherwise)
            bottom_ml (the height above the radar of the ML bottom for
                       every angle theta)
            top_ml (the height above the radar of the ML top for every angle
                    theta)

        ml_cart a dict with the following keys:
            x : x-coordinates of the Cartesian system (distance at ground)
            z : z-coordinates of the Cartesian system (height above surface)
            data (2D map with 1 where detected ML and 0 otherwise)
            bottom_ml (the height above the radar of the ML bottom for every
                       distance x)
            top_ml (the height above the radar of the ML top for every
                    distance x)
        ​ml_exists a boolean flag = 1 if a ML was detected
    '''

    # Get the labels that will be used for the melting layer
    mdata_ml = get_metadata(melting_layer_field)
    for label, value in zip(mdata_ml['labels'], mdata_ml['ticks']):
        mdata_ml[label] = value

    # Project to cartesian coordinates
    coords_c, refl_field_c, mapping = polar_to_cartesian(
        radar_sweep, refl_field, max_range=max_range)
    coords_c, rhohv_field_c, _ = polar_to_cartesian(
        radar_sweep, rhohv_field, mapping=mapping)
    cart_res = mapping['res']

    # Get Zh and Rhohv images
    refl_im = _normalize_image(refl_field_c, *ZH_IM_BOUNDS)
    rhohv_im = _normalize_image(rhohv_field_c, *RHOHV_IM_BOUNDS)

    # Combine images
    comb_im = (1 - rhohv_im) * refl_im
    comb_im[np.isnan(comb_im)] = 0.

    # Get vertical gradient
    size_filt = np.floor(SIZEFILT_M / cart_res).astype(int)
    gradient = _gradient_2D(_mean_filter(comb_im, (size_filt, size_filt)))
    gradient_z = gradient['Gy']
    gradient_z[np.isnan(rhohv_field_c)] = np.nan

    # First part, using ZH and RHOHV for top and bottom
    ###################################################################

    # First guess of ML without geometrical constraints
    bottom_ml, top_ml = _process_map_ml(
        gradient_z, rhohv_field_c, detect_threshold, *RHOHV_VALID_BOUNDS)

    # Restrict gradient using conditions on medians
    median_bot_height = np.nanmedian(bottom_ml)
    median_top_height = np.nanmedian(top_ml)

    if not np.isnan(median_bot_height):
        gradient_z[0:np.floor(LOWMLBOUND *
                              median_bot_height).astype(int), :] = np.nan
    if not np.isnan(median_top_height):
        gradient_z[np.floor(UPMLBOUND *
                            median_top_height).astype(int):, :] = np.nan

    # Identify top and bottom of ML with restricted gradient
    bottom_ml, top_ml = _process_map_ml(
        gradient_z, rhohv_field_c, detect_threshold, *RHOHV_VALID_BOUNDS)
    median_bot_height = np.nanmedian(bottom_ml)
    median_top_height = np.nanmedian(top_ml)

    # Final part - cleanup
    ###################################################################

    # Put NaN at pixels which have a ML thickness larger than max_thickness
    # Also put NaN where either the top or the bottom are undefined
    thickness = top_ml - bottom_ml
    bad_pixels = ~np.isnan(thickness)
    bad_pixels[bad_pixels] &= (
        bad_pixels[bad_pixels] > MAXTHICKNESS_ML/cart_res)
    top_ml[bad_pixels] = np.nan
    bottom_ml[bad_pixels] = np.nan
    top_ml[np.isnan(bottom_ml)] = np.nan
    bottom_ml[np.isnan(top_ml)] = np.nan

    median_bot_height = np.nanmedian(bottom_ml)
    median_top_height = np.nanmedian(top_ml)

    ###################################################################

    # If interpolation of small holes is activated (and if at least 2 valid
    # pts)
    if interp_holes and np.sum(np.isfinite(bottom_ml)) >= 2:
        # Find subsequences
        sub = _calc_sub_ind(bottom_ml)
        # If the first and last subset correspond to missing values we remove
        # them, as we want NO extrapolation
        if sub['values'][0] == 0:
            for k in ['values', 'lengths', 'idx']:
                sub[k] = sub[k][1:]
        if sub['values'][-1] == 0:
            for k in ['values', 'lengths', 'idx']:
                sub[k] = sub[k][:-1]
        # Find subset of subsequences where missing vals and length is
        # at most THRESHLENGTH
        sub2interp = {}
        cond = np.logical_and(sub['values'] == 0,
                              sub['lengths'] <= max_length_holes / cart_res)
        sub2interp['lengths'] = sub['lengths'][cond]
        sub2interp['idx'] = sub['idx'][cond]

        # Get corresponding indexes
        index2interp = []
        for k in range(0, len(sub2interp['idx'])):
            index2interp.extend(
                range(sub2interp['idx'][k],
                      sub2interp['idx'][k] +
                      sub2interp['lengths'][k]))
        # Interpolate all subsequences of less than threshLength [m] using
        # piecewise cubic hermite interpolation
        index2interp = np.array(index2interp)

        # Interpolate
        if index2interp.size > 0:
            idx_valid = np.where(np.isfinite(bottom_ml))[0]
            bottom_ml[index2interp] = pchip(idx_valid,
                                            bottom_ml[idx_valid])(index2interp)
            top_ml[index2interp] = pchip(idx_valid,
                                         top_ml[idx_valid])(index2interp)

    mid_ml = (median_top_height + median_bot_height) / 2

    # Check if ML is valid
    # 1) check if median_bot_height and median_top_height are defined
    if np.isnan(median_bot_height + median_top_height):
        invalid_ml = True
    else:
        invalid_ml = False
        # 2) Check how many values in the data are defined at the height of the
        # ML
        line_val = rhohv_field_c[np.int(mid_ml), :]

        # Check if ML is long enough
        if check_min_length:
            # the condition is that the ml is at least half as
            # long as the length of valid data at the ml height
            if np.logical_and(sum(np.isfinite(top_ml)) < 0.5,
                              sum(np.isfinite(line_val))):
                invalid_ml = True

    map_ml = np.zeros(gradient_z.shape)

    # 1 = below ML, 3 = in ML, 5 =  above ML

    # If ML is invalid, just fill top_ml and bottom_ml with NaNs
    if invalid_ml:
        top_ml = np.nan * np.zeros((gradient_z.shape[1]))
        bottom_ml = np.nan * np.zeros((gradient_z.shape[1]))
    else:
        for j in range(0, len(top_ml) - 1):
            if(not np.isnan(top_ml[j]) and not np.isnan(bottom_ml[j])):
                map_ml[np.int(top_ml[j]):, j] = mdata_ml['BELOW']
                map_ml[np.int(bottom_ml[j]):np.int(top_ml[j]), j] = mdata_ml[
                    'INSIDE']
                map_ml[0:np.int(bottom_ml[j]), j] = mdata_ml['ABOVE']

    # create dictionary of output ml

    # Cartesian coordinates
    ml_cart = {}
    ml_cart['data'] = np.array(map_ml)
    ml_cart['x'] = coords_c[0]
    ml_cart['z'] = coords_c[1]

    ml_cart['bottom_ml'] = np.array((bottom_ml) * cart_res)
    ml_cart['top_ml'] = np.array((top_ml) * cart_res)

    # Polar coordinates
    (theta, r), (bottom_ml, top_ml), map_ml = _remap_to_polar(
        radar_sweep, ml_cart['x'], ml_cart['bottom_ml'], ml_cart['top_ml'],
        interp=True)
    map_ml = np.ma.array(map_ml, mask=map_ml == 0, fill_value=fill_value)
    bottom_ml = np.ma.masked_invalid(bottom_ml)
    top_ml = np.ma.masked_invalid(top_ml)

    ml_pol = {}
    ml_pol['data'] = map_ml
    ml_pol['theta'] = theta
    ml_pol['range'] = r
    ml_pol['bottom_ml'] = bottom_ml
    ml_pol['top_ml'] = top_ml

    output = {}
    output['ml_cart'] = ml_cart
    output['ml_pol'] = ml_pol
    output['ml_exists'] = not invalid_ml

    return output


def _process_map_ml(gradient_z, rhohv, threshold, threshold_min_rhohv=0,
                    threshold_max_rhohv=np.Inf):

    n_cols = gradient_z.shape[1]
    bottom_ml = np.zeros((n_cols)) * np.nan
    top_ml = np.zeros((n_cols)) * np.nan

    # Loop on all vertical columns
    for j in range(0, n_cols):
        # Get a vertical slice of gradient
        grad_line = gradient_z[:, j]
        grad_nonan = grad_line

        grad_nonan = grad_nonan[~np.isnan(grad_nonan)]
        ind_bot = np.nan
        ind_top = np.nan

        if grad_nonan.size:
            # Sort by ascending order of gradient
            sortedGrad = np.sort(grad_nonan)

            max_val = sortedGrad[-1]
            min_val = sortedGrad[0]

            # Index of ML top is where gradient is min
            ind_top = np.where(grad_line == min_val)
            ind_top = ind_top[0][0] + 2
            # Index of ML bottom is where gradient is max
            ind_bot = np.where(grad_line == max_val)
            ind_bot = ind_bot[0][0]

            # We also check a condition on rhohv which should not be too small
            if not ind_bot.size or not ind_top.size or ind_top <= ind_bot:
                rhohv_thresh_cond = False
                rhohv_nan_cond = False
            else:
                rhohv_thresh_cond = (np.nanmax(rhohv[ind_bot:ind_top, j])
                                     <= threshold_max_rhohv
                                     and np.nanmin(rhohv[ind_bot:ind_top, j])
                                     >= threshold_min_rhohv)

                rhohv_nan_cond = np.sum(
                    np.isnan(rhohv[ind_bot:ind_top, j])) == 0

            if (min_val <= -threshold and max_val >= threshold and
                    rhohv_nan_cond and rhohv_thresh_cond):
                bottom_ml[j] = ind_bot - 1
                top_ml[j] = ind_top + 1

    return bottom_ml, top_ml


def _process_map_ml_only_zh(gradientZ):
    # Basically the same as previous routine, except only on Zh
    n_cols = gradientZ.shape[1]
    top_ml = np.zeros((n_cols)) * float('nan')

    for j in range(0, n_cols - 1):
        grad_line = gradientZ[:, j]
        grad_no_nan = grad_line
        grad_no_nan = grad_no_nan[~np.isnan(grad_no_nan)]
        if grad_no_nan.size:
            sorted_grad = np.sort(grad_no_nan)
            min_val = sorted_grad[0]
            ind_top = np.where(grad_line == min_val)
            ind_top = ind_top[0][0]
            top_ml[j] = ind_top + 1

    return top_ml


def _r_to_h(earth_radius, gate_range, gate_theta):
    '''
    Computes the height of radar gates knowing the earth radius at the given
    latitude and the range and elevation angle of the radar gate.

    Inputs:
        earth_radius : the radius of the earth for a given latitude in m.

        gate_range : the range of the gate(s) in m.

        gate_theta : elevation angle of the gate(s) in degrees.

    Outputs:
        height : the height above ground of all specified radar gates
    '''

    height = ((gate_range**2 + (KE * earth_radius)**2 +
               2 * gate_range * KE * earth_radius *
               np.sin(np.deg2rad(gate_theta)))**(0.5) - KE * earth_radius)

    return height


def _remap_to_polar(radar_sweep, x, bottom_ml, top_ml, tol=1.5, interp=True):
    '''
    This routine converts the ML in Cartesian coordinates back to polar
    coordinates.

    Inputs:
        radar_sweep : Radar
            A pyart radar instance containing the radar data in polar
            coordinates for a single sweep
        x: array of floats
            The horizontal distance in Cartesian coordinates.
        bottom_ml: array of floats
            Bottom of the ML detected in Cartesian coordinates.
        top_ml: array of floats
            Top of the ML detected on Cartesian coordinates.
        tol : float, optional
            Angular tolerance in degrees that is used when mapping elevation
            angles computed on the Cartesian image to the original angles in
            the polar data.
        interp : bool, optional
            Whether or not to interpolate the ML in polar coordinates
            (fill holes)

    Outputs:
        (theta, r) : tuple of elevation angle and range corresponding to the
                     polar coordinates
        (bottom_ml, top_ml) : tuple of ml bottom and top ranges for every
                              elevation angle theta
        map_ml_pol : a binary map of the ML in polar coordinates
    '''
    # This routine converts the ML in cartesian coordinates back to polar
    # coordinates

    # Get ranges of radar data
    r = radar_sweep.range['data']
    dr = r[1]-r[0]

    # Get angles of radar data
    theta = radar_sweep.elevation['data']

    # Vectors to store the heights of the ML top and bottom and matrix for the
    # map
    map_ml_pol = np.zeros((len(theta), len(r)))
    bottom_ml_pol = np.zeros(len(map_ml_pol)) + np.nan
    top_ml_pol = np.zeros(len(map_ml_pol)) + np.nan

    if np.sum(np.isfinite(bottom_ml)) > 0:
        # Convert cartesian to polar

        # Get ranges of all pixels located at the top and bottom of cartesian
        # ML
        theta_bottom_ml = np.degrees(-(np.arctan2(x, bottom_ml) - np.pi / 2))
        E = get_earth_radius(radar_sweep.latitude['data'])  # Earth radius
        r_bottom_ml = (
            np.sqrt((E * KE * np.sin(np.radians(theta_bottom_ml)))**2 +
                    2 * E * KE * bottom_ml + bottom_ml ** 2)
            - E * KE * np.sin(np.radians(theta_bottom_ml)))

        theta_top_ml = np.degrees(- (np.arctan2(x, top_ml) - np.pi / 2))
        E = get_earth_radius(radar_sweep.latitude['data'])  # Earth radius
        r_top_ml = (np.sqrt((E * KE * np.sin(np.radians(theta_top_ml))) ** 2 +
                            2 * E * KE * top_ml + top_ml ** 2) -
                    E * KE * np.sin(np.radians(theta_top_ml)))

        idx_r_bottom = np.zeros((len(theta))) * np.nan
        idx_r_top = np.zeros((len(theta))) * np.nan

        for i, t in enumerate(theta):
            # Find the pixel at the bottom of the ML with the closest angle
            # to theta
            idx_bot = np.nanargmin(np.abs(theta_bottom_ml - t))

            if np.abs(theta_bottom_ml[idx_bot] - t) < tol:
                # Same with pixel at top of ml
                idx_top = np.nanargmin(np.abs(theta_top_ml - t))
                if np.abs(theta_top_ml[idx_top] - t) < tol:

                    r_bottom = r_bottom_ml[idx_bot]
                    r_top = r_top_ml[idx_top]

                    idx_aux = np.where(r >= r_bottom)[0]
                    if idx_aux.size > 0:
                        idx_r_bottom[i] = idx_aux[0]

                    idx_aux = np.where(r >= r_top)[0]
                    if idx_aux.size > 0:
                        idx_r_top[i] = idx_aux[0]
        if interp:
            if np.sum(np.isfinite(idx_r_bottom)) >= 4:
                idx_valid = np.where(np.isfinite(idx_r_bottom))[0]
                idx_nan = np.where(np.isnan(idx_r_bottom))[0]
                bottom_ml_fill = InterpolatedUnivariateSpline(
                    idx_valid, idx_r_bottom[idx_valid], ext=1)(idx_nan)
                bottom_ml_fill[bottom_ml_fill == 0] = -9999
                idx_r_bottom[idx_nan] = bottom_ml_fill

            if np.sum(np.isfinite(idx_r_top)) >= 4:
                idx_valid = np.where(np.isfinite(idx_r_top))[0]
                idx_nan = np.where(np.isnan(idx_r_top))[0]
                top_ml_fill = InterpolatedUnivariateSpline(
                    idx_valid, idx_r_top[idx_valid], ext=1)(idx_nan)
                top_ml_fill[top_ml_fill == 0] = -9999
                idx_r_top[idx_nan] = top_ml_fill
        else:
            idx_r_bottom[np.isnan(idx_r_bottom)] = -9999
            idx_r_top[np.isnan(idx_r_top)] = -9999

        idx_r_bottom = idx_r_bottom.astype(int)
        idx_r_top = idx_r_top.astype(int)

        for i in range(len(map_ml_pol)):
            if idx_r_bottom[i] != -9999 and idx_r_top[i] != -9999:
                r_bottom_interp = min([len(r), idx_r_bottom[i]])*dr
                bottom_ml_pol[i] = _r_to_h(E, r_bottom_interp, theta[i])

                r_top_interp = min([len(r), idx_r_top[i]])*dr
                top_ml_pol[i] = _r_to_h(E, r_top_interp, theta[i])

                # check that data has plausible values
                if (bottom_ml_pol[i] > MAXHEIGHT_ML or
                        bottom_ml_pol[i] < MINHEIGHT_ML or
                        top_ml_pol[i] > MAXHEIGHT_ML or
                        top_ml_pol[i] < MINHEIGHT_ML or
                        bottom_ml_pol[i] >= top_ml_pol[i]):
                    bottom_ml_pol[i] = np.nan
                    top_ml_pol[i] = np.nan
                else:
                    map_ml_pol[i, 0:idx_r_bottom[i]] = 1
                    map_ml_pol[i, idx_r_bottom[i]:idx_r_top[i]] = 3
                    map_ml_pol[i, idx_r_top[i]:] = 5

    return (theta, r), (bottom_ml_pol, top_ml_pol), map_ml_pol


def _normalize_image(im, min_val, max_val):
    '''
    Uniformly normalizes a radar field to the [0-1] range

    Inputs:
        im : array
            A radar image in native units, ex. dBZ

        min_val : float
            All values smaller or equal to min_val in the original image
            will be set to zero

        max_val :
            All values larger or equal to min_val in the original image
            will be set to zero
    Outputs:
        out : the normalized radar image, with all values in [0,1]
    '''

    new_max = 1
    new_min = 0

    out = (im - min_val) * (new_max - new_min) / (max_val - min_val) + new_min
    valid = ~np.isnan(im)

    # set above max not max value
    above_max = deepcopy(valid)
    above_max[valid] &= im[valid] > max_val
    out[above_max] = new_max

    # set below min to min value
    below_min = deepcopy(valid)
    below_min[valid] &= im[valid] < min_val
    out[below_min] = new_min

    return out


def _gradient_2D(im):
    '''
    Computes the 2D gradient of a radar image

    Inputs:
        im : array
            A radar image in Cartesian coordinates

    Outputs:
        out : a gradient dictionary containing a field 'Gx' for the gradient
              in the horizontal and a field 'Gy' for the gradient in the
              vertical
    '''
    # Computes the 2D gradient of an image
    # dim = 1 = gradient along the rows (Y)
    # dim = 2 = gradient along the column (X)

    Gx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    Gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    out = {}

    out['Gx'] = convolve(im, Gx, mode='reflect')
    out['Gy'] = convolve(im, Gy, mode='reflect')

    return out


def _convolve_with_nan(input_array, kernel, boundary='mirror'):
    '''
    Convolves an image with a kernel while ignoring missing values

    Inputs:
        input_array : the input array can be 1D or 2D

        kernel : the kernel (filter) with which to convolve the input array

        boundary: how to treat the boundaries, see arguments of scipy's
                  convolve function.

    Outputs:
        conv: the convolved signal
    '''
    # Special 2D convolution procedure for data with holes (NaN)
    if isinstance(input_array, np.ma.masked_array):
        input_array = np.ma.masked(input_array, np.nan)

    # Flat function with NaNs for comparison.
    on = np.ones(input_array.shape)

    # Find all the NaNs in the input.
    n = np.isnan(input_array)

    # Replace NaNs with zero, both in 'a' and 'on'.
    input_array[n] = 0.
    on[n] = 0.

    # Check that the filter does not have NaNs.
    if np.isnan(kernel).any():
        print('Kernel contains NaN values.')
        return None

    # Calculate what a 'flat' function looks like after convolution.
    flat = convolve(on, kernel, mode=boundary)
    # Do the actual convolution
    conv = convolve(input_array, kernel, mode=boundary) / flat
    return conv


def _mean_filter(input_array, shape=(3, 3), boundary='mirror'):
    '''
    Local averaging (smoothing) while ignoring missing values

    Inputs:
        input_array : the input array can be 1D or 2D

        shape : the shape of the averaging (smoothing) filter

        boundary: how to treat the boundaries, see arguments of scipy's
                  convolve function.

    Outputs:
        out: the smoothed signal
    '''
    # 2D mean filter for data with holes (NaN)
    kernel = np.ones(shape)
    kernel = kernel / np.sum(kernel.ravel())
    out = _convolve_with_nan(input_array, kernel, boundary)

    return out


def _calc_sub_ind(inputVec):
    '''
    The code belows finds continuous subsequences of missing values, it fills
    a vector values containing 1 for values and 0 for missing values starting
    a new subsequence, a vector idx containing the index of the first value
    of the subsequence and a vector length containing the length of the
    subsequence.


    Inputs:
        inputVec : a binary input vector

    Outputs:
        sub: a dictionary with the keys:
            values : an array containing 1 for sequences of valid values
                    and 0 for sequences of missing values
            idx : an array containing the first index of the sequences
            length : an array containing the length of every sequence
    '''
    # the vector [1 2 3 NaN NaN 3 NaN 3 5 5 1 NaN NaN NaN]
    # would give for exemple :
    # lengths=[3 2 1 1 4 3], values=[1 0 1 0 1 0] and idx=[0 3 5 6 7 11]

    sub = {}
    sub['values'] = []
    sub['lengths'] = []
    sub['idx'] = []
    ind = None  # For PEP8...
    if np.size(inputVec) > 0:
        for ind in range(0, len(inputVec) - 1):
            if ind == 0:
                sub['idx'].append(ind)
                sub['values'].append(~np.isnan(inputVec[ind]))
            if ~np.isnan(inputVec[ind]) != sub['values'][-1]:
                sub['values'].append(~np.isnan(inputVec[ind]))
                sub['lengths'].append(ind - sub['idx'][-1])
                sub['idx'].append(ind)
        sub['lengths'].append(ind + 1 - sub['idx'][-1])

        sub['lengths'] = np.array(sub['lengths'])
        sub['idx'] = np.array(sub['idx'])
        sub['values'] = np.array(sub['values'])
    return sub
