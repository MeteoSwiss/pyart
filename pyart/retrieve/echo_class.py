"""
pyart.retrieve.echo_class
=========================

Functions for echo classification.

.. autosummary::
    :toctree: generated/

    steiner_conv_strat
    conv_strat_yuter
    hydroclass_semisupervised
    data_for_centroids
    compute_centroids
    centroids_iter
    store_centroids
    select_samples
    make_platykurtic
    search_medoids
    split_clusters
    compute_intermediate_medoids
    determine_medoids
    synthetic_obs_distribution
    compute_ks_threshold
    compare_samples
    bell_function
    gaussian_function
    trapezoidal_function
    sample_bell
    sample_trapezoidal
    get_freq_band
    _standardize
    _destandardize
    _assign_to_class
    _assign_to_class_scan
    _compute_coeff_transform
    _get_mass_centers
    _mass_centers_table
    _data_limits_table
    _data_limits_centroids_table
    _bell_function_table
    _trapezoidal_function_table

"""

import traceback
from copy import deepcopy
from warnings import warn

import numpy as np
from scipy import interpolate
from scipy.stats import ks_2samp

from ..config import get_field_name, get_fillvalue, get_metadata
from ..core import Grid
from ..util import ma_broadcast_to
from ._echo_class import _feature_detection, _revised_conv_strat, steiner_class_buff
from ._echo_class_wt import calc_scale_break, wavelet_reclass

try:
    from sklearn.model_selection import train_test_split
    from sklearn_extra.cluster import KMedoids

    _SKLEARN_AVAILABLE = True
    try:
        from sklearn_extra.cluster import CLARA

        _CLARA_AVAILABLE = True
    except ImportError:
        _CLARA_AVAILABLE = False
except ImportError:
    _SKLEARN_AVAILABLE = False

try:
    import dask

    _DASK_AVAILABLE = True
except ImportError:
    _DASK_AVAILABLE = False


def steiner_conv_strat(
    grid,
    dx=None,
    dy=None,
    intense=42.0,
    work_level=3000.0,
    peak_relation="default",
    area_relation="medium",
    bkg_rad=11000.0,
    use_intense=True,
    fill_value=None,
    refl_field=None,
):
    """
    Partition reflectivity into convective-stratiform using the Steiner et
    al. (1995) algorithm.

    Parameters
    ----------
    grid : Grid
        Grid containing reflectivity field to partition.
    dx, dy : float, optional
        The x- and y-dimension resolutions in meters, respectively. If None
        the resolution is determined from the first two axes values.
    intense : float, optional
        The intensity value in dBZ. Grid points with a reflectivity
        value greater or equal to the intensity are automatically
        flagged as convective. See reference for more information.
    work_level : float, optional
        The working level (separation altitude) in meters. This is the height
        at which the partitioning will be done, and should minimize bright band
        contamination. See reference for more information.
    peak_relation : 'default' or 'sgp', optional
        The peakedness relation. See reference for more information.
    area_relation : 'small', 'medium', 'large', or 'sgp', optional
        The convective area relation. See reference for more information.
    bkg_rad : float, optional
        The background radius in meters. See reference for more information.
    use_intense : bool, optional
        True to use the intensity criteria.
    fill_value : float, optional
         Missing value used to signify bad data points. A value of None
         will use the default fill value as defined in the Py-ART
         configuration file.
    refl_field : str, optional
         Field in grid to use as the reflectivity during partitioning. None
         will use the default reflectivity field name from the Py-ART
         configuration file.

    Returns
    -------
    eclass : dict
        Steiner convective-stratiform classification dictionary.

    References
    ----------
    Steiner, M. R., R. A. Houze Jr., and S. E. Yuter, 1995: Climatological
    Characterization of Three-Dimensional Storm Structure from Operational
    Radar and Rain Gauge Data. J. Appl. Meteor., 34, 1978-2007.

    """
    # Get fill value
    if fill_value is None:
        fill_value = get_fillvalue()

    # Parse field parameters
    if refl_field is None:
        refl_field = get_field_name("reflectivity")

    # parse dx and dy
    if dx is None:
        dx = grid.x["data"][1] - grid.x["data"][0]
    if dy is None:
        dy = grid.y["data"][1] - grid.y["data"][0]

    # Get coordinates
    x = grid.x["data"]
    y = grid.y["data"]
    z = grid.z["data"]

    # Get reflectivity data
    ze = np.ma.copy(grid.fields[refl_field]["data"])
    ze = ze.filled(np.nan)

    eclass = steiner_class_buff(
        ze,
        x,
        y,
        z,
        dx=dx,
        dy=dy,
        bkg_rad=bkg_rad,
        work_level=work_level,
        intense=intense,
        peak_relation=peak_relation,
        area_relation=area_relation,
        use_intense=use_intense,
    )

    return {
        "data": eclass.astype(np.int32),
        "standard_name": "echo_classification",
        "long_name": "Steiner echo classification",
        "valid_min": 0,
        "valid_max": 2,
        "comment_1": (
            "Convective-stratiform echo "
            "classification based on "
            "Steiner et al. (1995)"
        ),
        "comment_2": ("0 = Undefined, 1 = Stratiform, " "2 = Convective"),
    }


def conv_strat_yuter(
    grid,
    dx=None,
    dy=None,
    level_m=None,
    always_core_thres=42,
    bkg_rad_km=11,
    use_cosine=True,
    max_diff=5,
    zero_diff_cos_val=55,
    scalar_diff=1.5,
    use_addition=True,
    calc_thres=0.75,
    weak_echo_thres=5.0,
    min_dBZ_used=5.0,
    dB_averaging=True,
    remove_small_objects=True,
    min_km2_size=10,
    val_for_max_conv_rad=30,
    max_conv_rad_km=5.0,
    cs_core=3,
    nosfcecho=0,
    weakecho=3,
    sf=1,
    conv=2,
    refl_field=None,
    estimate_flag=True,
    estimate_offset=5,
):
    """
    Partition reflectivity into convective-stratiform using the Yuter et al. (2005)
    and Yuter and Houze (1997) algorithm.

    Parameters
    ----------
    grid : Grid
        Grid containing reflectivity field to partition.
    dx, dy : float, optional
        The x- and y-dimension resolutions in meters, respectively. If None
        the resolution is determined from the first two axes values parsed from grid object.
    level_m : float, optional
        Desired height in meters to classify with convective stratiform algorithm.
    always_core_thres : float, optional
        Threshold for points that are always convective. All values above the threshold are classifed as convective
        See Yuter et al. (2005) for more detail.
    bkg_rad_km : float, optional
        Radius to compute background reflectivity in kilometers. Default is 11 km. Recommended to be at least 3 x
        grid spacing
    use_cosine : bool, optional
        Boolean used to determine if a cosine scheme (see Yuter and Houze (1997)) should be used for identifying
        convective cores (True) or if a simpler scalar scheme (False) should be used.
    max_diff : float, optional
        Maximum difference between background average and reflectivity in order to be classified as convective.
        "a" value in Eqn. B1 in Yuter and Houze (1997)
    zero_diff_cos_val : float, optional
        Value where difference between background average and reflectivity is zero in the cosine function
        "b" value in Eqn. B1 in Yuter and Houze (1997)
    scalar_diff : float, optional
        If using a scalar difference scheme, this value is the multiplier or addition to the background average
    use_addition : bool, optional
        Determines if a multiplier (False) or addition (True) in the scalar difference scheme should be used
    calc_thres : float, optional
        Minimum percentage of points needed to be considered in background average calculation
    weak_echo_thres : float, optional
        Threshold for determining weak echo. All values below this threshold will be considered weak echo
    min_dBZ_used : float, optional
        Minimum dBZ value used for classification. All values below this threshold will be considered no surface echo
        See Yuter and Houze (1997) and Yuter et al. (2005) for more detail.
    dB_averaging : bool, optional
        True if using dBZ reflectivity values that need to be converted to linear Z before averaging. False for
        other non-dBZ values (i.e. snow rate)
    remove_small_objects : bool, optional
        Determines if small objects should be removed from convective core array. Default is True.
    min_km2_size : float, optional
        Minimum size of convective cores to be considered. Cores less than this size will be removed. Default is 10
        km^2.
    val_for_max_conv_rad : float, optional
        dBZ for maximum convective radius. Convective cores with values above this will have the maximum convective
        radius
    max_conv_rad_km : float, optional
        Maximum radius around convective cores to classify as convective. Default is 5 km
    cs_core : int, optional
        Value for points classified as convective cores
    nosfcecho : int, optional
        Value for points classified as no surface echo, based on min_dBZ_used
    weakecho : int, optional
        Value for points classified as weak echo, based on weak_echo_thres
    sf : int, optional
        Value for points classified as stratiform
    conv : int, optional
        Value for points classified as convective
    refl_field : str, optional
        Field in grid to use as the reflectivity during partitioning. None will use the default reflectivity
        field name from the Py-ART configuration file.
    estimate_flag : bool, optional
        Determines if over/underestimation should be applied. If true, the algorithm will also be run on the same field
        wih the estimate_offset added and the same field with the estimate_offset subtracted.
        Default is True (recommended)
    estimate_offset : float, optional
        Value used to offset the reflectivity values by for the over/underestimation application. Default value is 5
        dBZ.

    Returns
    -------
    convsf_dict : dict
        Convective-stratiform classification dictionary.

    References
    ----------
    Yuter, S. E., and R. A. Houze, Jr., 1997: Measurements of raindrop size
    distributions over the Pacific warm pool and implications for Z-R relations.
    J. Appl. Meteor., 36, 847-867.
    https://doi.org/10.1175/1520-0450(1997)036%3C0847:MORSDO%3E2.0.CO;2

    Yuter, S. E., R. A. Houze, Jr., E. A. Smith, T. T. Wilheit, and E. Zipser,
    2005: Physical characterization of tropical oceanic convection observed in
    KWAJEX. J. Appl. Meteor., 44, 385-415. https://doi.org/10.1175/JAM2206.1

    """

    # Maxmimum convective radius must be less than 5 km
    if max_conv_rad_km > 5:
        print("Max conv radius must be less than 5 km, exiting")
        raise

    # Parse field parameters
    if refl_field is None:
        refl_field = get_field_name("reflectivity")
        dB_averaging = True

    # parse dx and dy if None
    if dx is None:
        dx = grid.x["data"][1] - grid.x["data"][0]
    if dy is None:
        dy = grid.y["data"][1] - grid.y["data"][0]

    # add catch for background radius size
    if bkg_rad_km * 1000 < 2 * dx or bkg_rad_km * 1000 < 2 * dy:
        print(
            "Background radius for averaging must be at least 2 times dx and dy, exiting"
        )
        raise

    # Get coordinates
    z = grid.z["data"]

    # Get reflectivity data at desired level
    if level_m is None:
        try:
            ze = np.ma.copy(grid.fields[refl_field]["data"][0, :, :])
        except:
            ze = np.ma.copy(grid.fields[refl_field]["data"][:, :])
    else:
        zslice = np.argmin(np.abs(z - level_m))
        ze = np.ma.copy(grid.fields[refl_field]["data"][zslice, :, :])

    # run convective stratiform algorithm
    _, _, convsf_best = _revised_conv_strat(
        ze,
        dx,
        dy,
        always_core_thres=always_core_thres,
        bkg_rad_km=bkg_rad_km,
        use_cosine=use_cosine,
        max_diff=max_diff,
        zero_diff_cos_val=zero_diff_cos_val,
        scalar_diff=scalar_diff,
        use_addition=use_addition,
        calc_thres=calc_thres,
        weak_echo_thres=weak_echo_thres,
        min_dBZ_used=min_dBZ_used,
        dB_averaging=dB_averaging,
        remove_small_objects=remove_small_objects,
        min_km2_size=min_km2_size,
        val_for_max_conv_rad=val_for_max_conv_rad,
        max_conv_rad_km=max_conv_rad_km,
        cs_core=cs_core,
        nosfcecho=nosfcecho,
        weakecho=weakecho,
        sf=sf,
        conv=conv,
    )

    # put data into a dictionary to be added as a field
    convsf_dict = {
        "convsf": {
            "data": convsf_best,
            "standard_name": "convsf",
            "long_name": "Convective stratiform classification",
            "valid_min": 0,
            "valid_max": 3,
            "comment_1": (
                "Convective-stratiform echo "
                "classification based on "
                "Yuter and Houze (1997) and Yuter et al. (2005)"
            ),
            "comment_2": (
                "0 = No surface echo/Undefined, 1 = Stratiform, "
                "2 = Convective, 3 = weak echo"
            ),
        }
    }

    # If estimation is True, run the algorithm on the field with offset subtracted and the field with the offset added
    if estimate_flag:
        _, _, convsf_under = _revised_conv_strat(
            ze - estimate_offset,
            dx,
            dy,
            always_core_thres=always_core_thres,
            bkg_rad_km=bkg_rad_km,
            use_cosine=use_cosine,
            max_diff=max_diff,
            zero_diff_cos_val=zero_diff_cos_val,
            scalar_diff=scalar_diff,
            use_addition=use_addition,
            calc_thres=calc_thres,
            weak_echo_thres=weak_echo_thres,
            min_dBZ_used=min_dBZ_used,
            dB_averaging=dB_averaging,
            remove_small_objects=remove_small_objects,
            min_km2_size=min_km2_size,
            val_for_max_conv_rad=val_for_max_conv_rad,
            max_conv_rad_km=max_conv_rad_km,
            cs_core=cs_core,
            nosfcecho=nosfcecho,
            weakecho=weakecho,
            sf=sf,
            conv=conv,
        )

        _, _, convsf_over = _revised_conv_strat(
            ze + estimate_offset,
            dx,
            dy,
            always_core_thres=always_core_thres,
            bkg_rad_km=bkg_rad_km,
            use_cosine=use_cosine,
            max_diff=max_diff,
            zero_diff_cos_val=zero_diff_cos_val,
            scalar_diff=scalar_diff,
            use_addition=use_addition,
            calc_thres=calc_thres,
            weak_echo_thres=weak_echo_thres,
            min_dBZ_used=min_dBZ_used,
            dB_averaging=dB_averaging,
            remove_small_objects=remove_small_objects,
            min_km2_size=min_km2_size,
            val_for_max_conv_rad=val_for_max_conv_rad,
            max_conv_rad_km=max_conv_rad_km,
            cs_core=cs_core,
            nosfcecho=nosfcecho,
            weakecho=weakecho,
            sf=sf,
            conv=conv,
        )

        # save into dictionaries
        convsf_dict["convsf_under"] = {
            "data": convsf_under,
            "standard_name": "convsf_under",
            "long_name": "Convective stratiform classification (Underestimate)",
            "valid_min": 0,
            "valid_max": 3,
            "comment_1": (
                "Convective-stratiform echo "
                "classification based on "
                "Yuter and Houze (1997) and Yuter et al. (2005)"
            ),
            "comment_2": (
                "0 = Undefined, 1 = Stratiform, " "2 = Convective, 3 = weak echo"
            ),
        }

        convsf_dict["convsf_over"] = {
            "data": convsf_over,
            "standard_name": "convsf_under",
            "long_name": "Convective stratiform classification (Overestimate)",
            "valid_min": 0,
            "valid_max": 3,
            "comment_1": (
                "Convective-stratiform echo "
                "classification based on "
                "Yuter and Houze (1997)"
            ),
            "comment_2": (
                "0 = Undefined, 1 = Stratiform, " "2 = Convective, 3 = weak echo"
            ),
        }

    return convsf_dict


def feature_detection(
    grid,
    dx=None,
    dy=None,
    level_m=None,
    always_core_thres=42,
    bkg_rad_km=11,
    use_cosine=True,
    max_diff=5,
    zero_diff_cos_val=55,
    scalar_diff=1.5,
    use_addition=True,
    calc_thres=0.75,
    weak_echo_thres=5.0,
    min_val_used=5.0,
    dB_averaging=True,
    remove_small_objects=True,
    min_km2_size=10,
    binary_close=False,
    val_for_max_rad=30,
    max_rad_km=5.0,
    core_val=3,
    nosfcecho=0,
    weakecho=3,
    bkgd_val=1,
    feat_val=2,
    field=None,
    estimate_flag=True,
    estimate_offset=5,
    overest_field=None,
    underest_field=None,
):
    """
    This function can be used to detect features in a field (e.g. reflectivity, rain rate, snow rate,
    etc.) described by Tomkins et al. (2023) and based on original convective-stratiform algorithms developed by
    Steiner et al. (1995), Yuter et al. (2005) and Yuter and Houze (1997) algorithm.

    Author: Laura Tomkins (@lauratomkins)

    Parameters
    ----------
    grid : Grid
        Grid containing reflectivity field to partition.
    dx, dy : float, optional
        The x- and y-dimension resolutions in meters, respectively. If None
        the resolution is determined from the first two axes values parsed from grid object.
    level_m : float, optional
        Desired height in meters to run feature detection algorithm.
    always_core_thres : float, optional
        Threshold for points that are always features. All values above the threshold are classified as features.
    bkg_rad_km : float, optional
        Radius to compute background reflectivity in kilometers. Default is 11 km. Recommended to be at least 3 x
        grid spacing
    use_cosine : bool, optional
        Boolean used to determine if a cosine scheme (see Yuter and Houze (1997)) should be used for identifying
        cores (True) or if a simpler scalar scheme (False) should be used.
    max_diff : float, optional
        Maximum difference between background average and reflectivity in order to be classified as features.
        "a" value in Eqn. B1 in Yuter and Houze (1997)
    zero_diff_cos_val : float, optional
        Value where difference between background average and reflectivity is zero in the cosine function
        "b" value in Eqn. B1 in Yuter and Houze (1997)
    scalar_diff : float, optional
        If using a scalar difference scheme, this value is the multiplier or addition to the background average
    use_addition : bool, optional
        Determines if a multiplier (False) or addition (True) in the scalar difference scheme should be used
    calc_thres : float, optional
        Minimum percentage of points needed to be considered in background average calculation
    weak_echo_thres : float, optional
        Threshold for determining weak echo. All values below this threshold will be considered weak echo
    min_val_used : float, optional
        Minimum value used for classification. All values below this threshold will be considered no surface echo
        See Yuter and Houze (1997) and Yuter et al. (2005) for more detail. Units based on input field
    dB_averaging : bool, optional
        True if using dBZ reflectivity values that need to be converted to linear Z before averaging. False for
        other non-dBZ values (i.e. snow rate)
    remove_small_objects : bool, optional
        Determines if small objects should be removed from core array. Default is True.
    min_km2_size : float, optional
        Minimum size of Cores to be considered. Cores less than this size will be removed. Default is 10 km^2.
    binary_close : bool, optional
        Determines if a binary closing should be performed on the cores. Default is False.
    val_for_max_rad : float, optional
        value used for maximum radius. Cores with values above this will have the maximum radius incorporated.
    max_rad_km : float, optional
        Maximum radius around cores to classify as feature. Default is 5 km
    core_val : int, optional
        Value for points classified as cores
    nosfcecho : int, optional
        Value for points classified as no surface echo, based on min_val_used
    weakecho : int, optional
        Value for points classified as weak echo, based on weak_echo_thres.
    bkgd_val : int, optional
        Value for points classified as background echo.
    feat_val : int, optional
        Value for points classified as features.
    field : str, optional
        Field in grid to find objects in. None will use the default reflectivity field name from the Py-ART
        configuration file.
    estimate_flag : bool, optional
        Determines if over/underestimation should be applied. If true, the algorithm will also be run on the same field
        wih the estimate_offset added and the same field with the estimate_offset subtracted.
        Default is True (recommended)
    estimate_offset : float, optional
        Value used to offset the field values by for the over/underestimation application. Default value is 5 dBZ.
    overest_field : str, optional
        Name of field in grid object used to calculate the overestimate if estimate_flag is True.
    underest_field : str, optional
        Name of field in grid object used to calculate the underestimate if estimate_flag is True.

    Returns
    -------
    feature_dict : dict
        Feature detection classification dictionary.

    References
    ----------
    Steiner, M. R., R. A. Houze Jr., and S. E. Yuter, 1995: Climatological
    Characterization of Three-Dimensional Storm Structure from Operational
    Radar and Rain Gauge Data. J. Appl. Meteor., 34, 1978-2007.

    Yuter, S. E., and R. A. Houze, Jr., 1997: Measurements of raindrop size
    distributions over the Pacific warm pool and implications for Z-R relations.
    J. Appl. Meteor., 36, 847-867.
    https://doi.org/10.1175/1520-0450(1997)036%3C0847:MORSDO%3E2.0.CO;2

    Yuter, S. E., R. A. Houze, Jr., E. A. Smith, T. T. Wilheit, and E. Zipser,
    2005: Physical characterization of tropical oceanic convection observed in
    KWAJEX. J. Appl. Meteor., 44, 385-415. https://doi.org/10.1175/JAM2206.1

    Tomkins, L. M., S. E. Yuter, and M. A. Miller, 2024: Objective identification
    of faint and strong features in radar observations of winter storms. in prep.

    """

    # Maxmimum radius must be less than 5 km
    if max_rad_km > 5:
        print("Max radius must be less than 5 km, exiting")
        raise

    # Parse field parameters
    if field is None:
        field = get_field_name("reflectivity")
        dB_averaging = True

    # parse dx and dy if None
    if dx is None:
        dx = grid.x["data"][1] - grid.x["data"][0]
    if dy is None:
        dy = grid.y["data"][1] - grid.y["data"][0]

    # add catch for background radius size
    if bkg_rad_km * 1000 < 2 * dx or bkg_rad_km * 1000 < 2 * dy:
        print(
            "Background radius for averaging must be at least 2 times dx and dy, exiting"
        )
        raise

    # Get coordinates
    z = grid.z["data"]

    # Get reflectivity data at desired level
    if level_m is None:
        try:
            ze = np.ma.copy(grid.fields[field]["data"][0, :, :])
        except:
            ze = np.ma.copy(grid.fields[field]["data"][:, :])
    else:
        zslice = np.argmin(np.abs(z - level_m))
        ze = np.ma.copy(grid.fields[field]["data"][zslice, :, :])

    # run feature detection algorithm
    _, _, feature_best = _feature_detection(
        ze,
        dx,
        dy,
        always_core_thres=always_core_thres,
        bkg_rad_km=bkg_rad_km,
        use_cosine=use_cosine,
        max_diff=max_diff,
        zero_diff_cos_val=zero_diff_cos_val,
        scalar_diff=scalar_diff,
        use_addition=use_addition,
        calc_thres=calc_thres,
        weak_echo_thres=weak_echo_thres,
        min_val_used=min_val_used,
        dB_averaging=dB_averaging,
        remove_small_objects=remove_small_objects,
        min_km2_size=min_km2_size,
        binary_close=binary_close,
        val_for_max_rad=val_for_max_rad,
        max_rad_km=max_rad_km,
        core_val=core_val,
        nosfcecho=nosfcecho,
        weakecho=weakecho,
        bkgd_val=bkgd_val,
        feat_val=feat_val,
    )

    # put data into a dictionary to be added as a field
    feature_dict = {
        "feature_detection": {
            "data": feature_best[None, :, :],
            "standard_name": "feature_detection",
            "long_name": "Feature Detection",
            "valid_min": 0,
            "valid_max": 3,
            "comment_1": (
                f"{nosfcecho} = No surface echo/Undefined, {bkgd_val} = Background echo, {feat_val} = Features, {weakecho} = weak echo"
            ),
        }
    }

    # If estimation is True, run the algorithm on the field with offset subtracted and the field with the offset added
    if estimate_flag:
        # get underestimate field
        if underest_field is None:
            under_field = ze - estimate_offset
        elif underest_field is not None:
            under_field = np.ma.copy(grid.fields[underest_field]["data"][0, :, :])

        # run algorithm to get feature detection underestimate
        _, _, feature_under = _feature_detection(
            under_field,
            dx,
            dy,
            always_core_thres=always_core_thres,
            bkg_rad_km=bkg_rad_km,
            use_cosine=use_cosine,
            max_diff=max_diff,
            zero_diff_cos_val=zero_diff_cos_val,
            scalar_diff=scalar_diff,
            use_addition=use_addition,
            calc_thres=calc_thres,
            weak_echo_thres=weak_echo_thres,
            min_val_used=min_val_used,
            dB_averaging=dB_averaging,
            remove_small_objects=remove_small_objects,
            min_km2_size=min_km2_size,
            binary_close=binary_close,
            val_for_max_rad=val_for_max_rad,
            max_rad_km=max_rad_km,
            core_val=core_val,
            nosfcecho=nosfcecho,
            weakecho=weakecho,
            bkgd_val=bkgd_val,
            feat_val=feat_val,
        )

        # get overestimate field
        if overest_field is None:
            over_field = ze + estimate_offset
        elif overest_field is not None:
            over_field = np.ma.copy(grid.fields[overest_field]["data"][0, :, :])

        # run algorithm to get feature detection underestimate
        _, _, feature_over = _feature_detection(
            over_field,
            dx,
            dy,
            always_core_thres=always_core_thres,
            bkg_rad_km=bkg_rad_km,
            use_cosine=use_cosine,
            max_diff=max_diff,
            zero_diff_cos_val=zero_diff_cos_val,
            scalar_diff=scalar_diff,
            use_addition=use_addition,
            calc_thres=calc_thres,
            weak_echo_thres=weak_echo_thres,
            min_val_used=min_val_used,
            dB_averaging=dB_averaging,
            remove_small_objects=remove_small_objects,
            min_km2_size=min_km2_size,
            binary_close=binary_close,
            val_for_max_rad=val_for_max_rad,
            max_rad_km=max_rad_km,
            core_val=core_val,
            nosfcecho=nosfcecho,
            weakecho=weakecho,
            bkgd_val=bkgd_val,
            feat_val=feat_val,
        )

        # save into dictionaries
        feature_dict["feature_under"] = {
            "data": feature_under[None, :, :],
            "standard_name": "feature_detection_under",
            "long_name": "Feature Detection Underestimate",
            "valid_min": 0,
            "valid_max": 3,
            "comment_1": (
                f"{nosfcecho} = No surface echo/Undefined, {bkgd_val} = Background echo, {feat_val} = Features, {weakecho} = weak echo"
            ),
        }

        feature_dict["feature_over"] = {
            "data": feature_over[None, :, :],
            "standard_name": "feature_detection_over",
            "long_name": "Feature Detection Overestimate",
            "valid_min": 0,
            "valid_max": 3,
            "comment_1": (
                f"{nosfcecho} = No surface echo/Undefined, {bkgd_val} = Background echo, {feat_val} = Features, {weakecho} = weak echo"
            ),
        }

    return feature_dict


def hydroclass_semisupervised(
    radar,
    hydro_names=("AG", "CR", "LR", "RP", "RN", "VI", "WS", "MH", "IH"),
    var_names=("dBZ", "ZDR", "KDP", "RhoHV", "H_ISO0"),
    mass_centers=None,
    weights=np.array([1.0, 1.0, 1.0, 0.75, 0.5]),
    value=50.0,
    lapse_rate=-6.5,
    refl_field=None,
    zdr_field=None,
    rhv_field=None,
    kdp_field=None,
    temp_field=None,
    iso0_field=None,
    hydro_field=None,
    entropy_field=None,
    radar_freq=None,
    temp_ref="temperature",
    compute_entropy=False,
    output_distances=False,
    vectorize=False,
):
    """
    Classifies precipitation echoes following the approach by Besic et al
    (2016).

    Parameters
    ----------
    radar : radar
        Radar object.
    hydro_names : array of str
        name of the types of hydrometeors
    var_names : array of str
        name of the variables
    mass_centers : ndarray 2D, optional
        The centroids for each variable and hydrometeor class in (nclasses,
        nvariables).
    weights : ndarray 1D, optional
        The weight given to each variable. Ordered by [dBZ, ZDR, KDP, RhoHV,
        H_ISO0]
    value : float
        The value controlling the rate of decay in the distance transformation
    lapse_rate : float
        The decrease in temperature for each vertical km [deg/km]
    refl_field, zdr_field, rhv_field, kdp_field, temp_field, iso0_field : str
        Inputs. Field names within the radar object which represent the
        horizonal reflectivity, the differential reflectivity, the copolar
        correlation coefficient, the specific differential phase, the
        temperature (in deg celsius) and the height respect to the iso0 fields.
        A value of None for any of these parameters will use the default field
        name as defined in the Py-ART configuration file.
    hydro_field : str
        Output. Field name which represents the hydrometeor class field.
        A value of None will use the default field name as defined in the
        Py-ART configuration file.
    entropy_field : str
        Output. Field name which represents the entropy class field.
        A value of None will use the default field name as defined in the
        Py-ART configuration file.
    radar_freq : str, optional
        Radar frequency in Hertz (Hz) used for classification.
        This parameter will be ignored, if the radar object has frequency information.
    temp_ref : str
        the field use as reference for temperature. Can be either temperature
        or height_over_iso0
    compute_entropy : bool
        If true, the entropy is computed
    output_distances : bool
        If true, the normalized distances to the centroids for each
        hydrometeor are provided as output
    vectorize : bool
        If true, a vectorized version of the class assignation is going to be
        used

    Returns
    -------
    fields_dict : dict
        Dictionary containing the retrieved fields

    References
    ----------
    Besic, N., Figueras i Ventura, J., Grazioli, J., Gabella, M., Germann, U.,
    and Berne, A.: Hydrometeor classification through statistical clustering
    of polarimetric radar measurements: a semi-supervised approach,
    Atmos. Meas. Tech., 9, 4425-4445, doi:10.5194/amt-9-4425-2016, 2016

    """
    # select the centroids as a function of frequency band
    if mass_centers is None:
        if radar.instrument_parameters and "frequency" in radar.instrument_parameters:
            frequency = radar.instrument_parameters["frequency"]["data"][0]
            mass_centers = _get_mass_centers(frequency)
            warn(f"Using radar frequency from instrument parameters: {frequency}")
        elif radar_freq is not None:
            mass_centers = _get_mass_centers(radar_freq)
            warn(
                f"Radar instrument parameters are empty. Using user-supplied radar frequency: {radar_freq}"
            )
        else:
            mass_centers = _mass_centers_table()["C"]
            warn(
                "Radar instrument parameters and radar_freq param are empty."
                "So frequency is unknown. Default coefficients for C band will be applied."
            )

    if hydro_field is None:
        hydro_field = get_field_name("radar_echo_classification")
    if compute_entropy:
        if entropy_field is None:
            entropy_field = get_field_name("hydroclass_entropy")

    # Get the data fields
    fields_dict = {}
    for var_name in var_names:
        if var_name == "dBZ":
            if refl_field is None:
                refl_field = get_field_name("reflectivity")
            radar.check_field_exists(refl_field)
            fields_dict.update({var_name: deepcopy(radar.fields[refl_field]["data"])})
        elif var_name == "ZDR":
            if zdr_field is None:
                zdr_field = get_field_name("differential_reflectivity")
            radar.check_field_exists(zdr_field)
            fields_dict.update({var_name: deepcopy(radar.fields[zdr_field]["data"])})
        elif var_name == "KDP":
            if kdp_field is None:
                kdp_field = get_field_name("specific_differential_phase")
            radar.check_field_exists(kdp_field)
            fields_dict.update({var_name: deepcopy(radar.fields[kdp_field]["data"])})
        elif var_name == "RhoHV":
            if rhv_field is None:
                rhv_field = get_field_name("cross_correlation_ratio")
            radar.check_field_exists(rhv_field)
            fields_dict.update({var_name: deepcopy(radar.fields[rhv_field]["data"])})
        elif var_name == "H_ISO0":
            if temp_ref == "temperature":
                if temp_field is None:
                    temp_field = get_field_name("temperature")
                radar.check_field_exists(temp_field)
                # convert temp in relative height respect to iso0
                temp = deepcopy(radar.fields[temp_field]["data"])
                fields_dict.update({var_name: temp * (1000.0 / lapse_rate)})
            else:
                if iso0_field is None:
                    iso0_field = get_field_name("height_over_iso0")
                radar.check_field_exists(iso0_field)
                fields_dict.update(
                    {var_name: deepcopy(radar.fields[iso0_field]["data"])}
                )
        else:
            raise ValueError(
                "Variable "
                + var_name
                + " unknown. "
                + "Valid variable names for hydrometeor classification are: "
                + "H_ISO0, dBZ, ZDR, KDP and RhoHV"
            )

    # standardize data and centroids
    mc_std = np.empty(np.shape(mass_centers), dtype=fields_dict[var_names[0]].dtype)
    for i, var_name in enumerate(var_names):
        mc_std[:, i] = _standardize(mass_centers[:, i], var_name)
        fields_dict[var_name] = _standardize(fields_dict[var_name], var_name)

    # if entropy has to be computed get transformation parameters
    t_vals = None
    if compute_entropy:
        t_vals = _compute_coeff_transform(mc_std, weights=weights, value=value)

    # assign to class
    if vectorize:
        hydroclass_data, entropy_data, prop_data = _assign_to_class_scan(
            fields_dict, mc_std, var_names=var_names, weights=weights, t_vals=t_vals
        )
    else:
        hydroclass_data, entropy_data, prop_data = _assign_to_class(
            fields_dict, mc_std, weights=weights, t_vals=t_vals
        )

    # prepare output fields
    fields_dict = dict()
    hydro = get_metadata(hydro_field)
    hydro["data"] = hydroclass_data
    hydro.update({"_FillValue": 0})
    labels = ["NC"]
    ticks = [0]
    boundaries = [-0.5, 0.5]
    for i, hydro_name in enumerate(hydro_names):
        labels.append(hydro_name)
        ticks.append(i + 1)
        boundaries.append(i + 1.5)
    hydro.update({"labels": labels, "ticks": ticks, "boundaries": boundaries})
    fields_dict.update({"hydro": hydro})

    if compute_entropy:
        entropy = get_metadata(entropy_field)
        entropy["data"] = entropy_data
        fields_dict.update({"entropy": entropy})

        if output_distances:
            for field_name in hydro_names:
                field_name = "proportion_" + field_name
                prop = get_metadata(field_name)
                prop["data"] = prop_data[:, :, i]
                fields_dict.update({field_name: prop})
    return fields_dict


def data_for_centroids(
    radar,
    lapse_rate=-6.5,
    refl_field=None,
    zdr_field=None,
    rhv_field=None,
    kdp_field=None,
    temp_field=None,
    iso0_field=None,
    temp_ref="temperature",
    nsamples_max=20000,
):
    """
    Prepares the data to compute the centroids of the hydrometeor
    classification

    Parameters
    ----------
    radar : radar
        Radar object.
    lapse_rate : float
        The decrease in temperature for each vertical km [deg/km]
    refl_field, zdr_field, rhv_field, kdp_field, temp_field, iso0_field : str
        Inputs. Field names within the radar object which represent the
        horizonal reflectivity, the differential reflectivity, the copolar
        correlation coefficient, the specific differential phase, the
        temperature and the height respect to the iso0 fields. A value of None
        for any of these parameters will use the default field name as defined
        in the Py-ART configuration file.
    temp_ref : str
        the field use as reference for temperature. Can be either temperature
        or height_over_iso0
    nsamples_max : int
        Maximum number of samples to keep from each radar volume

    Returns
    -------
    refl_std, zdr_std, kdp_std, rhohv_std, relh_std : 1D-array
        The standardized valid data

    """
    # parse the field parameters
    if refl_field is None:
        refl_field = get_field_name("reflectivity")
    if zdr_field is None:
        zdr_field = get_field_name("differential_reflectivity")
    if rhv_field is None:
        rhv_field = get_field_name("cross_correlation_ratio")
    if kdp_field is None:
        kdp_field = get_field_name("specific_differential_phase")

    if temp_ref == "temperature":
        if temp_field is None:
            temp_field = get_field_name("temperature")
    else:
        if iso0_field is None:
            iso0_field = get_field_name("height_over_iso0")

    # extract fields and parameters from radar
    radar.check_field_exists(refl_field)
    radar.check_field_exists(zdr_field)
    radar.check_field_exists(rhv_field)
    radar.check_field_exists(kdp_field)
    if temp_ref == "temperature":
        radar.check_field_exists(temp_field)
    else:
        radar.check_field_exists(iso0_field)

    refl = deepcopy(radar.fields[refl_field]["data"])
    zdr = deepcopy(radar.fields[zdr_field]["data"])
    rhohv = deepcopy(radar.fields[rhv_field]["data"])
    kdp = deepcopy(radar.fields[kdp_field]["data"])

    if temp_ref == "temperature":
        # convert temp in relative height respect to iso0
        temp = deepcopy(radar.fields[temp_field]["data"])
        relh = temp * (1000.0 / lapse_rate)
    else:
        relh = deepcopy(radar.fields[iso0_field]["data"])

    # filter data out of limits
    dlimits_dict = _data_limits_centroids_table()

    mx, mn = dlimits_dict["dBZ"]
    refl[(refl < mn) | (refl > mx)] = np.ma.masked

    mx, mn = dlimits_dict["ZDR"]
    zdr[(zdr < mn) | (zdr > mx)] = np.ma.masked

    mx, mn = dlimits_dict["RhoHV"]
    rhohv[(rhohv < mn) | (rhohv > mx)] = np.ma.masked

    mx, mn = dlimits_dict["KDP"]
    kdp[(kdp < mn) | (kdp > mx)] = np.ma.masked

    mx, mn = dlimits_dict["H_ISO0"]
    relh[(relh < mn) | (relh > mx)] = np.ma.masked

    # keep only gates with all values valid
    mask = np.ma.getmaskarray(refl)
    mask = np.logical_or(mask, np.ma.getmaskarray(zdr))
    mask = np.logical_or(mask, np.ma.getmaskarray(rhohv))
    mask = np.logical_or(mask, np.ma.getmaskarray(kdp))
    mask = np.logical_or(mask, np.ma.getmaskarray(relh))
    valid = np.logical_not(mask)

    refl = refl[valid]
    zdr = zdr[valid]
    rhohv = rhohv[valid]
    kdp = kdp[valid]
    relh = relh[valid]

    # keep only nsamples_max values of the volume
    if refl.size <= nsamples_max:
        return refl, zdr, kdp, rhohv, relh

    rg = np.random.default_rng(seed=0)
    ind = rg.integers(low=0, high=refl.size, size=nsamples_max)

    refl = refl[ind]
    zdr = zdr[ind]
    rhohv = rhohv[ind]
    kdp = kdp[ind]
    relh = relh[ind]

    return refl, zdr, kdp, rhohv, relh


def compute_centroids(
    features_matrix,
    weight=(1.0, 1.0, 1.0, 1.0, 0.75),
    var_names=("dBZ", "ZDR", "KDP", "RhoHV", "H_ISO0"),
    hydro_names=("AG", "CR", "LR", "RP", "RN", "VI", "WS", "MH", "IH/HDG"),
    nsamples_iter=20000,
    external_iterations=30,
    internal_iterations=10,
    alpha=0.01,
    cv_approach=True,
    num_samples_arr=(30, 35, 40),
    n_samples_syn=50,
    nmedoids_min=1,
    acceptance_threshold=0.5,
    band="C",
    relh_slope=0.001,
    parallelized=False,
    sample_data=True,
    kmax_iter=100,
    nsamples_small=40000,
    sampling_size_clara=10000,
    niter_clara=5,
    keep_labeled_data=True,
    use_median=False,
    allow_label_duplicates=False,
):
    """
    Given a features matrix computes the centroids

    Parameters
    ----------
    features_matrix : 2D-array
        matrix of size (nsamples, nvariables)
    weight : tuple
        Weight given to each feature in the KS test
    var_names : tupple
        List of name variables
    hydro_names : tupple
        List of hydrometeor types
    nsamples_iter : int
        Number of samples of the features matrix in each external iteration
    external_iterations : int
        Number of iterations of the external loop. This number will determine
        how many medoids are computed for each hydrometeor class.
    internal_iterations : int
        Maximum number of iterations of the internal loop
    acceptance_threshold : float
        Threshold on the inter-quantile coefficient of dispersion of the
        medoids above which the medoid of the class is not acceptable.
    alpha : float
        Minimum value to accept the cluster according to p
    cv_approach : bool
        If true it is used a critical value approach to reject or accept
        similarity between observations and reference. If false it is used a
        p-value approach
    num_samples_arr : 1D-array
        Array containing the possible number of observation samples to use when
        comparing with reference
    n_samples_syn : int
        Number of samples from reference used in comparison
    nmedoids_min : int
        Minimum number of valid intermediate medoids to compute a final medoid
    band : str
        Frequency band of the radar data. Can be C, S or X
    relh_slope : float
        The slope used to transform the height relative to the iso0 into
        a sigmoid function.
    parallelized : bool
        If True the processing is going to be parallelized
    sample_data : bool
        If True the data is going to be sampled at each external loop
    kmax_iter : int
        Maximum number of iterations of the kmedoids algorithm
    nsamples_small : int
        Maximum number before using the k-medoids CLARA algorithm. If this
        number is exceeded the CLARA algorithm will be used
    sampling_size_clara : int
        Number of samples used in each iteration of the k-medoids CLARA
        algorithm.
    niter_clara : int
        Number of iterations performed by the k-medoids CLARA algorithm
    keep_labeled_data : bool
        If True the labeled data is going to be kept.
    use_median : bool
        If True the intermediate medoids are computed as the median of each
        variable and the final medoids are computed as the median of each.
        Otherwise they are computed using the kmedoids algorithm.
    allow_label_duplicates : bool
        If True allow to label multiple clusters with the same label

    Returns
    -------
    labeled_data : 2D-array
        matrix of size (nsamples, nvariables) containing the observations
    labels : 1D-array
        array with the labels index
    medoids_dict : dict
        Dictionary containing the intermediate medoids for each hydrometeor
        type
    final_medoids_dict : dict
        Dictionary containing the final medoids for each hydrometeor type

    """
    if not _SKLEARN_AVAILABLE:
        warn("Unable to compute centroids. scikit-learn package not available")
        return None, None, dict(), None
    if not _CLARA_AVAILABLE:
        warn(
            "CLARA k-medoids algorithm not available. "
            "Unable to process large datasets"
        )

    if parallelized:
        if not _DASK_AVAILABLE:
            warn("dask not available: The processing will not be parallelized")
            parallelized = False

    labels = None
    labeled_data = None
    medoids_dict = dict()
    if not parallelized:
        rg = np.random.default_rng(seed=0)
        for i in range(external_iterations):
            new_labels, new_labeled_data, inter_medoids_dict = centroids_iter(
                features_matrix,
                i,
                rg,
                weight=weight,
                var_names=var_names,
                hydro_names=hydro_names,
                nsamples_iter=nsamples_iter,
                external_iterations=external_iterations,
                internal_iterations=internal_iterations,
                alpha=alpha,
                cv_approach=cv_approach,
                num_samples_arr=num_samples_arr,
                n_samples_syn=n_samples_syn,
                band=band,
                relh_slope=relh_slope,
                sample_data=sample_data,
                kmax_iter=kmax_iter,
                nsamples_small=nsamples_small,
                sampling_size_clara=sampling_size_clara,
                niter_clara=niter_clara,
                keep_labeled_data=keep_labeled_data,
                use_median=use_median,
                allow_label_duplicates=allow_label_duplicates,
            )
            if not inter_medoids_dict:
                continue

            labels, labeled_data, medoids_dict = store_centroids(
                new_labels,
                new_labeled_data,
                inter_medoids_dict,
                hydro_names,
                labels=labels,
                labeled_data=labeled_data,
                medoids_dict=medoids_dict,
                keep_labeled_data=keep_labeled_data,
            )
    else:
        jobs = []
        for i in range(external_iterations):
            rg = np.random.default_rng(seed=None)
            features_matrix_aux = dask.delayed(features_matrix)
            jobs.append(
                dask.delayed(centroids_iter)(
                    features_matrix_aux,
                    i,
                    rg,
                    weight=weight,
                    var_names=var_names,
                    hydro_names=hydro_names,
                    nsamples_iter=nsamples_iter,
                    external_iterations=external_iterations,
                    internal_iterations=internal_iterations,
                    alpha=alpha,
                    cv_approach=cv_approach,
                    num_samples_arr=num_samples_arr,
                    n_samples_syn=n_samples_syn,
                    band=band,
                    relh_slope=relh_slope,
                    sample_data=sample_data,
                    kmax_iter=kmax_iter,
                    nsamples_small=nsamples_small,
                    sampling_size_clara=sampling_size_clara,
                    niter_clara=niter_clara,
                    keep_labeled_data=keep_labeled_data,
                    use_median=use_median,
                    allow_label_duplicates=allow_label_duplicates,
                )
            )
        try:
            jobs = dask.compute(*jobs)

            for i, (new_labels, new_labeled_data, inter_medoids_dict) in enumerate(
                jobs
            ):
                if keep_labeled_data:
                    if new_labels is None:
                        nlabeled = 0
                    else:
                        nlabeled = new_labels.size
                    print("iteration " + str(i + 1) + " labeled data " + str(nlabeled))
                if not inter_medoids_dict:
                    continue
                labels, labeled_data, medoids_dict = store_centroids(
                    new_labels,
                    new_labeled_data,
                    inter_medoids_dict,
                    hydro_names,
                    labels=labels,
                    labeled_data=labeled_data,
                    medoids_dict=medoids_dict,
                    keep_labeled_data=keep_labeled_data,
                )
            del jobs
        except Exception as ee:
            warn(str(ee))
            traceback.print_exc()

    if not medoids_dict:
        warn("Data could not be labeled")
        return None, None, dict(), None

    final_medoids_dict = determine_medoids(
        medoids_dict,
        var_names,
        hydro_names,
        nmedoids_min=nmedoids_min,
        acceptance_threshold=acceptance_threshold,
        kmax_iter=kmax_iter,
        use_median=use_median,
    )

    return labeled_data, labels, medoids_dict, final_medoids_dict


def centroids_iter(
    features_matrix,
    iteration,
    rg,
    weight=(1.0, 1.0, 1.0, 1.0, 0.75),
    var_names=("dBZ", "ZDR", "KDP", "RhoHV", "H_ISO0"),
    hydro_names=("AG", "CR", "LR", "RP", "RN", "VI", "WS", "MH", "IH/HDG"),
    nsamples_iter=20000,
    external_iterations=30,
    internal_iterations=10,
    alpha=0.01,
    cv_approach=True,
    num_samples_arr=(30, 35, 40),
    n_samples_syn=50,
    band="C",
    relh_slope=0.001,
    sample_data=True,
    kmax_iter=100,
    nsamples_small=40000,
    sampling_size_clara=10000,
    niter_clara=5,
    keep_labeled_data=True,
    use_median=False,
    allow_label_duplicates=False,
):
    """
    External iteration of the centroids computation

    Parameters
    ----------
    features_matrix : 2D-array
        matrix of size (nsamples, nvariables)
    iteration : int
        iteration number
    rg : Random Generator
        Random generator
    weight : tuple
        Weight given to each feature in the KS test
    var_names : tupple
        List of name variables
    hydro_names : tupple
        List of hydrometeor types
    nsamples_iter : int
        Number of samples of the features matrix in each external iteration
    external_iterations : int
        Number of iterations of the external loop. This number will determine
        how many medoids are computed for each hydrometeor class.
    internal_iterations : int
        Maximum number of iterations of the internal loop
    alpha : float
        Minimum value to accept the cluster according to p
    cv_approach : bool
        If true it is used a critical value approach to reject or accept
        similarity between observations and reference. If false it is used a
        p-value approach
    num_samples_arr : 1D-array
        Array containing the possible number of observation samples to use when
        comparing with reference
    n_samples_syn : int
        Number of samples from reference used in comparison
    band : str
        Frequency band of the radar data. Can be C, S or X
    relh_slope : float
        The slope used to transform the height relative to the iso0 into
        a sigmoid function.
    sample_data : Bool
        If True the feature matrix will be sampled
    kmax_iter : int
        Maximum number of iterations of the k-medoids algorithm
    nsamples_small : int
        Maximum number before using the k-medoids CLARA algorithm. If this
        number is exceeded the CLARA algorithm will be used
    sampling_size_clara : int
        Number of samples used in each iteration of the k-medoids CLARA
        algorithm.
    niter_clara : int
        Number of iterations performed by the k-medoids CLARA algorithm
    keep_labeled_data : Bool
        If True the labeled data is kept.
    use_median : Bool
        If True the intermediate medoids are computed as the median of each
        variable. Otherwise they are computed using the kmedoids algorithm.
    allow_label_duplicates : bool
        If True allow to label multiple clusters with the same label

    Returns
    -------
    labeled_data : 2D-array
        matrix of size (nsamples, nvariables) containing the observations
    labels : 1D-array
        array with the labels index
    medoids_dict : dict
        Dictionary containing the intermediate medoids for each hydrometeor
        type
    final_medoids_dict : dict
        Dictionary containing the final medoids for each hydrometeor type

    """
    print(
        "\n\n\nExternal loop. Iteration "
        + str(iteration + 1)
        + "/"
        + str(external_iterations)
    )
    # external loop to identify clusters
    ks_threshold, n_samples = compute_ks_threshold(
        rg, alpha=alpha, n_samples_syn=n_samples_syn, num_samples_arr=num_samples_arr
    )
    synthetic_obs = synthetic_obs_distribution(
        rg, var_names, hydro_names, band=band, relh_slope=relh_slope
    )

    if nsamples_iter > features_matrix.shape[0]:
        warn("Number of samples lower than number of samples per iteration")
        fm_sample = deepcopy(features_matrix)
    elif not sample_data:
        fm_sample = deepcopy(features_matrix)
    else:
        fm_sample, _ = train_test_split(features_matrix, train_size=nsamples_iter)

    # Uses sklearn.metrics.pairwise_distances for the metric
    # metric can be also those of scipy.spatial.distance
    if fm_sample.shape[0] > nsamples_small and _CLARA_AVAILABLE:
        kmedoids = CLARA(
            n_clusters=len(hydro_names),
            metric="seuclidean",
            init="k-medoids++",
            max_iter=kmax_iter,
            random_state=None,
            sampling_size=sampling_size_clara,
            samples=niter_clara,
        ).fit(fm_sample)
    else:
        if fm_sample.shape[0] < 3000:
            kmedoids = KMedoids(
                n_clusters=len(hydro_names),
                metric="seuclidean",
                method="pam",
                init="k-medoids++",
                max_iter=kmax_iter,
                random_state=None,
            ).fit(fm_sample)
        else:
            kmedoids = KMedoids(
                n_clusters=len(hydro_names),
                metric="seuclidean",
                method="alternate",
                init="k-medoids++",
                max_iter=kmax_iter,
                random_state=None,
            ).fit(fm_sample)

    new_labels, new_labeled_data, _ = search_medoids(
        fm_sample,
        kmedoids.labels_,
        synthetic_obs,
        var_names,
        hydro_names,
        weight,
        ks_threshold,
        alpha,
        cv_approach,
        n_samples_syn,
        n_samples,
        1,
        iteration_max=internal_iterations,
        relh_slope=relh_slope,
        kmax_iter=kmax_iter,
        nsamples_small=nsamples_small,
        sampling_size_clara=sampling_size_clara,
        niter_clara=niter_clara,
        allow_label_duplicates=allow_label_duplicates,
    )

    if new_labels is None:
        print("No data labeled in internal loops")
        return None, None, dict()

    print("labeled data in internal loop: " + str(new_labels.size))

    # Compute medoids as the median of the clustered data
    inter_medoids_dict = compute_intermediate_medoids(
        new_labeled_data,
        new_labels,
        hydro_names,
        kmax_iter=kmax_iter,
        nsamples_small=nsamples_small,
        sampling_size_clara=sampling_size_clara,
        niter_clara=niter_clara,
        use_median=use_median,
    )
    if not keep_labeled_data:
        new_labels = None
        new_labeled_data = None

    return new_labels, new_labeled_data, inter_medoids_dict


def store_centroids(
    new_labels,
    new_labeled_data,
    inter_medoids_dict,
    hydro_names,
    labels=None,
    labeled_data=None,
    medoids_dict=None,
    keep_labeled_data=True,
):
    """
    Store the centroids data to its respective recipients

    Parameters
    ----------
    new_labels : 1D-array
        array with the labels index
    new_labeled_data : 2D-array
        matrix of size (nsamples, nvariables) containing the observations
    inter_medoids_dict : dict
        Dictionary containing the intermediate medoids for each hydrometeor
        type
    hydro_names : array of str
        The name of the hydrometeor types
    labels : 1D-array or None
        array where to store the new labels
    labeled_data : 2D-array or None
        matrix of size (nsamples, nvariables) where to store the labeled data
    medoids_dict : dict
        Dictionary where to store the new medoids
    keep_labeled_data : Bool
        If True the labeled data is going to be stored

    Returns
    -------
    labels : 1D-array or None
        array with the labels index
    labeled_data : 2D-array or None
        matrix of size (nsamples, nvariables) containing the observations
    medoids_dict : dict
        Dictionary containing the intermediate medoids for each hydrometeor
        type

    """
    # store the correctly identified data and its labels
    if keep_labeled_data:
        if labels is None:
            labels = deepcopy(new_labels)
            labeled_data = deepcopy(new_labeled_data)
        else:
            labels = np.append(labels, new_labels, axis=-1)
            labeled_data = np.append(labeled_data, new_labeled_data, axis=0)
    del new_labels
    del new_labeled_data

    # store the new medoids
    for hydro_name in hydro_names:
        if hydro_name in inter_medoids_dict:
            if hydro_name in medoids_dict:
                medoids_dict[hydro_name] = np.append(
                    medoids_dict[hydro_name], inter_medoids_dict[hydro_name], axis=0
                )
            else:
                medoids_dict.update({hydro_name: inter_medoids_dict[hydro_name]})

    if labels is not None:
        print("total labeled data: " + str(labels.size))

    return labels, labeled_data, medoids_dict


def select_samples(
    fm,
    rg,
    nbins=110,
    pdf_zh_max=20000,
    pdf_relh_max=10000,
    sigma_zh=0.75,
    sigma_relh=1.5,
    randomize=True,
    platykurtic_dBZ=True,
    platykurtic_H_ISO0=True,
):
    """
    Selects the data to be used to compute the centroids

    Parameters
    ----------
    fm : 2D array (nsamples, nfeatures)
        Initial data
    rg : Random Generator
        The random generator used to randomize the polarimetric variables
    nbins : int
        Number of bins of the histograms used to make the data platykurtik
    pdf_zh_max : int
        Multiplication factor to the Guassian function over the
        reflectivity that determines the number of samples for each bin
    pdf_relh_max : int
        Multiplication factor to the Guassian function over the
        height relative to the iso-0 that determines the number of samples for
        each bin
    sigma_zh, sigma_relh : float
        sigma of the respective Gaussian functions
    randomize : bool
        if True the quantized data is randomized
    platykurtic_dBZ : Bool
        If True makes the reflectivity distribution platykurtic
    platykurtic_H_ISO0 : Bool
        If True makes the height relative to the iso-0 platykurtic

    Returns
    -------
    fm_sample : 2D array (nsamples, nfeatures)
        The selected data

    """
    nsamples = fm.shape[0]
    if randomize:
        print("Randomizing data")
        nfeatures = fm.shape[1]
        for i in range(nfeatures - 1):
            vals = np.unique(fm[:, i])
            step = np.median(vals[1:] - vals[:-1])
            print(f"Number of unique values before randomization: {vals.shape}")
            print(f"vmin: {vals.min()} vmax: {vals.max()}")
            print(f"Step between values: {step}")
            fm[:, i] += rg.random(nsamples) * step - step / 2.0
            print(f"Number of unique values after randomization: {fm[:, i].shape}")

    refl, zdr, kdp, rhohv, relh = make_platykurtic(
        fm[:, 0],
        fm[:, 1],
        fm[:, 2],
        fm[:, 3],
        fm[:, 4],
        nbins=nbins,
        pdf_zh_max=pdf_zh_max,
        pdf_relh_max=pdf_relh_max,
        sigma_zh=sigma_zh,
        sigma_relh=sigma_relh,
        platykurtic_dBZ=platykurtic_dBZ,
        platykurtic_H_ISO0=platykurtic_H_ISO0,
    )

    fm_sample = np.transpose(np.array([refl, zdr, kdp, rhohv, relh]))

    print("Selected " + str(fm_sample.shape[0]) + " samples out of " + str(nsamples))

    return fm_sample


def make_platykurtic(
    refl,
    zdr,
    kdp,
    rhohv,
    relh,
    nbins=110,
    pdf_zh_max=20000,
    pdf_relh_max=10000,
    sigma_zh=0.75,
    sigma_relh=1.5,
    platykurtic_dBZ=True,
    platykurtic_H_ISO0=True,
):
    """
    Prepares the data to compute the centroids of the hydrometeor
    classification

    Parameters
    ----------
    refl, zdr, kdp, rhohv, relh : 1D array
        Arrays containing the data
    nbins : int
        Number of bins of the histogram for each dataset
    pdf_zh_max : int
        Multiplication factor to the Guassian function over the
        reflectivity that determines the number of samples for each bin
    pdf_relh_max : int
        Multiplication factor to the Guassian function over the
        height relative to the iso-0 that determines the number of samples for
        each bin
    sigma_zh, sigma_relh : float
        sigma of the respective Gaussian functions
    platykurtic_dBZ : Bool
        If True makes the reflectivity distribution platykurtic
    platykurtic_H_ISO0 : Bool
        If True makes the height relative to the iso-0 platykurtic

    Returns
    -------
    refl, zdr, kdp, rhohv, relh : 1D array
        The selected data

    """
    x_vals = np.linspace(-1.1, 1.1, num=nbins)

    if platykurtic_dBZ:
        print("Making dBz distribution platykurtic")
        # make reflectivity platykurtik
        _, bin_edges = np.histogram(refl, bins=nbins)
        pdf = np.array(
            gaussian_function(x_vals, mu=0.0, sigma=sigma_zh, normal=True) * pdf_zh_max,
            dtype=int,
        )

        refl_aux = []
        zdr_aux = []
        rhohv_aux = []
        kdp_aux = []
        relh_aux = []
        for i in range(nbins):
            ind = np.where((refl >= bin_edges[i]) & (refl < bin_edges[i + 1]))[0]
            if ind.size > pdf[i]:
                ind = ind[: pdf[i]]

            refl_aux.extend(refl[ind])
            zdr_aux.extend(zdr[ind])
            rhohv_aux.extend(rhohv[ind])
            kdp_aux.extend(kdp[ind])
            relh_aux.extend(relh[ind])

        refl = np.array(refl_aux)
        zdr = np.array(zdr_aux)
        rhohv = np.array(rhohv_aux)
        kdp = np.array(kdp_aux)
        relh = np.array(relh_aux)

    if platykurtic_H_ISO0:
        # Make relative height platykurtik
        print("Making H_ISO0 distribution platykurtic")
        _, bin_edges = np.histogram(relh, bins=nbins)

        pdf = np.array(
            gaussian_function(x_vals, mu=0.0, sigma=sigma_relh, normal=True)
            * pdf_relh_max,
            dtype=int,
        )

        refl_aux = []
        zdr_aux = []
        rhohv_aux = []
        kdp_aux = []
        relh_aux = []
        for i in range(nbins):
            ind = np.where((relh >= bin_edges[i]) & (relh < bin_edges[i + 1]))[0]
            if ind.size > pdf[i]:
                ind = ind[: pdf[i]]

            refl_aux.extend(refl[ind])
            zdr_aux.extend(zdr[ind])
            rhohv_aux.extend(rhohv[ind])
            kdp_aux.extend(kdp[ind])
            relh_aux.extend(relh[ind])

        refl = np.array(refl_aux)
        zdr = np.array(zdr_aux)
        rhohv = np.array(rhohv_aux)
        kdp = np.array(kdp_aux)
        relh = np.array(relh_aux)

    return refl, zdr, kdp, rhohv, relh


def search_medoids(
    fm,
    clust_labels,
    synthetic_obs,
    var_names,
    hydro_names,
    weight,
    ks_threshold,
    alpha,
    cv_approach,
    n_samples_syn,
    n_samples,
    iteration,
    iteration_max=10,
    relh_slope=0.001,
    kmax_iter=100,
    nsamples_small=40000,
    sampling_size_clara=10000,
    niter_clara=5,
    allow_label_duplicates=False,
):
    """
    Given a features matrix computes the centroids. This function is recursive

    Parameters
    ----------
    fm : 2D-array
        feature matrix of size (nsamples, nvariables)
    clust_labels : 1D-array
        cluster labels
    synthetic_obs : dict of dicts
        dictionary of type dict[var_name][hydro_name] with the samples
        corresponding to each membership function distribution
    var_names : list
        variables names
    hydro_names : list
        hydrometeor types names
    weight : Tupple
        weight assigned to each variable when looking at the similarity
        between membership functions pdfs and real observations
    ks_threshold : float
        Treshold used in the Kolmogorov–Smirnov comparison. If a statistic is
        above this value the similarity is rejected
    alpha : float
        Parameter alpha
    cv_approach : bool
        If true it is used a critical value approach to reject or accept
        similarity between observations and reference. If false it is used a
        p-value approach
    n_samples_syn : int
        Number of samples of the synthetic observations used in the KS test
    n_samples : int
        Number of real observations used to compare with the membership
        functions
    iteration : int
        index of the current iteration in the internal loop
    iteration_max : int
        Maximum number of iterations allowed in the internal loop
    relh_slope : float
        slope of the sigmoid function used to standardize height relative to
        the iso-0 data into
    kmax_iter : int
        Maximum number of iterations of the k-medoids algorithm
    nsamples_small : int
        Maximum number before using the k-medoids CLARA algorithm. If this
        number is exceeded the CLARA algorithm will be used
    sampling_size_clara : int
        Number of samples used in each iteration of the k-medoids CLARA
        algorithm.
    niter_clara : int
        Number of iterations performed by the k-medoids CLARA algorithm
    allow_label_duplicates : Bool
        If True allow to label multiple clusters with the same label

    Returns
    -------
    new_labels : 1D-array of int or None
        Array containing the label of each new clustered observation. None if
        no observation has been clustered
    new_labeled_data : 2D_array of floats or None.
        Array containing the clustered observations. None if no observation
        has been clustered
    iteration : int
        current iteration

    """
    print("\n\nInternal loop. Iteration: " + str(iteration) + "/" + str(iteration_max))

    if iteration >= iteration_max:
        warn("Maximum number of iterations reached or not enough samples")
        return None, None, iteration

    (hydro_labels, labeled_data, cluster_labels, nonlabeled_data) = compare_samples(
        var_names,
        hydro_names,
        weight,
        synthetic_obs,
        fm,
        clust_labels,
        ks_threshold,
        alpha,
        cv_approach,
        n_samples,
        n_samples_syn=n_samples_syn,
        relh_slope=relh_slope,
        allow_label_duplicates=allow_label_duplicates,
    )

    n_labeled = 0
    if hydro_labels is not None:
        n_labeled = hydro_labels.size
    print(
        "iteration: "
        + str(iteration)
        + " labeled points: "
        + str(n_labeled)
        + "/"
        + str(fm.shape[0])
    )

    n_nonlabeled = 0
    if cluster_labels is not None:
        n_nonlabeled = cluster_labels.size
    if n_nonlabeled <= n_samples:
        return hydro_labels, labeled_data, iteration_max

    # split each non-identified cluster into two and compare with reference
    clusters = np.unique(cluster_labels)
    print(
        str(clusters.size) + " clusters are not valid. " + "Splitting invalid clusters"
    )

    iteration += 1
    iteration1 = np.empty(clusters.size)
    iteration2 = np.empty(clusters.size)
    for icluster, cluster_id in enumerate(clusters):
        fm1, clust_labels1, fm2, clust_labels2 = split_cluster(
            nonlabeled_data,
            cluster_labels,
            cluster_id,
            n_samples,
            kmax_iter=kmax_iter,
            nsamples_small=nsamples_small,
            sampling_size_clara=sampling_size_clara,
            niter_clara=niter_clara,
        )

        if fm1 is None:
            iteration1[icluster] = iteration_max
            hydro_labels1 = None
        else:
            (hydro_labels1, labeled_data1, iteration1[icluster]) = search_medoids(
                fm1,
                clust_labels1,
                synthetic_obs,
                var_names,
                hydro_names,
                weight,
                ks_threshold,
                alpha,
                cv_approach,
                n_samples_syn,
                n_samples,
                iteration,
                iteration_max=iteration_max,
                relh_slope=relh_slope,
                kmax_iter=kmax_iter,
                nsamples_small=nsamples_small,
                sampling_size_clara=sampling_size_clara,
                niter_clara=niter_clara,
                allow_label_duplicates=allow_label_duplicates,
            )

        if fm2 is None:
            iteration2[icluster] = iteration_max
            hydro_labels2 = None
        else:
            (hydro_labels2, labeled_data2, iteration2[icluster]) = search_medoids(
                fm2,
                clust_labels2,
                synthetic_obs,
                var_names,
                hydro_names,
                weight,
                ks_threshold,
                alpha,
                cv_approach,
                n_samples_syn,
                n_samples,
                iteration,
                iteration_max=iteration_max,
                relh_slope=relh_slope,
                kmax_iter=kmax_iter,
                nsamples_small=nsamples_small,
                sampling_size_clara=sampling_size_clara,
                niter_clara=niter_clara,
                allow_label_duplicates=allow_label_duplicates,
            )

        if hydro_labels1 is not None:
            # add the data
            if hydro_labels is None:
                hydro_labels = hydro_labels1
                labeled_data = labeled_data1
            else:
                hydro_labels = np.append(hydro_labels, hydro_labels1, axis=-1)
                labeled_data = np.append(labeled_data, labeled_data1, axis=0)

        if hydro_labels2 is not None:
            # add the data
            if hydro_labels is None:
                hydro_labels = hydro_labels2
                labeled_data = labeled_data2
            else:
                hydro_labels = np.append(hydro_labels, hydro_labels2, axis=-1)
                labeled_data = np.append(labeled_data, labeled_data2, axis=0)

    if np.all(iteration1 >= iteration_max) and np.all(iteration2 >= iteration_max):
        iteration = iteration_max

    return hydro_labels, labeled_data, iteration


def split_cluster(
    fm,
    labels,
    icluster,
    n_samples,
    kmax_iter=100,
    nsamples_small=40000,
    sampling_size_clara=10000,
    niter_clara=5,
):
    """
    Splits the elements of a features matrix corresponding to cluster icluster
    into 2 using the k-medoids algorithm

    Parameters
    ----------
    fm : 2D-array of floats
        feature matrix of size (nsamples, nvariables)
    labels : 1D-array
        labels corresponding to each sample
    icluster : int
        ID of the cluster to split
    n_samples : int
        minimum number of samples to consider the new set as valid
    kmax_iter : int
        Maximum number of iterations in the k-medoids algorithm
    nsamples_small : int
        Maximum number before using the k-medoids CLARA algorithm. If this
        number is exceeded the CLARA algorithm will be used
    sampling_size_clara : int
        Number of samples used in each iteration of the k-medoids CLARA
        algorithm.
    niter_clara : int
        Number of iterations performed by the k-medoids CLARA algorithm

    Returns
    -------
    fm1, fm2 : 2D-array of floats or None
        The feature matrix corresponding tot he new sets. None if the new set
        is too small
    clust_labels1, clust_labels2 : 1D_array of ints or None.
        labels of the new split

    """
    ind_cluster = np.where(labels == icluster)[0]
    fm_cluster = fm[ind_cluster, :]

    if ind_cluster.size > nsamples_small and _CLARA_AVAILABLE:
        kmedoids = CLARA(
            n_clusters=2,
            metric="seuclidean",
            init="k-medoids++",
            max_iter=kmax_iter,
            random_state=None,
            sampling_size=sampling_size_clara,
            samples=niter_clara,
        ).fit(fm_cluster)
    else:
        if ind_cluster.size < 3000:
            kmedoids = KMedoids(
                n_clusters=2,
                metric="seuclidean",
                method="pam",
                init="k-medoids++",
                max_iter=kmax_iter,
                random_state=None,
            ).fit(fm_cluster)
        else:
            kmedoids = KMedoids(
                n_clusters=2,
                metric="seuclidean",
                method="alternate",
                init="k-medoids++",
                max_iter=kmax_iter,
                random_state=None,
            ).fit(fm_cluster)

    ind1 = np.where(kmedoids.labels_ == 0)[0]
    ind2 = np.where(kmedoids.labels_ == 1)[0]

    print("\nCluster ID: " + str(icluster))
    print("Number of samples in fm1: " + str(ind1.size))
    print("Number of samples in fm2: " + str(ind2.size))

    # check if the number of samples is too small to proceed
    if ind1.size < n_samples:
        print(
            "Number of non-labeled samples left ("
            + str(ind1.size)
            + ") smaller than number of samples necessary for clustering ("
            + str(int(n_samples))
            + ")"
        )
        fm1 = None
        clust_labels1 = None
    else:
        fm1 = fm_cluster[ind1, :]
        clust_labels1 = kmedoids.labels_[ind1]

    # check if the number of samples is too small to proceed
    if ind2.size < n_samples:
        print(
            "Number of non-labeled samples left ("
            + str(ind2.size)
            + ") smaller than number of samples necessary for clustering ("
            + str(int(n_samples))
            + ")"
        )
        fm2 = None
        clust_labels2 = None
    else:
        fm2 = fm_cluster[ind2, :]
        clust_labels2 = kmedoids.labels_[ind2]

    return fm1, clust_labels1, fm2, clust_labels2


def compute_intermediate_medoids(
    fm,
    labels,
    hydro_names,
    kmax_iter=100,
    nsamples_small=40000,
    sampling_size_clara=10000,
    niter_clara=5,
    use_median=False,
):
    """
    Computes the intermediate medoids from the labeled data

    Parameters
    ----------
    fm : 2D-array of floats
        The features matrix
    labels : 1D-array of ints
        The label of each sample
    hydro_names : 1D-array of str
        Name of the hydrometeors
    kmax_iter : int
        Maximum number of iterations of the kmedoids algorithm
    nsamples_small : int
        Maximum number before using the k-medoids CLARA algorithm. If this
        number is exceeded the CLARA algorithm will be used
    sampling_size_clara : int
        Number of samples used in each iteration of the k-medoids CLARA
        algorithm.
    niter_clara : int
        Number of iterations performed by the k-medoids CLARA algorithm
    use_median : bool
        If True the intermediate medoids are computed as the median of each
        variable. Otherwise they are computed using the kmedoids algorithm

    Returns
    -------
    inter_medoids_dict : dict
        dictionary with the medoids for each hydrometeor class

    """
    inter_medoids_dict = dict()
    for ihydro, hydro_name in enumerate(hydro_names):
        ind = np.where(labels == ihydro)[0]
        if ind.size == 0:
            continue
        if use_median:
            inter_medoids_dict.update(
                {hydro_name: np.median(fm[ind, :], axis=0, keepdims=True)}
            )
            continue
        if ind.size > nsamples_small and _CLARA_AVAILABLE:
            kmedoids = CLARA(
                n_clusters=1,
                metric="seuclidean",
                init="k-medoids++",
                max_iter=kmax_iter,
                random_state=None,
                sampling_size=sampling_size_clara,
                samples=niter_clara,
            ).fit(fm[ind, :])
        else:
            if ind.size < 3000:
                kmedoids = KMedoids(
                    n_clusters=1,
                    metric="seuclidean",
                    method="pam",
                    init="k-medoids++",
                    max_iter=kmax_iter,
                    random_state=None,
                ).fit(fm[ind, :])
            else:
                kmedoids = KMedoids(
                    n_clusters=1,
                    metric="seuclidean",
                    method="alternate",
                    init="k-medoids++",
                    max_iter=kmax_iter,
                    random_state=None,
                ).fit(fm[ind, :])
        inter_medoids_dict.update({hydro_name: kmedoids.cluster_centers_})

    return inter_medoids_dict


def determine_medoids(
    medoids_dict,
    var_names,
    hydro_names,
    nmedoids_min=1,
    acceptance_threshold=0.5,
    kmax_iter=100,
    use_median=False,
):
    """
    Computes the final medoids from the medoids found at each iteration

    Parameters
    ----------
    medoids_dict : dict
        dictionary of type dict[hydro_name]= array(niterations, nvars)
        containing the medoids for each hydrometeor type computed at each
        iteration
    var_names : 1D-array of str
        Name of variables
    hydro_names : 1D-array of str
        Name of hydrometeors
    nmedoids_min : int
        Minimum number of intermediate medoids to compute an intermediate
        medoid
    acceptance_threshold : float
        Threshold on the inter-quantile coefficient of dispersion of the
        medoids above which the medoid of the class is not acceptable.
    use_median : bool
        If True the final medoid is compute as the median of each variable
        in the intermediate medoids. Otherwise is computed using the kmedoids
        algorithm

    Returns
    -------
    final_medoids_dict : dict
        dictionary of type dict[hydro_name]= medoids_arr containing the
        medoids for each hydrometeor type that has been accepted

    """
    final_medoids_dict = dict()
    nvars = len(var_names)
    for hydro_name in hydro_names:
        if hydro_name not in medoids_dict:
            warn(f"No intermediate medoids found for class {hydro_name}")
            continue
        coef = 0.0
        medoids = medoids_dict[hydro_name]
        if medoids.shape[0] < nmedoids_min:
            warn(f"Not enough intermediate medoids for class {hydro_name}")
            continue
        for ivar, var_name in enumerate(var_names):
            medoid_var = deepcopy(medoids[:, ivar])
            # shift the distribution towards positive values
            min_val = medoid_var.min()
            if min_val < 0:
                # warn('Distribution of variable {} for hydrometeor type {} has
                # negative values.'.format(
                #     var_name, hydro_name))
                medoid_var -= min_val
            quant75 = np.quantile(medoid_var, 0.75)
            quant25 = np.quantile(medoid_var, 0.25)
            if quant75 + quant25 == 0.0:
                if quant75 == 0.0:
                    pass
                else:
                    warn("Inter-quantile cannot be computed")
                    nvars -= 1
            else:
                cqv = (quant75 - quant25) / (quant75 + quant25)
                if cqv < 0.0:
                    warn(f"Variable {var_name} has negative inter-quantile {cqv}")
                    nvars -= 1
                else:
                    coef += cqv
        coef = coef / nvars
        if coef > acceptance_threshold:
            warn(
                "Inter-quantile coefficient of dispersion ("
                + str(coef)
                + ") larger than treshold ("
                + str(acceptance_threshold)
                + ") no valid centroids for class "
                + hydro_name
            )
            continue
        print(
            "medoids for "
            + hydro_name
            + " found. Inter-quantile coefficient of dispersion: "
            + str(coef)
        )
        if use_median:
            final_medoids_dict.update({hydro_name: np.median(medoids, axis=0)})
        else:
            kmedoids = KMedoids(
                n_clusters=1,
                metric="seuclidean",
                method="alternate",
                init="k-medoids++",
                max_iter=kmax_iter,
                random_state=None,
            ).fit(medoids)
            final_medoids_dict.update(
                {hydro_name: np.squeeze(kmedoids.cluster_centers_)}
            )
    return final_medoids_dict


def synthetic_obs_distribution(
    rg, var_names, hydro_names, band="C", relh_slope=0.001, par_var=0.05
):
    """
    Gets the samples corresponding to the theoretical probability density
    function of each hydrometeor and variable

    Parameters
    ----------
    rg : Random generator
        The random generator used to generate samples of a uniform
        distribution
    var_names : 1D-array
        Name of the variables
    hydro_names : 1D-array
        Name of the hydrometeor types
    nsamples : int
        Number of samples for each distribution
    band : str
        Frequency band of the radar data. Can be C, S or X
    relh_slope : float
        The slope used to transform the height relative to the iso0 into
        a sigmoid function.
    par_var : float
        Variability of each parameter of the membership function.

    Returns
    -------
    synthetic_obs : dict of dicts
        dictionary of type dict[var_name][hydro_name] with the samples
        corresponding to each distribution

    """
    uniform_samples = rg.random(size=1000000, dtype=np.float32)

    synthetic_obs = dict()
    for var_name in var_names[:-1]:
        var_dict = dict()
        m_arr = np.empty(len(hydro_names))
        a_arr = np.empty(len(hydro_names))
        b_arr = np.empty(len(hydro_names))
        for ihydro, hydro_name in enumerate(hydro_names):
            # transform the uniform distribution according to the bell-shaped
            # distribution with random variations
            m, a, b = _bell_function_table()[band][var_name][hydro_name]
            min_m, max_m = [m - par_var * m, m + par_var * m]
            min_a, max_a = [a - par_var * a, a + par_var * a]
            min_b, max_b = [b - par_var * b, b + par_var * b]

            m_arr[ihydro] = (max_m - min_m) * rg.random() + min_m
            a_arr[ihydro] = (max_a - min_a) * rg.random() + min_a
            b_arr[ihydro] = (max_b - min_b) * rg.random() + min_b
        mn = np.min(m_arr - 1.5 * a_arr)
        mx = np.max(m_arr + 1.5 * a_arr)

        for ihydro, hydro_name in enumerate(hydro_names):
            pdf_samples = sample_bell(
                m=m_arr[ihydro], a=a_arr[ihydro], b=b_arr[ihydro], mn=mn, mx=mx
            )(uniform_samples)
            if var_name == "RhoHV":
                pdf_samples[pdf_samples > 1.0] = 1.0
            var_dict.update({hydro_name: pdf_samples})
        synthetic_obs.update({var_name: var_dict})

    # synthetic observation for height relative to iso0
    var_dict = dict()
    v1_arr = np.empty(len(hydro_names))
    v2_arr = np.empty(len(hydro_names))
    v3_arr = np.empty(len(hydro_names))
    v4_arr = np.empty(len(hydro_names))
    for ihydro, hydro_name in enumerate(hydro_names):
        # transform the uniform distribution according to the trapezoidal
        # distribution with random variations
        v1, v2, v3, v4 = _trapezoidal_function_table()[hydro_name]
        min_v1, max_v1 = [v1 - par_var * v1, v1 + par_var * v1]
        min_v2, max_v2 = [v2 - par_var * v2, v2 + par_var * v2]
        min_v3, max_v3 = [v3 - par_var * v3, v3 + par_var * v3]
        min_v4, max_v4 = [v4 - par_var * v4, v3 + par_var * v4]

        v1_rand = (max_v1 - min_v1) * rg.random() + min_v1
        v2_rand = (max_v2 - min_v2) * rg.random() + min_v2
        v3_rand = (max_v3 - min_v3) * rg.random() + min_v3
        v4_rand = (max_v4 - min_v4) * rg.random() + min_v4

        if v1_rand < v2_rand < v3_rand < v4_rand:
            v1 = v1_rand
            v2 = v2_rand
            v3 = v3_rand
            v4 = v4_rand
        v1_arr[ihydro] = v1
        v2_arr[ihydro] = v2
        v3_arr[ihydro] = v3
        v4_arr[ihydro] = v4
    mn = np.min(v1_arr)
    mx = np.max(v4_arr)
    for ihydro, hydro_name in enumerate(hydro_names):
        pdf_samples = sample_trapezoidal(
            v1=v1_arr[ihydro],
            v2=v2_arr[ihydro],
            v3=v3_arr[ihydro],
            v4=v4_arr[ihydro],
            mn=mn,
            mx=mx,
        )(uniform_samples)

        # standardize the samples of the distribution
        pdf_samples = _standardize(pdf_samples, var_names[-1], relh_slope=relh_slope)
        var_dict.update({hydro_name: pdf_samples})
    synthetic_obs.update({var_names[-1]: var_dict})

    return synthetic_obs


def compute_ks_threshold(
    rg, alpha=0.01, n_samples_syn=50, num_samples_arr=(30, 35, 40)
):
    """
    Computes the threshold of acceptance for the Kolmogorov–Smirnov test

    Parameters
    ----------
    rg : Random generator
        The random generator used to generate samples of a uniform
        distribution
    alpha : float
        test significance (alpha)
    n_samples_syn : int
        Number of samples from reference used to compare
    num_samples_arr : 1D-array
        Number of observation samples used to compare with reference to choose
        from

    Returns
    -------
    ks_threshold : float
        Critical value
    num_samples : float
        The selected number of samples

    """
    num_samples = rg.choice(num_samples_arr)
    ks_threshold = np.sqrt(
        -np.log(alpha / 2.0)
        * (num_samples + n_samples_syn)
        / (2.0 * num_samples * n_samples_syn)
    )

    return ks_threshold, num_samples


def compare_samples(
    var_names,
    hydro_names,
    weight,
    synthetic_obs,
    fm,
    clust_labels,
    ks_threshold,
    alpha,
    cv_approach,
    n_samples_obs,
    n_samples_syn=50,
    margin_ratio=0.1,
    relh_slope=0.001,
    allow_label_duplicates=False,
):
    """
    Compares the distribution of the clustered samples with the expected
    distribution

    Parameters
    ----------
    var_names : 1D-array of str
        Name of the variables
    hydro_names : 1D-array of str
        Name of the hydrometeor types
    weight : dict
        Weight given to each variable
    synthetic_obs : dict of dicts
        dictionary of type dict[var_name][hydro_name] with the samples
        corresponding to each distribution
    fm : 2D-array
        Features matrix (nsamples, nvars)
    clust_labels : 1D-array of nsamples
        Labels of each sample
    ks_threshold : float
        Threshold to accept the cluster as similar to the synthetic
        distribution
    alpha : float
        parameter alpha
    cv_approach : bool
        If true it is used a critical value approach to reject or accept
        similarity between observations and reference. If false it is used a
        p-value approach
    n_samples_obs : int
        Number of observations used in the KS test
    n_samples_syn : int
        Number of samples of the synthetic observations used in the KS test
    margin_ratio : float
        Percentage over range of observations margin to make the synthetic
        observations local
    relh_slope : float
        The slope used to transform the height relative to the iso0 into
        a sigmoid function.
    allow_label_duplicates : bool
        If True allow to label multiple clusters with the same label

    Returns
    -------
    labels : 1D-array
        array with the index of the label for the labeled data and -1 for that
        not labeled

    """
    total_weight = np.sum(weight)
    labels = -1 + np.zeros(clust_labels.size, dtype=np.uint8)
    hydro_names_aux = list(hydro_names)
    for jhydro in np.unique(clust_labels):
        best_stat = 1e6
        best_p = -1.0
        for ihydro, hydro_name in enumerate(hydro_names_aux):
            total_stat = 0.0
            total_p = 0.0
            for ivar, var_name in enumerate(var_names):
                # select the MF and the real observations
                so_aux = deepcopy(synthetic_obs[var_name][hydro_name])
                real_obs = deepcopy(fm[:, ivar])
                if var_name == "H_ISO0":
                    real_obs = _standardize(real_obs, "H_ISO0", relh_slope=relh_slope)
                real_obs = real_obs[clust_labels == jhydro]

                # check if there are enough samples in cluster
                if real_obs.size <= n_samples_obs:
                    total_stat = total_weight
                    total_p = total_weight
                    continue

                # give the MF the same limits as the real obs
                obs_rng_min = real_obs.min()
                obs_rng_max = real_obs.max()
                margin = (obs_rng_max - obs_rng_min) * margin_ratio
                so_rng_min = obs_rng_min - margin
                so_rng_max = obs_rng_max + margin
                so_aux = so_aux[(so_aux >= so_rng_min) & (so_aux <= so_rng_max)]

                # compare the real observations with the synthetic data
                if so_aux.size <= n_samples_syn:
                    statistic = 1.0
                    p = 0.0
                else:
                    # sampling period of observations
                    isamp_obs = int(np.ceil(real_obs.size / n_samples_obs))
                    real_obs = real_obs[::isamp_obs]
                    isamp_syn = int(np.ceil(so_aux.size / n_samples_syn))
                    so_aux = so_aux[::isamp_syn]
                    statistic, p = ks_2samp(
                        so_aux, real_obs, alternative="two-sided", mode="auto"
                    )

                total_stat += statistic * weight[ivar]
                total_p += p * weight[ivar]
            total_stat /= total_weight
            total_p /= total_weight

            # check if data pass the test and is better than previous score
            if (cv_approach and (total_stat < ks_threshold)) or (
                not cv_approach and (total_p > alpha)
            ):
                if (cv_approach and (total_stat < best_stat)) or (
                    not cv_approach and (total_p > best_p)
                ):
                    jhydro_aux = jhydro
                    ihydro_aux = ihydro
                    hydro_name_aux = hydro_name
                    best_stat = total_stat
                    best_p = total_p

        if best_stat < 1e6 or best_p > -1.0:
            labels[clust_labels == jhydro_aux] = ihydro_aux
            if cv_approach:
                print(
                    "test passed for variable "
                    + hydro_name_aux
                    + " with total statistic "
                    + str(best_stat)
                    + " and required statistic "
                    + str(ks_threshold)
                )
            else:
                print(
                    "test passed for variable "
                    + hydro_name_aux
                    + " with total p "
                    + str(best_p)
                    + " and required alpha "
                    + str(alpha)
                )
            if not allow_label_duplicates:
                hydro_names_aux.remove(hydro_name_aux)

    ind_id = np.where(labels > -1)[0]
    ind_noid = np.where(labels == -1)[0]

    hydro_labels = None
    labeled_data = None
    if ind_id.size > 0:
        hydro_labels = labels[ind_id]
        labeled_data = fm[ind_id, :]

    cluster_labels = None
    nonlabeled_data = None
    if ind_noid.size > 0:
        cluster_labels = clust_labels[ind_noid]
        nonlabeled_data = fm[ind_noid, :]

    return hydro_labels, labeled_data, cluster_labels, nonlabeled_data


def bell_function(x_vals, m=39.0, a=19.0, b=10.0):
    """
    Bell-shaped probability function

    Parameters
    ----------
    x_vals : float or array
        the x values
    m, a, b : float
        Parameters describing the bell-shaped function. The default are those
        of reflectivity in rain at C-band. The input data is in dBZ

    Returns
    -------
    prob : float or array
        The probability values given the x_vals

    """
    y = 1.0 / (1.0 + np.power(np.abs((x_vals - m) / a), 2.0 * b))
    y[~np.isfinite(y)] = 0.0
    return y


def gaussian_function(x_vals, mu=25.0, sigma=19.0, normal=True):
    """
    Bell-shaped probability function

    Parameters
    ----------
    x_vals : float or array
        the x values
    m, a, b : float
        Parameters describing the bell-shaped function. The default are those
        of reflectivity in rain at C-band. The input data is in dBZ

    Returns
    -------
    prob : float or array
        The probability values given the x_vals

    """
    y = np.exp(-1 * np.power(x_vals - mu, 2.0) / (2 * np.power(sigma, 2.0)))
    if normal:
        y *= 1.0 / (sigma * np.sqrt(2.0 * np.pi))
    return y


def trapezoidal_function(x_vals, v1=-2500.0, v2=-2200, v3=-300, v4=0):
    """
    trapezoidal probability function

    Parameters
    ----------
    x_vals : float or array
        the x values
    v1, v2, v3, v4 : float
        Parameters describing the trapezoidal function. The default are those
        of rain. The input data is in m respect to the iso0

    Returns
    -------
    prob : float or array
        The probability values given the x_vals

    """
    prob = np.zeros(x_vals.shape)
    prob[(x_vals > v1) & (x_vals <= v2)] = (
        x_vals[(x_vals > v1) & (x_vals <= v2)] - v1
    ) / (v2 - v1)
    prob[(x_vals > v2) & (x_vals <= v3)] = 1.0
    prob[(x_vals > v3) & (x_vals <= v4)] = (
        v4 - x_vals[(x_vals > v3) & (x_vals <= v4)]
    ) / (v4 - v3)

    return prob


def sample_bell(m=39.0, a=19.0, b=10.0, mn=-10.0, mx=60.0):
    """
    returns the function that computes the inverse version of the normalized
    cumulative sum of a Bell-shaped function

    Parameters
    ----------
    m, a, b : float
        Parameters describing the bell-shaped function. The default are those
        of reflectivity in rain at C-band. The input data is in dBZ
    mn, mx : float
        Range of the data

    Returns
    -------
    inverse_cdf : func
        Inverse cumulative distribution function

    """
    x = np.linspace(mn, mx, num=200)
    y = bell_function(x, m=m, a=a, b=b)  # probability density function, pdf
    cdf_y = np.abs(np.cumsum(y + 1e-10))  # cumulative distribution func, cdf
    cdf_y = cdf_y / cdf_y.max()  # takes care of normalizing cdf to 1.0
    inverse_cdf = interpolate.interp1d(
        cdf_y, x, fill_value="extrapolate"
    )  # this is a function
    return inverse_cdf


def sample_trapezoidal(v1=-2500.0, v2=-2200, v3=-300, v4=0, mn=-5000.0, mx=5000.0):
    """
    returns the function that computes the inverse version of the normalized
    cumulative sum of a Trapezoidal distribution

    Parameters
    ----------
    v1, v2, v3, v4 : float
        Parameters describing the Trapezoidal function. The default are those
        of rain. The input data is in m respect to the iso-0
    mn, mx : float
        Range of the data

    Returns
    -------
    inverse_cdf : func
        Inverse cumulative distribution function

    """
    x = np.linspace(mn, mx, num=100)
    y = trapezoidal_function(x, v1=v1, v2=v2, v3=v3, v4=v4)  # pdf
    cdf_y = np.abs(np.cumsum(y + 1e-10))  # cdf
    cdf_y = cdf_y / cdf_y.max()  # takes care of normalizing cdf to 1.0
    inverse_cdf = interpolate.interp1d(
        cdf_y, x, fill_value="extrapolate"
    )  # this is a function
    return inverse_cdf


def get_freq_band(freq):
    """
    Returns the frequency band name (S, C, X, ...).

    Parameters
    ----------
    freq : float
        Radar frequency [Hz].

    Returns
    -------
    freq_band : str
        Frequency band name.

    """
    if 2e9 <= freq < 4e9:
        return "S"
    if 4e9 <= freq < 8e9:
        return "C"
    if 8e9 <= freq <= 12e9:
        return "X"

    warn("Unknown frequency band")

    return None


def _standardize(data, field_name, mx=None, mn=None, relh_slope=0.005):
    """
    Streches the radar data to -1 to 1 interval.

    Parameters
    ----------
    data : array
        Radar field.
    field_name : str
        Type of field (H_ISO0, dBZ, ZDR, KDP or RhoHV).
    mx, mn : floats or None, optional
        Data limits for array values.
    relh_slope : float
        The slope used to transform the relative height into a sigmoid
        function

    Returns
    -------
    field_std : dict
        Standardized radar data.

    """
    if field_name == "H_ISO0":
        field_std = 2.0 / (1.0 + np.ma.exp(-relh_slope * data)) - 1.0
        return field_std

    if (mx is None) or (mn is None):
        dlimits_dict = _data_limits_table()
        if field_name not in dlimits_dict:
            raise ValueError(
                "Field "
                + field_name
                + " unknown. "
                + "Valid field names for standardizing are: "
                + "H_ISO0, dBZ, ZDR, KDP and RhoHV"
            )
        mx, mn = dlimits_dict[field_name]

    if field_name == "KDP":
        data[data < -0.5] = -0.5
        data = 10.0 * np.ma.log10(data + 0.6)
    elif field_name == "RhoHV":
        # avoid infinite result
        data[data > 1.0] = 1.0
        data = 10.0 * np.ma.log10(1.0000000000001 - data)

    mask = np.ma.getmaskarray(data)
    field_std = 2.0 * (data - mn) / (mx - mn) - 1.0
    field_std[data < mn] = -1.0
    field_std[data > mx] = 1.0
    field_std[mask] = np.ma.masked

    return field_std


def _destandardize(data, field_name, mx=None, mn=None, relh_slope=0.005):
    """
    destandardize the data

    Parameters
    ----------
    data : array
        Radar field.
    field_name : str
        Type of field (H_ISO0, dBZ, ZDR, KDP or RhoHV).
    mx, mn : floats or None, optional
        Data limits for array values.
    relh_slope : float
        The slope used to transform the relative height into a sigmoid
        function

    Returns
    -------
    field_std : dict
        destandardized radar data.

    """
    if field_name == "H_ISO0":
        field_std = np.log(2.0 / (1.0 + data) - 1.0) / (-relh_slope)
        return field_std

    if (mx is None) or (mn is None):
        dlimits_dict = _data_limits_table()
        if field_name not in dlimits_dict:
            raise ValueError(
                "Field "
                + field_name
                + " unknown. "
                + "Valid field names for standardizing are: "
                + "H_ISO0, dBZ, ZDR, KDP and RhoHV"
            )
        mx, mn = dlimits_dict[field_name]

    if field_name == "KDP":
        data = np.power(10.0, 0.1 * data) - 0.6
    elif field_name == "RhoHV":
        data = 1.0000000000001 - np.power(10.0, 0.1 * data)

    mask = np.ma.getmaskarray(data)
    field_std = 0.5 * (data + 1.0) * (mx - mn) + mn
    field_std[mask] = np.ma.masked

    return field_std


def _assign_to_class(
    fields_dict,
    mass_centers,
    var_names=("dBZ", "ZDR", "KDP", "RhoHV", "H_ISO0"),
    weights=np.array([1.0, 1.0, 1.0, 0.75, 0.5]),
    t_vals=None,
):
    """
    Assigns an hydrometeor class to a radar range bin computing
    the distance between the radar variables an a centroid.

    Parameters
    ----------
    fields_dict : dict
        Dictionary containg the variables used for assigment normalized to
        [-1, 1] values
    mass_centers : matrix
        centroids normalized to [-1, 1] values (nclasses, nvariables)
    var_names : array of str
        Name of the variables
    weights : array
        optional. The weight given to each variable (nvariables)
    t_vals : array
        transformation values for the distance to centroids (nclasses)

    Returns
    -------
    hydroclass : int array
        the index corresponding to the assigned class
    entropy : float array
        the entropy
    t_dist : float matrix
        if entropy is computed, the transformed distances of each class
        (proxy for proportions of each hydrometeor) (nrays, nbins, nclasses)

    """
    # prepare data
    nrays = fields_dict[var_names[0]].shape[0]
    nbins = fields_dict[var_names[0]].shape[1]
    nclasses = mass_centers.shape[0]
    nvariables = mass_centers.shape[1]
    dtype = fields_dict[var_names[0]].dtype

    hydroclass = np.ma.empty((nrays, nbins), dtype=np.uint8)
    entropy = None
    t_dist = None
    if t_vals is not None:
        entropy = np.ma.empty((nrays, nbins), dtype=dtype)
        t_dist = np.ma.masked_all((nrays, nbins, nclasses), dtype=dtype)

    for ray in range(nrays):
        data = []
        for var_name in var_names:
            data.append(fields_dict[var_name][ray, :])
        data = np.ma.array(data, dtype=dtype)
        weights_mat = np.broadcast_to(
            weights.reshape(nvariables, 1), (nvariables, nbins)
        )
        dist = np.ma.zeros((nclasses, nbins), dtype=dtype)

        # compute distance: masked entries will not contribute to the distance
        mask = np.ma.getmaskarray(fields_dict[var_names[0]][ray, :])
        for i in range(nclasses):
            centroids_class = mass_centers[i, :]
            centroids_class = np.broadcast_to(
                centroids_class.reshape(nvariables, 1), (nvariables, nbins)
            )
            dist_ray = np.ma.sqrt(
                np.ma.sum(((centroids_class - data) ** 2.0) * weights_mat, axis=0)
            )
            dist_ray[mask] = np.ma.masked
            dist[i, :] = dist_ray

        # Get hydrometeor class
        class_vec = dist.argsort(axis=0, fill_value=10e40)
        hydroclass_ray = (class_vec[0, :] + 1).astype(np.uint8)
        hydroclass_ray[mask] = 0
        hydroclass[ray, :] = hydroclass_ray

        if t_vals is None:
            continue

        # Transform the distance using the coefficient of the dominant class
        t_vals_ray = np.ma.masked_where(mask, t_vals[class_vec[0, :]])
        t_vals_ray = ma_broadcast_to(t_vals_ray.reshape(1, nbins), (nclasses, nbins))
        t_dist_ray = np.ma.exp(-t_vals_ray * dist)

        # set transformed distances to a value between 0 and 1
        dist_total = np.ma.sum(t_dist_ray, axis=0)
        dist_total = ma_broadcast_to(dist_total.reshape(1, nbins), (nclasses, nbins))
        t_dist_ray /= dist_total

        # Compute entropy
        entropy_ray = -np.ma.sum(
            t_dist_ray * np.ma.log(t_dist_ray) / np.ma.log(nclasses), axis=0
        )
        entropy_ray[mask] = np.ma.masked
        entropy[ray, :] = entropy_ray

        t_dist[ray, :, :] = np.ma.transpose(t_dist_ray)

    if t_vals is not None:
        t_dist *= 100.0

    return hydroclass, entropy, t_dist


def _assign_to_class_scan(
    fields_dict,
    mass_centers,
    var_names=("dBZ", "ZDR", "KDP", "RhoHV", "H_ISO0"),
    weights=np.array([1.0, 1.0, 1.0, 0.75, 0.5]),
    t_vals=None,
):
    """
    assigns an hydrometeor class to a radar range bin computing
    the distance between the radar variables an a centroid.
    Computes the entire radar volume at once

    Parameters
    ----------
    fields_dict : dict
        Dictionary containg the variables used for assigment normalized to
        [-1, 1] values
    mass_centers : matrix
        centroids normalized to [-1, 1] values
    var_names : array of str
        Name of the variables
    weights : array
        optional. The weight given to each variable
    t_vals : matrix
        transformation values for the distance to centroids
        (nclasses, nvariables)

    Returns
    -------
    hydroclass : int array
        the index corresponding to the assigned class
    entropy : float array
        the entropy
    t_dist : float matrix
        if entropy is computed, the transformed distances of each class
        (proxy for proportions of each hydrometeor) (nrays, nbins, nclasses)

    """
    # prepare data
    nrays = fields_dict[var_names[0]].shape[0]
    nbins = fields_dict[var_names[0]].shape[1]
    nclasses = mass_centers.shape[0]
    nvariables = mass_centers.shape[1]
    dtype = fields_dict[var_names[0]].dtype

    data = []
    for var_name in var_names:
        data.append(fields_dict[var_name])
    data = np.ma.array(data, dtype=dtype)
    weights_mat = np.broadcast_to(
        weights.reshape(nvariables, 1, 1), (nvariables, nrays, nbins)
    )

    # compute distance: masked entries will not contribute to the distance
    mask = np.ma.getmaskarray(fields_dict[var_names[0]])
    dist = np.ma.zeros((nrays, nbins, nclasses), dtype=dtype)
    t_dist = None
    entropy = None
    for i in range(nclasses):
        centroids_class = mass_centers[i, :]
        centroids_class = np.broadcast_to(
            centroids_class.reshape(nvariables, 1, 1), (nvariables, nrays, nbins)
        )
        dist_aux = np.ma.sqrt(
            np.ma.sum(((centroids_class - data) ** 2.0) * weights_mat, axis=0)
        )
        dist_aux[mask] = np.ma.masked
        dist[:, :, i] = dist_aux

    del data
    del weights_mat

    # Get hydrometeor class
    class_vec = dist.argsort(axis=-1, fill_value=10e40)
    hydroclass = np.ma.asarray(class_vec[:, :, 0] + 1, dtype=np.uint8)
    hydroclass[mask] = 0

    if t_vals is not None:
        # Transform the distance using the coefficient of the dominant class
        t_vals_aux = np.ma.masked_where(mask, t_vals[class_vec[:, :, 0]])
        t_vals_aux = ma_broadcast_to(
            t_vals_aux.reshape(nrays, nbins, 1), (nrays, nbins, nclasses)
        )
        t_dist = np.ma.exp(-t_vals_aux * dist)
        del t_vals_aux

        # set distance to a value between 0 and 1
        dist_total = np.ma.sum(t_dist, axis=-1)
        dist_total = ma_broadcast_to(
            dist_total.reshape(nrays, nbins, 1), (nrays, nbins, nclasses)
        )
        t_dist /= dist_total
        del dist_total

        # compute entroy
        entropy = -np.ma.sum(t_dist * np.ma.log(t_dist) / np.ma.log(nclasses), axis=-1)
        entropy[mask] = np.ma.masked

        t_dist *= 100.0

    return hydroclass, entropy, t_dist


def _compute_coeff_transform(
    mass_centers, weights=np.array([1.0, 1.0, 1.0, 0.75, 0.5]), value=50.0
):
    """
    get the transformation coefficients

    Parameters
    ----------
    mass_centers : ndarray 2D
        The centroids for each class and variable (nclasses, nvariables)
    weights : array
        optional. The weight given to each variable (nvariables)
    value : float
        parameter controlling the rate of decay of the distance transformation

    Returns
    -------
    t_vals : ndarray 1D
        The coefficients used to transform the distances to each centroid for
        each class (nclasses)

    """
    nclasses, nvariables = np.shape(mass_centers)
    t_vals = np.empty((nclasses, nclasses), dtype=mass_centers.dtype)
    for i in range(nclasses):
        weights_mat = np.broadcast_to(
            weights.reshape(1, nvariables), (nclasses, nvariables)
        )
        centroids_class = mass_centers[i, :]
        centroids_class = np.broadcast_to(
            centroids_class.reshape(1, nvariables), (nclasses, nvariables)
        )
        t_vals[i, :] = np.sqrt(
            np.sum(
                weights_mat * np.power(np.abs(centroids_class - mass_centers), 2.0),
                axis=1,
            )
        )

    # pick the second lowest value (the first is 0)
    t_vals = np.sort(t_vals, axis=-1)[:, 1]
    t_vals = np.log(value) / t_vals

    return t_vals


def _get_mass_centers(freq):
    """
    Get mass centers for a particular frequency.

    Parameters
    ----------
    freq : float
        Radar frequency [Hz].

    Returns
    -------
    mass_centers : ndarray 2D
        The centroids for each variable and hydrometeor class in (nclasses,
        nvariables).

    """
    mass_centers_dict = _mass_centers_table()

    freq_band = get_freq_band(freq)
    if (freq_band is not None) and (freq_band in mass_centers_dict):
        return mass_centers_dict[freq_band]

    if freq < 4e9:
        freq_band_aux = "C"
    elif freq > 12e9:
        freq_band_aux = "X"

    mass_centers = mass_centers_dict[freq_band_aux]
    warn(
        "Radar frequency out of range. "
        + "Centroids only valid for C or X band. "
        + freq_band_aux
        + " band centroids will be applied"
    )

    return mass_centers


def _mass_centers_table():
    """
    Defines the mass centers look up table for each frequency band.

    Returns
    -------
    mass_centers_dict : dict
        A dictionary with the mass centers for each frequency band.

    """
    nclasses = 9
    nvariables = 5
    mass_centers = np.zeros((nclasses, nvariables))

    mass_centers_dict = dict()
    # C-band centroids derived for MeteoSwiss Albis radar
    #                       Zh        ZDR     kdp   RhoHV    delta_Z
    mass_centers[0, :] = [13.5829, 0.4063, 0.0497, 0.9868, 1330.3]  # DS
    mass_centers[1, :] = [02.8453, 0.2457, 0.0000, 0.9798, 0653.8]  # CR
    mass_centers[2, :] = [07.6597, 0.2180, 0.0019, 0.9799, -1426.5]  # LR
    mass_centers[3, :] = [31.6815, 0.3926, 0.0828, 0.9978, 0535.3]  # GR
    mass_centers[4, :] = [39.4703, 1.0734, 0.4919, 0.9876, -1036.3]  # RN
    mass_centers[5, :] = [04.8267, -0.5690, 0.0000, 0.9691, 0869.8]  # VI
    mass_centers[6, :] = [30.8613, 0.9819, 0.1998, 0.9845, -0066.1]  # WS
    mass_centers[7, :] = [52.3969, 2.1094, 2.4675, 0.9730, -1550.2]  # MH
    mass_centers[8, :] = [50.6186, -0.0649, 0.0946, 0.9904, 1179.9]  # IH/HDG

    mass_centers_dict.update({"C": mass_centers})

    mass_centers = np.zeros((nclasses, nvariables))
    # X-band centroids derived for MeteoSwiss DX50 radar
    #                       Zh        ZDR     kdp    RhoHV   delta_Z
    mass_centers[0, :] = [19.0770, 0.4139, 0.0099, 0.9841, 1061.7]  # DS
    mass_centers[1, :] = [03.9877, 0.5040, 0.0000, 0.9642, 0856.6]  # CR
    mass_centers[2, :] = [20.7982, 0.3177, 0.0004, 0.9858, -1375.1]  # LR
    mass_centers[3, :] = [34.7124, -0.3748, 0.0988, 0.9828, 1224.2]  # GR
    mass_centers[4, :] = [33.0134, 0.6614, 0.0819, 0.9802, -1169.8]  # RN
    mass_centers[5, :] = [08.2610, -0.4681, 0.0000, 0.9722, 1100.7]  # VI
    mass_centers[6, :] = [35.1801, 1.2830, 0.1322, 0.9162, -0159.8]  # WS
    mass_centers[7, :] = [52.4539, 2.3714, 1.1120, 0.9382, -1618.5]  # MH
    mass_centers[8, :] = [44.2216, -0.3419, 0.0687, 0.9683, 1272.7]  # IH/HDG

    mass_centers_dict.update({"X": mass_centers})

    mass_centers = np.zeros((nclasses, nvariables))
    # S-band centroids: Dummy centroids derived for MeteoSwiss C-band Albis
    # radar. To be substituted for real S-band ones
    #                       Zh        ZDR     kdp   RhoHV    delta_Z
    mass_centers[0, :] = [13.5829, 0.4063, 0.0497, 0.9868, 1330.3]  # DS
    mass_centers[1, :] = [02.8453, 0.2457, 0.0000, 0.9798, 0653.8]  # CR
    mass_centers[2, :] = [07.6597, 0.2180, 0.0019, 0.9799, -1426.5]  # LR
    mass_centers[3, :] = [31.6815, 0.3926, 0.0828, 0.9978, 0535.3]  # GR
    mass_centers[4, :] = [39.4703, 1.0734, 0.4919, 0.9876, -1036.3]  # RN
    mass_centers[5, :] = [04.8267, -0.5690, 0.0000, 0.9691, 0869.8]  # VI
    mass_centers[6, :] = [30.8613, 0.9819, 0.1998, 0.9845, -0066.1]  # WS
    mass_centers[7, :] = [52.3969, 2.1094, 2.4675, 0.9730, -1550.2]  # MH
    mass_centers[8, :] = [50.6186, -0.0649, 0.0946, 0.9904, 1179.9]  # IH/HDG

    mass_centers_dict.update({"S": mass_centers})
    return mass_centers_dict


def _data_limits_table():
    """
    Defines the data limits used in the standardization.

    Returns
    -------
    dlimits_dict : dict
        A dictionary with the limits for each variable.

    """
    dlimits_dict = dict()
    dlimits_dict.update({"dBZ": (60.0, -10.0)})
    dlimits_dict.update({"ZDR": (5.0, -1.5)})
    dlimits_dict.update({"KDP": (7.0, -10.0)})
    dlimits_dict.update({"RhoHV": (-5.23, -50.0)})
    dlimits_dict.update({"H_ISO0": (5000.0, -5000.0)})

    return dlimits_dict


def _data_limits_centroids_table():
    """
    Defines the data limits used in the computation of the centroids.

    Returns
    -------
    dlimits_dict : dict
        A dictionary with the limits for each variable.

    """
    dlimits_dict = dict()
    dlimits_dict.update({"dBZ": (60.0, -9.0)})
    dlimits_dict.update({"ZDR": (5.0, -1.5)})
    dlimits_dict.update({"KDP": (5.0, -0.5)})
    dlimits_dict.update({"RhoHV": (1.0, 0.7)})
    dlimits_dict.update({"H_ISO0": (2500.0, -2500.0)})

    return dlimits_dict


def _bell_function_table():
    """
    Defines the bell-shaped functions of the polarimetric variables for each
    hydrometeor type

    Returns
    -------
    bell_function_dict : dict of dicts
        A dictionary with the parameters defining the bell-shaped function for
        each frequency, variable and hydrometeor type

    """
    bell_function_dict = dict()

    # C-band m, a, b
    dBZ_dict = {
        "CR": (-2.8, 12.0, 5.0),
        "AG": (17.0, 18.1, 10.0),
        "LR": (1.75, 29.0, 10.0),
        "RN": (39.0, 19.0, 10.0),
        "RP": (37.0, 9.2, 5),  # was 0.8 originally
        "VI": (-1.0, 11.0, 5.0),
        "WS": (24.0, 21.3, 10.0),
        "MH": (58.18, 8.0, 10.0),
        "IH/HDG": (48.8, 8.0, 10.0),
    }

    ZDR_dict = {
        "CR": (2.9, 2.7, 10.0),
        "AG": (1.0, 1.1, 7.0),
        "LR": (0.46, 0.46, 5.0),
        "RN": (2.3, 2.2, 9.0),
        "RP": (0.9, 0.9, 6.0),
        "VI": (-0.9, 0.9, 10),
        "WS": (1.3, 0.9, 10.0),
        "MH": (2.19, 1.5, 10.0),
        "IH/HDG": (0.36, 0.5, 10.0),
    }

    KDP_dict = {
        "CR": (0.08, 0.08, 6.0),
        "AG": (-0.008, 0.3, 1.0),
        "LR": (0.03, 0.03, 2.0),
        "RN": (5.5, 5.5, 10.0),
        "RP": (0.1, 0.08, 3.0),
        "VI": (-0.75, 0.75, 30.0),
        "WS": (0.25, 0.43, 6.0),
        "MH": (1.08, 2.0, 6.0),
        "IH/HDG": (0.07, 0.15, 6.0),
    }

    RhoHV_dict = {
        "CR": (0.98, 0.025, 3.0),
        "AG": (0.93, 0.07, 3.0),
        "LR": (1.0, 0.018, 3.0),
        "RN": (1.0, 0.025, 3.0),
        "RP": (1.0, 0.025, 1.0),
        "VI": (0.975, 0.022, 3.0),
        "WS": (0.8, 0.10, 10.0),
        "MH": (0.95, 0.05, 3.0),
        "IH/HDG": (0.99, 0.05, 3.0),
    }

    vars_dict = {"dBZ": dBZ_dict, "ZDR": ZDR_dict, "KDP": KDP_dict, "RhoHV": RhoHV_dict}

    bell_function_dict.update({"C": vars_dict})

    # X-band m, a, b
    dBZ_dict = {
        "CR": (-3, 12.0, 5.0),
        "AG": (16.0, 17.0, 10.0),
        "LR": (2.0, 29.0, 10.0),
        "RN": (42.0, 17.0, 10.0),
        "RP": (34.0, 10.0, 5),  # was 0.8 originally
        "VI": (3.5, 14.0, 5.0),
        "WS": (30.0, 20.0, 10.0),
        "MH": (53.37, 8.0, 10.0),
        "IH/HDG": (45.5, 8.0, 10.0),
    }

    ZDR_dict = {
        "CR": (3.2, 2.6, 10.0),
        "AG": (0.7, 0.7, 7.0),
        "LR": (0.5, 0.5, 5.0),
        "RN": (2.7, 2.8, 9.0),
        "RP": (0.3, 1.0, 6.0),
        "VI": (-0.8, 1.3, 10.0),
        "WS": (1.3, 0.9, 10.0),
        "MH": (2.2, 1.4, 10.0),
        "IH/HDG": (2.6, 0.5, 10.0),
    }

    KDP_dict = {
        "CR": (0.15, 0.15, 6.0),
        "AG": (0.2, 0.2, 1.0),
        "LR": (0.18, 0.18, 2.0),
        "RN": (12.6, 12.9, 10.0),
        "RP": (0.7, 2.1, 3.0),
        "VI": (-0.1, 0.08, 30.0),
        "WS": (1.0, 1.0, 6.0),
        "MH": (1.37, 2.0, 6.0),
        "IH/HDG": (0.1, 0.15, 6.0),
    }

    RhoHV_dict = {
        "CR": (0.985, 0.015, 3.0),
        "AG": (0.989, 0.011, 3.0),
        "LR": (0.992, 0.007, 3.0),
        "RN": (0.99, 0.01, 3.0),
        "RP": (0.993, 0.007, 1.0),
        "VI": (0.965, 0.035, 3.0),
        "WS": (0.835, 0.135, 10.0),
        "MH": (0.96, 0.05, 3.0),
        "IH/HDG": (0.97, 0.05, 3.0),
    }

    vars_dict = {"dBZ": dBZ_dict, "ZDR": ZDR_dict, "KDP": KDP_dict, "RhoHV": RhoHV_dict}

    bell_function_dict.update({"X": vars_dict})

    # S-band m, a, b: WS, MH, IH to be reviewed
    dBZ_dict = {
        "CR": (-3, 12.0, 5.0),
        "AG": (17.0, 17.0, 10.0),
        "LR": (-3.0, 24.0, 10.0),
        "RN": (41.5, 15.5, 10.0),
        "RP": (35.0, 10.0, 5),  # was 0.8 originally
        "VI": (3.0, 14.0, 5.0),
        "WS": (35.0, 20.0, 10.0),
        "MH": (55.0, 8.0, 10.0),
        "IH/HDG": (45.0, 8.0, 10.0),
    }

    ZDR_dict = {
        "CR": (2.9, 2.9, 10.0),
        "AG": (0.6, 0.6, 7.0),
        "LR": (0.35, 0.35, 5.0),
        "RN": (2.6, 2.5, 9.0),
        "RP": (0.3, 0.8, 6.0),
        "VI": (-0.8, 1.3, 10.0),
        "WS": (1.5, 0.9, 10.0),
        "MH": (2.0, 1.5, 10.0),
        "IH/HDG": (0.0, 0.5, 10.0),
    }

    KDP_dict = {
        "CR": (0.045, 0.045, 6.0),
        "AG": (0.04, 0.04, 1.0),
        "LR": (0.01, 0.01, 2.0),
        "RN": (3.7, 3.7, 10.0),
        "RP": (0.2, 0.6, 3.0),
        "VI": (-0.02, 0.02, 30.0),
        "WS": (0.1, 0.4, 6.0),
        "MH": (0.5, 2.0, 6.0),
        "IH/HDG": (0.05, 0.15, 6.0),
    }

    RhoHV_dict = {
        "CR": (0.99, 0.01, 3.0),
        "AG": (0.989, 0.011, 3.0),
        "LR": (0.995, 0.005, 3.0),
        "RN": (0.99, 0.01, 3.0),
        "RP": (0.995, 0.005, 1.0),
        "VI": (0.965, 0.035, 3.0),
        "WS": (0.85, 0.1, 10.0),
        "MH": (0.95, 0.05, 3.0),
        "IH/HDG": (0.98, 0.05, 3.0),
    }

    vars_dict = {"dBZ": dBZ_dict, "ZDR": ZDR_dict, "KDP": KDP_dict, "RhoHV": RhoHV_dict}

    bell_function_dict.update({"S": vars_dict})

    return bell_function_dict


def _trapezoidal_function_table():
    """
    Defines the trapezoidal functions of the height respect to iso-0 for each
    hydrometeor type

    Returns
    -------
    trapezoidal_function_dict : dict
        A dictionary with the parameters defining the trapezoidal function for
        each hydrometeor type

    """
    # v1, v2, v3, v4
    return {
        "CR": (0.0, 1000.0, 2200.0, 2500.0),
        "AG": (0.0, 500.0, 2000.0, 2500.0),
        "LR": (-2500.0, -2200.0, -300.0, 0.0),
        "RN": (-2500.0, -2200.0, -300.0, 0.0),
        "RP": (0.0, 500.0, 2000.0, 2200.0),
        "VI": (0.0, 1000.0, 2200.0, 2500.0),
        "WS": (-500.0, -300.0, 300.0, 500.0),
        "MH": (-2500.0, -2200.0, -300.0, 0.0),
        "IH/HDG": (0.0, 500.0, 2000.0, 2500.0),
    }


def conv_strat_raut(
    grid,
    refl_field,
    cappi_level=0,
    zr_a=200,
    zr_b=1.6,
    core_wt_threshold=5,
    conv_wt_threshold=1.5,
    conv_scale_km=25,
    min_reflectivity=5,
    conv_min_refl=25,
    conv_core_threshold=42,
    override_checks=False,
):
    """
    A computationally efficient method to classify radar echoes into convective cores, mixed convection,
    and stratiform regions for gridded radar reflectivity field.

    This function uses à trous wavelet transform (ATWT) for multiresolution (i.e. scale) analysis of radar field,
    focusing on precipitation structure over reflectivity thresholds for robust echo classification (Raut et al 2008, 2020).

    Parameters
    ----------
    grid : PyART Grid
        Grid object containing radar data.
    refl_field : str
        Field name for reflectivity data in the Py-ART grid object.
    zr_a : float, optional
        Coefficient 'a' in the Z-R relationship Z = a*R^b for reflectivity to rain rate conversion.
        The algorithm is not sensitive to precise values of 'zr_a' and 'zr_b'; however,
        they must be adjusted based on the type of radar used.
        Default is 200.
    zr_b : float, optional
        Coefficient 'b' in the Z-R relationship Z = a*R^b. Default is 1.6.
    core_wt_threshold : float, optional
        Threshold for wavelet components to separate convective cores from mix-intermediate type.
        Default is 5. Recommended values are between 4 and 6.
    conv_wt_threshold : float, optional
        Threshold for significant wavelet components to separate all convection from stratiform.
        Default is 1.5. Recommended values are between 1 and 2.
    conv_scale_km : float, optional
        Approximate scale break (in km) between convective and stratiform scales.
        Scale break may vary over different regions and seasons
        (Refere to Raut et al 2018 for more discussion on scale-breaks). Note that the
        algorithm is insensitive to small variations in the scale break due to the
        dyadic nature of the scaling. The actual scale break used in the calculation of wavelets
        is returned in the output dictionary by parameter `scale_break_used`.
        Default is 25 km. Recommended values are between 16 and 32 km.
    min_reflectivity : float, optional
        Minimum reflectivity threshold. Reflectivities below this value are not classified.
        Default is 5 dBZ. This value must be greater than or equal to '0'.
    conv_min_refl : float, optional
        Reflectivity values lower than this threshold will be always considered as non-convective.
        Default is 25 dBZ. Recommended values are between 25 and 30 dBZ.
    conv_core_threshold : float, optional
        Reflectivities above this threshold are classified as convective cores if wavelet components are significant (See: conv_wt_threshold).
        Default is 42 dBZ.
        Recommended value must be is greater than or equal to 40 dBZ. The algorithm is not sensitive to this value.
    override_checks : bool, optional
        If set to True, the function will bypass the sanity checks for above parameter values.
        This allows the user to use custom values for parameters, even if they fall outside
        the recommended ranges. The default is False.

    Returns
    -------

    dict :
    A dictionary structured as a Py-ART grid field, suitable for adding to a Py-ART Grid object. The dictionary
    contains the classification data and associated metadata. The classification categories are as follows:
        - 3: Convective Cores: associated with strong updrafts and active collision-coalescence.
        - 2: Mixed-Intermediate: capturing a wide range of convective activities, excluding the convective cores.
        - 1: Stratiform: remaining areas with more uniform and less intense precipitation.
        - 0: Unclassified: for reflectivity below the minimum threshold.


    References
    ----------
    Raut, B. A., Karekar, R. N., & Puranik, D. M. (2008). Wavelet-based technique to extract convective clouds from
    infrared satellite images. IEEE Geosci. Remote Sens. Lett., 5(3), 328-330.

    Raut, B. A., Seed, A. W., Reeder, M. J., & Jakob, C. (2018). A multiplicative cascade model for high‐resolution
    space‐time downscaling of rainfall. J. Geophys. Res. Atmos., 123(4), 2050-2067.

    Raut, B. A., Louf, V., Gayatri, K., Murugavel, P., Konwar, M., & Prabhakaran, T. (2020). A multiresolution technique
    for the classification of precipitation echoes in radar data. IEEE Trans. Geosci. Remote Sens., 58(8), 5409-5415.
    """

    # Check if the grid is a Py-ART Grid object
    if not isinstance(grid, Grid):
        raise TypeError("The 'grid' is not a Py-ART Grid object.")

    # Check if dx and dy are the same, and warn if not
    dx = grid.x["data"][1] - grid.x["data"][0]
    dy = grid.y["data"][1] - grid.y["data"][0]
    if dx != dy:
        warn(
            "Warning: Grid resolution `dx` and `dy` should be comparable for correct results.",
            UserWarning,
        )

    # Compute scale break (dyadic) here to paas it on as parameter to user dictionary
    scale_break = calc_scale_break(res_meters=dx, conv_scale_km=conv_scale_km)

    # From dyadic scale to km
    scale_break_km = (2 ** (scale_break - 1)) * dx / 1000

    # Sanity checks for parameters if override_checks is False
    if not override_checks:
        conv_core_threshold = max(
            40, conv_core_threshold
        )  # Ensure conv_core_threshold is at least 40 dBZ
        core_wt_threshold = max(
            4, min(core_wt_threshold, 6)
        )  # core_wt_threshold should be between 4 and 6
        conv_wt_threshold = max(
            1, min(conv_wt_threshold, 2)
        )  # conv_wt_threshold should be between 1 and 2
        conv_scale_km = max(
            16, min(conv_scale_km, 32)
        )  # conv_scale_km should be between 15 and 30 km
        min_reflectivity = max(
            0, min_reflectivity
        )  # min_reflectivity should be non-negative
        conv_min_refl = max(
            25, min(conv_min_refl, 30)
        )  # conv_min_refl should be between 25 and 30 dBZ

    # Call the actual wavelet_relass function to obtain radar echo classificatino
    reclass = wavelet_reclass(
        grid,
        refl_field,
        cappi_level,
        zr_a,
        zr_b,
        core_wt_threshold=core_wt_threshold,
        conv_wt_threshold=conv_wt_threshold,
        scale_break=scale_break,
        min_reflectivity=min_reflectivity,
        conv_min_refl=conv_min_refl,
        conv_core_threshold=conv_core_threshold,
    )

    reclass = np.expand_dims(reclass, axis=0)

    # put data into a dictionary to be added as a field
    reclass_dict = {
        "wt_reclass": {
            "data": reclass,
            "standard_name": "wavelet_echo_class",
            "long_name": "Wavelet-based multiresolution radar echo classification",
            "valid_min": 0,
            "valid_max": 3,
            "classification_description": "0: Unclassified, 1: Stratiform, 2: Mixed-Intermediate, 3: Convective Cores",
            "parameters": {
                "refl_field": refl_field,
                "cappi_level": cappi_level,
                "zr_a": zr_a,
                "zr_b": zr_b,
                "core_wt_threshold": core_wt_threshold,
                "conv_wt_threshold": conv_wt_threshold,
                "conv_scale_km": conv_scale_km,
                "scale_break_used": int(scale_break_km),
                "min_reflectivity": min_reflectivity,
                "conv_min_refl": conv_min_refl,
                "conv_core_threshold": conv_core_threshold,
            },
        }
    }

    return reclass_dict
