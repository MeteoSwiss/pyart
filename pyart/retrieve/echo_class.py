"""
pyart.retrieve.echo_class
=========================

Functions for echo classification.

.. autosummary::
    :toctree: generated/

    steiner_conv_strat
    hydroclass_semisupervised
    data_for_centroids
    compute_centroids
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

from warnings import warn
from copy import deepcopy

import numpy as np
from scipy import interpolate
from scipy.stats import ks_2samp

from ..config import get_fillvalue, get_field_name, get_metadata
from ._echo_class import steiner_class_buff
from ..util import ma_broadcast_to

try:
    from sklearn_extra.cluster import KMedoids
    from sklearn.model_selection import train_test_split
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


def steiner_conv_strat(grid, dx=None, dy=None, intense=42.0,
                       work_level=3000.0, peak_relation='default',
                       area_relation='medium', bkg_rad=11000.0,
                       use_intense=True, fill_value=None,
                       refl_field=None):
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
        refl_field = get_field_name('reflectivity')

    # parse dx and dy
    if dx is None:
        dx = grid.x['data'][1] - grid.x['data'][0]
    if dy is None:
        dy = grid.y['data'][1] - grid.y['data'][0]

    # Get coordinates
    x = grid.x['data']
    y = grid.y['data']
    z = grid.z['data']

    # Get reflectivity data
    ze = np.ma.copy(grid.fields[refl_field]['data'])
    ze = ze.filled(np.NaN)

    eclass = steiner_class_buff(ze, x, y, z, dx=dx, dy=dy, bkg_rad=bkg_rad,
                                work_level=work_level, intense=intense,
                                peak_relation=peak_relation,
                                area_relation=area_relation,
                                use_intense=use_intense,)

    return {'data': eclass.astype(np.int32),
            'standard_name': 'echo_classification',
            'long_name': 'Steiner echo classification',
            'valid_min': 0,
            'valid_max': 2,
            'comment_1': ('Convective-stratiform echo '
                          'classification based on '
                          'Steiner et al. (1995)'),
            'comment_2': ('0 = Undefined, 1 = Stratiform, '
                          '2 = Convective')}


def hydroclass_semisupervised(radar,
                              hydro_names=('AG', 'CR', 'LR', 'RP', 'RN', 'VI',
                                           'WS', 'MH', 'IH/HDG'),
                              var_names=('dBZ', 'ZDR', 'KDP', 'RhoHV',
                                         'H_ISO0'),
                              mass_centers=None,
                              weights=np.array([1., 1., 1., 0.75, 0.5]),
                              value=50., lapse_rate=-6.5, refl_field=None,
                              zdr_field=None, rhv_field=None, kdp_field=None,
                              temp_field=None, iso0_field=None,
                              hydro_field=None, entropy_field=None,
                              temp_ref='temperature', compute_entropy=False,
                              output_distances=False, vectorize=False):
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
        temperature and the height respect to the iso0 fields. A value of None
        for any of these parameters will use the default field name as defined
        in the Py-ART configuration file.
    hydro_field : str
        Output. Field name which represents the hydrometeor class field.
        A value of None will use the default field name as defined in the
        Py-ART configuration file.
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
        # assign coefficients according to radar frequency
        if 'frequency' in radar.instrument_parameters:
            mass_centers = _get_mass_centers(
                radar.instrument_parameters['frequency']['data'][0])
        else:
            mass_centers = _mass_centers_table()['C']
            warn('Radar frequency unknown. ' +
                 'Default coefficients for C band will be applied')

    if hydro_field is None:
        hydro_field = get_field_name('radar_echo_classification')
    if compute_entropy:
        if entropy_field is None:
            entropy_field = get_field_name('hydroclass_entropy')

    # Get the data fields
    fields_dict = {}
    for var_name in var_names:
        if var_name == 'dBZ':
            if refl_field is None:
                refl_field = get_field_name('reflectivity')
            radar.check_field_exists(refl_field)
            fields_dict.update({
                var_name: deepcopy(radar.fields[refl_field]['data'])})
        elif var_name == 'ZDR':
            if zdr_field is None:
                zdr_field = get_field_name('differential_reflectivity')
            radar.check_field_exists(zdr_field)
            fields_dict.update({
                var_name: deepcopy(radar.fields[zdr_field]['data'])})
        elif var_name == 'KDP':
            if kdp_field is None:
                kdp_field = get_field_name('specific_differential_phase')
            radar.check_field_exists(kdp_field)
            fields_dict.update({
                var_name: deepcopy(radar.fields[kdp_field]['data'])})
        elif var_name == 'RhoHV':
            if rhv_field is None:
                rhv_field = get_field_name('cross_correlation_ratio')
            radar.check_field_exists(rhv_field)
            fields_dict.update({
                var_name: deepcopy(radar.fields[rhv_field]['data'])})
        elif var_name == 'H_ISO0':
            if temp_ref == 'temperature':
                if temp_field is None:
                    temp_field = get_field_name('temperature')
                radar.check_field_exists(temp_field)
                # convert temp in relative height respect to iso0
                temp = deepcopy(radar.fields[temp_field]['data'])
                fields_dict.update({var_name: temp*(1000./lapse_rate)})
            else:
                if iso0_field is None:
                    iso0_field = get_field_name('height_over_iso0')
                radar.check_field_exists(iso0_field)
                fields_dict.update({
                    var_name: deepcopy(radar.fields[iso0_field]['data'])})
        else:
            raise ValueError(
                'Variable ' + var_name + ' unknown. '
                + 'Valid variable names for hydrometeor classification are: '
                + 'H_ISO0, dBZ, ZDR, KDP and RhoHV')

    # standardize data and centroids
    mc_std = np.empty(
        np.shape(mass_centers), dtype=fields_dict[var_names[0]].dtype)
    for i, var_name in enumerate(var_names):
        mc_std[:, i] = _standardize(mass_centers[:, i], var_name)
        fields_dict[var_name] = _standardize(fields_dict[var_name], var_name)

    # if entropy has to be computed get transformation parameters
    t_vals = None
    if compute_entropy:
        t_vals = _compute_coeff_transform(
            mc_std, weights=weights, value=value)

    # assign to class
    if vectorize:
        hydroclass_data, entropy_data, prop_data = _assign_to_class_scan(
            fields_dict, mc_std, var_names=var_names, weights=weights,
            t_vals=t_vals)
    else:
        hydroclass_data, entropy_data, prop_data = _assign_to_class(
            fields_dict, mc_std, weights=weights, t_vals=t_vals)

    # prepare output fields
    fields_dict = dict()
    hydro = get_metadata(hydro_field)
    hydro['data'] = hydroclass_data
    hydro.update({'_FillValue': 0})
    labels = ['NC']
    ticks = [1]
    boundaries = [0.5, 1.5]
    for i, hydro_name in enumerate(hydro_names):
        labels.append(hydro_name)
        ticks.append(i+2)
        boundaries.append(i+2.5)
    hydro.update({
        'labels': labels,
        'ticks': ticks,
        'boundaries': boundaries})
    fields_dict.update({'hydro': hydro})

    if compute_entropy:
        entropy = get_metadata(entropy_field)
        entropy['data'] = entropy_data
        fields_dict.update({'entropy': entropy})

        if output_distances:
            for hydro_name in hydro_names:
                field_name = 'proportion_'+field_name
                prop = get_metadata(field_name)
                prop['data'] = prop_data[:, :, i]
                fields_dict.update({field_name: prop})

    return fields_dict


def data_for_centroids(radar, lapse_rate=-6.5, refl_field=None,
                       zdr_field=None, rhv_field=None, kdp_field=None,
                       temp_field=None, iso0_field=None,
                       temp_ref='temperature', nsamples_max=20000):
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
        refl_field = get_field_name('reflectivity')
    if zdr_field is None:
        zdr_field = get_field_name('differential_reflectivity')
    if rhv_field is None:
        rhv_field = get_field_name('cross_correlation_ratio')
    if kdp_field is None:
        kdp_field = get_field_name('specific_differential_phase')

    if temp_ref == 'temperature':
        if temp_field is None:
            temp_field = get_field_name('temperature')
    else:
        if iso0_field is None:
            iso0_field = get_field_name('height_over_iso0')

    # extract fields and parameters from radar
    radar.check_field_exists(refl_field)
    radar.check_field_exists(zdr_field)
    radar.check_field_exists(rhv_field)
    radar.check_field_exists(kdp_field)
    if temp_ref == 'temperature':
        radar.check_field_exists(temp_field)
    else:
        radar.check_field_exists(iso0_field)

    refl = deepcopy(radar.fields[refl_field]['data'])
    zdr = deepcopy(radar.fields[zdr_field]['data'])
    rhohv = deepcopy(radar.fields[rhv_field]['data'])
    kdp = deepcopy(radar.fields[kdp_field]['data'])

    if temp_ref == 'temperature':
        # convert temp in relative height respect to iso0
        temp = deepcopy(radar.fields[temp_field]['data'])
        relh = temp*(1000./lapse_rate)
    else:
        relh = deepcopy(radar.fields[iso0_field]['data'])

    # filter data out of limits
    dlimits_dict = _data_limits_centroids_table()

    mx, mn = dlimits_dict['dBZ']
    refl[(refl < mn) | (refl > mx)] = np.ma.masked

    mx, mn = dlimits_dict['ZDR']
    zdr[(zdr < mn) | (zdr > mx)] = np.ma.masked

    mx, mn = dlimits_dict['RhoHV']
    rhohv[(rhohv < mn) | (rhohv > mx)] = np.ma.masked

    mx, mn = dlimits_dict['KDP']
    kdp[(kdp < mn) | (kdp > mx)] = np.ma.masked

    mx, mn = dlimits_dict['H_ISO0']
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


def compute_centroids(features_matrix, weight=(1., 1., 1., 1., 0.75),
                      var_names=('dBZ', 'ZDR', 'KDP', 'RhoHV', 'H_ISO0'),
                      hydro_names = ('CR', 'AG', 'LR', 'RN', 'RP', 'VI', 'WS',
                                     'MH', 'IH/HDG'),
                      nsamples_iter=20000, external_iterations=30,
                      internal_iterations=10, alpha=0.01,
                      num_samples_arr=(30, 35, 40), n_samples_syn=50,
                      nmedoids_min=1, acceptance_threshold=0.5, band='C',
                      relh_slope=0.001):
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
        warn(
            'Unable to compute centroids. scikit-learn package not available')
        return None, None, None, None

    rg = np.random.default_rng(seed=0)

    labels = None
    labeled_data = None
    medoids_dict = dict()
    for i in range(external_iterations):
        print('\n\n\nExternal loop. Iteration '+str(i+1)+'/' +
              str(external_iterations))
        # external loop to identify clusters
        ks_threshold, n_samples = compute_ks_threshold(
            rg, alpha=alpha, n_samples_syn=n_samples_syn,
            num_samples_arr=num_samples_arr)
        synthetic_obs = synthetic_obs_distribution(
            rg, var_names, hydro_names, band=band, relh_slope=relh_slope)

        if nsamples_iter > features_matrix.shape[0]:
            warn('Number of samples lower than number of samples per iteration')
            fm_sample = deepcopy(features_matrix)
        else:
            fm_sample, _ = train_test_split(
                features_matrix, train_size=nsamples_iter)

        # Uses sklearn.metrics.pairwise_distances for the metric
        # metric can be also those of scipy.spatial.distance
        kmedoids = KMedoids(
            n_clusters=len(hydro_names), metric='seuclidean', method='alternate',
            init='k-medoids++', max_iter=300, random_state=None).fit(
                fm_sample)

        new_labels, new_labeled_data, _ = search_medoids(
            fm_sample, kmedoids.labels_, synthetic_obs, var_names,
            hydro_names, weight, ks_threshold, alpha, n_samples_syn, n_samples,
            1, iteration_max=internal_iterations, relh_slope=relh_slope)

        if new_labels is None:
            print('No data labeled in internal loops')
            continue

        print('labeled data in internal loop: '+str(new_labels.size))

        # Compute medoids as the median of the clustered data
        inter_medoids_dict = compute_intermediate_medoids(
            new_labeled_data, new_labels, hydro_names)

        # store the correctly identified data and its labels
        if labels is None:
            labels = new_labels
            labeled_data = new_labeled_data
        else:
            labels = np.append(labels, new_labels, axis=-1)
            labeled_data = np.append(
                labeled_data, new_labeled_data, axis=0)

        # store the new medoids
        for hydro_name in hydro_names:
            if hydro_name in inter_medoids_dict:
                if hydro_name in medoids_dict:
                    medoids_dict[hydro_name] = np.append(
                        medoids_dict[hydro_name],
                        inter_medoids_dict[hydro_name], axis=0)
                else:
                    medoids_dict.update({
                        hydro_name: inter_medoids_dict[hydro_name]})
        if labels is not None:
            print('total labeled data: '+str(labels.size))

    if labels is None:
        warn('Data could not be labeled')
        return None, None, None, None

    final_medoids_dict = determine_medoids(
        medoids_dict, var_names, nmedoids_min=nmedoids_min,
        acceptance_threshold=acceptance_threshold)

    return labeled_data, labels, medoids_dict, final_medoids_dict


def select_samples(fm, rg, nbins=110, pdf_zh_max=20000, pdf_relh_max=10000,
                   sigma_zh=0.75, sigma_relh=1.5, randomize=True):
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

    Returns
    -------
    fm_sample : 2D array (nsamples, nfeatures)
        The selected data

    """
    nsamples = fm.shape[0]
    if randomize:
        nfeatures = fm.shape[1]
        for i in range(nfeatures-1):
            vals = np.unique(fm[:, i])
            step = np.median(vals[1:]-vals[:-1])
            fm[:, i] += rg.random()*step-step/2.

    # random shuffle of the data
    rg.shuffle(fm)

    refl, zdr, kdp, rhohv, relh = make_platykurtic(
        fm[:, 0], fm[:, 1], fm[:, 2], fm[:, 3], fm[:, 4],
        nbins=nbins, pdf_zh_max=pdf_zh_max, pdf_relh_max=pdf_relh_max,
        sigma_zh=sigma_zh, sigma_relh=sigma_relh)

    fm_sample = np.transpose(np.array([refl, zdr, kdp, rhohv, relh]))

    print('Selected '+str(fm_sample.shape[0])+' samples out of ' +
          str(nsamples))

    return fm_sample


def make_platykurtic(refl, zdr, kdp, rhohv, relh, nbins=110,
                     pdf_zh_max=20000, pdf_relh_max=10000, sigma_zh=0.75,
                     sigma_relh=1.5):
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

    Returns
    -------
    refl, zdr, kdp, rhohv, relh : 1D array
        The selected data

    """
    # select data so that reflectivity is platykurtik
    _, bin_edges = np.histogram(refl, bins=nbins)

    x_vals = np.linspace(-1.1, 1.1, num=nbins)
    pdf = np.array(
        gaussian_function(x_vals, mu=0., sigma=sigma_zh, normal=True) *
        pdf_zh_max, dtype=int)

    refl_aux = []
    zdr_aux = []
    rhohv_aux = []
    kdp_aux = []
    relh_aux = []
    for i in range(nbins):
        ind = np.where((refl >= bin_edges[i]) & (refl < bin_edges[i+1]))[0]
        if ind.size > pdf[i]:
            ind = ind[:pdf[i]]

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

    # select data so that relative height is platykurtik
    _, bin_edges = np.histogram(relh, bins=nbins)

    pdf = np.array(
        gaussian_function(x_vals, mu=0., sigma=sigma_relh, normal=True) *
        pdf_relh_max, dtype=int)

    refl_aux = []
    zdr_aux = []
    rhohv_aux = []
    kdp_aux = []
    relh_aux = []
    for i in range(nbins):
        ind = np.where((relh >= bin_edges[i]) & (relh < bin_edges[i+1]))[0]
        if ind.size > pdf[i]:
            ind = ind[:pdf[i]]

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


def search_medoids(fm, clust_labels, synthetic_obs, var_names, hydro_names, weight,
                   ks_threshold, alpha, n_samples_syn, n_samples, iteration,
                   iteration_max=10, relh_slope=0.001):
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
    print('\n\nInternal loop. Iteration: '+str(iteration)+'/' +
          str(iteration_max))

    if iteration >= iteration_max:
        warn('Maximum number of iterations reached or not enough samples')
        return None, None, iteration

    (hydro_labels, labeled_data, cluster_labels,
     nonlabeled_data) = compare_samples(
        var_names, hydro_names, weight, synthetic_obs, fm, clust_labels,
        ks_threshold, alpha, n_samples, n_samples_syn=n_samples_syn,
        relh_slope=relh_slope)

    n_labeled = 0
    if hydro_labels is not None:
        n_labeled = hydro_labels.size
    print('iteration: '+str(iteration)+' labeled points: '+str(n_labeled) +
          '/'+str(fm.shape[0]))

    n_nonlabeled = 0
    if cluster_labels is not None:
        n_nonlabeled = cluster_labels.size
    if n_nonlabeled <= n_samples:
        return hydro_labels, labeled_data, iteration_max

    # split each non-identified cluster into two and compare with reference
    clusters = np.unique(cluster_labels)
    print(str(clusters.size)+' clusters are not valid. ' +
          'Splitting invalid clusters')

    iteration += 1
    iteration1 = np.empty(clusters.size)
    iteration2 = np.empty(clusters.size)
    for icluster, cluster_id in enumerate(clusters):
        fm1, clust_labels1, fm2, clust_labels2 = split_cluster(
            nonlabeled_data, cluster_labels, cluster_id, n_samples)

        if fm1 is None:
            iteration1[icluster] = iteration_max
            hydro_labels1 = None
        else:
            (hydro_labels1, labeled_data1,
             iteration1[icluster]) = search_medoids(
                fm1, clust_labels1, synthetic_obs, var_names, hydro_names,
                weight, ks_threshold, alpha, n_samples_syn, n_samples,
                iteration, iteration_max=iteration_max, relh_slope=relh_slope)

        if fm2 is None:
            iteration2[icluster] = iteration_max
            hydro_labels2 = None
        else:
            (hydro_labels2, labeled_data2,
             iteration2[icluster]) = search_medoids(
                fm2, clust_labels2, synthetic_obs, var_names, hydro_names,
                weight, ks_threshold, alpha, n_samples_syn, n_samples,
                iteration, iteration_max=iteration_max, relh_slope=relh_slope)

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

    if (np.all(iteration1 >= iteration_max) and
            np.all(iteration2 >= iteration_max)):
        iteration = iteration_max

    return hydro_labels, labeled_data, iteration


def split_cluster(fm, labels, icluster, n_samples):
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

    kmedoids = KMedoids(
            n_clusters=2, metric='seuclidean', method='alternate',
            init='k-medoids++', max_iter=300, random_state=None).fit(
                fm_cluster)
    ind1 = np.where(kmedoids.labels_ == 0)[0]
    ind2 = np.where(kmedoids.labels_ == 1)[0]

    print('\nCluster ID: '+str(icluster))
    print('Number of samples in fm1: '+str(ind1.size))
    print('Number of samples in fm2: '+str(ind2.size))

    # check if the number of samples is too small to proceed
    if ind1.size < n_samples:
        print('Number of non-labeled samples left ('+str(ind1.size) +
              ') smaller than number of samples necessary for clustering (' +
              str(int(n_samples))+')')
        fm1 = None
        clust_labels1 = None
    else:
        fm1 = fm_cluster[ind1, :]
        clust_labels1 = kmedoids.labels_[ind1]

    # check if the number of samples is too small to proceed
    if ind2.size < n_samples:
        print('Number of non-labeled samples left ('+str(ind2.size) +
              ') smaller than number of samples necessary for clustering (' +
              str(int(n_samples))+')')
        fm2 = None
        clust_labels2 = None
    else:
        fm2 = fm_cluster[ind2, :]
        clust_labels2 = kmedoids.labels_[ind2]

    return fm1, clust_labels1, fm2, clust_labels2


def compute_intermediate_medoids(fm, labels, hydro_names):
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
        # medoids = np.median(fm[ind, :], axis=0, keepdims=True)
        kmedoids = KMedoids(
            n_clusters=1, metric='seuclidean', method='alternate',
            init='k-medoids++', max_iter=300, random_state=None).fit(
                fm[ind, :])
        medoids = kmedoids.cluster_centers_

        inter_medoids_dict.update({hydro_name: medoids})

    return inter_medoids_dict


def determine_medoids(medoids_dict, var_names, nmedoids_min=1,
                      acceptance_threshold=0.5):
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
    nmedoids_min : int
        Minimum number of intermediate medoids to compute an intermediate
        medoid
    acceptance_threshold : float
        Threshold on the inter-quantile coefficient of dispersion of the
        medoids above which the medoid of the class is not acceptable.

    Returns
    -------
    final_medoids_dict : dict
        dictionary of type dict[hydro_name]= medoids_arr containing the
        medoids for each hydrometeor type that has been accepted

    """
    final_medoids_dict = dict()
    nvars = len(var_names)
    for hydro_name in medoids_dict.keys():
        coef = 0.
        medoids = medoids_dict[hydro_name]
        if medoids.shape[0] < nmedoids_min:
            warn('Not enough intermediate medoids')
            continue
        for ivar in range(nvars):
            medoid_var = medoids[:, ivar]
            quant75 = np.quantile(medoid_var, 0.75)
            quant25 = np.quantile(medoid_var, 0.25)
            if quant75+quant25 == 0.:
                if quant75 == 0.:
                    pass
                else:
                    warn('Inter-quantile cannot be computed')
                    nvars -= 1
            else:
                coef += (quant75-quant25)/(quant75+quant25)
        coef = coef/nvars
        if coef > acceptance_threshold:
            warn('Inter-quantile coefficient of dispersion (' +
                 str(coef)+') larger than treshold (' +
                 str(acceptance_threshold) +
                 ') no valid centroids for class '+hydro_name)
            continue
        final_medoids_dict.update({hydro_name: np.median(medoids, axis=0)})
    return final_medoids_dict


def synthetic_obs_distribution(rg, var_names, hydro_names, band='C',
                               relh_slope=0.001, par_var=0.05):
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
            min_m, max_m = [m-par_var*m, m+par_var*m]
            min_a, max_a = [a-par_var*a, a+par_var*a]
            min_b, max_b = [b-par_var*b, b+par_var*b]

            m_arr[ihydro] = (max_m-min_m)*rg.random()+min_m
            a_arr[ihydro] = (max_a-min_a)*rg.random()+min_a
            b_arr[ihydro] = (max_b-min_b)*rg.random()+min_b
        mn = np.min(m_arr-1.5*a_arr)
        mx = np.max(m_arr+1.5*a_arr)

        for ihydro, hydro_name in enumerate(hydro_names):
            pdf_samples = sample_bell(
                m=m_arr[ihydro], a=a_arr[ihydro], b=b_arr[ihydro], mn=mn,
                mx=mx)(uniform_samples)
            if var_name == 'RhoHV':
                pdf_samples[pdf_samples > 1.] = 1.
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
        min_v1, max_v1 = [v1-par_var*v1, v1+par_var*v1]
        min_v2, max_v2 = [v2-par_var*v2, v2+par_var*v2]
        min_v3, max_v3 = [v3-par_var*v3, v3+par_var*v3]
        min_v4, max_v4 = [v4-par_var*v4, v3+par_var*v4]

        v1_rand = (max_v1-min_v1)*rg.random()+min_v1
        v2_rand = (max_v2-min_v2)*rg.random()+min_v2
        v3_rand = (max_v3-min_v3)*rg.random()+min_v3
        v4_rand = (max_v4-min_v4)*rg.random()+min_v4

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
            v1=v1_arr[ihydro], v2=v2_arr[ihydro], v3=v3_arr[ihydro],
            v4=v4_arr[ihydro], mn=mn, mx=mx)(uniform_samples)

        # standardize the samples of the distribution
        pdf_samples = _standardize(
            pdf_samples, var_names[-1], relh_slope=relh_slope)
        var_dict.update({hydro_name: pdf_samples})
    synthetic_obs.update({var_names[-1]: var_dict})

    return synthetic_obs


def compute_ks_threshold(rg, alpha=0.01, n_samples_syn=50,
                         num_samples_arr=(30, 35, 40)):
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
        (-np.log(alpha/2.)*(num_samples+n_samples_syn) /
        (2.*num_samples*n_samples_syn)))

    return ks_threshold, num_samples


def compare_samples(var_names, hydro_names, weight, synthetic_obs, fm,
                    clust_labels, ks_threshold, alpha, n_samples_obs,
                    n_samples_syn=50, margin_ratio=0.1, relh_slope=0.001):
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

    Returns
    -------
    labels : 1D-array
        array with the index of the label for the labeled data and -1 for that
        not labeled

    """
    total_weight = np.sum(weight)
    labels = -1+np.zeros(clust_labels.size, dtype=np.uint8)
    hydro_names_aux = list(hydro_names)
    for jhydro in np.unique(clust_labels):
        best_stat = 1e6
        best_p = -1.
        for ihydro, hydro_name in enumerate(hydro_names_aux):
            total_stat = 0.
            total_p = 0.
            for ivar, var_name in enumerate(var_names):
                # select the MF and the real observations
                so_aux = synthetic_obs[var_name][hydro_name]
                real_obs = deepcopy(fm[:, ivar])
                if var_name == 'H_ISO0':
                    real_obs = _standardize(
                        real_obs, 'H_ISO0', relh_slope=relh_slope)
                real_obs = real_obs[clust_labels == jhydro]

                # check if there are enough samples in cluster
                if real_obs.size <= n_samples_obs:
                    total_stat = total_weight
                    total_p = total_weight
                    continue

                # give the MF the same limits as the real obs
                obs_rng_min = real_obs.min()
                obs_rng_max = real_obs.max()
                margin = (obs_rng_max-obs_rng_min)*margin_ratio
                so_rng_min = obs_rng_min-margin
                so_rng_max = obs_rng_max+margin
                so_aux = so_aux[
                    (so_aux >= so_rng_min) & (so_aux <= so_rng_max)]

                # compare the real observations with the synthetic data
                if so_aux.size <= n_samples_syn:
                    statistic = 1.
                    p = 0.
                else:
                    # sampling period of observations
                    isamp_obs = int(real_obs.size/n_samples_obs)
                    real_obs = real_obs[::isamp_obs]
                    isamp_syn = int(so_aux.size/n_samples_syn)
                    so_aux = so_aux[::isamp_syn]
                    statistic, p = ks_2samp(
                        so_aux, real_obs, alternative='two-sided',
                        mode='auto')
                total_stat += statistic*weight[ivar]
                total_p += p*weight[ivar]
            total_stat /= total_weight
            total_p /= total_weight

            # check if data pass the test
            if total_stat < ks_threshold or total_p > alpha:
                # check if test better than previous
                if best_stat > total_stat or total_p > best_p:
                    jhydro_aux = jhydro
                    ihydro_aux = ihydro
                    hydro_name_aux = hydro_name
                    best_stat = total_stat
                    best_p = total_p

        if best_stat < 1e6 or best_p > -1.:
            labels[clust_labels == jhydro_aux] = ihydro_aux
            print('test passed for variable '+hydro_name_aux +
                  ' with total statistic '+str(best_stat) +
                  ' and required statistic '+str(ks_threshold) +
                  ' and/or total p '+str(best_p)+' and required alpha ' +
                  str(alpha))
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


def bell_function(x_vals, m=39., a=19., b=10.):
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
    y = 1./(1.+np.power(np.abs((x_vals-m)/a), 2.*b))
    y[~np.isfinite(y)] = 0.
    return y


def gaussian_function(x_vals, mu=25., sigma=19., normal=True):
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
    y = np.exp(-1*np.power(x_vals-mu, 2.)/(2*np.power(sigma, 2.)))
    if normal:
        y *= 1./(sigma*np.sqrt(2.*np.pi))
    return y


def trapezoidal_function(x_vals, v1=-2500., v2=-2200, v3=-300, v4=0):
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
        (x_vals[(x_vals > v1) & (x_vals <= v2)]-v1)/(v2-v1))
    prob[(x_vals > v2) & (x_vals <= v3)] = 1.
    prob[(x_vals > v3) & (x_vals <= v4)] = (
        (v4-x_vals[(x_vals > v3) & (x_vals <= v4)])/(v4-v3))

    return prob


def sample_bell(m=39., a=19., b=10., mn=-10., mx=60.):
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
    cdf_y = np.abs(np.cumsum(y+1e-10))   # cumulative distribution function, cdf
    cdf_y = cdf_y/cdf_y.max()       # takes care of normalizing cdf to 1.0
    inverse_cdf = interpolate.interp1d(
        cdf_y, x, fill_value='extrapolate')  # this is a function
    return inverse_cdf


def sample_trapezoidal(v1=-2500., v2=-2200, v3=-300, v4=0, mn=-5000., mx=5000.):
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
    y = trapezoidal_function(
        x, v1=v1, v2=v2, v3=v3, v4=v4)  # probability density function, pdf
    cdf_y = np.cumsum(y+1e-10)          # cumulative distribution function, cdf
    cdf_y = cdf_y/cdf_y.max()       # takes care of normalizing cdf to 1.0
    inverse_cdf = interpolate.interp1d(
        cdf_y, x, fill_value='extrapolate')  # this is a function
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
        return 'S'
    if 4e9 <= freq < 8e9:
        return 'C'
    if 8e9 <= freq <= 12e9:
        return 'X'

    warn('Unknown frequency band')

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
    if field_name == 'H_ISO0':
        field_std = 2./(1.+np.ma.exp(-relh_slope*data))-1.
        return field_std

    if (mx is None) or (mn is None):
        dlimits_dict = _data_limits_table()
        if field_name not in dlimits_dict:
            raise ValueError(
                'Field ' + field_name + ' unknown. '
                + 'Valid field names for standardizing are: '
                + 'H_ISO0, dBZ, ZDR, KDP and RhoHV')
        mx, mn = dlimits_dict[field_name]

    if field_name == 'KDP':
        data[data < -0.5] = -0.5
        data = 10.*np.ma.log10(data+0.6)
    elif field_name == 'RhoHV':
        # avoid infinite result
        data[data > 1.] = 1.
        data = 10.*np.ma.log10(1.0000000000001-data)

    mask = np.ma.getmaskarray(data)
    field_std = 2.*(data-mn)/(mx-mn)-1.
    field_std[data < mn] = -1.
    field_std[data > mx] = 1.
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
    if field_name == 'H_ISO0':
        field_std = np.log(2./(1.+data)-1.)/(-relh_slope)
        return field_std

    if (mx is None) or (mn is None):
        dlimits_dict = _data_limits_table()
        if field_name not in dlimits_dict:
            raise ValueError(
                'Field ' + field_name + ' unknown. '
                + 'Valid field names for standardizing are: '
                + 'H_ISO0, dBZ, ZDR, KDP and RhoHV')
        mx, mn = dlimits_dict[field_name]

    if field_name == 'KDP':
        data = np.power(10., 0.1*data)-0.6
    elif field_name == 'RhoHV':
        data = 1.0000000000001-np.power(10., 0.1*data)

    mask = np.ma.getmaskarray(data)
    field_std = 0.5*(data+1.)*(mx-mn)+mn
    field_std[mask] = np.ma.masked

    return field_std


def _assign_to_class(fields_dict, mass_centers,
                     var_names=('dBZ', 'ZDR', 'KDP', 'RhoHV', 'H_ISO0'),
                     weights=np.array([1., 1., 1., 0.75, 0.5]),
                     t_vals=None):
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
            data.append(fields_dict[var_name])
        data = np.ma.array(data, dtype=dtype)
        weights_mat = np.broadcast_to(
            weights.reshape(nvariables, 1), (nvariables, nbins))
        dist = np.ma.zeros((nclasses, nbins), dtype=dtype)

        # compute distance: masked entries will not contribute to the distance
        mask = np.ma.getmaskarray(fields_dict[var_names[0]][ray, :])
        for i in range(nclasses):
            centroids_class = mass_centers[i, :]
            centroids_class = np.broadcast_to(
                centroids_class.reshape(nvariables, 1), (nvariables, nbins))
            dist_ray = np.ma.sqrt(np.ma.sum(
                ((centroids_class-data)**2.)*weights_mat, axis=0))
            dist_ray[mask] = np.ma.masked
            dist[i, :] = dist_ray

        # Get hydrometeor class
        class_vec = dist.argsort(axis=0, fill_value=10e40)
        hydroclass_ray = (class_vec[0, :]+2).astype(np.uint8)
        hydroclass_ray[mask] = 1
        hydroclass[ray, :] = hydroclass_ray

        if t_vals is None:
            continue

        # Transform the distance using the coefficient of the dominant class
        t_vals_ray = np.ma.masked_where(mask, t_vals[class_vec[0, :]])
        t_vals_ray = ma_broadcast_to(
            t_vals_ray.reshape(1, nbins), (nclasses, nbins))
        t_dist_ray = np.ma.exp(-t_vals_ray*dist)

        # set transformed distances to a value between 0 and 1
        dist_total = np.ma.sum(t_dist_ray, axis=0)
        dist_total = ma_broadcast_to(
            dist_total.reshape(1, nbins), (nclasses, nbins))
        t_dist_ray /= dist_total

        # Compute entropy
        entropy_ray = -np.ma.sum(
            t_dist_ray*np.ma.log(t_dist_ray)/np.ma.log(nclasses), axis=0)
        entropy_ray[mask] = np.ma.masked
        entropy[ray, :] = entropy_ray

        t_dist[ray, :, :] = np.ma.transpose(t_dist_ray)

    if t_vals is not None:
        t_dist *= 100.

    return hydroclass, entropy, t_dist


def _assign_to_class_scan(fields_dict, mass_centers,
                          var_names=('dBZ', 'ZDR', 'KDP', 'RhoHV', 'H_ISO0'),
                          weights=np.array([1., 1., 1., 0.75, 0.5]),
                          t_vals=None):
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
        weights.reshape(nvariables, 1, 1), (nvariables, nrays, nbins))

    # compute distance: masked entries will not contribute to the distance
    mask = np.ma.getmaskarray(fields_dict[var_names[0]])
    dist = np.ma.zeros((nrays, nbins, nclasses), dtype=dtype)
    t_dist = None
    entropy = None
    for i in range(nclasses):
        centroids_class = mass_centers[i, :]
        centroids_class = np.broadcast_to(
            centroids_class.reshape(nvariables, 1, 1),
            (nvariables, nrays, nbins))
        dist_aux = np.ma.sqrt(np.ma.sum(
            ((centroids_class-data)**2.)*weights_mat, axis=0))
        dist_aux[mask] = np.ma.masked
        dist[:, :, i] = dist_aux

    del data
    del weights_mat

    # Get hydrometeor class
    class_vec = dist.argsort(axis=-1, fill_value=10e40)
    hydroclass = np.ma.asarray(class_vec[:, :, 0]+2, dtype=np.uint8)
    hydroclass[mask] = 1

    if t_vals is not None:
        # Transform the distance using the coefficient of the dominant class
        t_vals_aux = np.ma.masked_where(mask, t_vals[class_vec[:, :, 0]])
        t_vals_aux = ma_broadcast_to(
            t_vals_aux.reshape(nrays, nbins, 1), (nrays, nbins, nclasses))
        t_dist = np.ma.exp(-t_vals_aux*dist)
        del t_vals_aux

        # set distance to a value between 0 and 1
        dist_total = np.ma.sum(t_dist, axis=-1)
        dist_total = ma_broadcast_to(
            dist_total.reshape(nrays, nbins, 1), (nrays, nbins, nclasses))
        t_dist /= dist_total
        del dist_total

        # compute entroy
        entropy = -np.ma.sum(
            t_dist*np.ma.log(t_dist)/np.ma.log(nclasses), axis=-1)
        entropy[mask] = np.ma.masked

        t_dist *= 100.

    return hydroclass, entropy, t_dist


def _compute_coeff_transform(mass_centers,
                             weights=np.array([1., 1., 1., 0.75, 0.5]),
                             value=50.):
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
            weights.reshape(1, nvariables), (nclasses, nvariables))
        centroids_class = mass_centers[i, :]
        centroids_class = np.broadcast_to(
            centroids_class.reshape(1, nvariables), (nclasses, nvariables))
        t_vals[i, :] = np.sqrt(
            np.sum(weights_mat*np.power(
                np.abs(centroids_class-mass_centers), 2.), axis=1))

    # pick the second lowest value (the first is 0)
    t_vals = np.sort(t_vals, axis=-1)[:, 1]
    t_vals = np.log(value)/t_vals

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
        freq_band_aux = 'C'
    elif freq > 12e9:
        freq_band_aux = 'X'

    mass_centers = mass_centers_dict[freq_band_aux]
    warn('Radar frequency out of range. ' +
         'Centroids only valid for C or X band. ' +
         freq_band_aux + ' band centroids will be applied')

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

    mass_centers_dict.update({'C': mass_centers})

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

    mass_centers_dict.update({'X': mass_centers})

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
    dlimits_dict.update({'dBZ': (60., -10.)})
    dlimits_dict.update({'ZDR': (5., -1.5)})
    dlimits_dict.update({'KDP': (7., -10.)})
    dlimits_dict.update({'RhoHV': (5.23, -50.)})
    dlimits_dict.update({'H_ISO0': (5000., -5000.)})

    return dlimits_dict


def _data_limits_centroids_table():
    """
    Defines the data limits used in the standardization.

    Returns
    -------
    dlimits_dict : dict
        A dictionary with the limits for each variable.

    """
    dlimits_dict = dict()
    dlimits_dict.update({'dBZ': (60., -10.)})
    dlimits_dict.update({'ZDR': (5., -1.5)})
    dlimits_dict.update({'KDP': (5., -0.5)})
    dlimits_dict.update({'RhoHV': (1., 0.7)})
    dlimits_dict.update({'H_ISO0': (2000., -2000.)})

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
        'CR': (-2.8, 12., 5.),
        'AG': (17., 18.1, 10.),
        'LR': (1.75, 29., 10.),
        'RN': (39., 19., 10.),
        'RP': (37., 9.2, 0.8),
        'VI': (-1., 11., 5.),
        'WS': (24., 21.3, 10.),
        'MH': (58.18, 8., 10.),
        'IH/HDG': (48.8, 8., 10.)}

    ZDR_dict = {
        'CR': (2.9, 2.7, 10.),
        'AG': (1., 1.1, 7.),
        'LR': (0.46, 0.46, 5.),
        'RN': (2.3, 2.2, 9.),
        'RP': (0.9, 0.9, 6.),
        'VI': (-0.9, 0.9, 10),
        'WS': (1.3, 0.9, 10.),
        'MH': (2.19, 1.5, 10.),
        'IH/HDG': (0.36, 0.5, 10.)}

    KDP_dict = {
        'CR': (0.08, 0.08, 6.),
        'AG': (-0.008, 0.3, 1.),
        'LR': (0.03, 0.03, 2.),
        'RN': (5.5, 5.5, 10.),
        'RP': (0.1, 0.08, 3.),
        'VI': (-0.75, 0.75, 30.),
        'WS': (0.25, 0.43, 6.),
        'MH': (1.08, 2., 6.),
        'IH/HDG': (0.07, 0.15, 6.)}

    RhoHV_dict = {
        'CR': (0.98, 0.025, 3.),
        'AG': (0.93, 0.07, 3.),
        'LR': (1., 0.018, 3.),
        'RN': (1., 0.025, 3.),
        'RP': (1., 0.025, 1.),
        'VI': (0.975, 0.022, 3.),
        'WS': (0.8, 0.10, 10.),
        'MH': (0.95, 0.05, 3.),
        'IH/HDG': (0.99, 0.05, 3.)}

    vars_dict = {
        'dBZ': dBZ_dict,
        'ZDR': ZDR_dict,
        'KDP': KDP_dict,
        'RhoHV': RhoHV_dict}

    bell_function_dict.update({'C': vars_dict})

    # X-band m, a, b
    dBZ_dict = {
        'CR': (-3, 12., 5.),
        'AG': (16., 17., 10.),
        'LR': (2., 29., 10.),
        'RN': (42., 17., 10.),
        'RP': (34., 10., 0.8),
        'VI': (3.5, 14., 5.),
        'WS': (30., 20., 10.),
        'MH': (53.37, 8., 10.),
        'IH/HDG': (45.5, 8., 10.)}

    ZDR_dict = {
        'CR': (3.2, 2.6, 10.),
        'AG': (0.7, 0.7, 7.),
        'LR': (0.5, 0.5, 5.),
        'RN': (2.7, 2.8, 9.),
        'RP': (0.3, 1., 6.),
        'VI': (-0.8, 1.3, 10.),
        'WS': (1.3, 0.9, 10.),
        'MH': (2.2, 1.4, 10.),
        'IH/HDG': (2.6, 0.5, 10.)}

    KDP_dict = {
        'CR': (0.15, 0.15, 6.),
        'AG': (0.2, 0.2, 1.),
        'LR': (0.18, 0.18, 2.),
        'RN': (12.6, 12.9, 10.),
        'RP': (0.7, 2.1, 3.),
        'VI': (-0.1, 0.08, 30.),
        'WS': (1., 1., 6.),
        'MH': (1.37, 2., 6.),
        'IH/HDG': (0.1, 0.15, 6.)}

    RhoHV_dict = {
        'CR': (0.985, 0.015, 3.),
        'AG': (0.989, 0.011, 3.),
        'LR': (0.992, 0.007, 3.),
        'RN': (0.99, 0.01, 3.),
        'RP': (0.993, 0.007, 1.),
        'VI': (0.965, 0.035, 3.),
        'WS': (0.835, 0.135, 10.),
        'MH': (0.96, 0.05, 3.),
        'IH/HDG': (0.97, 0.05, 3.)}

    vars_dict = {
        'dBZ': dBZ_dict,
        'ZDR': ZDR_dict,
        'KDP': KDP_dict,
        'RhoHV': RhoHV_dict}

    bell_function_dict.update({'X': vars_dict})

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
        'CR': (0., 1000., 2200., 2500.),
        'AG': (0., 500., 2000., 2500.),
        'LR': (-2500., -2200., -300., 0.),
        'RN': (-2500., -2200., -300., 0.),
        'RP': (0., 500., 2000., 2200.),
        'VI': (0., 1000., 2200., 2500.),
        'WS': (-500., -300., 300., 500.),
        'MH': (-2500., -2200., -300., 0.),
        'IH/HDG': (0., 500., 2000., 2500.)}
