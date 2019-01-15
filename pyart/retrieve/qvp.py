"""
pyart.retrieve.quasi_vertical_profile
=====================================

Retrieval of QVPs from a radar object

.. autosummary::
    :toctree: generated/

    quasi_vertical_profile
    compute_qvp
    compute_rqvp
    compute_directional_stats
    project_to_vertical
    get_target_elevations
    _create_qvp_object
    _update_qvp_metadata

"""

from copy import deepcopy
from warnings import warn

import numpy as np
from scipy.interpolate import interp1d
from netCDF4 import num2date

from ..core.transforms import antenna_to_cartesian
from ..io.common import make_time_unit_str
from ..util.xsect import cross_section_rhi


def quasi_vertical_profile(radar, desired_angle=None, fields=None, gatefilter=None):

    """
    Quasi Vertical Profile.

    Creates a QVP object containing fields from a radar object that can
    be used to plot and produce the quasi vertical profile

    Parameters
    ----------
    radar : Radar
        Radar object used.
    field : string
        Radar field to use for QVP calculation.
    desired_angle : float
        Radar tilt angle to use for indexing radar field data.
        None will result in wanted_angle = 20.0

    Other Parameters
    ----------------
    gatefilter : GateFilter
        A GateFilter indicating radar gates that should be excluded
        from the import qvp calculation

    Returns
    -------
    qvp : Dictonary
        A quasi vertical profile object containing fields
        from a radar object

    References
    ----------
    Troemel, S., M. Kumjian, A. Ryzhkov, and C. Simmer, 2013: Backscatter
    differential phase - estimation and variability. J Appl. Meteor. Clim..
    52, 2529 - 2548.

    Troemel, S., A. Ryzhkov, P. Zhang, and C. Simmer, 2014: Investigations
    of backscatter differential phase in the melting layer. J. Appl. Meteorol.
    Clim. 54, 2344 - 2359.

    Ryzhkov, A., P. Zhang, H. Reeves, M. Kumjian, T. Tschallener, S. Tromel,
    C. Simmer, 2015: Quasi-vertical profiles - a new way to look at polarimetric
    radar data. Submitted to J. Atmos. Oceanic Technol.

    """

    # Creating an empty dictonary
    qvp = {}

    # Setting the desired radar angle and getting index value for desired radar angle
    if desired_angle is None:
        desired_angle = 20.0
    index = abs(radar.fixed_angle['data'] - desired_angle).argmin()
    radar_slice = radar.get_slice(index)

    # Printing radar tilt angles and radar elevation
    print(radar.fixed_angle['data'])
    print(radar.elevation['data'][-1])

    # Setting field parameters
    # If fields is None then all radar fields pulled else defined field is used
    if fields is None:
        fields = radar.fields

        for field in fields:

            # Filtering data based on defined gatefilter
            # If none is defined goes to else statement
            if gatefilter is not None:
                get_fields = radar.get_field(index, field)
                mask_fields = np.ma.masked_where(gatefilter.gate_excluded[radar_slice],
                                                 get_fields)
                radar_fields = np.ma.mean(mask_fields, axis=0)
            else:
                radar_fields = radar.get_field(index, field).mean(axis=0)

            qvp.update({field:radar_fields})

    else:
        # Filtering data based on defined gatefilter
        # If none is defined goes to else statement
        if gatefilter is not None:
            get_field = radar.get_field(index, fields)
            mask_field = np.ma.masked_where(gatefilter.gate_excluded[radar_slice],
                                            get_field)
            radar_field = np.ma.mean(mask_field, axis=0)
        else:
            radar_field = radar.get_field(index, fields).mean(axis=0)

        qvp.update({fields:radar_field})

    # Adding range, time, and height fields
    qvp.update({'range': radar.range['data'], 'time': radar.time})
    _, _, z = antenna_to_cartesian(qvp['range']/1000.0, 0.0,
                                   radar.fixed_angle['data'][index])
    qvp.update({'height': z})
    return qvp


def compute_qvp(radar, field_names, ref_time=None, angle=0., ang_tol=1.,
                hmax=10000., hres=50., avg_type='mean', nvalid_min=30,
                interp_kind='none', qvp=None):
    """
    Computes range defined quasi vertical profiles, by averaging over height
    levels PPI data.

    Parameters
    ----------
    radar : Radar
        Radar object used.
    field_names : list of str
        list of field names to add to the QVP
    ref_time : datetime object
        reference time for current radar volume
    angle : int or float
        If the radar object contains a PPI volume, the sweep number to
        use, if it contains an RHI volume the elevation angle.
    ang_tol : float
        If the radar object contains an RHI volume, the tolerance in the
        elevation angle for the conversion into PPI
    hmax : float
        The maximum height to plot [m].
    hres : float
        The height resolution [m].
    avg_type : str
        The type of averaging to perform. Can be either "mean" or "median"
    nvalid_min : int
        Minimum number of valid points to accept average.
    interp_kind : str
        type of interpolation when projecting to vertical grid: 'none',
        or 'nearest', etc. Default 'none'
        'none' will select from all data points within the regular grid
        height bin the closest to the center of the bin.
        'nearest' will select the closest data point to the center of the
        height bin regardless if it is within the height bin or not.
        Data points can be masked values
        If another type of interpolation is selected masked values will be
        eliminated from the data points before the interpolation
    qvp : QVP object or None
        If it is None this is the QVP object where to store the data from the
        current time step. Otherwise a new QVP object will be created


    Returns
    -------
    qvp : qvp object
        The computed QVP object

    Reference
    ---------
    Ryzhkov A., Zhang P., Reeves H., Kumjian M., Tschallener T., Trömel S.,
    Simmer C. 2016: Quasi-Vertical Profiles: A New Way to Look at Polarimetric
    Radar Data. JTECH vol. 33 pp 551-562

    """
    if avg_type not in ('mean', 'median'):
        warn('Unsuported statistics '+avg_type)
        return None

    radar_aux = deepcopy(radar)
    # transform radar into ppi over the required elevation
    if radar_aux.scan_type == 'rhi':
        radar_aux = cross_section_rhi(radar_aux, [angle], el_tol=ang_tol)
    elif radar_aux.scan_type == 'ppi':
        radar_aux = radar_aux.extract_sweeps([int(angle)])
    else:
        warn('Error: unsupported scan type.')
        return None

    if qvp is None:
        qvp = _create_qvp_object(
            radar_aux, field_names, qvp_type='qvp',
            start_time=ref_time, hmax=hmax, hres=hres)

    # modify metadata
    if ref_time is None:
        ref_time = radar_aux.time['units']
    qvp = _update_qvp_metadata(qvp, ref_time=ref_time)

    for field_name in field_names:
        # compute QVP data
        values, _ = compute_directional_stats(
            radar_aux.fields[field_name]['data'], avg_type=avg_type,
            nvalid_min=nvalid_min, axis=0)

        # Project to vertical grid:
        qvp_data = project_to_vertical(
            values, radar_aux.gate_altitude['data'][0, :], qvp.range['data'],
            interp_kind=interp_kind)

        # Put data in radar object
        if np.size(qvp.fields[field_name]['data']) == 0:
            qvp.fields[field_name]['data'] = qvp_data.reshape(1, qvp.ngates)
        else:
            qvp.fields[field_name]['data'] = np.ma.concatenate(
                (qvp.fields[field_name]['data'],
                 qvp_data.reshape(1, qvp.ngates)))

    return qvp


def compute_rqvp(radar, field_names, ref_time=None, hmax=10000., hres=2.,
                 avg_type='mean', nvalid_min=30, interp_kind='none',
                 rmax=50000., weight_power=2., qvp=None):
    """
    Computes quasi vertical profiles, by averaging over height levels
    PPI data.

    Parameters
    ----------
    radar : Radar
        Radar object used.
    field_names : list of str
        list of field names to add to the QVP
    ref_time : datetime object
        reference time for current radar volume
    hmax : float
        The maximum height to plot [m].
    hres : float
        The height resolution [m].
    avg_type : str
        The type of averaging to perform. Can be either "mean" or "median"
    nvalid_min : int
        Minimum number of valid points to accept average.
    interp_kind : str
        type of interpolation when projecting to vertical grid: 'none',
        or 'nearest', etc. Default 'none'
        'none' will select from all data points within the regular grid
        height bin the closest to the center of the bin.
        'nearest' will select the closest data point to the center of the
        height bin regardless if it is within the height bin or not.
        Data points can be masked values
        If another type of interpolation is selected masked values will be
        eliminated from the data points before the interpolation
    rmax : float
        ground range up to which the data is intended for use [m].
    weight_power : float
        Power p of the weighting function 1/abs(grng-(rmax-1))**p given to
        the data outside the desired range. -1 will set the weight to 0.
    qvp : QVP object or None
        If it is None this is the QVP object where to store the data from the
        current time step. Otherwise a new QVP object will be created


    Returns
    -------
    qvp : qvp object
        The computed QVP object

    Reference
    ---------
    Tobin D.M., Kumjian M.R. 2017: Polarimetric Radar and Surface-Based
    Precipitation-Type Observations of ice Pellet to Freezing Rain
    Transitions. Weather and Forecasting vol. 32 pp 2065-2082

    """
    if avg_type not in ('mean', 'median'):
        warn('Unsuported statistics '+avg_type)
        return None

    radar_aux = deepcopy(radar)
    # transform radar into ppi over the required elevation
    if radar_aux.scan_type == 'rhi':
        target_elevations, el_tol = get_target_elevations(radar_aux)
        radar_ppi = cross_section_rhi(
            radar_aux, target_elevations, el_tol=el_tol)
    elif radar_aux.scan_type == 'ppi':
        radar_ppi = radar_aux
    else:
        warn('Error: unsupported scan type.')
        return None

    radar_aux = radar_ppi.extract_sweeps([0])

    if qvp is None:
        qvp = _create_qvp_object(
            radar_aux, field_names, qvp_type='rqvp',
            start_time=ref_time, hmax=hmax, hres=hres)

    # modify metadata
    if ref_time is None:
        ref_time = radar_aux.time['units']
    qvp = _update_qvp_metadata(qvp, ref_time=ref_time)

    rmax_km = rmax/1000.
    for field_name in field_names:
        val_interp = np.ma.masked_all((radar_ppi.nsweeps, qvp.ngates))
        grng_interp = np.ma.masked_all((radar_ppi.nsweeps, qvp.ngates))
        for sweep in range(radar_ppi.nsweeps):
            radar_aux = deepcopy(radar_ppi)
            radar_aux = radar_aux.extract_sweeps([sweep])

            # Compute QVP for this sweep
            values, _ = compute_directional_stats(
                radar_aux.fields[field_name]['data'], avg_type=avg_type,
                nvalid_min=nvalid_min, axis=0)

            height = radar_aux.gate_altitude['data'][0, :]

            # Project to grid
            val_interp[sweep, :] = project_to_vertical(
                values, height, qvp.range['data'], interp_kind=interp_kind)

            # compute ground range [Km]
            grng = np.sqrt(
                np.power(radar_aux.gate_x['data'][0, :], 2.) +
                np.power(radar_aux.gate_y['data'][0, :], 2.))/1000.

            # Project ground range to grid
            f = interp1d(
                height, grng, kind=interp_kind, bounds_error=False,
                fill_value='extrapolate')
            grng_interp[sweep, :] = f(qvp.range['data'])

        # Compute weight
        weight = np.ma.abs(grng_interp-(rmax_km-1.))
        weight[grng_interp <= rmax_km-1.] = 1./np.power(
            weight[grng_interp <= rmax_km-1.], 0.)

        if weight_power == -1:
            weight[grng_interp > rmax_km-1.] = 0.
        else:
            weight[grng_interp > rmax_km-1.] = 1./np.power(
                weight[grng_interp > rmax_km-1.], weight_power)

        # mask weights where there is no data
        mask = np.ma.getmaskarray(val_interp)
        weight = np.ma.masked_where(mask, weight)

        # Weighted average
        qvp_data = (
            np.ma.sum(val_interp*weight, axis=0)/np.ma.sum(weight, axis=0))

        # Put data in radar object
        if np.size(qvp.fields[field_name]['data']) == 0:
            qvp.fields[field_name]['data'] = qvp_data.reshape(1, qvp.ngates)
        else:
            qvp.fields[field_name]['data'] = np.ma.concatenate(
                (qvp.fields[field_name]['data'],
                 qvp_data.reshape(1, qvp.ngates)))

    return qvp


def compute_directional_stats(field, avg_type='mean', nvalid_min=1, axis=0):
    """
    Computes the mean or the median along one of the axis (ray or range)

    Parameters
    ----------
    field : ndarray
        the radar field
    avg_type :str
        the type of average: 'mean' or 'median'
    nvalid_min : int
        the minimum number of points to consider the stats valid. Default 1
    axis : int
        the axis along which to compute (0=ray, 1=range)

    Returns
    -------
    values : ndarray 1D
        The resultant statistics
    nvalid : ndarray 1D
        The number of valid points used in the computation

    """
    if avg_type == 'mean':
        values = np.ma.mean(field, axis=axis)
    else:
        values = np.ma.median(field, axis=axis)

    # Set to non-valid if there is not a minimum number of valid gates
    valid = np.logical_not(np.ma.getmaskarray(field))
    nvalid = np.sum(valid, axis=0, dtype=int)
    values[nvalid < nvalid_min] = np.ma.masked

    return values, nvalid


def project_to_vertical(data_in, data_height, grid_height, interp_kind='none',
                        fill_value=-9999.):
    """
    Projects radar data to a regular vertical grid

    Parameters
    ----------
    data_in : ndarray 1D
        the radar data to project
    data_height : ndarray 1D
        the height of each radar point
    grid_height : ndarray 1D
        the regular vertical grid to project to
    interp_kind : str
        The type of interpolation to use: 'none' or 'nearest'
    fill_value : float
        The fill value used for interpolation

    Returns
    -------
    data_out : ndarray 1D
        The projected data

    """
    if data_in.size == 0:
        data_out = np.ma.masked_all(grid_height.size)
        return data_out

    if interp_kind == 'none':
        hres = grid_height[1]-grid_height[0]
        data_out = np.ma.masked_all(grid_height.size)
        for ind_r, h in enumerate(grid_height):
            ind_h = find_rng_index(data_height, h, rng_tol=hres/2.)
            if ind_h is None:
                continue
            data_out[ind_r] = data_in[ind_h]
    elif interp_kind == 'nearest':
        data_filled = data_in.filled(fill_value=fill_value)
        f = interp1d(
            data_height, data_filled, kind=interp_kind, bounds_error=False,
            fill_value=fill_value)
        data_out = np.ma.masked_values(f(grid_height), fill_value)
    else:
        valid = np.logical_not(np.ma.getmaskarray(data_in))
        height_valid = data_height[valid]
        data_valid = data_in[valid]
        f = interp1d(
            height_valid, data_valid, kind=interp_kind, bounds_error=False,
            fill_value=fill_value)
        data_out = np.ma.masked_values(f(grid_height), fill_value)

    return data_out


def find_rng_index(rng_vec, rng, rng_tol=0.):
    """
    Find the range index corresponding to a particular range

    Parameters
    ----------
    rng_vec : float array
        The range data array where to look for
    rng : float
        The range to search
    rng_tol : float
        Tolerance [m]

    Returns
    -------
    ind_rng : int
        The range index

    """
    dist = np.abs(rng_vec-rng)
    ind_rng = np.argmin(dist)
    if dist[ind_rng] > rng_tol:
        return None

    return ind_rng


def get_target_elevations(radar_in):
    """
    Gets RHI target elevations

    Parameters
    ----------
    radar_in : Radar object
        current radar object

    Returns
    -------
    target_elevations : 1D-array
        Azimuth angles
    el_tol : float
        azimuth tolerance
    """
    sweep_start = radar_in.sweep_start_ray_index['data'][0]
    sweep_end = radar_in.sweep_end_ray_index['data'][0]
    target_elevations = np.sort(
        radar_in.elevation['data'][sweep_start:sweep_end+1])
    el_tol = np.median(target_elevations[1:]-target_elevations[:-1])

    return target_elevations, el_tol


def _create_qvp_object(radar, field_names, qvp_type='qvp', start_time=None,
                       hmax=10000., hres=200.):
    """
    Creates a QVP object containing fields from a radar object that can
    be used to plot and produce the quasi vertical profile

    Parameters
    ----------
    radar : Radar
        Radar object used.
    field_names : list of strings
        Radar fields to use for QVP calculation.
    qvp_type : str
        Type of QVP. Can be qvp, rqvp, evp
    start_time : datetime object
        the QVP start time
    hmax : float
        The maximum height of the QVP [m]. Default 10000.
    hres : float
        The QVP range resolution [m]. Default 50

    Returns
    -------
    qvp : Radar-like object
        A quasi vertical profile object containing fields
        from a radar object

    """
    qvp = deepcopy(radar)

    # prepare space for field
    qvp.fields = dict()
    for field_name in field_names:
        qvp.add_field(field_name, deepcopy(radar.fields[field_name]))
        qvp.fields[field_name]['data'] = np.array([], dtype='float64')

    # fixed radar objects parameters
    qvp.range['data'] = np.arange(hmax/hres)*hres+hres/2.
    qvp.ngates = len(qvp.range['data'])

    if start_time is None:
        qvp.time['units'] = radar.time['units']
    else:
        qvp.time['units'] = make_time_unit_str(start_time)

    qvp.time['data'] = np.array([], dtype='float64')
    qvp.scan_type = qvp_type
    qvp.sweep_mode['data'] = np.array([qvp_type])
    qvp.sweep_start_ray_index['data'] = np.array([0], dtype='int32')

    if qvp_type in ('rqvp', 'time_height'):
        qvp.fixed_angle['data'] = np.array([90.], dtype='float64')

    # ray dependent radar objects parameters
    qvp.sweep_end_ray_index['data'] = np.array([-1], dtype='int32')
    qvp.rays_per_sweep['data'] = np.array([0], dtype='int32')
    qvp.azimuth['data'] = np.array([], dtype='float64')
    qvp.elevation['data'] = np.array([], dtype='float64')
    qvp.nrays = 0

    return qvp


def _update_qvp_metadata(qvp, ref_time):
    """
    updates a QVP object metadata with data from the current radar volume

    Parameters
    ----------
    qvp : QVP object
        QVP object
    ref_time : datetime object
        the current radar volume reference time

    Returns
    -------
    qvp : QVP object
        The updated QVP object

    """
    start_time = num2date(0, qvp.time['units'], qvp.time['calendar'])
    qvp.time['data'] = np.append(
        qvp.time['data'], (ref_time - start_time).total_seconds())
    qvp.sweep_end_ray_index['data'][0] += 1
    qvp.rays_per_sweep['data'][0] += 1
    qvp.nrays += 1

    qvp.azimuth['data'] = np.ones((qvp.nrays, ), dtype='float64')*0.
    qvp.elevation['data'] = (
        np.ones((qvp.nrays, ), dtype='float64')*qvp.fixed_angle['data'][0])

    qvp.gate_longitude['data'] = (
        np.ones((qvp.nrays, qvp.ngates), dtype='float64') *
        qvp.longitude['data'][0])
    qvp.gate_latitude['data'] = (
        np.ones((qvp.nrays, qvp.ngates), dtype='float64') *
        qvp.latitude['data'][0])
    qvp.gate_altitude['data'] = np.broadcast_to(
        qvp.range['data'], (qvp.nrays, qvp.ngates))

    return qvp
