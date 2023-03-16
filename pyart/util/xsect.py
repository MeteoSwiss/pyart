"""
pyart.util.xsect
================
Function for extracting cross sections from radar volumes.
.. autosummary::
    :toctree: generated/
    cross_section_ppi
    cross_section_rhi
    colocated_gates
    colocated_gates2
    intersection
    find_intersection_volume
    find_intersection_limits
    find_equal_vol_region
    get_ground_distance
    get_range
    get_vol_diameter
    get_target_elevations
    interpolate_trajectory
    _construct_xsect_radar
    _copy_dic
"""

from copy import copy
from warnings import warn

import numpy as np
from scipy.spatial import cKDTree


from ..core import Radar, geographic_to_cartesian_aeqd
from ..config import get_metadata, get_field_name


try:
    import pyproj
    _PYPROJ_AVAILABLE = True
except ImportError:
    _PYPROJ_AVAILABLE = False


def cross_section_ppi(radar, target_azimuths, az_tol=None):
    """
    Extract cross sections from a PPI volume along one or more azimuth angles.
    Parameters
    ----------
    radar : Radar
        Radar volume containing PPI sweeps from which azimuthal
        cross sections will be extracted.
    target_azimuth : list
        Azimuthal angles in degrees where cross sections will be taken.
    az_tol : float, optional
        Azimuth angle tolerance in degrees. If none the nearest angle is used.
        If valid only angles within the tolerance distance are considered.
    Returns
    -------
    radar_rhi : Radar
        Radar volume containing RHI sweeps which contain azimuthal
        cross sections from the original PPI volume.
    """
    # determine which rays from the ppi radar make up the pseudo RHI
    prhi_rays = []
    valid_azimuths = []
    rhi_sweep_start_ray_index = []
    rhi_sweep_end_ray_index = []
    ray_index = -1
    for target_azimuth in target_azimuths:
        found_first = False
        for sweep_slice in radar.iter_slice():
            sweep_azimuths = radar.azimuth['data'][sweep_slice]
            d_az = np.abs(sweep_azimuths - target_azimuth)
            if az_tol is None:
                ray_number = np.argmin(d_az)
                prhi_rays.append(ray_number + sweep_slice.start)
            else:
                d_az_min = np.min(d_az)
                if d_az_min > az_tol:
                    warn('WARNING: No azimuth found whithin tolerance ' +
                         'for angle '+str(target_azimuth) +
                         '. Minimum distance to radar azimuth ' +
                         str(d_az_min)+' larger than tolerance ' +
                         str(az_tol))
                    continue
                ray_number = np.argmin(d_az)
                prhi_rays.append(ray_number + sweep_slice.start)

            ray_index += 1
            if not found_first:
                valid_azimuths.append(target_azimuth)
                rhi_sweep_start_ray_index.append(ray_index)
                found_first = True
        if found_first:
            rhi_sweep_end_ray_index.append(ray_index)

    rhi_nsweeps = len(valid_azimuths)
    if rhi_nsweeps == 0:
        raise ValueError('No azimuth found within tolerance')

    radar_rhi = _construct_xsect_radar(
        radar, 'rhi', prhi_rays, rhi_sweep_start_ray_index,
        rhi_sweep_end_ray_index, valid_azimuths)

    return radar_rhi


def cross_section_rhi(radar, target_elevations, el_tol=None):
    """
    Extract cross sections from an RHI volume along one or more elevation
    angles.
    Parameters
    ----------
    radar : Radar
        Radar volume containing RHI sweeps from which azimuthal
        cross sections will be extracted.
    target_elevations : list
        Elevation angles in degrees where cross sections will be taken.
    el_tol : float, optional
        Elevation angle tolerance in degrees. If none the nearest angle is
        used. If valid only angles within the tolerance distance are
        considered.
    Returns
    -------
    radar_ppi : Radar
        Radar volume containing PPI sweeps which contain azimuthal
        cross sections from the original RHI volume.
    """

    # determine which rays from the rhi radar make up the pseudo PPI
    pppi_rays = []
    valid_elevations = []
    ppi_sweep_start_ray_index = []
    ppi_sweep_end_ray_index = []
    ray_index = -1
    for target_elevation in target_elevations:
        found_first = False
        for sweep_slice in radar.iter_slice():
            sweep_elevations = radar.elevation['data'][sweep_slice]
            d_el = np.abs(sweep_elevations - target_elevation)
            if el_tol is None:
                ray_number = np.argmin(d_el)
                pppi_rays.append(ray_number + sweep_slice.start)
            else:
                d_el_min = np.min(d_el)
                if d_el_min > el_tol:
                    warn('WARNING: No elevation found whithin tolerance ' +
                         'for angle '+str(target_elevation) +
                         '. Minimum distance to radar elevation ' +
                         str(d_el_min) + ' larger than tolerance ' +
                         str(el_tol))
                    continue
                ray_number = np.argmin(d_el)
                pppi_rays.append(ray_number + sweep_slice.start)

            ray_index += 1
            if not found_first:
                valid_elevations.append(target_elevation)
                ppi_sweep_start_ray_index.append(ray_index)
                found_first = True
        if found_first:
            ppi_sweep_end_ray_index.append(ray_index)

    ppi_nsweeps = len(valid_elevations)
    if ppi_nsweeps == 0:
        raise ValueError('No elevation found within tolerance')

    radar_ppi = _construct_xsect_radar(
        radar, 'ppi', pppi_rays, ppi_sweep_start_ray_index,
        ppi_sweep_end_ray_index, valid_elevations)

    return radar_ppi


def colocated_gates(radar1, radar2, h_tol=0., latlon_tol=0.,
                    coloc_gates_field=None):
    """
    Flags radar gates of radar1 colocated with radar2
    Parameters
    ----------
    radar1 : Radar
        radar object that is going to be flagged
    radar2 : Radar
        radar object
    h_tol : float
        tolerance in altitude [m]
    latlon_tol : float
        tolerance in latitude/longitude [deg]
    coloc_gates_field : string
        Name of the field to retrieve the data
    Returns
    -------
    coloc_dict : dict
        a dictionary containing the colocated positions of radar 1
        (ele, azi, rng) and radar 2
    coloc_rad1 :
        field with the colocated gates of radar1 flagged, i.e:
        1: not colocated gates 2: colocated (0 is reserved)
    """
    # parse the field parameters
    if coloc_gates_field is None:
        coloc_gates_field = get_field_name('colocated_gates')

    coloc_dict = {
        'rad1_ele': [],
        'rad1_azi': [],
        'rad1_rng': [],
        'rad1_ray_ind': [],
        'rad1_rng_ind': [],
        'rad2_ele': [],
        'rad2_azi': [],
        'rad2_rng': [],
        'rad2_ray_ind': [],
        'rad2_rng_ind': []}

    coloc_rad1 = radar1.fields[coloc_gates_field]
    coloc_rad2 = radar2.fields[coloc_gates_field]

    ind_ray_rad1, ind_rng_rad1 = np.where(coloc_rad1['data'] == 2)
    ngates = len(ind_ray_rad1)
    # debug output:
    # print('looking whether '+str(ngates) +
    #       ' gates of radar1 are colocated with radar2. ' +
    #       'This may take a while...')

    # Make region preselection for radar 2
    i_ray_psel, i_rng_psel = np.where(coloc_rad2['data'] == 2)

    # compute Cartesian position of radar1 respect to radar 2
    x0, y0 = geographic_to_cartesian_aeqd(
        radar1.longitude['data'], radar1.latitude['data'],
        radar2.longitude['data'][0], radar2.latitude['data'][0], R=6370997.)
    z0 = radar1.altitude['data'][0]-radar2.altitude['data'][0]

    for i in range(ngates):
        rad1_alt = radar1.gate_altitude['data'][
            ind_ray_rad1[i], ind_rng_rad1[i]]
        rad1_lat = radar1.gate_latitude['data'][
            ind_ray_rad1[i], ind_rng_rad1[i]]
        rad1_lon = radar1.gate_longitude['data'][
            ind_ray_rad1[i], ind_rng_rad1[i]]

        inds = np.where(
            np.logical_and(
                np.logical_and(
                    radar2.gate_altitude['data'][i_ray_psel, i_rng_psel] <
                    rad1_alt+h_tol,
                    radar2.gate_altitude['data'][i_ray_psel, i_rng_psel] >
                    rad1_alt-h_tol),
                np.logical_and(
                    np.logical_and(
                        radar2.gate_latitude['data'][i_ray_psel, i_rng_psel] <
                        rad1_lat+latlon_tol,
                        radar2.gate_latitude['data'][i_ray_psel, i_rng_psel] >
                        rad1_lat-latlon_tol),
                    np.logical_and(
                        radar2.gate_longitude['data'][i_ray_psel, i_rng_psel] <
                        rad1_lon+latlon_tol,
                        radar2.gate_longitude['data'][i_ray_psel, i_rng_psel] >
                        rad1_lon-latlon_tol)
                    )
                ))

        if inds[0].size == 0:
            # not colocated: set co-located flag to 1
            coloc_rad1['data'][ind_ray_rad1[i], ind_rng_rad1[i]] = 1
            continue

        ind_ray_rad2 = i_ray_psel[inds]
        ind_rng_rad2 = i_rng_psel[inds]

        if len(ind_ray_rad2) == 1:
            ind_ray_rad2 = ind_ray_rad2[0]
            ind_rng_rad2 = ind_rng_rad2[0]
        else:
            # compute minimum distance
            # position of radar 1 gate respect to radar 2
            rad1_x = radar1.gate_x['data'][ind_ray_rad1[i], ind_rng_rad1[i]]+x0
            rad1_y = radar1.gate_y['data'][ind_ray_rad1[i], ind_rng_rad1[i]]+y0
            rad1_z = radar1.gate_z['data'][ind_ray_rad1[i], ind_rng_rad1[i]]+z0

            rad2_x = radar2.gate_x['data'][ind_ray_rad2, ind_rng_rad2]
            rad2_y = radar2.gate_y['data'][ind_ray_rad2, ind_rng_rad2]
            rad2_z = radar2.gate_z['data'][ind_ray_rad2, ind_rng_rad2]

            dist = np.sqrt(
                (rad2_x-rad1_x)**2.+(rad2_y-rad1_y)**2.+(rad2_z-rad1_z)**2.)
            ind_min = np.argmin(dist)

            ind_ray_rad2 = ind_ray_rad2[ind_min]
            ind_rng_rad2 = ind_rng_rad2[ind_min]

        # colocated and valid gate
        coloc_dict['rad1_ele'].append(
            radar1.elevation['data'][ind_ray_rad1[i]])
        coloc_dict['rad1_azi'].append(
            radar1.azimuth['data'][ind_ray_rad1[i]])
        coloc_dict['rad1_rng'].append(
            radar1.range['data'][ind_rng_rad1[i]])
        coloc_dict['rad1_ray_ind'].append(
            ind_ray_rad1[i])
        coloc_dict['rad1_rng_ind'].append(
            ind_rng_rad1[i])
        coloc_dict['rad2_ele'].append(
            radar2.elevation['data'][ind_ray_rad2])
        coloc_dict['rad2_azi'].append(
            radar2.azimuth['data'][ind_ray_rad2])
        coloc_dict['rad2_rng'].append(
            radar2.range['data'][ind_rng_rad2])
        coloc_dict['rad2_ray_ind'].append(
            ind_ray_rad2)
        coloc_dict['rad2_rng_ind'].append(
            ind_rng_rad2)

    #     # debug output:
    #     print(
    #         radar1.elevation['data'][ind_ray_rad1[i]],
    #         radar1.azimuth['data'][ind_ray_rad1[i]],
    #         radar1.range['data'][ind_rng_rad1[i]],
    #         radar2.elevation['data'][ind_ray_rad2],
    #         radar2.azimuth['data'][ind_ray_rad2],
    #         radar2.range['data'][ind_rng_rad2])
    #     print(
    #         radar1.gate_longitude['data'][ind_ray_rad1[i], ind_rng_rad1[i]],
    #         radar1.gate_latitude['data'][ind_ray_rad1[i], ind_rng_rad1[i]],
    #         radar1.gate_altitude['data'][ind_ray_rad1[i], ind_rng_rad1[i]],
    #         radar2.gate_longitude['data'][ind_ray_rad2, ind_rng_rad2],
    #         radar2.gate_latitude['data'][ind_ray_rad2, ind_rng_rad2],
    #         radar2.gate_altitude['data'][ind_ray_rad2, ind_rng_rad2])
    #
    # ind_ray_rad1, ind_rng_rad1 = np.where(coloc_rad1['data'])
    # ngates = len(ind_ray_rad1)
    # print(str(ngates)+' gates of radar1 are colocated with radar2.')

    return coloc_dict, coloc_rad1


def colocated_gates2(radar1, radar2, distance_upper_bound=1000.,
                     coloc_gates_field=None):
    """
    Flags radar gates of radar1 co-located with radar2. Uses nearest neighbour
    calculation with cKDTree
    Parameters
    ----------
    radar1 : Radar
        radar object that is going to be flagged
    radar2 : Radar
        radar object
    distance_upper_bound : float
        upper bound of the distance between neighbours (m)
    coloc_gates_field : string
        Name of the field to retrieve the data
    Returns
    -------
    coloc_dict : dict
        a dictionary containing the colocated positions of radar 1
        (ele, azi, rng) and radar 2
    coloc_rad1 :
        field with the colocated gates of radar1 flagged, i.e:
        1: not colocated gates 2: colocated (0 is reserved)
    """
    # parse the field parameters
    if coloc_gates_field is None:
        coloc_gates_field = get_field_name('colocated_gates')

    coloc_dict = {
        'rad1_ele': [],
        'rad1_azi': [],
        'rad1_rng': [],
        'rad1_ray_ind': [],
        'rad1_rng_ind': [],
        'rad2_ele': [],
        'rad2_azi': [],
        'rad2_rng': [],
        'rad2_ray_ind': [],
        'rad2_rng_ind': []}

    coloc_rad1 = radar1.fields[coloc_gates_field]
    coloc_rad2 = radar2.fields[coloc_gates_field]

    ind_ray_rad1, ind_rng_rad1 = np.where(coloc_rad1['data'] == 2)

    # Make region pre-selection for radar 2
    i_ray_psel, i_rng_psel = np.where(coloc_rad2['data'] == 2)

    # compute Cartesian position of radar1 respect to radar 2
    x0, y0 = geographic_to_cartesian_aeqd(
        radar1.longitude['data'], radar1.latitude['data'],
        radar2.longitude['data'][0], radar2.latitude['data'][0], R=6370997.)
    z0 = radar1.altitude['data'][0]-radar2.altitude['data'][0]

    # Position of radar 1 gates respect to radar 2
    rad1_x = radar1.gate_x['data'][ind_ray_rad1, ind_rng_rad1]+x0
    rad1_y = radar1.gate_y['data'][ind_ray_rad1, ind_rng_rad1]+y0
    rad1_z = radar1.gate_z['data'][ind_ray_rad1, ind_rng_rad1]+z0

    rad2_x = radar2.gate_x['data'][i_ray_psel, i_rng_psel]
    rad2_y = radar2.gate_y['data'][i_ray_psel, i_rng_psel]
    rad2_z = radar2.gate_z['data'][i_ray_psel, i_rng_psel]

    tree = cKDTree(
        np.transpose((rad2_x, rad2_y, rad2_z)),
        compact_nodes=False, balanced_tree=False)
    dist, ind_vec = tree.query(np.transpose((rad1_x, rad1_y, rad1_z)), k=1)

    ind_not_col = np.where(dist > distance_upper_bound)
    ind_col = np.where(dist <= distance_upper_bound)
    ind_vec = ind_vec[ind_col]

    # colocated and valid gate
    coloc_dict['rad1_ele'] = radar1.elevation['data'][ind_ray_rad1[ind_col]]
    coloc_dict['rad1_azi'] = radar1.azimuth['data'][ind_ray_rad1[ind_col]]
    coloc_dict['rad1_rng'] = radar1.range['data'][ind_rng_rad1[ind_col]]
    coloc_dict['rad1_ray_ind'] = ind_ray_rad1[ind_col]
    coloc_dict['rad1_rng_ind'] = ind_rng_rad1[ind_col]
    coloc_dict['rad2_ele'] = radar2.elevation['data'][i_ray_psel[ind_vec]]
    coloc_dict['rad2_azi'] = radar2.azimuth['data'][i_ray_psel[ind_vec]]
    coloc_dict['rad2_rng'] = radar2.range['data'][i_rng_psel[ind_vec]]
    coloc_dict['rad2_ray_ind'] = i_ray_psel[ind_vec]
    coloc_dict['rad2_rng_ind'] = i_rng_psel[ind_vec]

    # not colocated: set co-located flag to 1
    coloc_rad1['data'][
        ind_ray_rad1[ind_not_col], ind_rng_rad1[ind_not_col]] = 1

    return coloc_dict, coloc_rad1


def intersection(radar1, radar2, h_tol=0., latlon_tol=0., vol_d_tol=None,
                 vismin=None, hmin=None, hmax=None, rmin=None, rmax=None,
                 elmin=None, elmax=None, azmin=None, azmax=None,
                 visib_field=None, intersec_field=None):
    """
    Flags region of radar1 that is intersecting with radar2 and complies with
    criteria regarding visibility, altitude, range, elevation angle and
    azimuth angle
    Parameters
    ----------
    radar1 : Radar
        radar object that is going to be flagged
    radar2 : Radar
        radar object checked for intersecting region
    h_tol : float
        tolerance in altitude [m]
    latlon_tol : float
        latitude and longitude tolerance [decimal deg]
    vol_d_tol : float
        pulse volume diameter tolerance [m]
    vismin : float
        minimum visibility [percentage]
    hmin, hmax : floats
        min and max altitude [m MSL]
    rmin, rmax : floats
        min and max range from radar [m]
    elmin, elmax : floats
        min and max elevation angle [deg]
    azmin, azmax : floats
        min and max azimuth angle [deg]
    Returns
    -------
    intersec_rad1_dict : dict
        the field with the gates of radar1 in the same region as radar2
        flagged, i.e.: 1 not intersecting, 2 intersecting, 0 is reserved
    """
    # parse the field parameters
    if intersec_field is None:
        intersec_field = get_field_name('colocated_gates')
    if visib_field is None:
        visib_field = get_field_name('visibility')

    # define common volume
    intersec_rad1 = find_intersection_volume(
        radar1, radar2, h_tol=h_tol, latlon_tol=latlon_tol)

    # check for equal volume of rad1
    if vol_d_tol is not None:
        intersec_rad1[np.logical_not(find_equal_vol_region(
            radar1, radar2, vol_d_tol=vol_d_tol))] = 1

    # check for visibility
    if visib_field in radar1.fields and vismin is not None:
        intersec_rad1[radar1.fields[visib_field]['data'] < vismin] = 1

    # check for altitude limits
    if hmin is not None:
        intersec_rad1[radar1.gate_altitude['data'] < hmin] = 1
    if hmax is not None:
        intersec_rad1[radar1.gate_altitude['data'] > hmax] = 1

    # check for range limits
    if rmin is not None:
        intersec_rad1[:, radar1.range['data'] < rmin] = 1
    if rmax is not None:
        intersec_rad1[:, radar1.range['data'] > rmax] = 1

    # check elevation angle limits
    if elmin is not None:
        intersec_rad1[radar1.elevation['data'] < elmin, :] = 1
    if elmax is not None:
        intersec_rad1[radar1.elevation['data'] > elmax, :] = 1

    # check min and max azimuth angle
    if azmin is not None and azmax is not None:
        if azmin <= azmax:
            intersec_rad1[radar1.azimuth['data'] < azmin, :] = 1
            intersec_rad1[radar1.azimuth['data'] > azmax, :] = 1
        if azmin > azmax:
            intersec_rad1[np.logical_and(
                radar1.azimuth['data'] < azmin,
                radar1.azimuth['data'] > azmax), :] = 1
    elif azmin is not None:
        intersec_rad1[radar1.azimuth['data'] < azmin, :] = 1
    elif azmax is not None:
        intersec_rad1[radar1.azimuth['data'] > azmax, :] = 1

    intersec_rad1_dict = get_metadata(intersec_field)
    intersec_rad1_dict['data'] = intersec_rad1
    intersec_rad1_dict.update({'_FillValue': 0})

    return intersec_rad1_dict


def find_intersection_volume(radar1, radar2, h_tol=0., latlon_tol=0.):
    """
    Flags region of radar1 that is intersecting with radar2
    Parameters
    ----------
    radar1 : Radar
        radar object that is going to be flagged
    radar2 : Radar
        radar object checked for intersecting region
    h_tol : float
        tolerance in altitude [m]
    latlon_tol : float
        latitude and longitude tolerance [decimal deg]
    Returns
    -------
    intersec : 2d array
        the field with gates within the common volume flagged, i.e.
        1: Not intersecting, 2: intersecting (0 is reserved)
    """
    intersec = np.ma.ones((radar1.nrays, radar1.ngates), dtype=np.uint8)

    min_lat, max_lat, min_lon, max_lon, min_alt, max_alt = (
        find_intersection_limits(
            radar1.gate_latitude['data'], radar1.gate_longitude['data'],
            radar1.gate_altitude['data'], radar2.gate_latitude['data'],
            radar2.gate_longitude['data'], radar2.gate_altitude['data'],
            h_tol=h_tol, latlon_tol=latlon_tol))

    intersec[np.logical_and(
        np.logical_and(radar1.gate_altitude['data'] > min_alt,
                       radar1.gate_altitude['data'] < max_alt),
        np.logical_and(
            np.logical_and(radar1.gate_latitude['data'] > min_lat,
                           radar1.gate_latitude['data'] < max_lat),
            np.logical_and(radar1.gate_longitude['data'] > min_lon,
                           radar1.gate_longitude['data'] < max_lon)))] = 2

    return intersec


def find_intersection_limits(lat1, lon1, alt1, lat2, lon2, alt2, h_tol=0.,
                             latlon_tol=0.):
    """
    Find the limits of the intersection between two volumes
    Parameters
    ----------
    lat1, lon1, alt1 : float array
        array with the positions of first volume. lat, lon in decimal
        degrees, alt in m MSL.
    lat2, lon2, alt2 : float array
        array with the positions of second volume. lat, lon in decimal
        degrees, alt in m MSL.
    h_tol: float
        altitude tolerance [m MSL]
    latlon_tol: float
        latitude and longitude tolerance [decimal deg]
    Returns
    -------
    min_lat, max_lat, min_lon, max_lon, min_alt, max_alt : floats
        the limits of the intersecting region
    """

    min_lat = np.max([np.min(lat1), np.min(lat2)])-latlon_tol
    max_lat = np.min([np.max(lat1), np.max(lat2)])+latlon_tol

    min_lon = np.max([np.min(lon1), np.min(lon2)])-latlon_tol
    max_lon = np.min([np.max(lon1), np.max(lon2)])+latlon_tol

    min_alt = np.max([np.min(alt1), np.min(alt2)])-h_tol
    max_alt = np.min([np.max(alt1), np.max(alt2)])+h_tol

    return min_lat, max_lat, min_lon, max_lon, min_alt, max_alt


def find_equal_vol_region(radar1, radar2, vol_d_tol=0):
    """
    Flags regions of radar1 that are equivolumetric
    (similar pulse volume diameter) with radar2
    Parameters
    ----------
    radar1 : Radar
        radar object that is going to be flagged
    radar2 : Radar
        radar object
    vol_d_tol : float
        pulse volume diameter tolerance
    Returns
    -------
    equal_vol : 2D boolean array
        field with true where both radars have a similar pulse volume diameter
    """
    rng_ground = get_ground_distance(
        radar1.gate_latitude['data'], radar1.gate_longitude['data'],
        radar2.latitude['data'], radar2.longitude['data'])
    rng_rad2 = get_range(
        rng_ground, radar1.gate_altitude['data'], radar2.altitude['data'])
    if (radar2.instrument_parameters is not None and
            'radar_beam_width_h' in radar2.instrument_parameters):
        bwidth2 = radar2.instrument_parameters['radar_beam_width_h']['data'][0]
    else:
        warn('Unknown radar 2 beamwidth. Assumed 1 deg')
        bwidth2 = 1.
    if (radar1.instrument_parameters is not None and
            'radar_beam_width_h' in radar1.instrument_parameters):
        bwidth1 = radar1.instrument_parameters['radar_beam_width_h']['data'][0]
    else:
        warn('Unknown radar 1 beamwidth. Assumed 1 deg')
        bwidth1 = 1.
    vol_d_rad2 = get_vol_diameter(bwidth2, rng_rad2)
    vol_d_rad1 = get_vol_diameter(
        bwidth1,
        np.broadcast_to(
            radar1.range['data'].reshape(1, radar1.ngates),
            (radar1.nrays, radar1.ngates)))

    return np.isclose(vol_d_rad1, vol_d_rad2, rtol=0., atol=vol_d_tol)


def get_ground_distance(lat_array, lon_array, lat0, lon0):
    """
    Computes the ground distance to a fixed point
    Parameters
    ----------
    lat_array : float array
        array of latitudes [decimal deg]
    lon_array : float array
        array of longitudes [decimal deg]
    lat0: float
        latitude of fix point
    lon0: float
        longitude of fix point
    Returns
    -------
    rng_ground : float array
        the ground range [m]
    """
    # distance of each gate of rad1 from rad2
    r_earth = 6371e3  # [m]

    dlat_rad = (lat_array-lat0)*np.pi/180.
    dlon_rad = (lon_array-lon0)*np.pi/180.
    a = (np.sin(dlat_rad/2.)*np.sin(dlat_rad/2.) +
         np.cos(lat_array*np.pi/180.)*np.cos(lat0*np.pi/180.) *
         np.sin(dlon_rad/2.)*np.sin(dlon_rad/2.))

    return 2.*np.arctan2(np.sqrt(a), np.sqrt(1.-a))*r_earth


def get_range(rng_ground, alt_array, alt0):
    """
    Computes the range to a fixed point from the ground distance and the
    altitudes
    Parameters
    ----------
    rng_ground : float array
        array of ground distances [m]
    alt_array : float array
        array of altitudes [m MSL]
    alt0: float
        altitude of fixed point [m MSL]
    Returns
    -------
    rng : float array
        the range [m]
    """
    alt_from0 = np.abs(alt_array-alt0)

    return np.sqrt(alt_from0*alt_from0+rng_ground*rng_ground)


def get_vol_diameter(beamwidth, rng):
    """
    Computes the pulse volume diameter from the antenna beamwidth and the
    range from the radar
    Parameters
    ----------
    beamwidth : float
        the radar beamwidth [deg]
    rng : float array
        the range from the radar [m]
    Returns
    -------
    vol_d : float array
        the pulse volume diameter
    """

    return beamwidth*np.pi/180.*rng


def get_target_elevations(radar):
    """
    Gets RHI target elevations
    Parameters
    ----------
    radar : Radar object
        radar object

    Returns
    -------
    target_elevations : 1D-array
        Azimuth angles
    el_tol : float
        azimuth tolerance
    """
    sweep_start = radar.sweep_start_ray_index['data'][0]
    sweep_end = radar.sweep_end_ray_index['data'][0]
    target_elevations = np.sort(
        radar.elevation['data'][sweep_start:sweep_end+1])
    el_tol = np.median(target_elevations[1:]-target_elevations[:-1])

    return target_elevations, el_tol


def _construct_xsect_radar(
        radar, scan_type, pxsect_rays, xsect_sweep_start_ray_index,
        xsect_sweep_end_ray_index, target_angles):
    """
    Constructs a new radar object that contains cross-sections at fixed angles
    of a PPI or RHI volume scan.
    Parameters
    ----------
    radar : Radar
        Radar volume containing RHI/PPI sweeps from which a cross sections will
        be extracted.
    scan_type : str
        Type of cross section scan (ppi or rhi).
    pxsect_rays : list
        list of rays from the radar volume to be copied in the cross-sections
        radar object
    xsect_sweep_start_ray_index, xsect_sweep_end_ray_index : array of ints
        start and end sweep ray index of each cross-section scan
    target_angles : array
        the target fixed angles
    Returns
    -------
    radar_xsect : Radar
        Radar volume containing sweeps which contain cross sections from the
        original volume.
    """
    xsect_nsweeps = len(target_angles)

    _range = _copy_dic(radar.range)
    latitude = _copy_dic(radar.latitude)
    longitude = _copy_dic(radar.longitude)
    altitude = _copy_dic(radar.altitude)
    metadata = _copy_dic(radar.metadata)

    time = _copy_dic(radar.time, excluded_keys=['data'])
    time['data'] = radar.time['data'][pxsect_rays].copy()

    azimuth = _copy_dic(radar.azimuth, excluded_keys=['data'])
    azimuth['data'] = radar.azimuth['data'][pxsect_rays].copy()

    elevation = _copy_dic(radar.elevation, excluded_keys=['data'])
    elevation['data'] = radar.elevation['data'][pxsect_rays].copy()

    fields = {}
    for field_name, orig_field_dic in radar.fields.items():
        field_dic = _copy_dic(orig_field_dic, excluded_keys=['data'])
        field_dic['data'] = orig_field_dic['data'][pxsect_rays].copy()
        fields[field_name] = field_dic

    sweep_number = _copy_dic(radar.sweep_number, excluded_keys=['data'])
    sweep_number['data'] = np.arange(xsect_nsweeps, dtype='int32')

    sweep_mode = _copy_dic(radar.sweep_mode, excluded_keys=['data'])
    sweep_mode['data'] = np.array([scan_type]*xsect_nsweeps)

    fixed_angle = _copy_dic(radar.fixed_angle, excluded_keys=['data'])
    fixed_angle['data'] = np.array(target_angles, dtype='float32')

    sweep_start_ray_index = _copy_dic(
        radar.sweep_start_ray_index, excluded_keys=['data'])
    ssri = np.array(xsect_sweep_start_ray_index, dtype='int32')
    sweep_start_ray_index['data'] = ssri

    sweep_end_ray_index = _copy_dic(
        radar.sweep_end_ray_index, excluded_keys=['data'])
    seri = np.array(xsect_sweep_end_ray_index, dtype='int32')
    sweep_end_ray_index['data'] = seri

    radar_xsect = Radar(
        time, _range, fields, metadata, scan_type,
        latitude, longitude, altitude,
        sweep_number, sweep_mode, fixed_angle,
        sweep_start_ray_index, sweep_end_ray_index,
        azimuth, elevation)

    return radar_xsect

def interpolate_pts_xsect(ref_points, bin, exact = True):
    """ Interpolates points along a cross-section

    Parameters
    ----------
    ref_points : ndarray
        N x 2 array containing the lon, lat coordinates of N reference points along the trajectory
        in WGS84 coordinates, for example [[11, 46], [10, 45], [9, 47]]
    bin : float
        Reference distance between the points to interpolate in meters

    Other Parameters
    ----------------
    exact : bool
        If set to true the last point in xyxy will be appended to the computed points
        this means that the last distance will be smaller than the value of 'bin'

    Returns
    -------
    refdist : ndarray
        Array of length N containing the cumulative distance between the reference points (xyxy),
        note that the reference distance 0, is in the center of the cross-section
    ipts_dist : ndarray
        Array of length N containing the cumulative distance between the interpolated points,
        note that the reference distance 0, is in the center of the cross-section
    ipts : ndarray
        L x 2 array of all computed points lon/lat along the cross-section, where L is equal to
        the total distance of the cross-section divided by bin.
    """

    if not _PYPROJ_AVAILABLE:
        raise MissingOptionalDependency(
            "pyproj is required to use interpolate_pts_xsect but is not installed")

    ref_points = np.array(ref_points)
    geod = pyproj.Geod(ellps='clrk66')

    dist = [0]
    az = []
    for i in range(len(ref_points) - 1):
        a,_,d = geod.inv(ref_points[i, 0], ref_points[i, 1], 
            ref_points[i+1, 0], ref_points[i+1, 1])
        dist.append(d)
        az.append(a)
    
    refdist = np.cumsum(dist)
    totdist = np.sum(dist)

    ipts_dist = np.arange(0, totdist, bin)
    if exact:
        ipts_dist = np.append(ipts_dist, totdist)

    ipts = []
    for d in ipts_dist:
        # Find reference points
        idx = np.where(d >= refdist)[0][-1]
        if idx >= len(ref_points) - 2: # Happens if exact = True for the last pt
            idx = len(ref_points) - 2
        offset = d - refdist[idx]

        # calculate pt from offset
        pt_lon, pt_lat, _ = geod.fwd(ref_points[idx, 0], 
            ref_points[idx, 1], az[idx], offset)

        ipts.append([pt_lon, pt_lat])
    ipts = np.array(ipts)
    return refdist, ipts_dist, ipts

def interpolate_grid_to_xsection(grid, field_name, points, z_level = 0):
    """ Interpolates a grid to a set of points

    Parameters
    ----------
    grid : Grid
        Grid object that contains data to interpolate
    field_name : str
        Name of the field to interpolate
    points : ndarray
        N x 2 array containing the lon, lat coordinates of N points along the trajectory
        in WGS84 coordinates, for example [[11, 46], [10, 45], [9, 47]]
        
    Other Parameters
    ----------------
    z_level : int
        vertical level of the grid to consider

    Returns
    -------
    xsection : ndarray
        Array of length N containing the grid values interpolated at the location of the
        cross-section (at ground level)
    """
    
    if not _PYPROJ_AVAILABLE:
        raise MissingOptionalDependency(
            "pyproj is required to use interpolate_grid_to_xsection but is not installed")

    # Convert points to grid proj
    inProj = pyproj.Proj("+init=EPSG:4326")
    transformer = pyproj.Transformer.from_proj(inProj, grid.projection)

    xr_grid, yr_grid = transformer.transform(points[:,0], points[:,1])

    # Get closest points
    profile = []

    grid_data = grid.fields[field_name]['data']
    for i in range(len(points)):
        idx_x = np.argmin(np.abs(xr_grid[i] - grid.x['data']))
        idx_y = np.argmin(np.abs(yr_grid[i] - grid.y['data']))
        profile.append(grid.fields[field_name]['data'][z_level][idx_y, idx_x])
    profile = np.array(profile).astype(float)

    return profile


def _copy_dic(orig_dic, excluded_keys=None):
    """ Return a copy of the original dictionary copying each element. """
    if excluded_keys is None:
        excluded_keys = []
    dic = {}
    for k, v in orig_dic.items():
        if k not in excluded_keys:
            dic[k] = copy(v)
    return dic
