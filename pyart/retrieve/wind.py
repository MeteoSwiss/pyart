"""
pyart.retrieve.wind
=========================================

Functions for wind estimation

.. autosummary::
    :toctree: generated/

    est_wind_vel
    est_vertical_windshear

"""

from warnings import warn

import numpy as np

from ..config import get_metadata, get_field_name, get_fillvalue


def est_wind_vel(radar, vert_proj=False, vel_field=None, wind_field=None):
    """
    Estimates wind velocity. Projects the radial wind component to the
    horizontal or vertical of the azimuth plane. It assumes that the
    orthogonal component is negligible.

    The horizontal wind component is given by:
        v = v_r*cos(el)-v_el*sin(el)+v_az
    where:
        v_r is the radial wind component (measured by the radar)
        v_el is the perpendicular wind component in the azimuth plane.
        v_az is the horizontal component perpendicular to the radial
        direction and the azimuth plane
        el is the elevation

    The horizontal wind component in the azimuth plane is given by:
        v_h = v_r*cos(el) - v_el*sin(el)
    which since we do not know v_el we assume:
        v_h ~ v_r*cos(el)
    This assumption holds for small elevation angles

    The vertical wind component in the azimuth plane is given by:
        v_h = v_r*sin(el) - v_el*cos(el)
    which since we do not know v_el we assume:
        v_h ~ v_r*sin(el)
    This assumption holds for angles close to 90 deg

    Parameters
    ----------
    radar : Radar
        Radar object
    vert_proj : Boolean
        If true estimates the vertical projection, otherwise the horizontal
    vel_field : str
        name of the velocity field
    wind_field : str
        name of the velocity field

    Returns
    -------
    wind : dict
        Field dictionary containing the estimated wind velocity

    """

    # parse the field parameters
    if vel_field is None:
        vel_field = get_field_name('velocity')
    if wind_field is None:
        wind_field = get_field_name('azimuthal_horizontal_wind_component')
        if vert_proj:
            wind_field = get_field_name('vertical_wind_component')

    radar.check_field_exists(vel_field)

    vel = radar.fields[vel_field]['data']
    ele = radar.elevation['data']
    ele[ele > 90.] = 180.-ele[ele > 90.]
    ele = np.broadcast_to(ele.reshape(radar.nrays, 1),
                          (radar.nrays, radar.ngates))
    if vert_proj:
        wind_data = vel * np.sin(ele*np.pi/180.)
    else:
        wind_data = vel * np.cos(ele*np.pi/180.)

    wind = get_metadata(wind_field)
    wind['data'] = wind_data

    return wind


def est_vertical_windshear(radar, az_tol=0.5, wind_field=None,
                           windshear_field=None):
    """
    Estimates wind shear.

    Parameters
    ----------
    radar : Radar
        Radar object
    az_tol : float
        azimuth tolerance to consider gate on top of selected one
    wind_field : str
        name of the horizontal wind velocity field
    windshear_field : str
        name of the vertical wind shear field

    Returns
    -------
    windshear : dict
        Field dictionary containing the wind shear field

    """

    # parse the field parameters
    if wind_field is None:
        wind_field = get_field_name('azimuthal_horizontal_wind_component')
    if windshear_field is None:
        windshear_field = get_field_name('vertical_wind_shear')

    radar.check_field_exists(wind_field)
    wind = radar.fields[wind_field]['data']

    # compute ground range
    ele = radar.elevation['data']
    ele[ele > 90.] = 180.-ele[ele > 90.]
    ele = np.broadcast_to(ele.reshape(radar.nrays, 1),
                          (radar.nrays, radar.ngates))

    rng_mat = np.broadcast_to(radar.range['data'].reshape(1, radar.ngates),
                              (radar.nrays, radar.ngates))
    rng_ground = rng_mat*np.cos(ele*np.pi/180.)

    # initialize output
    windshear_data = np.ma.empty((radar.nrays, radar.ngates), dtype=float)
    windshear_data[:] = np.ma.masked

    azi_vec = radar.azimuth['data']
    ele_vec = radar.elevation['data']
    alt_mat = radar.gate_altitude['data']
    mask = np.ma.getmaskarray(wind)
    for ray in range(radar.nrays):
        # look for the elevation on top of the current ray
        ind_rays_top = np.where(
            np.logical_and(
                ele_vec > ele_vec[ray],
                np.abs(azi_vec-azi_vec[ray]) < az_tol))[0]
        if len(ind_rays_top) == 0:
            continue
        ind_rays_top = np.where(ele_vec == np.min(ele_vec[ind_rays_top]))[0]
        ele_nearest = ele_vec[ind_rays_top[0]]
        azi_top = azi_vec[ind_rays_top]
        azi_nearest = azi_top[np.argmin(np.abs(azi_top-azi_vec[ray]))]
        ind_ray = np.where(
            np.logical_and(
                ele_vec == ele_nearest, azi_vec == azi_nearest))[0][0]

        for rng in range(radar.ngates):
            if mask[ray, rng]:
                continue
            # look for the nearest gate on top of selected gate
            ind_rng = np.argmin(
                np.abs(rng_ground[ind_ray, :]-rng_ground[ray, rng]))
            if mask[ind_ray, ind_rng]:
                continue
            # interpolate the two closest gates on top of the one
            # examined
            if rng_ground[ind_ray, ind_rng] < rng_ground[ray, rng]:
                if ind_rng+1 >= radar.ngates:
                    continue
                if mask[ind_ray, ind_rng+1]:
                    continue
                rng_ground_near = rng_ground[ind_ray, ind_rng:ind_rng+2]
                wind_near = wind[ind_ray, ind_rng:ind_rng+2]
                alt_near = alt_mat[ind_ray, ind_rng:ind_rng+2]
            else:
                if ind_rng-1 < 0:
                    continue
                if mask[ind_ray, ind_rng-1]:
                    continue
                rng_ground_near = rng_ground[ind_ray, ind_rng-1:ind_rng+1]
                wind_near = wind[ind_ray, ind_rng-1:ind_rng+1]
                alt_near = alt_mat[ind_ray, ind_rng-1:ind_rng+1]

            wind_top = np.interp(
                rng_ground[ray, rng], rng_ground_near, wind_near)
            gate_altitude_top = np.interp(
                rng_ground[ray, rng], rng_ground_near, alt_near)
            # compute wind shear
            windshear_data[ray, rng] = (
                1000.*(wind_top-wind[ray, rng]) /
                (gate_altitude_top-alt_mat[ray, rng]))

    windshear = get_metadata(windshear_field)
    windshear['data'] = windshear_data

    return windshear
