"""
pyart.retrieve.wind
=========================================

Functions for wind estimation

.. autosummary::
    :toctree: generated/

    est_wind_vel
    est_vertical_windshear
    est_vertical_windshear_lidar
    est_wind_profile
    _wind_coeff
    _vad
    _vel_variance
    _compute_gate_altitudes

"""

from copy import deepcopy

import numpy as np

from ..config import get_field_name, get_metadata


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
        vel_field = get_field_name("velocity")
    if wind_field is None:
        wind_field = get_field_name("azimuthal_horizontal_wind_component")
        if vert_proj:
            wind_field = get_field_name("vertical_wind_component")

    radar.check_field_exists(vel_field)

    vel = radar.fields[vel_field]["data"]
    ele = deepcopy(radar.elevation["data"])
    ele[ele > 90.0] = 180.0 - ele[ele > 90.0]
    ele = np.broadcast_to(ele.reshape(radar.nrays, 1), (radar.nrays, radar.ngates))
    if vert_proj:
        wind_data = vel * np.sin(ele * np.pi / 180.0)
    else:
        wind_data = vel * np.cos(ele * np.pi / 180.0)

    wind = get_metadata(wind_field)
    wind["data"] = wind_data

    return wind


def est_vertical_windshear(radar, az_tol=0.5, wind_field=None, windshear_field=None):
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
        wind_field = get_field_name("azimuthal_horizontal_wind_component")
    if windshear_field is None:
        windshear_field = get_field_name("vertical_wind_shear")

    radar.check_field_exists(wind_field)
    wind = radar.fields[wind_field]["data"]

    # compute ground range
    ele = deepcopy(radar.elevation["data"])
    ele[ele > 90.0] = 180.0 - ele[ele > 90.0]
    ele = np.broadcast_to(ele.reshape(radar.nrays, 1), (radar.nrays, radar.ngates))

    rng_mat = np.broadcast_to(
        radar.range["data"].reshape(1, radar.ngates), (radar.nrays, radar.ngates)
    )
    rng_ground = rng_mat * np.cos(ele * np.pi / 180.0)

    # initialize output
    windshear_data = np.ma.empty((radar.nrays, radar.ngates), dtype=float)
    windshear_data[:] = np.ma.masked

    azi_vec = radar.azimuth["data"]
    ele_vec = radar.elevation["data"]
    alt_mat = radar.gate_altitude["data"]
    mask = np.ma.getmaskarray(wind)
    for ray in range(radar.nrays):
        # look for the elevation on top of the current ray
        ind_rays_top = np.where(
            np.logical_and(
                ele_vec > ele_vec[ray], np.abs(azi_vec - azi_vec[ray]) < az_tol
            )
        )[0]
        if ind_rays_top.size == 0:
            continue
        ind_rays_top = np.where(ele_vec == np.min(ele_vec[ind_rays_top]))[0]
        ele_nearest = ele_vec[ind_rays_top[0]]
        azi_top = azi_vec[ind_rays_top]
        azi_nearest = azi_top[np.argmin(np.abs(azi_top - azi_vec[ray]))]
        ind_ray = np.where(
            np.logical_and(ele_vec == ele_nearest, azi_vec == azi_nearest)
        )[0][0]

        for rng in range(radar.ngates):
            if mask[ray, rng]:
                continue
            # look for the nearest gate on top of selected gate
            ind_rng = np.argmin(np.abs(rng_ground[ind_ray, :] - rng_ground[ray, rng]))
            if mask[ind_ray, ind_rng]:
                continue
            # interpolate the two closest gates on top of the one
            # examined
            if rng_ground[ind_ray, ind_rng] < rng_ground[ray, rng]:
                if ind_rng + 1 >= radar.ngates:
                    continue
                if mask[ind_ray, ind_rng + 1]:
                    continue
                rng_ground_near = rng_ground[ind_ray, ind_rng : ind_rng + 2]
                wind_near = wind[ind_ray, ind_rng : ind_rng + 2]
                alt_near = alt_mat[ind_ray, ind_rng : ind_rng + 2]
            else:
                if ind_rng - 1 < 0:
                    continue
                if mask[ind_ray, ind_rng - 1]:
                    continue
                rng_ground_near = rng_ground[ind_ray, ind_rng - 1 : ind_rng + 1]
                wind_near = wind[ind_ray, ind_rng - 1 : ind_rng + 1]
                alt_near = alt_mat[ind_ray, ind_rng - 1 : ind_rng + 1]

            wind_top = np.interp(rng_ground[ray, rng], rng_ground_near, wind_near)
            gate_altitude_top = np.interp(
                rng_ground[ray, rng], rng_ground_near, alt_near
            )
            # compute wind shear
            windshear_data[ray, rng] = (
                1000.0
                * (wind_top - wind[ray, rng])
                / (gate_altitude_top - alt_mat[ray, rng])
            )

    windshear = get_metadata(windshear_field)
    windshear["data"] = windshear_data

    return windshear


def est_vertical_windshear_lidar(
    radar, az_tol=0.5, wind_field=None, windshear_field=None
):
    """
    Estimates wind shear.

    Parameters
    ----------
    radar : Radar
        Radar object with lidar data
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
        wind_field = get_field_name("azimuthal_horizontal_wind_component")
    if windshear_field is None:
        windshear_field = get_field_name("vertical_wind_shear")

    azi_vec = radar.azimuth["data"]
    ele_vec = radar.elevation["data"]
    ran_vec = radar.range["data"]

    radar.check_field_exists(wind_field)
    wind = radar.fields[wind_field]["data"]

    # compute ground range
    ele = deepcopy(radar.elevation["data"])
    ele[ele > 90.0] = 180.0 - ele[ele > 90.0]
    ele = np.broadcast_to(
        ele.reshape(np.size(ele_vec), 1), (np.size(ele_vec), np.size(ran_vec))
    )

    rng_mat = np.broadcast_to(
        ran_vec.reshape(1, np.size(ran_vec)), (np.size(ele_vec), np.size(ran_vec))
    )
    rng_ground = rng_mat * np.cos(ele * np.pi / 180.0)

    alt_mat = _compute_gate_altitudes(ele, rng_ground)

    # initialize output
    windshear_data = np.ma.empty((np.size(ele_vec), np.size(ran_vec)), dtype=float)
    windshear_data[:] = np.ma.masked

    ngates = np.size(ele_vec) * np.size(ran_vec)

    mask = np.ma.getmaskarray(wind)

    for ray in range(np.size(ele_vec)):
        # look for the elevation on top of the current ray
        ind_rays_top = np.where(
            np.logical_and(
                ele_vec > ele_vec[ray], np.abs(azi_vec - azi_vec[ray]) < az_tol
            )
        )[0]
        if ind_rays_top.size == 0:
            continue
        ind_rays_top = np.where(ele_vec == np.min(ele_vec[ind_rays_top]))[0]
        ele_nearest = ele_vec[ind_rays_top[0]]
        azi_top = azi_vec[ind_rays_top]
        azi_nearest = azi_top[np.argmin(np.abs(azi_top - azi_vec[ray]))]
        ind_ray = np.where(
            np.logical_and(ele_vec == ele_nearest, azi_vec == azi_nearest)
        )[0][0]

        for rng in range(np.size(ran_vec)):
            if mask[ray, rng]:
                continue
            # look for the nearest gate on top of selected gate
            ind_rng = np.argmin(np.abs(rng_ground[ind_ray, :] - rng_ground[ray, rng]))
            if mask[ind_ray, ind_rng]:
                continue
            # interpolate the two closest gates on top of the one
            # examined
            if rng_ground[ind_ray, ind_rng] < rng_ground[ray, rng]:
                if ind_rng + 1 >= ngates:
                    continue
                if mask[ind_ray, ind_rng]:
                    continue
                rng_ground_near = rng_ground[ind_ray, ind_rng : ind_rng + 2]
                wind_near = wind[ind_ray, ind_rng : ind_rng + 2]
                alt_near = alt_mat[ind_ray, ind_rng : ind_rng + 2]
            else:
                if ind_rng - 1 < 0:
                    continue
                if mask[ind_ray, ind_rng - 1]:
                    continue
                rng_ground_near = rng_ground[ind_ray, ind_rng - 1 : ind_rng + 1]
                wind_near = wind[ind_ray, ind_rng - 1 : ind_rng + 1]
                alt_near = alt_mat[ind_ray, ind_rng - 1 : ind_rng + 1]

            wind_top = np.interp(rng_ground[ray, rng], rng_ground_near, wind_near)
            gate_altitude_top = np.interp(
                rng_ground[ray, rng], rng_ground_near, alt_near
            )
            # compute wind shear
            windshear_data[ray, rng] = (
                1000.0
                * (wind_top - wind[ray, rng])
                / (gate_altitude_top - alt_mat[ray, rng])
            )

    windshear = get_metadata(windshear_field)
    windshear["data"] = windshear_data

    return windshear


def est_wind_profile(
    radar,
    npoints_min=6,
    azi_spacing_max=45.0,
    vel_diff_max=10.0,
    sign=1,
    rad_vel_field=None,
    u_vel_field=None,
    v_vel_field=None,
    w_vel_field=None,
    vel_est_field=None,
    vel_std_field=None,
    vel_diff_field=None,
):
    """
    Estimates the vertical wind profile using VAD techniques

    Parameters
    ----------
    radar : Radar
        Radar object
    npoints_min : int
        Minimum number of points in the VAD to retrieve wind components.
        0 will retrieve them regardless
    azi_spacing_max : float
        Maximum spacing between valid gates in the VAD to retrieve wind
        components. 0 will retrieve them regardless.
    vel_diff_max : float
        Maximum velocity difference allowed between retrieved and measured
        radial velocity at each range gate. Gates exceeding this threshold
        will be removed and VAD will be recomputed. If -1 there will not be
        a second pass.
    sign : int, optional
        Sign convention which the radial velocities in the volume created
        from the sounding data will will. This should match the convention
        used in the radar data. A value of 1 represents when positive values
        velocities are towards the radar, -1 represents when negative
        velocities are towards the radar.
    rad_vel_field : str
        name of the measured radial velocity field
    u_vel_field, v_vel_field, w_vel_field : str
        names of the 3 wind components fields
    vel_est_field : str
        name of the retrieved radial Doppler velocity field
    vel_std_field : str
        name of the standard deviation of the velocity retrieval field
    vel_diff_field : str
        name of the diference between retrieved and measured radial velocity
        field

    Returns
    -------
    wind : dict
        Field dictionary containing the estimated wind velocity

    """
    # parse the field parameters
    if rad_vel_field is None:
        rad_vel_field = get_field_name("velocity")
    if u_vel_field is None:
        u_vel_field = get_field_name("eastward_wind_component")
    if v_vel_field is None:
        v_vel_field = get_field_name("northward_wind_component")
    if w_vel_field is None:
        w_vel_field = get_field_name("vertical_wind_component")
    if vel_est_field is None:
        vel_est_field = get_field_name("retrieved_velocity")
    if vel_std_field is None:
        vel_std_field = get_field_name("retrieved_velocity_std")
    if vel_diff_field is None:
        vel_diff_field = get_field_name("velocity_difference")

    radar.check_field_exists(rad_vel_field)
    vel = deepcopy(radar.fields[rad_vel_field]["data"])

    # Compute wind coefficients
    u_coeff, v_coeff, w_coeff = _wind_coeff(radar)

    # first guess VAD
    u_vel, v_vel, w_vel, vel_est = _vad(
        radar,
        u_coeff,
        v_coeff,
        w_coeff,
        vel,
        npoints_min=npoints_min,
        azi_spacing_max=azi_spacing_max,
        sign=sign,
    )

    # Remove gates where velocity difference exceeds threshold and recompute
    # VAD if applicable
    if vel_diff_max > -1:
        vel = np.ma.masked_where(np.ma.abs(vel - vel_est) > vel_diff_max, vel)

        # Final VAD
        u_vel, v_vel, w_vel, vel_est = _vad(
            radar,
            u_coeff,
            v_coeff,
            w_coeff,
            vel,
            npoints_min=npoints_min,
            azi_spacing_max=azi_spacing_max,
        )

    # Compute velocity estimation variance
    vel_std, vel_diff = _vel_std(radar, vel, vel_est)

    # prepare output
    u_vel_dict = get_metadata(u_vel_field)
    u_vel_dict["data"] = u_vel

    v_vel_dict = get_metadata(v_vel_field)
    v_vel_dict["data"] = v_vel

    w_vel_dict = get_metadata(w_vel_field)
    w_vel_dict["data"] = w_vel

    vel_est_dict = get_metadata(vel_est_field)
    vel_est_dict["data"] = vel_est

    vel_std_dict = get_metadata(vel_std_field)
    vel_std_dict["data"] = vel_std

    vel_diff_dict = get_metadata(vel_diff_field)
    vel_diff_dict["data"] = vel_diff

    return (
        u_vel_dict,
        v_vel_dict,
        w_vel_dict,
        vel_est_dict,
        vel_std_dict,
        vel_diff_dict,
    )


def _wind_coeff(radar):
    """
    Computes the coefficients to transform 3-D wind vectors into radial
    velocity at each range gate

    Parameters
    ----------
    radar : Radar
        Radar object

    Returns
    -------
    u_coeff, v_coeff, w_coeff : 2D float arrays
        The coefficients for each wind component

    """
    cos_ele = np.cos(radar.elevation["data"] * np.pi / 180.0)
    sin_ele = np.sin(radar.elevation["data"] * np.pi / 180.0)
    cos_azi = np.cos(radar.azimuth["data"] * np.pi / 180.0)
    sin_azi = np.sin(radar.azimuth["data"] * np.pi / 180.0)

    u_coeff = sin_azi * cos_ele
    v_coeff = cos_azi * cos_ele
    w_coeff = sin_ele

    u_coeff = np.reshape(np.tile(u_coeff, radar.ngates), (radar.ngates, radar.nrays)).T
    v_coeff = np.reshape(np.tile(v_coeff, radar.ngates), (radar.ngates, radar.nrays)).T
    w_coeff = np.reshape(np.tile(w_coeff, radar.ngates), (radar.ngates, radar.nrays)).T

    return u_coeff, v_coeff, w_coeff


def _vad(
    radar, u_coeff, v_coeff, w_coeff, vel, npoints_min=6, azi_spacing_max=45.0, sign=1
):
    """
    Estimates wind components using VAD techniques

    Parameters
    ----------
    radar : Radar
        Radar object
    u_coeff, v_coeff, w_coeff : 2D float arrays
        the coefficients to transform 3D winds into radial velocity
    vel : 2D float array
        The measured radial velocity field
    npoints_min : int
        Minimum number of points in the VAD to retrieve wind components.
        0 will retrieve them regardless
    azi_spacing_max : float
        Maximum spacing between valid gates in the VAD to retrieve wind
        components. 0 will retrieve them regardless.
    sign : int, optional
        Sign convention which the radial velocities in the volume created
        from the sounding data will will. This should match the convention
        used in the radar data. A value of 1 represents when positive values
        velocities are towards the radar, -1 represents when negative
        velocities are towards the radar.

    Returns
    -------
    u_vel, v_vel, w_vel : 2D float arrays
        The 3 estimated wind components at each range gate
    vel_est : 2D float array
        The estimated radial velocity at each range gate

    """
    vel_aux = deepcopy(vel)
    if sign == 1:
        vel_aux = -vel_aux

    # prepare wind matrices
    u_vel = np.ma.empty((radar.nrays, radar.ngates), dtype=float)
    u_vel.mask = np.ma.masked
    v_vel = np.ma.empty((radar.nrays, radar.ngates), dtype=float)
    v_vel.mask = np.ma.masked
    w_vel = np.ma.empty((radar.nrays, radar.ngates), dtype=float)
    w_vel.mask = np.ma.masked

    # first guess VAD
    for ind_sweep in range(radar.nsweeps):
        ind_start = radar.sweep_start_ray_index["data"][ind_sweep]
        ind_end = radar.sweep_end_ray_index["data"][ind_sweep]
        for ind_rng in range(radar.ngates):
            vel_azi = vel_aux[ind_start : ind_end + 1, ind_rng]

            # check minimum number of valid points
            if npoints_min > 0:
                if vel_azi.compressed().size < npoints_min:
                    continue

            # get position of valid gates
            is_valid_azi = np.logical_not(np.ma.getmaskarray(vel_azi))

            # check maximum allowed gap between data
            if azi_spacing_max > 0.0:
                valid_azi = np.sort(
                    radar.azimuth["data"][ind_start : ind_end + 1][is_valid_azi]
                )
                delta_azi_max = np.max(
                    np.append(
                        valid_azi[1:] - valid_azi[:-1],
                        valid_azi[0] - (valid_azi[-1] - 360),
                    )
                )
                if delta_azi_max > azi_spacing_max:
                    continue

            # get wind coefficients for this azimuth
            u_coeff_aux = u_coeff[ind_start : ind_end + 1, ind_rng][is_valid_azi]
            v_coeff_aux = v_coeff[ind_start : ind_end + 1, ind_rng][is_valid_azi]
            w_coeff_aux = w_coeff[ind_start : ind_end + 1, ind_rng][is_valid_azi]
            coeff_arr = np.array([u_coeff_aux, v_coeff_aux.T, w_coeff_aux]).T

            # retrieve velocity using least square method
            vel_ret, _, _, _ = np.linalg.lstsq(
                coeff_arr, vel_azi.compressed(), rcond=None
            )
            u_vel[ind_start : ind_end + 1, ind_rng] = vel_ret[0]
            v_vel[ind_start : ind_end + 1, ind_rng] = vel_ret[1]
            w_vel[ind_start : ind_end + 1, ind_rng] = vel_ret[2]

    # Compute estimated radial velocity
    vel_est = u_vel * u_coeff + v_vel * v_coeff + w_vel * w_coeff
    if sign == 1:
        vel_est = -vel_est

    return u_vel, v_vel, w_vel, vel_est


def _vel_std(radar, vel, vel_est):
    """
    Computes the variance of the retrieved wind velocity

    Parameters
    ----------
    radar : Radar
        Radar object
    vel : 2D float array
        The measured radial velocity field
    vel_est : 2D float array
        The estimated radial velocity field

    Returns
    -------
    vel_std : 2D float arrays
        The estimated standard deviation at each range gate (one for VAD)
    vel_diff : 2D float array
        The actual velocity difference between estimated and measured radial
        velocities

    """
    vel_diff = vel - vel_est

    vel_std = np.ma.empty((radar.nrays, radar.ngates), dtype=float)
    for ind_sweep in range(radar.nsweeps):
        ind_start = radar.sweep_start_ray_index["data"][ind_sweep]
        ind_end = radar.sweep_end_ray_index["data"][ind_sweep]
        for ind_rng in range(radar.ngates):
            vel_diff_aux = vel_diff[ind_start : ind_end + 1, ind_rng]
            nvalid = vel_diff_aux.compressed().size
            vel_std[ind_start : ind_end + 1, ind_rng] = np.ma.sqrt(
                np.ma.sum(np.ma.power(vel_diff_aux, 2.0)) / (nvalid - 3)
            )

    return vel_std, vel_diff


def _compute_gate_altitudes(elevations, ranges):
    """
    Computes the lidar gate altitude above the lidar location.

    Parameters
    ----------
    elevations : 1d float array
        Elevations of the rays
    ranges : 1d float array
        Ranges of the lidar gates

    Returns
    -------
    gate_altitudes : 1D float array
        The estimated lidar gate altitude above ground in meters.

    """
    earth_radius = 6371  # Radius of the Earth in kilometers
    radar_height = 0.0  # Height of the radar above the ground in kilometers

    # Ensure that the elevation and range arrays have the same shape
    if elevations.shape != ranges.shape:
        raise ValueError("The elevation and range arrays must have the same shape.")

    # Convert elevation from degrees to radians
    elevation_rad = np.radians(elevations)

    # Compute the gate altitudes using the radar equation
    gate_altitudes = earth_radius * np.sin(elevation_rad) + np.sqrt(
        (earth_radius * np.sin(elevation_rad)) ** 2
        + (2 * (ranges + radar_height) * earth_radius)
    )

    return gate_altitudes * 1000
