"""
pyart.graph.common
==================

Common graphing routines.

.. autosummary::
    :toctree: generated/

    parse_ax
    parse_ax_fig
    parse_cmap
    parse_vmin_vmax
    parse_lon_lat
    generate_colorbar_label
    generate_field_name
    generate_radar_name
    generate_grid_name
    generate_radar_time_begin
    generate_radar_time_sweep
    generate_grid_time_begin
    generate_filename
    generate_grid_filename
    generate_title
    generate_grid_title
    generate_longitudinal_level_title
    generate_latitudinal_level_title
    generate_cross_section_level_title
    generate_vpt_title
    generate_ray_title
    generate_xsection_title
    set_limits

"""

import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import num2date

from ..config import get_field_colormap, get_field_limits, get_rgba_bounds

########################
# Common radar methods #
########################


def mask_data_outside(flag, data, v1, v2):
    """Return the data masked outside of v1 and v2 when flag is True."""
    if flag:
        data = np.ma.masked_invalid(data)
        data = np.ma.masked_outside(data, v1, v2)
    return data


def parse_ax(ax):
    """Parse and return ax parameter."""
    if ax is None:
        ax = plt.gca()
    return ax


def parse_ax_fig(ax, fig):
    """Parse and return ax and fig parameters."""
    if ax is None:
        ax = plt.gca()
    if fig is None:
        fig = plt.gcf()
    return ax, fig


def parse_cmap(cmap, field=None):
    """Parse and return the cmap parameter."""
    if cmap is None:
        cmap = get_field_colormap(field)
    return cmap


def parse_rgba_bounds(fields):
    """Parse and return the rgba bounds."""
    rgba_bounds = get_rgba_bounds(fields)
    return rgba_bounds


def parse_vmin_vmax(container, field, vmin, vmax):
    """Parse and return vmin and vmax parameters."""
    field_dict = container.fields[field]
    field_default_vmin, field_default_vmax = get_field_limits(field)
    if vmin is None:
        if "valid_min" in field_dict:
            vmin = field_dict["valid_min"]
        else:
            vmin = field_default_vmin
    if vmax is None:
        if "valid_max" in field_dict:
            vmax = field_dict["valid_max"]
        else:
            vmax = field_default_vmax
    return vmin, vmax


def parse_lon_lat(grid, lon, lat):
    """Parse lat and lon parameters"""
    if lat is None:
        lat = grid.origin_latitude["data"][0]
    if lon is None:
        lon = grid.origin_longitude["data"][0]
    return lon, lat


def generate_colorbar_label(standard_name, units):
    """Generate and return a label for a colorbar."""
    return str(standard_name).replace("_", " ") + " (" + units + ")"


def generate_field_name(container, field):
    """
    Return a nice field name for a particular field.

    If field is a tuple/list of 3 strings, generate an RGB composite label.
    """

    def _single_field_name(f):
        if "standard_name" in container.fields.get(f, {}):
            name = container.fields[f]["standard_name"]
        elif "long_name" in container.fields.get(f, {}):
            name = container.fields[f]["long_name"]
        else:
            name = str(f)
        name = name.replace("_", " ")
        return name[0].upper() + name[1:]

    # --- RGB composite ---
    if isinstance(field, (list, tuple)) and len(field) == 3:
        fr, fg, fb = field
        name_r = _single_field_name(fr)
        name_g = _single_field_name(fg)
        name_b = _single_field_name(fb)

        # Compact, clear RGB label
        return f"RGB composite\nR: {name_r} | G: {name_g} | B: {name_b}"

    # --- Legacy single-field behavior ---
    return _single_field_name(field)


def generate_radar_name(radar):
    """Return radar name."""
    if "instrument_name" in radar.metadata:
        return radar.metadata["instrument_name"]
    return ""


def generate_grid_name(grid):
    """Return grid name."""
    if "instrument_name" in grid.metadata:
        iname = grid.metadata["instrument_name"]
        if isinstance(iname, bytes):
            iname = iname.decode("utf-8")
        return iname
    return ""


def generate_radar_time_begin(radar):
    """Return time begin in datetime instance."""
    # datetime object describing first sweep time
    times = radar.time["data"][0]
    units = radar.time["units"]
    calendar = radar.time["calendar"]
    return num2date(times, units, calendar)


def generate_radar_time_sweep(radar, sweep):
    """Return time that a specific sweep began in a datetime instance."""
    first_ray = radar.sweep_start_ray_index["data"][sweep]
    times = radar.time["data"][first_ray]
    units = radar.time["units"]
    calendar = radar.time["calendar"]
    return num2date(times, units, calendar)


def generate_grid_time_begin(grid):
    """Return time begin in datetime instance."""
    times = grid.time["data"][0]
    units = grid.time["units"]
    if "calendar" in grid.time:
        calendar = grid.time["calendar"]
    else:
        calendar = "standard"
    return num2date(times, units, calendar)


def generate_filename(
    radar, field, sweep, ext="png", datetime_format="%Y%m%d%H%M%S", use_sweep_time=False
):
    """
    Generate a filename for a plot.

    Generated filename has form:
        radar_name_field_sweep_time.ext

    Parameters
    ----------
    radar : Radar
        Radar structure.
    field : str
        Field plotted.
    sweep : int
        Sweep plotted.
    ext : str
        Filename extension.
    datetime_format : str
        Format of datetime (using strftime format).
    use_sweep_time : bool
        If true, the current sweep's beginning time is used.

    Returns
    -------
    filename : str
        Filename suitable for saving a plot.

    """
    name_s = generate_radar_name(radar).replace(" ", "_")
    field_s = field.replace(" ", "_")
    if use_sweep_time:
        time_s = generate_radar_time_sweep(radar, sweep).strftime(datetime_format)
    else:
        time_s = generate_radar_time_begin(radar).strftime(datetime_format)
    sweep_s = str(sweep).zfill(2)
    return f"{name_s}_{field_s}_{sweep_s}_{time_s}.{ext}"


def generate_grid_filename(grid, field, level, ext="png"):
    """
    Generate a filename for a plot.

    Generated filename has form:
        grid_name_field_level_time.ext

    Parameters
    ----------
    grid : Grid
        Grid structure.
    field : str
        Field plotted.
    level : int
        Level plotted.
    ext : str
        Filename extension.

    Returns
    -------
    filename : str
        Filename suitable for saving a plot.

    """
    name_s = generate_grid_name(grid).replace(" ", "_")
    field_s = field.replace(" ", "_")
    time_s = generate_grid_time_begin(grid).strftime("%Y%m%d%H%M%S")
    level_s = str(level).zfill(2)
    return f"{name_s}_{field_s}_{level_s}_{time_s}.{ext}"


def generate_title(radar, field, sweep, datetime_format=None, use_sweep_time=True):
    """
    Generate a title for a plot.

    Parameters
    ----------
    radar : Radar
        Radar structure.
    field : str or tuple of str
        Field(s) plotted. Either a single str or a tuple of 3 str (for RGB plots)
    sweep : int
        Sweep plotted.
    datetime_format : str
        Format of datetime (using strftime format).
    use_sweep_time : bool
        If true, the current sweep's beginning time is used.

    Returns
    -------
    title : str
        Plot title.

    """
    if use_sweep_time:
        begin_time = generate_radar_time_sweep(radar, sweep)
    else:
        begin_time = generate_radar_time_begin(radar)
    if datetime_format:
        time_str = begin_time.strftime(datetime_format)
    else:
        time_str = begin_time.isoformat() + "Z"
    fixed_angle = radar.fixed_angle["data"][sweep]
    l1 = f"{generate_radar_name(radar)} {fixed_angle:.1f} Deg. {time_str} "
    field_name = generate_field_name(radar, field)
    return l1 + "\n" + field_name


def generate_grid_title(grid, field, level):
    """
    Generate a title for a plot.

    Parameters
    ----------
    grid : Grid
        Radar structure.
    field : str
        Field plotted.
    level : int
        Verical level plotted.


    Returns
    -------
    title : str
        Plot title.

    """
    time_str = generate_grid_time_begin(grid).isoformat() + "Z"
    height = grid.z["data"][level] / 1000.0
    l1 = f"{generate_grid_name(grid)} {height:.1f} km {time_str} "
    field_name = generate_field_name(grid, field)
    return l1 + "\n" + field_name


def generate_longitudinal_level_title(grid, field, level):
    """
    Generate a title for a plot.

    Parameters
    ----------
    grid : Grid
        Radar structure.
    field : str
        Field plotted.
    level : int
        Longitudinal level plotted.


    Returns
    -------
    title : str
        Plot title.

    """
    time_str = generate_grid_time_begin(grid).isoformat() + "Z"
    disp = grid.x["data"][level] / 1000.0
    if disp >= 0:
        direction = "east"
    else:
        direction = "west"
        disp = -disp
    l1 = f"{generate_grid_name(grid)} {disp:.1f} km {direction} of origin\n{time_str} "
    field_name = generate_field_name(grid, field)
    return l1 + "\n" + field_name


def generate_latitudinal_level_title(grid, field, level):
    """
    Generate a title for a plot.

    Parameters
    ----------
    grid : Grid
        Radar structure.
    field : str
        Field plotted.
    level : int
        Latitudinal level plotted.


    Returns
    -------
    title : str
        Plot title.

    """
    time_str = generate_grid_time_begin(grid).isoformat() + "Z"
    disp = grid.y["data"][level] / 1000.0
    if disp >= 0:
        direction = "north"
    else:
        direction = "south"
        disp = -disp
    l1 = f"{generate_grid_name(grid)} {disp:.1f} km {direction} of origin\n{time_str} "
    field_name = generate_field_name(grid, field)
    return l1 + "\n" + field_name


def generate_cross_section_level_title(grid, field):
    """
    Generate a title for a plot.

    Parameters
    ----------
    grid : Grid
        Radar structure.
    field : str
        Field plotted.

    Returns
    -------
    title : str
        Plot title.

    """
    time_str = generate_grid_time_begin(grid).isoformat() + "Z"
    field_name = generate_field_name(grid, field)
    return (
        generate_grid_name(grid)
        + " cross-section slice\n"
        + time_str
        + "\n"
        + field_name
    )


def generate_vpt_title(radar, field):
    """
    Generate a title for a VPT plot.

    Parameters
    ----------
    radar : Radar
        Radar structure.
    field : str
        Field plotted.

    Returns
    -------
    title : str
        Plot title.

    """
    time_str = generate_radar_time_begin(radar).isoformat() + "Z"
    l1 = f"{generate_radar_name(radar)} {time_str} "
    field_name = generate_field_name(radar, field)
    return l1 + "\n" + field_name


def generate_ray_title(radar, field, ray):
    """
    Generate a title for a ray plot.

    Parameters
    ----------
    radar : Radar
        Radar structure.
    field : str
        Field plotted.
    ray : int
        Ray plotted.

    Returns
    -------
    title : str
        Plot title.

    """
    time_str = generate_radar_time_begin(radar).isoformat() + "Z"
    l1 = f"{generate_radar_name(radar)} {time_str}"
    azim = radar.azimuth["data"][ray]
    elev = radar.elevation["data"][ray]
    l2 = f"Ray: {int(ray)}  Elevation: {elev:.1f} Azimuth: {azim:.1f}"
    field_name = generate_field_name(radar, field)
    return l1 + "\n" + l2 + "\n" + field_name


def generate_az_rhi_title(radar, field, azimuth):
    """
    Generate a title for a pseudo-RHI from PPI azimuth plot.

    Parameters
    ----------
    radar : Radar
        Radar structure.
    field : str
        Field plotted.
    azimuth : float
        Azimuth plotted.

    Returns
    -------
    title : str
        Plot title.

    """
    time_str = generate_radar_time_begin(radar).isoformat() + "Z"
    l1 = f"{generate_radar_name(radar)} {time_str} "
    l2 = f"Azimuth: {azimuth:.1f} deg"
    field_name = generate_field_name(radar, field)
    return l1 + "\n" + l2 + "\n" + field_name


def generate_xsection_title(radar, field, points):
    """
    Generate a title for a cross-section plot

    Parameters
    ----------
    radar : Radar
        Radar structure.
    field : str
        Field plotted.
    points : ndarray
        N x 2 array containing the lon/lat coordinates of the reference points

    Returns
    -------
    title : str
        Plot title.

    """
    points_fmt = ",".join([f"{pt[0]:2.1f}°/{pt[1]:2.1f}°" for pt in points])
    time_str = generate_radar_time_begin(radar).isoformat() + "Z"
    l1 = f"{generate_radar_name(radar)} {time_str} "
    l2 = f"Points: {points_fmt}"
    field_name = generate_field_name(radar, field)
    return l1 + "\n" + l2 + "\n" + field_name


def set_limits(xlim=None, ylim=None, ax=None):
    """
    Set the display limits.

    Parameters
    ----------
    xlim : tuple, optional
        2-Tuple containing y-axis limits in km. None uses default limits.
    ylim : tuple, optional
        2-Tuple containing x-axis limits in km. None uses default limits.
    ax : Axis
        Axis to adjust.  None will adjust the current axis.

    """
    if ax is None:
        ax = plt.gca()
    if ylim is not None:
        ax.set_ylim(ylim)
    if xlim is not None:
        ax.set_xlim(xlim)


def normalize_rgba(fields_rgb, rgba_bounds):
    """
    Normalize 3 radar fields into (r,g,b,a) in [0,1] using piecewise-linear
    mappings described by rgba_bounds.

    Parameters
    ----------
    fields_rgb : tuple
        (field_r, field_g, field_b) as numpy arrays or numpy masked arrays.
        All must be broadcastable to the same shape.
    rgba_bounds : tuple
        Length-4 tuple/list: (rb, gb, bb, ab) where each element is:
          ((x0, x1, ...), (y0, y1, ...))  OR  ((x0, x1), (y0, y1))

        - The first three bounds map their corresponding field to rn/gn/bn.
        - The 4th bounds maps *field_r* to alpha (an). based on values of the
          FIRST field (rn)

        Example:
          rgba_bounds[0] = ((30, 60), (0, 1))
          rgba_bounds[1] = ((100, 70), (0, 1))  # decreasing x is allowed
          rgba_bounds[2] = ((0, 4), (0, 1))
          rgba_bounds[3] = ((-10,0,10,...,40), (0.05,0.12,...,1.0))

    Returns
    -------
    rn, gn, bn, an : numpy.ma.MaskedArray
        Normalized channels in [0,1], preserving a combined mask:
        if a gate is masked in the source field(s), the output is masked there.
        Alpha is additionally clipped to [0,1].
    """
    field_r, field_g, field_b = fields_rgb

    def _as_ma(x):
        return x if np.ma.isMaskedArray(x) else np.ma.array(x)

    field_r = _as_ma(field_r)
    field_g = _as_ma(field_g)
    field_b = _as_ma(field_b)

    # Broadcast to a common shape (preserving masks)
    r, g, b = np.broadcast_arrays(field_r, field_g, field_b)
    combined_mask = (
        np.ma.getmaskarray(r) | np.ma.getmaskarray(g) | np.ma.getmaskarray(b)
    )

    r = np.ma.array(r, mask=combined_mask)
    g = np.ma.array(g, mask=combined_mask)
    b = np.ma.array(b, mask=combined_mask)

    def _piecewise_map(data_ma, bounds):
        """
        Piecewise-linear map from x->y with clipping.
        bounds: ((x...), (y...))
        x can be increasing or decreasing; mapping is defined by the given points.
        """
        (x_pts, y_pts) = bounds
        x = np.asarray(x_pts, dtype=float)
        y = np.asarray(y_pts, dtype=float)

        if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
            raise ValueError(
                "Each rgba_bounds[i] must be ((x0,x1,...),(y0,y1,...)) with same length."
            )
        if x.size < 2:
            raise ValueError("Each rgba_bounds[i] must have at least 2 control points.")

        # Work on filled array, then reapply mask
        mask = np.ma.getmaskarray(data_ma)
        data = np.asarray(data_ma.filled(np.nan), dtype=float)

        # np.interp requires xp increasing. If provided decreasing, reverse.
        if x[0] > x[-1]:
            x = x[::-1]
            y = y[::-1]

        # Map with clipping to endpoints
        out = np.interp(data, x, y, left=y[0], right=y[-1])

        # Re-mask NaNs and original mask
        out_mask = mask | ~np.isfinite(data)
        return np.ma.array(out, mask=out_mask)

    # Apply mappings
    rn = _piecewise_map(r, rgba_bounds[0])
    gn = _piecewise_map(g, rgba_bounds[1])
    bn = _piecewise_map(b, rgba_bounds[2])

    # Alpha uses FIRST field (r)
    an = _piecewise_map(r, rgba_bounds[3])

    # Clip all channels to [0,1] (y-values should already be in range, but be safe)
    rn = np.ma.clip(rn, 0.0, 1.0)
    gn = np.ma.clip(gn, 0.0, 1.0)
    bn = np.ma.clip(bn, 0.0, 1.0)
    an = np.ma.clip(an, 0.0, 1.0)

    # Optionally: enforce a combined mask across RGB so they disappear together
    combined_mask = (
        np.ma.getmaskarray(rn) | np.ma.getmaskarray(gn) | np.ma.getmaskarray(bn)
    )
    rn.mask = combined_mask
    gn.mask = combined_mask
    bn.mask = combined_mask
    # alpha should also be masked where RGB is masked
    an.mask = combined_mask | np.ma.getmaskarray(an)
    # Combine masks: if any channel masked, hide that pixel
    m = np.ma.getmaskarray(rn) | np.ma.getmaskarray(gn) | np.ma.getmaskarray(bn)

    # Set alpha=1 where valid, nan where masked
    an[m] = np.nan
    rgba = np.stack(
        [
            np.ma.array(rn, mask=m).filled(0.0),
            np.ma.array(gn, mask=m).filled(0.0),
            np.ma.array(bn, mask=m).filled(0.0),
            an,
        ],
        axis=-1,
    ).astype(float)

    rgba_pm = rgba.copy()
    # Premultiply RGB by alpha
    rgba_pm[..., :3] *= rgba_pm[..., 3:4]
    # Make everything opaque
    rgba_pm[..., 3] = 1.0
    return rgba_pm


def get_rgba_data(
    fields, display, sweep, mask_tuple, filter_transitions, gatefilter, mask_outside
):
    """
    Build an (M, N, 4) RGBA array from three radar fields.

    Parameters
    ----------
    field : (str, str, str)
        Tuple/list of three field names (R, G, B).
    display : pyart Display object
        Display object to use to retrieve data
    sweep : int, optional
        Sweep number to plot.
    mask_tuple : (str, float)
        Tuple containing the field name and value below which to mask
        field prior to plotting, for example to mask all data where
        NCP < 0.5 set mask_tuple to ['NCP', 0.5]. None performs no masking.
    filter_transitions : bool
        True to remove rays where the antenna was in transition between
        sweeps from the plot. False will include these rays in the plot.
        No rays are filtered when the antenna_transition attribute of the
        underlying radar is not present.
    gatefilter : GateFilter
        GateFilter instance. None will result in no gatefilter mask being
        applied to data.
    mask_outside : bool
        True to mask data outside of vmin, vmax. False performs no
        masking.

    Returns
    -------
    rgba : numpy.ma.MaskedArray
        Masked array of shape (ny, nx, 4), float in [0, 1].
    """
    fr, fg, fb = fields

    rgba_bounds = parse_rgba_bounds(fields)

    # Get per-channel data
    field_r = display._get_data(fr, sweep, mask_tuple, filter_transitions, gatefilter)
    field_g = display._get_data(fg, sweep, mask_tuple, filter_transitions, gatefilter)
    field_b = display._get_data(fb, sweep, mask_tuple, filter_transitions, gatefilter)

    # Optional: mask outside per-channel limits
    if mask_outside:
        field_r = mask_data_outside(
            True, field_r, rgba_bounds[0][0][0], rgba_bounds[0][0][-1]
        )
        field_g = mask_data_outside(
            True, field_g, rgba_bounds[1][0][0], rgba_bounds[1][1][-1]
        )
        field_b = mask_data_outside(
            True, field_b, rgba_bounds[2][0][0], rgba_bounds[2][1][-1]
        )

    rgba_pm = normalize_rgba((field_r, field_g, field_b), rgba_bounds)
    return rgba_pm


def get_rgba_data_grid(fields, ds, mask_outside=True):
    """
    Build an (M, N, 4) RGBA array from three grid fields.

    Parameters
    ----------
    field : (str, str, str)
        Tuple/list of three field names (R, G, B).
    ds : xarray.Dataset
        Dataset from ``grid.grid.to_xarray()``.
    level : int, default 0
        Vertical level index.
    mask_outside : bool
        True to mask data outside of vmin, vmax. False performs no
        masking.

    Returns
    -------
    rgba : numpy.ma.MaskedArray
        Masked array of shape (ny, nx, 4), float in [0, 1].
    """
    fr, fg, fb = fields

    rgba_bounds = parse_rgba_bounds(fields)

    # Get per-channel data
    field_r = np.ma.masked_invalid(ds[fr].data)
    field_g = np.ma.masked_invalid(ds[fg].data)
    field_b = np.ma.masked_invalid(ds[fb].data)

    # Optional: mask outside per-channel limits
    if mask_outside:
        field_r = np.ma.masked_outside(
            field_r, rgba_bounds[0][0][0], rgba_bounds[0][0][-1]
        )
        field_g = np.ma.masked_outside(
            field_g, rgba_bounds[1][0][0], rgba_bounds[1][1][-1]
        )
        field_b = np.ma.masked_outside(
            field_b, rgba_bounds[2][0][0], rgba_bounds[2][1][-1]
        )

    rgba_pm = normalize_rgba((field_r, field_g, field_b), rgba_bounds)
    return rgba_pm
