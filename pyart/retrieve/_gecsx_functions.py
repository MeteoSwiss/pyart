"""
pyart.retrieve._gecsx_functions
==================

Helping functions for visibility and ground echoes estimation from a DEM.

"""

import logging
import warnings

import numpy as np
from scipy.special import erfc

from ._gecsx_functions_cython import vis_weighting

RCS_MIN = 1.0  # [m^2] remove RCS below this level


def antenna_pattern_gauss(
    d_az,
    d_el,
    antenna_3dB,
    db=False,
    twoway=True,
    az_conv=None,
    el_conv=None,
    units="rad",
):
    """
    Get the antenna weighting factor due to the azimuth and elevation offsets
    from the main antenna direction. The weighting factor is meant in terms of
    power (not amplitude of the E-field). An Gaussian antenna pattern is
    assumed.

    Parameters
    ----------
    d_az : array
        Azimuth offsets to the main antenna axis [deg]
    d_el : array
        Elevation offset to the main antenna axis [deg]
    antenna_3dB : float
        Half power beam width [deg]
    db : bool (optional)
        If true return the result in dB instead linear.
    twoway: bool (optional)
        If true, return the two-way weighting factor instead of
        the one-way factor.
    az_conv (optional): float
        If set, assumes that the antenna moves in azimuth direction (PPI) and
        averages over the angle given by this keyword [deg].
    el_conv (optional): float
        If set, assumes that the antenna moves in elevation direction (RHI)
        and averages over the angle given by this keyword [deg].
    units (optional) : str
        Specify if inputs quantities are given in radians ( "rad" ) or
        degrees ( "deg" )

    Returns
    -------
    fa : array
        Weighting factor.
    """
    if az_conv is not None:
        if az_conv <= 0:
            az_conv = None
    if el_conv is not None:
        if el_conv <= 0:
            el_conv = None

    if units not in ["deg", "rad"]:
        print(
            'Invalid units, must be either "rad" or "deg", ' + 'assuming they are "rad"'
        )

    if units == "deg":
        # Convert all quantities to rad
        if az_conv is not None:
            az_conv *= np.pi / 180.0
        if el_conv is not None:
            el_conv *= np.pi / 180.0
        d_az = d_az.copy() * np.pi / 180.0
        d_el = d_el.copy() * np.pi / 180.0
        antenna_3dB *= np.pi / 180.0

    if az_conv is None or el_conv is None:
        if az_conv is not None:
            """The antenna is moving in azimuth direction and is averaging the
            received pulses over 'az_conv' deg. The norm azimuth position
            of the antenna is reached, when the antenna moved half of the
            azimuth distance (az_offset = 'az_conv'/2 deg).
            The weighting factor at the azimuth position 'daz' from the norm
            position is given by the following integral:

                               1    / daz+az_offset
              fa(daz, del) = ---- * |               f(daz) d(daz)   * f(del)
                             Norm   / daz-az_offset
            where
              daz : Is the azimuth deviation from the norm antenna position
              del : Is the elevation deviation from the norm antenna position
              fa(daz,del) : Weighting factor at point (daz,del) f(0,0) must be 1.
              Norm : Normalization such that f(0,0)=1
              f(x) : Weighting factor of the non moving antenna (Gaussian
                    function, see below)

            Solving the integral above leads to:

                            K1
            fa(daz, del) = ---- * ( erf(K*(daz+az_offset)) -erf(K*(daz-az_offset)) )
                           Norm
                                                                       * f(del)
            where
                  2 * sqrt(ln(2))
              K = ---------------
                      phi3db

                   sqrt(!PI)
              K1 = ---------
                     2 * K

              erf : the error function
              phi3db : the half power beam width
            """

            az_offset = az_conv / 2.0
            K = 2.0 * np.sqrt(np.log(2)) / antenna_3dB
            K1 = np.sqrt(np.pi) / 2.0 / K
            Norm = 2.0 * K1 * erfc(K * az_offset)
            faz = (
                K1
                / Norm
                * (erfc(K * (d_az + az_offset)) - erfc(K * (d_az - az_offset)))
            )
        else:
            da = (2.0 * d_az / antenna_3dB) ** 2
            ind = da > 20.0
            da[ind] = 20
            faz = np.exp(-da * np.log10(2))

        if el_conv is not None:
            # see explanation for az_conv above
            el_offset = el_conv / 2.0 * np.pi / 180.0
            K = 2.0 * np.sqrt(np.log(2)) / antenna_3dB
            K1 = np.sqrt(np.pi) / 2.0 / K
            Norm = 2.0 * K1 * erfc(K * el_offset)
            fel = (
                K1
                / Norm
                * (erfc(K * (d_el + el_offset)) - erfc(K * (d_el - el_offset)))
            )
        else:
            de = (2.0 * d_el / antenna_3dB) ** 2
            ind = de > 20.0
            de[ind] = 20
            fel = np.exp(-de * np.log10(2))

        fa = faz * fel
    else:
        # Gaussian antenna pattern:
        #
        # f(daz,del) = e^(-( (2*daz/phi3db_el)^2 + (2*del/phi3db_az)^2 ) * ln(2))
        #
        # from Gauss normal distribution N(x) with N(x)=1 and N(x=X0/2)=1/2 :
        #
        # N(x) = e^(-(2*x/X0)^2 * ln(2))

        da = 2.0 * d_az / antenna_3dB
        de = 2.0 * d_el / antenna_3dB
        dr = da**2 + de**2

        ind = dr > 20.0
        dr[ind] = 20

        fa = np.exp(-dr * np.log(2))

    if twoway:
        fa = fa**2

    if db:
        fa = 10.0 * np.log10(fa)

    return fa


def clip_grid(grid, xr, yr, extra_m=5000):
    """
    Clips a grid by limiting to an area defined by vectors xr and yr

    Parameters
    ----------
    grid : pyart.Core.Grid
        grid object that contains all Cartesian fields to be clipped
    xr : array
        1D array of x-coordinates outside of which the grid will be clipped
    extra_m : int
        Additional distance outside of domain defined by xr and yr which will
        be kept in the grid, i.e. the lower left corner of the final grid will
        be (min(xr) - extra_m, min(yr) - extra_m) and the upper right corner
        will be (max(xr) + extra_m, max(yr) + extra_m)

    Returns
    -------
    grid : pyart.Core.Grid
        The clipped grid
    """

    min_x = np.min(xr)
    min_y = np.min(yr)
    max_x = np.max(xr)
    max_y = np.max(yr)

    mask_x = np.logical_and(
        grid.x["data"] > min_x - extra_m, grid.x["data"] < max_x + extra_m
    )
    mask_y = np.logical_and(
        grid.y["data"] > min_y - extra_m, grid.y["data"] < max_y + extra_m
    )

    grid.x["data"] = grid.x["data"][mask_x]
    grid.y["data"] = grid.y["data"][mask_y]
    for f in grid.fields.keys():
        nz = len(grid.fields[f]["data"])  # Nb of z levels
        grid.fields[f]["data"] = grid.fields[f]["data"][
            np.ix_(range(nz), mask_y, mask_x)
        ]
    grid.nx = len(grid.x["data"])
    grid.ny = len(grid.y["data"])
    return grid


def range_weights(rangemap, rr, pulselength, db=False):
    """
    Get the contribution of a radar target at distance 'r' to the radar signal
    at distance 'R'. Assuming a rectangular pulse and a matched filter with
    the same shape as the radar pulse. Assuming that the radar receiver is not
    averaging over multiple samples around R. The contribution is linear with
    1 at the r=R and 0 if |r - R| >= pulselength.

    Parameters
    ----------
    rangemap : array
        Array with distances (r) to the radar [m]
    rr : float
        Distance of the reference point to the radar [m]
    pulselength : float
        Pulse length of the radar pulse [m]
    db : bool
        If true return the result in dB instead linear.

    Returns
    -------
    fr : array
        Weighting factors for each range point.
    """

    dr = rangemap - rr
    fr = 1.0 - np.abs(dr) / pulselength

    ind0 = fr < 0
    fr[ind0] = 0.0
    fr[~ind0] = fr[~ind0] * (rr / rangemap[~ind0]) ** 4

    if db:
        fr = 10.0 * np.log10(fr)

    return fr


def rcs(
    azmap,
    rmap,
    elmap,
    areaeffmap,
    sigma0map,
    vismap,
    rpol,
    azpol,
    elpol,
    DEM_res,
    DEM_xmin,
    DEM_ymin,
    rad_x,
    rad_y,
    beamwidth,
    pulsewidth,
    range_weighting=True,
    az_conv=0,
    raster_oversampling=1,
    verbose=True,
):
    """
    Computes the radar cross section of ground clutter in polar coordinates

    Parameters
    ----------
    azmap : array
        Cartesian array with radar azimuth angles
    elmap : array
        Cartesian array with radar elevation angles
    rmap : array
        Cartesian array with distances from radar
    areaeffmap : array
        Cartesian array with effective areas
    sigma0map : array
        Cartesian array with sigma0 : the ratio between effective backscatter-
        ing areas and RCS
    vismap : array
        Cartesian array with radar visibility
    rpol : array
        Radar ranges at which to compute the RCS in polar coordinates
    azpol : array
        Radar azimuths at which to compute the RCS in polar coordinates
    elpol : array
        Radar elevations at which to compute the RCS in polar coordinates
    DEM_res : int
        Resolution of the DEM grid
    DEM_xmin : float
        minimum x coord of the DEM (W-E)
    DEM_ymin : float
        minimum y coord of the DEM (S-N)
    rad_x : float
        Radar position x coordinate
    rad y : float
        Radar position y coordinate
    beamwidth : float
        Antenna 3dB beamwidth in deg
    pulsewidth : float
        Radar pulsewidth in sec
    range_weighting : bool, optional
        If true, weight the backscattering cross sections of each cell
        according to their range offset using the radar pulse length.
    az_conv : float, optional
        If set, assumes that the antenna moves in azimuth direction
        (PPI) and averages over the angle given by this keyword [deg].
    raster_oversampling : int (optional)
        The raster resolution of the DEM should be smaller than
        the range resolution of the radar (defined by the pulse length).
        If this is not the case, this keyword can be set to increase the
        raster resolution. The values for the elevation, sigma naught,
        visibility are repeated. The other values are recalculated.
        Values for RASTER_OVERSAMPLING:
        0 or undefined: No oversampling is done
        1: Oversampling is done. The factor N is automatically calculated
        such that 2*dx/N < pulse length
        2 or larger: Oversampling is done with this value as N
    verbose : bool, optional
        If true, will print info about current progress

    Returns
    -------
    rcspolarmap : array
        2D Array of RCS in polar coordinates in PyART format, with all az
        angles for all elevations stacked
    """

    nrows, ncols = azmap.shape
    area_unweighted = areaeffmap * sigma0map

    pulselength = pulsewidth * 3.0e8 / 2.0  # [m]
    if az_conv is not None:
        az_conv_offset = az_conv / 2.0
    else:
        az_conv_offset = 0

    beamwidth_rad = beamwidth * np.pi / 180.0

    if not range_weighting:
        range_weight = 1  # unity

    if raster_oversampling == 0:
        N = 1
    elif raster_oversampling == 1:
        N = int(np.ceil(2 * DEM_res / pulselength))
    else:
        N = raster_oversampling

    if N != 1:
        # New dimensions
        nc = N * ncols
        nr = N * nrows

        # repeat the values NxN, equivalent of rebin in IDL
        elvals = np.repeat(np.repeat(elmap, N, axis=0), N, axis=1)
        areavals = np.repeat(np.repeat(area_unweighted / N**2, N, axis=0), N, axis=1)
        visvals = np.repeat(np.repeat(vismap, N, axis=0), N, axis=1)

        # New x- and y-vectors
        xvec = np.arange(nc) * DEM_res / N + DEM_xmin
        yvec = np.arange(nr) * DEM_res / N + DEM_ymin

        xdiff = xvec - rad_x
        ydiff = yvec - rad_y

        # New distance from radar map
        X, Y = np.meshgrid(xdiff, ydiff)
        rvals = np.sqrt(X**2 + Y**2)

        # New azimuth map
        azmap_rad = (np.arctan2(X, Y) + 2 * np.pi) % (2 * np.pi)
        azvals = azmap_rad * 180.0 / np.pi
    else:
        rvals = rmap
        azvals = azmap
        azmap_rad = azvals * np.pi / 180.0
        elvals = elmap
        areavals = area_unweighted
        visvals = vismap
    elmap_rad = elvals * np.pi / 180.0
    elevations_rad = np.array(elpol) * np.pi / 180.0

    # Define the area around a point P(range, azimuth) where the cells
    # have a contribution to the RCS. This area is defined with the
    # range limits from range-dr_offset to range+dr_offset and the
    # azimuth limits from azimuth-daz_offset to azimuth+daz_offset.
    #
    # For a Gaussian antenna, azimuth offset more than 2*HPBW does not a
    # have remarkable contribution.
    # With a rectangular pulse and a matched filter cells farer away
    # than pulse length does not a have remarkable contribution.

    daz_offset = (2.0 * beamwidth) + az_conv_offset  # [deg]
    dr_offset = pulselength  # [m]

    azpol_unique = np.unique(azpol)
    nazim = len(azpol_unique)
    nrange = len(rpol)

    # pyart storage format: 2D arrays (naz * nel, nranges)
    rcspolarmap = np.zeros((nazim * len(elpol), nrange)) + np.nan

    for rind in range(nrange):
        if verbose:
            logging.info(f"Computing range bin {rpol[rind]:2.1f}")
        rr = rpol[rind]

        indr = np.logical_and(
            np.logical_and(rvals >= rr - dr_offset, rvals < rr + dr_offset), visvals > 0
        )

        if not np.any(indr):
            continue

        indr = np.where(indr)

        for azind in range(nazim):
            az = azpol_unique[azind]
            # Inside the loops over range (rr) and azimuth (az), the
            # coordinates (rr, az) describe the point P(rr, az) for which
            # the RCS is calculated. If more than one DEM cell is within
            # the area from az-daz/2 to az+daz/2 and from rr-dr/2 to
            # rr+dr/2, the calculated RCS value is set to all of these
            # cells (next neighbor).

            # Get area around rr and az
            azmin = az - daz_offset
            azmax = az + daz_offset
            if azmin < 0:
                azmin = 360.0 + azmin
                indaz = np.logical_or(
                    np.logical_and(azvals[indr] >= 0, azvals[indr] < azmax),
                    np.logical_and(azvals[indr] >= azmin, azvals[indr] <= 360.0),
                )
            elif azmax > 360:
                azmax = azmax - 360.0
                indaz = np.logical_or(
                    np.logical_and(azvals[indr] >= azmin, azvals[indr] <= 360),
                    np.logical_and(azvals[indr] >= 0, azvals[indr] < azmax),
                )
            else:
                indaz = np.logical_and(azvals[indr] >= azmin, azvals[indr] < azmax)

            # Cells that contribute to the cells to set indset
            inda = tuple([indr[0][indaz], indr[1][indaz]])

            # Calculate offsets in azimuth and elevation to the
            # point P(rr,az) and the elevation angle of the antenna.

            daz_area = azmap_rad[inda] - (az * np.pi / 180.0)

            indaz = daz_area > np.pi
            daz_area[indaz] = daz_area[indaz] - 2.0 * np.pi

            indaz = daz_area < -np.pi
            daz_area[indaz] = daz_area[indaz] + 2.0 * np.pi

            if range_weighting:
                # Get the weighting factor due to the range offset.
                range_weight = range_weights(rvals[inda], rr, pulselength)

            ind_rzero = rvals[inda] <= 0.0
            if np.any(ind_rzero):
                continue

            for iel, el in enumerate(elevations_rad):
                del_area = elmap_rad[inda] - el

                # Get the two-way weighting factor due to the azimuth offset
                # to the main antenna direction (assuming a Gaussian antenna
                # pattern).
                ant_weight = antenna_pattern_gauss(
                    daz_area,
                    del_area,
                    beamwidth_rad,
                    twoway=True,
                    az_conv=az_conv * np.pi / 180.0,
                    units="rad",
                )

                # RCS = SUM_j sigma_j
                # = SUM_j sigma0_j * A_eff_j * fa(dphi_j,dteta_j)^2 * fr(drange)
                # where
                # sigma_j  : Backscattering cross section of each cell [m^2]
                # sigma0_j : Sigma naught of each cell [1]
                # A_eff_j  : Effective area of each cell [m^2]
                # fa       : One-way weighting function due to the azimuth
                #            and elevation offsets.
                # fr       : Range weighting function due to the range offset

                # RCS contribution of each cell inside the contribution
                # area.
                rcs_area = ant_weight * range_weight * areavals[inda]
                # Sum up all the contributions
                rcs = np.nansum(rcs_area)

                if rcs < RCS_MIN:
                    rcs = np.nan

                # Set rcs to all values inside the set area.

                rcspolarmap[azind + iel * nazim, rind] = rcs
    # Correctly map the vispol to the actual azimuth angles
    rcspolar_remapped = []
    for i, az in enumerate(azpol):
        idx_az = np.searchsorted(azpol_unique, az)
        rcspolar_remapped.extend(rcspolarmap[nazim * i + idx_az, :])
    rcspolar_remapped = np.array(rcspolar_remapped)
    return rcspolar_remapped


def sigma0(inc_ang, frequency, method="Gabella"):
    """
    Estimates the sigma0 factor (ratio between effective backscattering area)
    and Radar Cross Section

    Parameters
    ----------
    inc_ang : array
        Incidence angles [deg]
    frequency_ghz : float
        Frequency in GHz
    method : str, optional
        Which estimation method to use, either 'Gabella' or 'Delrieu'

    Returns
    -------
    sigma_0 : array
        RCS to Backscattering area ratio

    References
    ----------
    Delrieu, G., J. D. Creutin, and H. Andrieu, 1995: Simulation of radar
    mountain returns using a digitized terrain model.
    J. Atmos. Oceanic Technol.,12, 1038–1049.

    Gabella, M., & Perona, G. (1998). Simulation of the Orographic Influence
    on Weather Radar Using a Geometric–Optics Approach, Journal of Atmospheric
    and Oceanic Technology, 15(6), 1485-1494.
    """

    inc_angr = inc_ang * np.pi / 180.0
    sigma_0 = np.zeros(inc_ang.shape)

    if method not in ["Gabella", "Delrieu"]:
        warnings.warn(
            'Invalid method for sigma_0: use "gabella" or '
            + '"delrieu", using "Gabella" instead'
        )
        method = "Gabella"

    if method == "Gabella":
        # GABELLA/PERONA 1997 (GECS)
        te1 = 80.0  # Maximum Angle of "Plateau" Region      [deg]
        te2 = 87.0  # Dimming Start Angle at Grazing Angles  [deg]
        te3 = 88.0  # Border Incidence Angle                 [deg]
        k1 = 0.1  # Offset Index
        k2 = 1.0  # Dimming Index near Grazing Angles
        bbdeg = 2.75  # Tuning Angle in Exp Rising above Border Angle [deg]

        te1r = te1 * np.pi / 180.0
        te2r = te2 * np.pi / 180.0
        te3r = te3 * np.pi / 180.0
        bbdegr = bbdeg * np.pi / 180.0

        factor = k1 * np.cos(te2r) * ((np.pi / 2.0 - te2r) / (np.pi / 2.0 - te1r)) ** k2

        ind0 = inc_ang <= te1

        if len(ind0) != 0:
            sigma_0[ind0] = k1 * np.cos(inc_angr[ind0])

        ind1 = np.logical_and(inc_ang > te1, inc_ang <= te2)

        if len(ind1) != 0:
            iang = inc_angr[ind1]
            sigma_0[ind1] = (
                k1 * np.cos(iang) * ((np.pi / 2.0 - iang) / (np.pi / 2.0 - te1r)) ** k2
            )

        ind2 = np.logical_and(inc_ang > te2, inc_ang <= te3)

        if len(ind2) != 0:
            sigma_0[ind2] = factor

        ind3 = np.logical_and(inc_ang > te3, inc_ang <= 90)

        if len(ind3) != 0:
            iang = inc_angr[ind3]
            sigma_0[ind3] = factor * np.exp((iang - te3r) / bbdegr)

        ind4 = inc_ang > 90

        if len(ind4) != 0:
            sigma_0[ind4] = factor * np.exp((np.pi / 2.0 - te3r) / bbdegr)

    elif method == "Delrieu":
        # DELRIEU 1995 (X-BAND) / SKOLNIK 1990
        lim_ang_del = 63.75  # Borderline Angle separating curve branche
        # First Branch  [0..lim_ang_del]
        a1 = -9.1  # Skolnik's Model "a1" parameter [dB]
        b1 = -0.12  # Skolnik's Model "b1" parameter [dB/deg]
        c1 = 0.25  # Skolnik's Model "c1" parameter [dB/GHz]
        d1 = 0.0  # Skolnik's Model "d1" parameter [dB/(deg*GHz))]
        # Second Branch  [lim_ang_del..90]
        a1_2 = 12.93  # Skolnik's Model "a1" parameter [dB]
        b1_2 = -0.37  # Skolnik's Model "b1" parameter [dB/deg]
        c1_2 = 0.0  # Skolnik's Model "c1" parameter [dB/GHz]
        d1_2 = 0.0  # Skolnik's Model "d1" parameter [dB/(deg*GHz))]

        sigma_db = np.zeros(sigma_0.shape) + np.nan

        ind = inc_ang < lim_ang_del

        if len(ind) != 0:
            iang = inc_angr[ind]
            sigma_db[ind] = a1 + b1 * iang + c1 * frequency + d1 * frequency * iang
        ind = inc_ang >= lim_ang_del

        if len(ind) != 0:
            iang = inc_angr[ind]
            sigma_db[ind] = (
                a1_2 + b1_2 * iang + c1_2 * frequency + d1_2 * frequency * iang
            )

        sigma_0 = 10 ** (sigma_db / 10.0)

    return sigma_0


def visibility(
    azmap, rmap, elmap, DEM_res, DEM_xmin, DEM_ymin, rad_x, rad_y, dr, daz, verbose=True
):
    """
    Computes the radar visibility over the DEM grid (in Cartesian coords)

    Parameters
    ----------
    azmap : array
        Cartesian array with radar azimuth angles
    rmap : array
        Cartesian array with distances from radar
    elmap : array
        Cartesian array with radar elevation angles
    DEM_res : int
        Resolution of the DEM grid
    DEM_xmin : float
        minimum x coord of the DEM (W-E)
    DEM_ymin : float
        minimum y coord of the DEM (S-N)
    rad_x : float
        Radar position x coordinate
    rad y : float
        Radar position y coordinate
    dr : float
        Range discretization used when computing the Cartesian visibility field
        the larger the better but the slower the processing will be
    daz : float
        Azimuth discretization used when computing the Cartesian visibility
        field, the larger the better but the slower the processing will be
    verbose : bool, optional
        If true, will print info about current progress

    Returns
    -------
    visibility : array
        2D Array of visibility along the Cartesian DEM grid, 100 for pixels
        which are visible, zero otherwise.
    minviselev : array
        2D array of minimum visible elevation angle along the Cartesian
        DEM grid.
    """

    range_max = np.max(rmap)
    nr = int(range_max / dr)
    nrows = len(elmap)
    ncols = len(elmap[0])
    visib = np.zeros([nrows, ncols])
    minviselev = np.zeros([nrows, ncols])

    rr_start = dr / 2

    azmin = 0.0
    azmax = 0.0
    rrmin = 0.0
    rrmax = 0.0
    azmin_sin = 0.0
    azmax_sin = 0.0
    azmin_cos = 0.0
    azmax_cos = 0.0
    el_rmin_azmin = 0.0
    el_rmin_azmax = 0.0
    el_rmax_azmin = 0.0
    el_rmax_azmax = 0.0
    el_max = 0.0
    el_max_prev = 0.0

    kx_rmin_azmin = 0
    kx_rmin_azmax = 0
    kx_rmax_azmin = 0
    kx_rmax_azmax = 0
    ky_rmin_azmin = 0
    ky_rmin_azmax = 0
    ky_rmax_azmin = 0
    ky_rmax_azmax = 0

    radkx = int(np.round((rad_x - DEM_xmin) / DEM_res))
    radky = int(np.round((rad_y - DEM_ymin) / DEM_res))
    for kx in range(radkx - 1, radkx + 2):
        for ky in range(radky - 1, radky + 2):
            visib[ky, kx] = 100
            minviselev[ky, kx] = elmap[ky, kx]

    az_ = np.arange(0, 360 + daz, daz)

    for azind in range(len(az_)):
        if verbose:
            logging.info(f"Computing azimuth {az_[azind]:2.1f}")
        az = az_[azind]
        azmin = az - daz / 2.0
        azmax = az + daz / 2.0
        if azmin < 0:
            azmin = 360.0 + azmin
            indseta = np.logical_or(
                np.logical_and(azmap >= 0, azmap < azmax),
                np.logical_and(azmap >= azmin, azmap <= 360.0),
            )
        elif azmax > 360:
            azmax = azmax - 360.0
            indseta = np.logical_or(
                np.logical_and(azmap >= azmin, azmap <= 360),
                np.logical_and(azmap >= 0, azmap < azmax),
            )
        else:
            indseta = np.logical_and(azmap >= azmin, azmap < azmax)

        azmin_sin = np.sin(azmin * np.pi / 180.0)
        azmax_sin = np.sin(azmax * np.pi / 180.0)
        azmin_cos = np.cos(azmin * np.pi / 180.0)
        azmax_cos = np.cos(azmax * np.pi / 180.0)

        indseta = np.where(indseta)

        el_max_prev = -90
        for rind in range(nr):
            rr = rr_start + rind * dr

            rrmin = rr - dr / 2
            rrmax = rr + dr / 2

            if np.any(indseta):
                indsetr = np.logical_and(rmap[indseta] >= rrmin, rmap[indseta] < rrmax)

                indsetr = tuple(
                    [indseta[0][indsetr], indseta[1][indsetr]]
                )  # Cells to set

                kx_rmin_azmin = radkx + int(round((rrmin * azmin_sin) / DEM_res))
                kx_rmin_azmax = radkx + int(round((rrmin * azmax_sin) / DEM_res))
                kx_rmax_azmin = radkx + int(round((rrmax * azmin_sin) / DEM_res))
                kx_rmax_azmax = radkx + int(round((rrmax * azmax_sin) / DEM_res))

                ky_rmin_azmin = radky + int(round((rrmin * azmin_cos) / DEM_res))
                ky_rmin_azmax = radky + int(round((rrmin * azmax_cos) / DEM_res))
                ky_rmax_azmin = radky + int(round((rrmax * azmin_cos) / DEM_res))
                ky_rmax_azmax = radky + int(round((rrmax * azmax_cos) / DEM_res))

                el_rmin_azmin = -90.0
                el_rmin_azmax = -90.0
                el_rmax_azmin = -90.0
                el_rmax_azmax = -90.0

                if (
                    (kx_rmin_azmin >= 0)
                    and (kx_rmin_azmin < ncols)
                    and (ky_rmin_azmin >= 0)
                    and (ky_rmin_azmin < nrows)
                ):
                    el_rmin_azmin = elmap[ky_rmin_azmin, kx_rmin_azmin]
                if (
                    (kx_rmin_azmax >= 0)
                    and (kx_rmin_azmax < ncols)
                    and (ky_rmin_azmax >= 0)
                    and (ky_rmin_azmax < nrows)
                ):
                    el_rmin_azmin = elmap[ky_rmin_azmax, kx_rmin_azmax]
                if (
                    (kx_rmax_azmin >= 0)
                    and (kx_rmax_azmin < ncols)
                    and (ky_rmax_azmin >= 0)
                    and (ky_rmax_azmin < nrows)
                ):
                    el_rmin_azmin = elmap[ky_rmax_azmin, kx_rmax_azmin]
                if (
                    (kx_rmax_azmax >= 0)
                    and (kx_rmax_azmax < ncols)
                    and (ky_rmax_azmax >= 0)
                    and (ky_rmax_azmax < nrows)
                ):
                    el_rmin_azmin = elmap[ky_rmax_azmax, kx_rmax_azmax]

                el_max = max(
                    [el_rmin_azmin, el_rmin_azmax, el_rmax_azmin, el_rmax_azmax]
                )

                if np.any(indsetr):
                    el_max = max([el_max, np.max(elmap[indsetr])])
                    if el_max >= el_max_prev:
                        visib[indsetr] = 100
                        minviselev[indsetr] = el_max
                    else:
                        minviselev[indsetr] = el_max_prev

                el_max = max([el_max, el_max_prev])
                el_max_prev = el_max

    return visib, minviselev


def visibility_angle(
    minviselmap,
    azmap,
    rmap,
    rpol,
    azpol,
    elpol,
    DEM_res,
    DEM_xmin,
    DEM_ymin,
    rad_x,
    rad_y,
    beamwidth,
    pulsewidth,
    az_conv=0,
    raster_oversampling=1,
    verbose=True,
):
    """
    Computes the radar visibility in polar coordinates

    Parameters
    ----------
    minviselmap : array
        Cartesian array with minimum visible elevation angles
    azmap: array
        Cartesian array with azimuth angles
    rmap : array
        Cartesian array with distances from radar
    rpol : array
        Radar ranges at which to compute the RCS in polar coordinates
    azpol : array
        Radar azimuths at which to compute the RCS in polar coordinates
    elpol : array
        Radar elevations at which to compute the RCS in polar coordinates
    DEM_res : int
        Resolution of the DEM grid
    DEM_xmin : float
        minimum x coord of the DEM (W-E)
    DEM_ymin : float
        minimum y coord of the DEM (S-N)
    rad_x : float
        Radar position x coordinate
    rad y : float
        Radar position y coordinate
    beamwidth : float
        Antenna 3dB beamwidth in deg
    pulsewidth : float
        Radar pulsewidth in sec
    az_conv : float, optional
        If set, assumes that the antenna moves in azimuth direction
        (PPI) and averages over the angle given by this keyword [deg].
    raster_oversampling : int (optional)
        The raster resolution of the DEM should be smaller than
        the range resolution of the radar (defined by the pulse length).
        If this is not the case, this keyword can be set to increase the
        raster resolution. The values for the elevation, sigma naught,
        visibility are repeated. The other values are recalculated.
        Values for raster_oversampling:
        0 or undefined: No oversampling is done
        1: Oversampling is done. The factor N is automatically calculated
        such that 2*dx/N < pulse length
        2 or larger: Oversampling is done with this value as N
    verbose : bool, optional
        If true, will print info about current progress

    Returns
    -------
    vispol : array
        2D Array of RCS in polar coordinates in PyART format, with all az
        angles for all elevations stacked
    """
    nrows, ncols = minviselmap.shape

    pulselength = pulsewidth * 3.0e8 / 2.0  # [m]
    az_conv_offset = az_conv / 2.0

    if raster_oversampling == 0:
        N = 1
    elif raster_oversampling == 1:
        N = int(np.ceil(2 * DEM_res / pulselength))
    else:
        N = raster_oversampling

    if N != 1:
        # New dimensions
        nc = N * ncols
        nr = N * nrows

        # repeat the values NxN, equivalent of rebin in IDL
        minvisvals = np.repeat(np.repeat(minviselmap, N, axis=0), N, axis=1)

        xvec = np.arange(nc) * DEM_res / N + DEM_xmin
        yvec = np.arange(nr) * DEM_res / N + DEM_ymin

        xdiff = xvec - rad_x
        ydiff = yvec - rad_y

        # New distance from radar map
        X, Y = np.meshgrid(xdiff, ydiff)
        rvals = np.sqrt(X**2 + Y**2)

        # New azimuth map
        azmap_rad = (np.arctan2(X, Y) + 2 * np.pi) % (2 * np.pi)
        azvals = azmap_rad * 180.0 / np.pi
    else:
        rvals = rmap
        azvals = azmap
        minvisvals = minviselmap

    """
    Define the area around a point P(range, azimuth) where the cells
    have a contribution to the visibility. This area is defined with the
    range limits from range-dr/2 to range+dr/2 and the
    azimuth limits from azimuth-daz_offset to azimuth+daz_offset.

    For a Gaussian antenna, azimuth offset more than 2*HPBW does not a
    have remarkable contribution.
    With a rectangular pulse and a matched filter cells farer away
    than pulse length does not a have remarkable contribution.
    """

    daz_offset = (2.0 * beamwidth) + az_conv_offset  # [deg]
    del_offset = 2.0 * beamwidth  # [deg]

    delta_deg = 0.1  # [deg]
    ndaz = int(2 * daz_offset / delta_deg)
    ndel = int(2 * del_offset / delta_deg)

    daz_vec = np.arange(ndaz + 1) * delta_deg - daz_offset  # [deg]
    del_vec = np.arange(ndel + 1) * delta_deg - del_offset  # [deg]

    daz_area_antenna, del_area_antenna = np.meshgrid(daz_vec, del_vec)

    # Get the two-way weighting factor due to the azimuth and elevation offsets
    # to the main antenna direction (assuming a Gaussian antenna pattern).
    ant_weight = antenna_pattern_gauss(
        daz_area_antenna, del_area_antenna, beamwidth, az_conv=az_conv, units="deg"
    )

    ant_weight_total = np.nansum(ant_weight)

    azpol_unique = np.unique(azpol)
    nazim = len(azpol_unique)
    nrange = len(rpol)
    range_resolution = rpol[1] - rpol[0]

    # pyart storage format: 2D arrays (naz * nel, nranges)
    vispol = np.zeros((nazim * len(elpol), nrange))
    # Inside the loops over range (rr) and azimuth (az), the
    # coordinates (rr, az) describe the point P(rr, az) for which
    # the visibility is calculated. If more than one DEM cell is within
    # the area from az-daz/2 to az+daz/2 and from rr-dr/2 to
    # rr+dr/2, the calculated visibility value is set to all of these
    # cells (next neighbor).
    for iaz in range(nazim):
        logging.info(f"Computing azimuth {azpol_unique[iaz]:2.1f}")
        # Get azimuth values to explore
        azmin = azpol_unique[iaz] - daz_offset
        azmax = azpol_unique[iaz] + daz_offset
        if azmin < 0:
            azmin = 360.0 + azmin
            indaz = np.logical_or(
                np.logical_and(azvals >= 0, azvals < azmax),
                np.logical_and(azvals >= azmin, azvals <= 360.0),
            )

        elif azmax > 360:
            azmax = azmax - 360.0
            indaz = np.logical_or(
                np.logical_and(azvals >= azmin, azvals <= 360),
                np.logical_and(azvals >= 0, azvals < azmax),
            )
        else:
            indaz = np.logical_and(azvals >= azmin, azvals < azmax)

        if not np.any(indaz):
            logging.warning(f"Visibility for azim {azpol[iaz]:f} not known")
            continue

        indaz = np.where(indaz)

        # These declaration avoids reindexing at every loop turn
        # and save a lot of time
        rvals_indaz = rvals[indaz]

        for iel, el in enumerate(elpol):
            # range bins (center pulse) [m]rvec[ir]-dr/2
            rmin_ground = rpol * np.cos(el * np.pi / 180.0)
            # range bins (center pulse) [m]rvec[ir]+dr/2
            rmax_ground = (rpol + range_resolution) * np.cos(el * np.pi / 180.0)

            for ir in range(0, nrange):
                # Get range values to explore
                indr = np.logical_and(
                    rvals_indaz >= rmin_ground[ir], rvals_indaz < rmax_ground[ir]
                )

                if not np.any(indr):
                    logging.warning(
                        f"Visibility for az {azpol_unique[iaz]:f} deg and range {rpol[ir]:f} not known"
                    )
                    vispol[iaz + iel * nazim, ir] = 100.0
                    continue

                # DEM cells that contribute to the cells to set indset
                indcells = tuple([indaz[0][indr], indaz[1][indr]])

                ind = minvisvals[indcells] < el - del_offset
                if np.all(ind):  # radar beam completely above min vis ang
                    if ir > 0:
                        if vispol[iaz + iel * nazim, ir - 1] == 100:
                            vispol[iaz + iel * nazim, ir] = 100
                        else:
                            vispol[iaz + iel * nazim, ir] = vispol[
                                iaz + iel * nazim, ir - 1
                            ]
                    else:
                        vispol[iaz + iel * nazim, ir] = 100.0
                    continue

                ind = minvisvals[indcells] > el + del_offset
                if np.all(ind):  # radar beam completely below min vis ang
                    continue

                # Calculate offsets in azimuth to the point P(rr,az)
                daz_area = azvals[indcells] - azpol_unique[iaz]

                ind = daz_area > 180.0
                daz_area[ind] = daz_area[ind] - 360.0
                ind = daz_area < -180.0
                daz_area[ind] = daz_area[ind] + 360.0

                vis = 0.0
                ind_rzero = rvals[indcells] <= 0.0

                if not np.any(ind_rzero):
                    vis = vis_weighting(
                        daz_vec, del_vec, daz_area, ant_weight, minvisvals[indcells], el
                    )

                    vis = (vis / ant_weight_total) * 100.0

                # Set vis to all values inside the set area.
                if ir > 0:
                    if vispol[iaz + iel * nazim, ir - 1] < vis:
                        vispol[iaz + iel * nazim, ir] = vispol.data[
                            iaz + iel * nazim, ir - 1
                        ]
                    else:
                        vispol[iaz + iel * nazim, ir] = vis
                else:
                    vispol[iaz + iel * nazim, ir] = vis

    # Correctly map the vispol to the actual azimuth angles
    vispol_remapped = []
    for i, az in enumerate(azpol):
        idx_az = np.searchsorted(azpol_unique, az)
        vispol_remapped.extend(vispol[nazim * i + idx_az, :])
    return vispol_remapped
