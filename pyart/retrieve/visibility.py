"""
pyart.retrieve.visibility
==================

Functions for visibility and ground echoes estimation from a DEM.

"""

import numpy as np
import warnings

from itertools import product, repeat
from scipy.interpolate import interp1d
from functools import partial
from pyproj import Transformer, Proj
import multiprocessing as mp


from ..config import get_field_name, get_metadata, get_fillvalue
from ..core import antenna_vectors_to_cartesian as rad_to_cart

# These 2 are non picklable and need to defined as globals
nngrid = 0
transformer = 0

# Define worker function for parallelization on azimuth angles
def _worker_function(angles, ranges, dem_data, radar_params, 
                     numeric_params, atm_params):
    
    print(angles)
    rad_alt = radar_params['radar_altitude']
    freq = radar_params['freq']
    power = radar_params['power']
    gain = radar_params['gain']
    loss = radar_params['loss'] 
    beamwidth = radar_params['beamwidth']
    tau = radar_params['tau']
    
    atm_att = atm_params['atm_att']
    ke = atm_params['ke']
    mosotti = atm_params['mosotti']
    
    grid_res = numeric_params['grid_res']
    quad_pts_range = numeric_params['quad_pts_range']
    quad_pts_GH = numeric_params['GH_pts']
    GH_weights = numeric_params['GH_weights']
    sigma_0_method = numeric_params['sigma_0_method']
    
    # Get original length of the ray, before oversampling (see below)
    original_range_length = int(len(ranges) / quad_pts_range)
    visib_integ = np.ma.zeros((original_range_length,)) 
    rcs_integ = np.ma.zeros((original_range_length,)) 
    clutter_dBm_integ = np.ma.zeros((original_range_length,)) 
    clutter_dBZ_integ = np.ma.zeros((original_range_length,)) 
    
    i = 0
    for ph, pv in quad_pts_GH: # Loop on quadrature points
        # Get radar Cartesian coords at quadrature pt elevation and azimuth
        xr, yr, zr = rad_to_cart(ranges, angles[0] + ph, angles[1] + pv, 
                                 ke = ke)
        # Add radar altitude to z coordinates
        zr = zr[0] + rad_alt
        
        # Project to DEM coordinates and get corresponding grid indexes
        xr_proj, yr_proj = transformer.transform(xr, yr)
        i_idx,j_idx = nngrid.get_idx(xr_proj, yr_proj)
        
        
        # Create ray and compute relevant variabls
        ray = _Ray(angles[0], ranges, i_idx, j_idx, dem_data)
        ray.calc_dem_bent(dem_data, zr)
        ray.calc_slope_and_aspect(dem_data, grid_res)
        ray.calc_visibility()
        if ray.topography_encounter_idx != None:
            ray.calc_theta(onlytopo = True)
            ray.calc_incident_ang(onlytopo = True)
            ray.calc_eff_area(grid_res, onlytopo = True)
            ray.calc_sigma_0(sigma_0_method, freq, onlytopo = True)
            ray.calc_rcs()
            ray.calc_clutter_mW(power, gain, loss, freq, atm_att)
            ray.calc_clutter_Z(freq, mosotti, tau, beamwidth)
 
        # aggregate
        visib = np.nanmean(np.reshape(ray.visibility,
                                      (original_range_length, 
                                       quad_pts_range)), 
                                       axis=1)
        rcs = np.nanmean(np.reshape(ray.rcs,
                                      (original_range_length, 
                                       quad_pts_range)), 
                                       axis=1)
        clutter_W = np.nanmean(np.reshape(ray.clutter_W,
                                      (original_range_length, 
                                       quad_pts_range)), 
                                       axis=1)
        clutter_Z = np.nanmean(np.reshape(ray.clutter_Z,
                                      (original_range_length, 
                                       quad_pts_range)), 
                                       axis=1)
         
        # sum up with quadrature GH_weights
        visib_integ += visib * GH_weights[i]
        rcs_integ += rcs * GH_weights[i]
        clutter_dBm_integ += clutter_W * GH_weights[i]
        clutter_dBZ_integ += clutter_Z * GH_weights[i]
        i += 1
    
    clutter_dBm_integ = 10 * np.log10(clutter_dBm_integ) # Convert mW -> dBm
    clutter_dBZ_integ = 10 * np.log10(clutter_dBZ_integ) # Convert mm6m-3 -> dBZ
    return(visib_integ, rcs_integ, clutter_dBm_integ, clutter_dBZ_integ)

class _IndexNNGrid(object):
    """
    A class for fast interpolation in a 2D grid. It is able to find the closest
    point in a 2D grid for an arbitrary set of coordinates.

    Attributes
    ----------
    interp_i : Scipy 1D Interpolator
        Returns the i index (rows) of the closest point in the grid for a
        given x coordinate.
    interp_j : Scipy 1D Interpolator
        Returns the j index (rows) of the closest point in the grid for a
        given y coordinate.
    """
    
    def __init__(self, x, y):
        """Creates the interpolators from the Grid coordinates x and y"""
        I = np.arange(len(x))
        J = np.arange(len(y))
        self.interp_i = interp1d(x, I, bounds_error = False, kind = 'nearest') 
        self.interp_j = interp1d(y, J, bounds_error = False, kind = 'nearest') 
        
    def get_idx(self, xi, yi):
        """Returns the indexes of the closest grid points for a set of x 
        and y coordinates """
        i = self.interp_i(xi).astype(int)[0]
        j = self.interp_j(yi).astype(int)[0]
        i = np.ma.array(i, mask = i < 0)
        j = np.ma.array(j, mask = j < 0)
        return i, j    

class _Ray(object): 
    """
    A class that computes visibility variables on a single radar ray

    Attributes
    ----------
    i_idx : array
        Row indexes of all ray coordinates in the DEM grid.
    j_idx : array
        Column indexes of all ray coordinates in the DEM grid.
    valid_idx : bool array
        Boolean array that gives which ray indexes are valid (i.e. with
        a valid associated DEM value).
    i_idx_valid : array
        Contains only the valid row indexes.
    j_idx_valid : array
        Contains only the valid column indexes.
    dem : array
        Terrain altitude along ray
    dem_bent : array
        Terrain altitude corrected for radar beam height along ray
    slope : array
        Terrain slope along ray
    aspect : array
        Terrain aspect along ray
    theta : array
        Incidence elevation angle at terrain altitude along ray
    min_vis_theta : array
        Minimum elevation angle with sufficient visibility along ray
    min_vis_alt : array
        Minimum altitude with sufficient visibility along ray
    visib : array
        Radar visibility along ray ( 1 or 0)
    topography_encounter_idx : array
        Index along ray where the beam hits the topography
    """
    
    def __init__(self, az, ranges, i_idx, j_idx, dem_data):
        """
        Initializes the class
        
        Parameters
        ----------
        az : float
            Azimuth angle of that ray
        ranges : array
            Ranges in meters to all gates in the ray
        i_idx : array
            Row indexes of all ray coordinates in the DEM grid, as obtained
            with the _IndexNNGrid class.
        j_idx : array
            Column indexes of all ray coordinates in the DEM grid, as obtained
            with the _IndexNNGrid class.
        dem_data : array
            2D array that contains the DEM terrain altitude data (the grid
            for which to use i_idx, j_idx).
        """
        self.az = az
        self.ranges = ranges
        self.ranges_log = np.log10(ranges)
        
        self.i_idx = i_idx
        self.j_idx = j_idx
        self.calc_valid_gates(dem_data)
        
        self.i_idx_valid = self.i_idx[self.valid_idx]
        self.j_idx_valid = self.j_idx[self.valid_idx]
        self.N = len(self.valid_idx) # Nb of range gates
        
        self.dem_bent = np.zeros((self.N))
        self.slope = np.zeros((self.N))
        self.aspect = np.zeros((self.N))
        self.theta = np.zeros((self.N))
        self.min_vis_theta = np.zeros((self.N))
        self.min_vis_alt = np.zeros((self.N))
        self.area_eff = np.zeros((self.N))
        self.incident_ang = np.zeros((self.N))
        self.rcs = np.zeros((self.N))
        self.clutter_Z = np.zeros((self.N))
        self.clutter_W = np.zeros((self.N))
        self.sigma_0 = np.zeros((self.N))
        
    def calc_valid_gates(self, dem_data):
        """
        Calculates the valid gates of the ray 
        
        Parameters
        ----------
        dem_data : array
            2D array that contains the DEM terrain altitude data (the grid
            for which to use i_idx, j_idx)
        """
        # valid gates are those were i and j indexes are defined and DEM
        # data is available (no hole in DEM grid)
        valid = np.logical_and(~self.i_idx.mask, ~self.j_idx.mask)
        valid[self.i_idx == dem_data.shape[0] - 1] = False
        valid[self.j_idx == dem_data.shape[1] - 1] = False
        valid[self.i_idx == 0] = False
        valid[self.j_idx == 0] = False
        valid[valid] = ~dem_data[self.i_idx[valid], self.j_idx[valid]].mask
        
        self.valid_idx = valid
        
    def calc_dem_bent(self, dem_data, alt):
        """
        Bends the DEM data along the ray (i.e. subtracts the altitude of
        the radar gates from the terrain altitude).
        
        Parameters
        ----------        
        alt : array
            Altitude of all radar gates along the ray
        """
        self.dem = np.ma.zeros((self.N))
        self.dem[self.valid_idx] = (dem_data[self.i_idx_valid, 
                                             self.j_idx_valid])
        self.dem.mask = ~self.valid_idx
        self.dem_bent = self.dem - alt

        
    def calc_slope_and_aspect(self, dem_data, grid_res, onlytopo = False):
        """
        Calculates the slope and aspect from the DEM data using Sobel filters
        to compute the gradient in X and Y directions.
        
        Parameters
        ----------
        dem_data : array
            2D array that contains the DEM terrain altitude data (the grid
            for which to use i_idx, j_idx)
        
        grid_res : int or float
            Spatial resolution of the grid on which the DEM is defined, for
            example 25 meters.
        
        idx_along_ray : array, optional
            Array of 1D indexes along ray where to compute the slope and aspect
            if not provided it will be computed all along the ray
        """
        # simplify notation
        i = self.i_idx
        j = self.j_idx
        valid = self.valid_idx
        if onlytopo:
            i = i[self.topography_encounter_idx]
            j = j[self.topography_encounter_idx]
            valid = valid[self.topography_encounter_idx]
            
        d = dem_data
        
        # Sobel filter in Y direction
        gy = np.ma.zeros((self.N)) + np.nan
        gy[self.valid_idx] = (d[i[valid]-1,j[valid]-1] + 
                   2 * d[i[valid],j[valid]-1] + 
                   d[i[valid]+1,j[valid]-1] -
                   d[i[valid]-1,j[valid]+1] -
                   2 * d[i[valid],j[valid]+1] -
                   d[i[valid]+1,j[valid]+1])/(8 * grid_res)
        
        # Sobel filter in X direction
        gx = np.ma.zeros((self.N)) + np.nan
        gx[valid] = (d[i[valid]+1,j[valid]-1] + 
                   2 * d[i[valid]+1,j[valid]] + 
                   d[i[valid]+1,j[valid]+1] -
                   d[i[valid]-1,j[valid]-1] -
                   2 * d[i[valid]-1,j[valid]] -
                   d[i[valid]-1,j[valid]+1])/(8 * grid_res)
        
        slope = np.ma.zeros((self.N)) 
        slope[valid] = np.arctan(np.sqrt(gy[valid]**2 + 
                                         gx[valid]**2)) * 180 / np.pi
        aspect = np.ma.zeros((self.N)) + np.nan
        aspect[valid] = (np.arctan2(gy[valid],
                                    -gx[valid]) + np.pi) * 180 / np.pi
        
        slope.mask = ~valid
        aspect.mask = ~valid
        self.slope = slope
        self.aspect = aspect
        
    def calc_theta(self, onlytopo = False):
        """
        Calculates the theta angle, i.e. the elevation angle at which the radar
        ray touches the topography
        
        Parameters
        ----------
        ranges : array
            range of every gate from the radar.
        
        """
    
        if not hasattr(self, 'dem_bent'):
            warnings.warn('Please use bend_dem() function first...')
            return None

        i = self.i_idx
        j = self.j_idx
        valid = self.valid_idx
        if onlytopo:
            i = np.array([i[self.topography_encounter_idx]])
            j = np.array([j[self.topography_encounter_idx]])
            valid = np.array([self.topography_encounter_idx])
            
        theta = np.ma.zeros((self.N))
        theta[valid] = (np.arctan2(self.dem_bent[valid], self.ranges[valid]) *
                                180 / np.pi)
        self.theta = theta
    
    def calc_min_theta(self, beamwidth):
        """
        Calculates the minimum theta angle, at which there is sufficient 
        visibility. 
        """
        if not hasattr(self, 'theta'):
           warnings.warn('Please use calc_theta() function first...')
           return None
       
        min_vis_theta = np.zeros((self.N)) + np.nan 
        current = self.theta[0]
        for i in range(1, self.N):
            if np.isnan(self.theta[i]):
                min_vis_theta[i] = np.nan
            elif self.theta[i] > current:
                current = self.theta[i]
        min_vis_theta[i] = current + beamwidth / 2.
        
        self.min_vis_theta = min_vis_theta

        
    def calc_min_vis_alt(self, ke):
        """
        Calculates the minimum visible altitude along the ray
        
        Parameters
        ----------
        ranges : array
            range of every gate from the radar.
         beamwidth : float, optional
            3dB beamwidth of the antenna pattern.
        ke : float, optional
            Effective radius scale factor 
        
        """

        if not hasattr(self, 'min_vis_theta'):
            warnings.warn('Please use calc_min_theta() function first...')
            return None
        
        # elevation angle in radians.
        theta_e = (self.min_vis_theta) * np.pi / 180.0   
        R = 6371.0 * 1000.0 * ke     # effective radius of earth in meters.
    
        self.min_vis_alt = (self.ranges ** 2 + R ** 2 + 2.0 * self.ranges * R *
                                    np.sin(theta_e)) ** 0.5 - R

    def calc_visibility(self):
        """
        Calculates the visibility along the ray : all gates after first 
        encounter with topography have zero visibility, as well as the index
        along the ray where the beam first "hits" the topography
        """
        if not hasattr(self, 'dem_bent'):
            warnings.warn('Please use bend_dem() function first...')
            return None
        
        visib = np.ma.ones((self.N))
        idx_pos = np.where(self.dem_bent > 0)[0]
        if  len(idx_pos):
            visib[idx_pos[0]:] = 0
            self.topography_encounter_idx = idx_pos[0]
        else:
            self.topography_encounter_idx = None
            
        visib.mask = self.dem_bent.mask

        self.visibility = visib

        
    def calc_incident_ang(self, onlytopo = False):
        if not hasattr(self, 'theta'):
            warnings.warn('Please use calc_theta() function first...')
            return None
        if not hasattr(self, 'slope'):
            warnings.warn('Please use calc_slope_aspect() function first...')
            return None

        i = self.i_idx
        j = self.j_idx
        valid = self.valid_idx
        if onlytopo:
            i = np.array([i[self.topography_encounter_idx]])
            j = np.array([j[self.topography_encounter_idx]])
            valid = np.array([self.topography_encounter_idx])
            
        slope  = self.slope[valid] * np.pi / 180.0
        aspect = self.aspect[valid] * np.pi / 180.0
        zenith = (90. - self.theta[valid]) * np.pi / 180.0 
        az     = self.az * np.pi / 180.0 
        
        inc_ang = np.zeros((self.N))
        inc_ang[valid] = np.arccos(-( np.sin(slope) * np.sin(zenith) * 
             (np.sin(aspect) * np.sin(az) + np.cos(aspect) * np.cos(az)) + 
                     np.cos(slope) * np.cos(zenith))) * 180 / np.pi
        self.incident_ang = inc_ang

        
    def calc_eff_area(self, grid_res, onlytopo = False):
        if not hasattr(self, 'slope'):
            warnings.warn('Please use calc_slope_aspect() function first...')
            return None
        i = self.i_idx
        j = self.j_idx
        valid = self.valid_idx
        if onlytopo:
            i = np.array([i[self.topography_encounter_idx]])
            j = np.array([j[self.topography_encounter_idx]])
            valid = np.array([self.topography_encounter_idx])
            
        area_eff = np.zeros((self.N))
        area_eff[valid] = grid_res**2 / np.cos(self.slope[valid] *
                                                np.pi / 180.0)
        self.area_eff = area_eff
        
    def calc_rcs(self):
        if not hasattr(self, 'area_eff'):
            warnings.warn('Please use calc_area_eff() function first...')
            return None
        if not hasattr(self, 'sigma_0'):
            warnings.warn('Please use calc_area_eff() function first...')
            return None
        
        self.rcs = self.area_eff * self.sigma_0
        
    def calc_clutter_mW(self, freq, Pt, G, L, atm_att):
        if not hasattr(self, 'rcs'):
            warnings.warn('Please use calc_rcs() function first...')
        lambd = 3. / (freq * 10.) 
        pconst = (10 * np.log10(Pt) + 2 * G + 20 * np.log10(lambd) - L  - 
                  30 * np.log10(4 * np.pi))
       
        self.clutter_mW = 10**(0.1 * (pconst + 40 * self.ranges_log - 
                            2 * atm_att * self.ranges / 1000. + 
                            10 * np.log10(self.rcs)))

        
    def calc_clutter_Z(self, freq, mosotti, tau, beamwidth):
        if not hasattr(self, 'rcs'):
            warnings.warn('Please use calc_rcs() function first...')
        lambd = 3. / (freq * 10.) 
        dbzconst = (10 * np.log10(2) + 40 * np.log10(lambd) -
                   10 * np.log10(tau * 3e8) - 20 *np.log10(beamwidth) - 
                   60 * np.log10(np.pi) - 20 * np.log10(mosotti))
       
        convert_dbzm_to_dbz = 180. # 10*log10(1 m^6 / 1 mm^6) = 180

        self.clutter_Z = (self.rcs - 20 * self.ranges_log + dbzconst + 
                          convert_dbzm_to_dbz)

    def calc_sigma_0(self, method = 'gabella', freq_ghz = 5.6,
                     onlytopo = False):
        """
        Calculates sigma_0: the ratio between the backscattering cross section 
        and the physical backscattering area
    
        Parameters
        ----------
        method : str, optional
            Method to use in the calculation of sigma_0, either "gabella" or
            "delrieu"
        freq_ghz : float, optional
            Frequency of the radar in GHz, is used only for the Delrieu method
      
        """
        if not hasattr(self, 'incident_ang'):
            warnings.warn('Please use calc_incident_ang() function first...')
            return None
        
        i = self.i_idx
        j = self.j_idx
        valid = self.valid_idx
        if onlytopo:
            i = np.array([i[self.topography_encounter_idx]])
            j = np.array([j[self.topography_encounter_idx]])
            valid = np.array([self.topography_encounter_idx])
            
        sigma_0 = np.zeros((self.N))[valid]
        inc_ang = self.incident_ang[valid]
        inc_angr = inc_ang * np.pi / 180.0
        if method == 'gabella':
            # GABELLA/PERONA 1997 (GECS)
            te1 = 80.     # Maximum Angle of "Plateau" Region      [deg]
            te2 = 87.     # Dimming Start Angle at Grazing Angles  [deg]
            te3 = 88.     # Border Incidence Angle                 [deg]
            k1 = 0.1      # Offset Index
            k2 = 1.       # Dimming Index near Grazing Angles
            bbdeg = 2.75  # Tuning Angle in Exp Rising above Border Angle [deg]

            te1r = te1 * np.pi / 180.0
            te2r = te2 * np.pi / 180.0
            te3r = te3 * np.pi / 180.0
            bbdegr = bbdeg * np.pi / 180.0

            factor =  k1 * np.cos(te2r) * ((np.pi / 2. - te2r) / 
                                           (np.pi / 2.-te1r) ) ** k2
            
            ind0 = np.where(inc_ang <= te1)[0]
            if len(ind0):
                sigma_0[ind0] = k1 * np.cos(inc_angr[ind0])
            
            ind1 = np.where(np.logical_and(inc_ang > te1, inc_ang <= te2))[0]
            if len(ind1):
                iang = inc_angr[ind1]
                sigma_0[ind1] =  k1 * np.cos(iang) * ((np.pi / 2.
                                                              - iang) / 
                                               (np.pi / 2.-te1r) ) ** k2

            ind2 = np.where(np.logical_and(inc_ang > te2, inc_ang <= te3))[0]
            if len(ind2):
                sigma_0[ind2] = factor

            ind3 = np.where(np.logical_and(inc_ang > te3, inc_ang <= 90))[0]
            if len(ind3):
                iang = inc_angr[ind3]
                sigma_0[ind3] = factor * np.exp((iang - te3r) / bbdegr)

            ind4 = np.where(inc_ang > 90)
            if len(ind4):
                sigma_0[ind4] = factor * np.exp((np.pi / 2. - te3r) 
                                                       / bbdegr)
            
        elif method == 'delrieu':
            # DELRIEU 1995 (X-BAND) / SKOLNIK 1990
            lim_ang_del = 63.75  # Borderline Angle separating curve branche
            # First Branch  [0..lim_ang_del]
            a1 = -9.1    # Skolnik's Model "a1" parameter [dB]
            b1 = -0.12   # Skolnik's Model "b1" parameter [dB/deg]
            c1 = 0.25    # Skolnik's Model "c1" parameter [dB/GHz]
            d1 = 0.      # Skolnik's Model "d1" parameter [dB/(deg*GHz))]
            # Second Branch  [lim_ang_del..90]
            a1_2 = 12.93 # Skolnik's Model "a1" parameter [dB]
            b1_2 = -0.37 # Skolnik's Model "b1" parameter [dB/deg]
            c1_2 = 0.    # Skolnik's Model "c1" parameter [dB/GHz]
            d1_2 = 0.    # Skolnik's Model "d1" parameter [dB/(deg*GHz))]

            sigma_db = np.zeros((self.N))[valid]

            ind = np.where( inc_ang < lim_ang_del)[0]
            if len(ind):
                iang = inc_angr[ind]
                sigma_db[ind] = (a1 + b1 * iang + c1 * freq_ghz +
                                 d1 * freq_ghz * iang)
            ind = np.where( inc_ang >= lim_ang_del)[0]
            if len(ind):
               iang = inc_angr[ind]
               sigma_db[ind] = (a1_2 + b1_2 * iang + c1_2 * freq_ghz +
                                 d1_2 * freq_ghz * iang) 

            sigma_0 = 10 ** (sigma_db / 10.)
        else:
            warnings.warn('Invalid method for sigma_0: use "gabella" or '+\
                          '"delrieu"')
            return
        
        self.sigma_0[valid] = sigma_0
    
    
        
def _GH_quadrature(npts_el, npts_az, beamwidth):
    """
    Retrieves the Gauss-Hermite quadrature points and GH_weights for Gaussian
    antenna integration in azimuthal and elevational directions.

    Parameters
    ----------
    npts_el : int
        Number of quadrature points in the elevational direction.
    npts_az : int
        Number of quadrature points in the azimuthal direction.        
    beamwidth : float, optional
        3dB beamwidth of the antenna pattern.

        
    Returns
    -------
    quad_pts : array
        Quadrature points as an N x 2 array that contains the azimuthal
        offset in the first column and the elevational column in the second.
        The number of rows is equal to npts_el x npts_az.
    GH_weights : array
        Normalized quadrature GH_weights of all quadrature points.

    References
    ----------
    Wolfensberger, D. and Berne, A.: From model to radar variables: a new
    forward polarimetric radar operator for COSMO, Atmos. Meas. Tech., 11,
    3883–3916, https://doi.org/10.5194/amt-11-3883-2018, 2018. 
    """
    
    sigma = beamwidth/(2 * np.sqrt(2 * np.log(2)))
    
    pts_az, weights_az = np.polynomial.hermite.hermgauss(npts_az)
    pts_az *= sigma
    
    pts_el, weights_el = np.polynomial.hermite.hermgauss(npts_el)
    pts_el *= sigma
    
    # Create 2D array of GH_weights
    GH_weights = np.outer(weights_az * sigma, weights_el * sigma)
    GH_weights *= np.abs(np.cos(np.deg2rad(pts_el)))
    
    sum_weights = np.sum(GH_weights.ravel())
    GH_weights /= sum_weights # Normalize GH_weights
    
    quad_pts = list(product(pts_az,pts_el))
    GH_weights = GH_weights.ravel()
    
    return quad_pts, GH_weights

    
def visibility_processing(radar, dem_grid,
                          frequency, 
                          beamwidth, 
                          tau,
                          power,
                          gain,
                          loss,
                          fill_value = None, 
                          terrain_altitude_field = None,
                          bent_terrain_altitude_field = None,
                          terrain_slope_field = None,
                          terrain_aspect_field = None,
                          theta_angle_field = None,
                          visibility_field = None,
                          min_vis_altitude_field = None,
                          min_vis_theta_field = None,
                          incident_angle_field = None,
                          sigma_0_field = None,
                          rcs_ground_clutter_field = None,
                          dBm_ground_clutter_field = None,
                          dBZ_ground_clutter_field = None,
                          atm_att = 0.05,
                          mosotti = 0.9644,
                          ke = 4/3., 
                          sigma_0_method = 'gabella',
                          quad_pts_range = 1,
                          quad_pts_az = 9, 
                          quad_pts_el = 9, 
                          parallel = True):
            
    # parse fill value
    if fill_value is None:
        fill_value = get_fillvalue()

    # parse field names
    if terrain_altitude_field is None:
        terrain_altitude_field = get_field_name('terrain_altitude')
    if bent_terrain_altitude_field is None:
        bent_terrain_altitude_field = get_field_name('bent_terrain_altitude')
    if terrain_slope_field is None:
        terrain_slope_field = get_field_name('terrain_slope')
    if terrain_aspect_field is None:
        terrain_aspect_field = get_field_name('terrain_aspect')
    if theta_angle_field is None:
        theta_angle_field = get_field_name('theta_angle')
    if visibility_field is None:
        visibility_field = get_field_name('visibility')
    if min_vis_altitude_field is None:
        min_vis_altitude_field = get_field_name('min_vis_altitude')
    if incident_angle_field is None:
        incident_angle_field = get_field_name('incident_angle')    
    if sigma_0_field is None:
        sigma_0_field = get_field_name('sigma_0')   
    if rcs_ground_clutter_field is None:
        rcs_ground_clutter_field = get_field_name('rcs_ground_clutter')   
    if dBm_ground_clutter_field is None:
        dBm_ground_clutter_field = get_field_name('dBm_ground_clutter')   
    if dBZ_ground_clutter_field is None:
        dBZ_ground_clutter_field = get_field_name('dBZ_ground_clutter')   
        
    # Define aeqd projection for radar local Cartesian coords
    pargs = Proj(proj="aeqd", lat_0 = radar.latitude['data'][0], 
                 lon_0 = radar.longitude['data'][0], datum = "WGS84", 
                 units="m")    
    
    # Define coordinate transform: (local radar Cart coords) -> (DEM coords)
    global transformer
    transformer = Transformer.from_proj(pargs, dem_grid.projection)
    
    # Get quadrature pts and GH_weights in az,el directions
    GH_pts, GH_weights = _GH_quadrature(quad_pts_el, quad_pts_az, beamwidth)
    
    # Create grid interpolator
    global nngrid
    nngrid = _IndexNNGrid(dem_grid.x['data'], dem_grid.y['data'])
    
    # Initialize output fields
    
    nazimuth = len(radar.azimuth['data'])
    ngates = len(radar.range['data'])
    
    visibility = np.ma.zeros([nazimuth, ngates], fill_value = fill_value)
    theta =  np.ma.zeros([nazimuth, ngates], fill_value = fill_value)
    dem_bent = np.ma.zeros([nazimuth, ngates], fill_value = fill_value)
    terrain_slope =  np.ma.zeros([nazimuth, ngates], fill_value = fill_value)
    terrain_aspect =  np.ma.zeros([nazimuth, ngates], fill_value = fill_value)
    min_vis_theta =  np.ma.zeros([nazimuth, ngates], fill_value = fill_value)
    min_vis_altitude =  np.ma.zeros([nazimuth, ngates], fill_value = fill_value)
    incident_angle =  np.ma.zeros([nazimuth, ngates], fill_value = fill_value)
    sigma_0 =  np.ma.zeros([nazimuth, ngates], fill_value = fill_value)
    clutter_rcs =  np.ma.zeros([nazimuth, ngates], fill_value = fill_value)
    clutter_dBZ =  np.ma.zeros([nazimuth, ngates], fill_value = fill_value)
    clutter_dBm =  np.ma.zeros([nazimuth, ngates], fill_value = fill_value)
    ##
    
    
    dem_data = dem_grid.fields[terrain_altitude_field]['data']
    # In the first step we estimate the variables at topography level but
    # only over the radar coordinates        
    for i, az in enumerate(radar.get_azimuth(0)): # Loop on az angles
    
        # Get radar Cartesian coords at elevation 0
        xr, yr, zr = rad_to_cart(radar.range['data'],
                                 az, 0,
                                 ke = ke)
        # Add radar altitude to z coordinates
        zr = zr[0] + radar.altitude['data']
        
        # Project to DEM coordinates and get corresponding grid indexes
        xr_proj, yr_proj = transformer.transform(xr,yr)
        i_idx,j_idx = nngrid.get_idx(xr_proj,yr_proj)
        
        # Create ray and compute relevant variabls
        ray = _Ray(az, radar.range['data'], i_idx, j_idx, dem_data)
        ray.calc_dem_bent(dem_data, zr)
        ray.calc_slope_and_aspect(dem_data, dem_grid.metadata['resolution'])
        ray.calc_theta()
        ray.calc_min_theta(beamwidth)
        ray.calc_min_vis_alt( ke)
        ray.calc_incident_ang()
        ray.calc_sigma_0(freq_ghz = frequency)
        
        theta[i] = ray.theta
        min_vis_theta[i] = ray.min_vis_theta
        min_vis_altitude[i] = ray.min_vis_alt
        dem_bent[i] = ray.dem_bent
        terrain_slope[i] = ray.slope
        terrain_aspect[i] = ray.aspect
        sigma_0[i] = ray.sigma_0
        incident_angle[i] = ray.incident_ang
        
    theta_dic = get_metadata(theta_angle_field)
    theta_dic['data'] = theta
    
    slope_dic = get_metadata(terrain_slope_field)
    slope_dic['data'] = terrain_slope
    
    aspect_dic = get_metadata(terrain_aspect_field)
    aspect_dic['data'] = terrain_aspect
    
    min_vis_theta_dic = get_metadata(min_vis_theta_field)
    min_vis_theta_dic['data'] = min_vis_theta

    min_vis_altitude_dic = get_metadata(min_vis_altitude_field)
    min_vis_altitude_dic['data'] = min_vis_altitude  
    
    bent_terrain_altitude_dic = get_metadata(bent_terrain_altitude_field)
    bent_terrain_altitude_dic['data'] = dem_bent  

    incident_angle_dic = get_metadata(incident_angle_field)
    incident_angle_dic['data'] = incident_angle  
    
    sigma_0_dic = get_metadata(sigma_0_field)
    sigma_0_dic['data'] = sigma_0  
    
    if quad_pts_range >= 3:
        """ 
        Interpolate range array based on how many quadrature points in range
        are wanted (at least 3)
        For example if quad_pts_range = 5 and original range array is 
        [25,75,125] (bin centers), it will give [0,25,50,50,75,100,125,150]
        i.e. bin 0 - 50 (with center 25) is decomposed into ranges [0,25,50]
        """
        ranges = radar.range['data']
        nrange = len(radar.range['data'])
        dr = np.mean(np.diff(radar.range['data'])) # range res
        intervals = np.arange(ranges[0] - dr / 2, dr * nrange + dr / 2, dr)
        range_resampled = []
        for i in range(len(intervals) - 1):
            range_resampled.extend(np.linspace(intervals[i], intervals[i + 1],
                                               quad_pts_range))
    else:
        # No resampling
        range_resampled = radar.range['data']
    print('done')
    
    # create parallel computing instance
    if parallel:
        pool = mp.Pool(processes=mp.cpu_count(), maxtasksperchild=1)
        map_ = pool.map
    else:
        map_ = map
        
    # Create parameter dictionaries for workers
    radar_params = {}
    radar_params['radar_altitude'] = radar.altitude['data']
    radar_params['freq'] = frequency
    radar_params['power'] = power
    radar_params['gain'] = gain
    radar_params['loss'] = loss
    radar_params['beamwidth'] = beamwidth
    radar_params['tau'] = tau
    
    atm_params = {}
    atm_params['atm_att'] = atm_att
    atm_params['ke'] = ke
    atm_params['mosotti'] = mosotti
    
    numeric_params = {}
    numeric_params['grid_res'] = dem_grid.metadata['resolution']
    numeric_params['quad_pts_range'] = quad_pts_range
    numeric_params['GH_pts'] = GH_pts
    numeric_params['GH_weights'] = GH_weights
    numeric_params['sigma_0_method'] = sigma_0_method
    
    # Loop on fixed angles : el for ppi, az for rhi
    idx = 0
    for i, fixangle in enumerate(radar.fixed_angle['data']): 
        # Create partial worker func that takes only angle as input
        if radar.scan_type == 'ppi':
            angles = list((zip(radar.get_azimuth(i), repeat(fixangle))))
        elif radar.scan_type == 'rhi':
            angles = list((zip(repeat(fixangle), radar.get_elevation(i))))
            
        partialworker = partial(_worker_function, 
                                ranges = range_resampled,
                                dem_data = dem_data,
                                radar_params = radar_params,
                                atm_params = atm_params,
                                numeric_params = numeric_params)
   
        results = list(map_(partialworker, angles))
     
        visibility[idx : idx + len(results), :] = np.array([r[0] for r 
                                                            in results])
        clutter_rcs[idx : idx + len(results), :] = np.array([r[1] for r 
                                                            in results])
        clutter_dBm[idx : idx + len(results), :] = np.array([r[2] for r
                                                            in results])
        clutter_dBZ[idx : idx + len(results), :] = np.array([r[3] for r 
                                                            in results])
        
        idx += len(results)
        
        if parallel:
            pool.close()
       
    visibility_dic = get_metadata(visibility_field)
    visibility_dic['data'] = visibility  

    rcs_ground_clutter_dic = get_metadata(rcs_ground_clutter_field)
    rcs_ground_clutter_dic['data'] = clutter_rcs  
    
    dBm_ground_clutter_dic = get_metadata(dBm_ground_clutter_field)
    dBm_ground_clutter_dic['data'] = clutter_dBm 
    
    dBZ_ground_clutter_dic = get_metadata(dBZ_ground_clutter_field)
    dBZ_ground_clutter_dic['data'] = clutter_dBZ  
    
    return (bent_terrain_altitude_dic, slope_dic, aspect_dic, 
            theta_dic, min_vis_theta_dic, min_vis_altitude_dic,
            visibility_dic, incident_angle_dic, sigma_0_dic,
            rcs_ground_clutter_dic, dBm_ground_clutter_dic,
            dBZ_ground_clutter_dic)
