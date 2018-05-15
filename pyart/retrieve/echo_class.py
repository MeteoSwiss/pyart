"""
pyart.retrieve.echo_class
=========================

Functions for echo classification

.. autosummary::
    :toctree: generated/

    steiner_conv_strat
    melting_layer_giangrande
    hydroclass_semisupervised
    _standardize
    _assign_to_class
    _assign_to_class_scan
    _get_mass_centers
    _mass_centers_table
    _data_limits_table
    get_freq_band

"""

import numpy as np
import sys

from ..config import get_fillvalue, get_field_name, get_metadata
from ..exceptions import MissingOptionalDependency
from ..util.xsect import cross_section_ppi
from ..core.transforms import antenna_vectors_to_cartesian
from scipy.interpolate import RegularGridInterpolator
from copy import deepcopy

try:
    from . import _echo_steiner
    _F90_EXTENSIONS_AVAILABLE = True
except ImportError:
    _F90_EXTENSIONS_AVAILABLE = False

from warnings import warn


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

    Other Parameters
    ----------------
    dx, dy : float
        The x- and y-dimension resolutions in meters, respectively.  If None
        the resolution is determined from the first two axes values.
    intense : float
        The intensity value in dBZ. Grid points with a reflectivity
        value greater or equal to the intensity are automatically
        flagged as convective. See reference for more information.
    work_level : float
        The working level (separation altitude) in meters. This is the height
        at which the partitioning will be done, and should minimize bright band
        contamination. See reference for more information.
    peak_relation : 'default' or 'sgp'
        The peakedness relation. See reference for more information.
    area_relation : 'small', 'medium', 'large', or 'sgp'
        The convective area relation. See reference for more information.
    bkg_rad : float
        The background radius in meters. See reference for more information.
    use_intense : bool
        True to use the intensity criteria.
    fill_value : float
         Missing value used to signify bad data points. A value of None
         will use the default fill value as defined in the Py-ART
         configuration file.
    refl_field : str
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
    # check that Fortran extensions is available
    if not _F90_EXTENSIONS_AVAILABLE:
        raise MissingOptionalDependency(
            "Py-ART must be built on a system with a Fortran compiler to "
            "use the steiner_conv_strat function.")

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
    ze = np.ma.filled(ze, fill_value).astype(np.float64)

    # Call Fortran routine
    eclass = _echo_steiner.classify(
        ze, x, y, z, dx=dx, dy=dy, bkg_rad=bkg_rad, work_level=work_level,
        intense=intense, peak_relation=peak_relation,
        area_relation=area_relation, use_intense=use_intense,
        fill_value=fill_value)

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


def melting_layer_giangrande(radar,
                            rhomin=0.75, rhomax=0.94,
                            nml_points_min=1500, percentile_bottom=0.3,
                            refl_field=None, zdr_field=None,
                            rhv_field=None, temp_field=None,
                            iso0_field=None, ml_field=None,
                            temp_ref='temperature', ml_globdata=None):
    """
    Detects the melting layer following the approach by Giangrande et al
    (2008)

    Parameters
    ----------
    radar : radar
        radar object

    Other Parameters
    ----------------
    rhomin : float
        min rhohv to consider pixel potential melting layer pixel
    rhomax : float
        max rhohv to consider pixel potential melting layer pixel
    nml_points_min : int
        minimum number of melting layer points to consider valid melting layer detection
    percentile_bottom : float [0,1]
        percentile of ml points above which is considered that the bottom of the melting layer starts
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
    temp_ref : str
        the field use as reference for temperature. Can be either temperature
        or height_over_iso0
    ml_globdata :
        stack of previous volume data to introduce some time dependency. Its max size is
        controlled by the nVol parameter. It is always in (pseudo-)RHI mode.

    Returns
    -------
    ml : dict
        melting layer detection field

    References
    ----------
    Giangrande, S.E., Krause, J.M., Ryzhkov, A.V.: Automatic Designation of
    the Melting Layer with a Polarimetric Prototype of the WSR-88D Radar,
    J. of Applied Meteo. and Clim., 47, 1354-1364, doi:10.1175/2007JAMC1634.1,
    2008

    """

    # input data=======================================

    # parse the field parameters
    
    if refl_field is None:
        refl_field = get_field_name('reflectivity')
    if zdr_field is None:
        zdr_field = get_field_name('differential_reflectivity')
    if rhv_field is None:
        rhv_field = get_field_name('cross_correlation_ratio')
    if ml_field is None:
        ml_field = get_field_name('melting_layer')

    if temp_ref == 'temperature':
        if temp_field is None:
            temp_field = get_field_name('temperature')
    else:
        if iso0_field is None:
            iso0_field = get_field_name('height_over_iso0')

    # check if fieldnames exists
    
    radar.check_field_exists(refl_field)
    radar.check_field_exists(zdr_field)
    radar.check_field_exists(rhv_field)
    if temp_ref == 'temperature':
        radar.check_field_exists(temp_field)
    else:
        radar.check_field_exists(iso0_field)

    # hard-coded parameters=======================================

    zhmin = 20.                         # min zh to consider pixel potential melting layer pixel [dBZ]
    hwindow = 500.                      # window above the suspected melting layer range bin where to look for a peak of zh and zdr [m]
    htol = 500.                         # height above iso-0° where we allow the presence of melting layer points [m]
    mlzhmin = 30.                       # min peak reflectivity within the melting layer [dBZ]
    mlzhmax = 50.                       # max peak reflectivity within the melting layer [dBZ]
    mlzdrmin = 1.0                      # min peak zdr within the melting layer [dB]
    mlzdrmax = 5.0                      # max peak zdr within the melting layer [dB]
    elmin = 4.0                         # min elevation angle used [°]
    elmax = 10.                         # max elevation angle used [°]
    ind_rmin = 3                        # index of min range where to start looking for melting layer

    deltah = 50.                        # height resolution [m]
    maxh = 6000.                        # maximum freezing level height [m]
    nVol = 3                            # number of consecutive volumes to combine data from
    nVol_nodata = 12                    # number of consecutive volumes where melting layer value is valid

    percentile_top = 0.9                # percentile of ml points below which is considered that the top of the melting layer starts
    ml_height_bottom_diff = 1000.       # maximum difference allowed between height of the previous melting layer bottom and current melting layer point

    a = 6378100.                        # equatorial earth radius [m]
    ke = 4./3.                          # constant to calculate the effective radius

    ml_thickness = 400.                 # default melting layer thickness [m]

    azimuth_angle_pseudo_rhi = 1        # azimuth angle interval for pseudo-RHI [°]
    az_tol_pseudo_rhi = None            # azimuth angle tolerance for pseudo-RHI [°]

    ## initialization=======================================

    # in case of a PPI scan, a pseudo-RHI is produced.
    # to ensure a regular grid over time, each successive scan is resampled on the same grid of the initial radar object.
    
    print('RADAR PPI')
    print('n sweeps = ',radar.nsweeps)
    print('n range bins = ',radar.ngates)
    print('n rays = ',radar.nrays)

    if ml_globdata is None: # first volume for melting layer detection


        if radar.scan_type == 'ppi':

            # generate reference grid from list of azimuths

            azimuths = radar.azimuth['data']
            minaz = azimuths.min()
            maxaz = azimuths.max()
            target_azimuths = np.arange(radar.azimuth['data'].min(), radar.azimuth['data'].max(), azimuth_angle_pseudo_rhi).tolist()
            if maxaz > target_azimuths[-1]:
                target_azimuths.append(maxaz)
            print('min = ', radar.azimuth['data'].min(), ', max = ', radar.azimuth['data'].max())
            print('n target azimuths: ', len(target_azimuths))    

            # produce pseudo-RHI
            
            radar_rhi = cross_section_ppi(radar, target_azimuths, az_tol_pseudo_rhi)
            
            # pseudo-RHI volume parameters
            nAzimuth    = radar_rhi.nsweeps
            nRays       = radar_rhi.nrays
            nElevation  = nRays/nAzimuth  
            nRangeBins  = radar_rhi.ngates    
            nHeight     = maxh/deltah
            
        elif radar.scan_type == 'rhi':

            # first RHI volume defines reference grid in stack
            radar_rhi = deepcopy(radar)
            
            # RHI volume parameters
            nAzimuth    = radar_rhi.nsweeps
            nRays       = radar_rhi.nrays
            nElevation  = nRays/nAzimuth  
            nRangeBins  = radar_rhi.ngates    
            nHeight     = maxh/deltah

        else:
            warn('Error: unsupported scan type.')
            return None, None



        ml_globdata={
            'iVol_nodata'           :   0,
            'iVol'                  :   0,
            'timevec'               :   0,
            'ml_points'             :   np.ma.masked_all((nAzimuth, nHeight, nVol), dtype='int32'),
            'ml_height_top'         :   np.ma.masked_all(nAzimuth, dtype='float32'),
            'ml_height_bottom'      :   np.ma.masked_all(nAzimuth, dtype='float32'),
            'iso0_height'           :   np.ma.masked_all(nAzimuth, dtype='float32'),
            'azimuth_angles'        :   radar_rhi.fixed_angle['data'],
            'radar_initial'         :   deepcopy(radar_rhi)
        }
        ml_globdata['radar_initial'].fields = []

    else:                   # not the first volume for melting layer detection

        if radar.scan_type == 'ppi':

            # produce pseudo-RHI from PPI scan
            radar_rhi = cross_section_ppi(radar, ml_globdata['azimuth_angles'], az_tol_pseudo_rhi)
            
            # pseudo-RHI volume parameters
            nAzimuth    = radar_rhi.nsweeps#len(np.unique(radar_rhi.fixed_angle['data']))
            nRays       = radar_rhi.nrays
            nElevation  = nRays/nAzimuth  
            nRangeBins  = radar_rhi.ngates    
            nHeight     = maxh/deltah

        elif radar.scan_type == 'rhi':

            radar_rhi = deepcopy(radar)

            # resample on initial grid
            refl = interpol_field(ml_globdata['radar_initial'], radar_rhi, refl_field)
            zdr = interpol_field(ml_globdata['radar_initial'], radar_rhi, zdr_field)
            rhohv = interpol_field(ml_globdata['radar_initial'], radar_rhi, rhv_field)
            if temp_ref == 'temperature':
                temp = interpol_field(ml_globdata['radar_initial'], radar_rhi, field_name)
            else:
                iso0 = interpol_field(ml_globdata['radar_initial'], radar_rhi, iso0_field)

            # update fields
            radar_rhi.fields[refl_field]['data'] = refl
            radar_rhi.fields[refl_field]['data'] = zdr
            radar_rhi.fields[refl_field]['data'] = rhohv
            if temp_ref == 'temperature':
                radar_rhi.fields[temp_field]['data'] = temp
            else:
                radar_rhi.fields[iso0_field]['data'] = iso0
                
            # RHI volume parameters
            nAzimuth    = radar_rhi.nsweeps
            nRays       = radar_rhi.nrays
            nElevation  = nRays/nAzimuth  
            nRangeBins  = radar_rhi.ngates    
            nHeight     = maxh/deltah

        else:
            warn('Error: unsupported scan type.')
            return None, None

    # extract the fields (refl, zdr, rhohv, temp, iso0)
    
    refl = radar_rhi.fields[refl_field]['data']
    zdr = radar_rhi.fields[zdr_field]['data']
    rhohv = radar_rhi.fields[rhv_field]['data']
    if temp_ref == 'temperature':
        temp = radar_rhi.fields[temp_field]['data']
    else:
        iso0 = radar_rhi.fields[iso0_field]['data']

    # initialize melting layer detection flag
    
    mldetected=False
    
    print('RADAR PSEUDO-RHI')
    print('n elevations = ',nElevation)
    print('n azimuths = ',nAzimuth)
    print('n range bins = ',nRangeBins)
    print('n rays = ',nRays)
    print('data shape = ', rhohv.shape)

    # set half length of the window over azimuth data, whalflength [°]
    # if necessary, adjust minimum nml_points_min, the number of melting layer points necessary to get a valid melting layer detection
    
    if nAzimuth <= 3:
        whalflength = 0
        nml_points_min = 75
    elif nAzimuth >= 21:
        whalflength = 10
    else:
        whalflength = np.round(nAzimuth/2-1)
        nml_points_min = np.round(1500/whalflength)

    # altitude over sea level of each radar bin [m] >>(l. 226)
    
    deltar = radar_rhi.range['data'][1]-radar_rhi.range['data'][0]
    beamwidth = radar_rhi.metadata.get('beamwidth',1)
    
    print('delta r = ',deltar)
    print('beamwidth = ',beamwidth)
    print('radar altitude = ',radar_rhi.altitude['data'][0])
        
    _, _, hcenter = antenna_vectors_to_cartesian(radar_rhi.range['data'], radar_rhi.azimuth['data'], radar_rhi.elevation['data']) + radar_rhi.altitude['data'][0]
    _, _, hlowerleft = antenna_vectors_to_cartesian(radar_rhi.range['data']-deltar/2, 
                                                    radar_rhi.azimuth['data'], 
                                                    radar_rhi.elevation['data']-beamwidth/2) + radar_rhi.altitude['data'][0]
    _, _, hlowerright = antenna_vectors_to_cartesian(radar_rhi.range['data']+deltar/2, 
                                                     radar_rhi.azimuth['data'], 
                                                     radar_rhi.elevation['data']-beamwidth/2) + radar_rhi.altitude['data'][0]
    _, _, hupperleft = antenna_vectors_to_cartesian(radar_rhi.range['data']-deltar/2, 
                                                    radar_rhi.azimuth['data'], 
                                                    radar_rhi.elevation['data']+beamwidth/2) + radar_rhi.altitude['data'][0]
    _, _, hupperright = antenna_vectors_to_cartesian(radar_rhi.range['data']+deltar/2, 
                                                     radar_rhi.azimuth['data'], 
                                                     radar_rhi.elevation['data']+beamwidth/2) + radar_rhi.altitude['data'][0]
    
    # median azimuth iso0 height per ray
    
    iso0_height = np.ma.masked_all(nRays)
    if temp_ref == 'temperature':
        first_0deg_gate_per_ray = np.argmax(temp<0,axis=1)
        first_0deg_gate_per_ray[~np.any(temp<0,axis=1)] = nRangeBins        
        for n,sweep_slice in enumerate(radar_rhi.iter_slice()):
            iso0_height[sweep_slice] = np.median(hcenter[first_0deg_gate_per_ray[sweep_slice]])
    else:
        for n,sweep_slice in enumerate(radar_rhi.iter_slice()):
            ind_iso0 = np.argmin(iso0[sweep_slice])
            iso0_height[sweep_slice] = hcenter[ind_iso0]                                

    # initialize suspected melting layer points matrix
    
    if ml_globdata['iVol'] == nVol:
        ml_globdata['iVol'] = 0
    ml_globdata['ml_points'][:, :, ml_globdata['iVol']] = 0
    ml_globdata['ml_points'].mask[:, :, ml_globdata['iVol']] = False

    # look for candidate melting layer points ================================
    
    # loop azimuths
    for n,sweep_slice in enumerate(radar_rhi.iter_slice()):
        print('sweep = ',n)
        
        # extract sweep (fixed azimuth, excluding close ranges)
        elevation_sweep = radar_rhi.elevation['data'][sweep_slice]
        rhohv_sweep = rhohv[sweep_slice,ind_rmin:]
        refl_sweep = refl[sweep_slice,ind_rmin:]
        hcenter_Sweep = hcenter[sweep_slice,ind_rmin:]
        
        # gates in sweep that fulfill conditions on elevation angle, rhohv, refl and gate altitude
        ind_ml = np.logical_and.reduce((
                    np.logical_or(
                        np.logical_and(elevation_sweep >= elmin, elevation_sweep <= elmax),
                        np.logical_and(elevation_sweep <= 180-elmin, elevation_sweep >= 180-elmax)
                        )[:,None],
                    rhohv_sweep <= rhomax,
                    rhohv_sweep >= rhomin,
                    refl_sweep >= zhmin,
                    hcenter_sweep < maxh,
                    ))
        
        nml = ind_ml.sum()
        
        print(ind_ml.shape,nml)
        
        if nml > 0:
        
                # validate individual gates according to a peak in zh and zdr observed above 
                ind_rays, ind_range = np.where(ind_ml) # all valid gates in nth sweep
                
                # loop over valid gates
                for ii in range(nml):
                               
                    # and find all valid gates that are within hwindow meters above
                    ind_gates_above = np.logical_and( 
                                        hcenter_sweep[ind_rays[ii],:]-hcenter_sweep[ind_rays[ii],ind_range[ii]] < hwindow,
                                        hcenter_sweep[ind_rays[ii],:]-hcenter_sweep[ind_rays[ii],ind_range[ii]] >= 0 
                                        )
        
                    # compute peaks of zh and zdr
                    # zhmax = refl_sweep[ind_gates_above].max()                    
                    # zdrmax = zdr_sweep[ind_gates_above].max()     
        
        sys.exit()
        
        
        # rays in sweep within elevation limits
    
        ind_elvalid = np.logical_or(
                        np.logical_and(elevation_angles >= elmin, elevation_angles <= elmax),
                        np.logical_and(elevation_angles <= 180-elmin, elevation_angles >= 180-elmax)
                        )
                      
        nelvalid = ind_elvalid.sum()
    
        if nelvalid > 0:
    
            # gates in sweep that are within elevation, range, rhohv, zh and altitude limits 

            ind_ml = np.logical_and.reduce((
                        rhohv[sweep_slice,:][ind_elvalid, ind_rmin:] <= rhomax,
                        rhohv[sweep_slice,:][ind_elvalid, ind_rmin:] >= rhomin,
                        refl[sweep_slice,:][ind_elvalid, ind_rmin:] >= zhmin,
                        hcenter[sweep_slice,:][ind_elvalid, ind_rmin:] < maxh,
                        ))
                
            nml = ind_ml.sum()
            if nml > 0:
        
                # validate individual gates according to a peak in zh and zdr observed above 
        
                ind_rays, ind_range = np.where(ind_ml) # all valid gates in nth sweep
            
                # loop over valid gates
                for ii in range(nml):
                               
                    # and find all valid gates that are within hwindow meters above
                    ind_gates_above = np.logical_and( 
                                        hcenter[sweep_slice,:][ind_elvalid, ind_rmin:][ind_rays[ii],:]-hcenter[sweep_slice,:][ind_elvalid, ind_rmin:][ind_rays[ii],ind_range[ii]] < hwindow,
                                        hcenter[sweep_slice,:][ind_elvalid, ind_rmin:][ind_rays[ii],:]-hcenter[sweep_slice,:][ind_elvalid, ind_rmin:][ind_rays[ii],ind_range[ii]] >= 0 
                                        )
                
                    # compute peaks of zh and zdr
                    zhmax = refl[sweep_slice,:][ind_elvalid, ind_rmin:][ind_rays[ii],:][ind_gates_above].max()                    
                    zdrmax = zdr[sweep_slice,:][ind_elvalid, ind_rmin:][ind_rays[ii],:][ind_gates_above].max()     

                    if (zhmax >= mlzhmin 
                        and zhmax <= mlzhmax 
                        and zdrmax >= mlzdrmin 
                        and zdrmax <= mlzdrmax 
                        and hcenter[sweep_slice,:][ind_elvalid, ind_rmin:][ind_rays[ii],ind_range[ii]] < iso0_height[sweep_slice,:][ind_elvalid, ind_rmin:][0]+htol):
                        
                            
                            ml_globdata['ml_points'][n, int(hcenter[sweep_slice,:][ind_elvalid, ind_rmin:][ind_rays[ii],ind_range[ii]]/deltah), ml_globdata['iVol']] += 1
                            
                            print(ml_globdata['azimuth_angles'][n], 
                                 int(hcenter[sweep_slice,:][ind_elvalid, ind_rmin:][ind_rays[ii],ind_range[ii]]/deltah),
                                 ml_globdata['ml_points'][n, int(hcenter[sweep_slice,:][ind_elvalid, ind_rmin:][ind_rays[ii],ind_range[ii]]/deltah), ml_globdata['iVol']])
    
    
    
    sys.exit()

    # prepare output fields
    ml = get_metadata(ml_field)
    ml['data'] = ml_data

    # update the global stack of ml volumes


    # if radar.scan_type == 'ppi':
    # convert back to ppi

    return ml

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
    field_dest['data'] = np.ma.empty((radar_dest.nrays, radar_dest.ngates))
    field_dest['data'][:] = np.ma.masked

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


def hydroclass_semisupervised(radar, mass_centers=None,
                              weights=np.array([1., 1., 1., 0.75, 0.5]),
                              refl_field=None, zdr_field=None, rhv_field=None,
                              kdp_field=None, temp_field=None, iso0_field=None,
                              hydro_field=None, temp_ref='temperature'):
    """
    Classifies precipitation echoes following the approach by
    Besic et al (2016)

    Parameters
    ----------
    radar : radar
        radar object

    Other Parameters
    ----------------
    mass_centers : ndarray 2D
        The centroids for each variable and hydrometeor class in (nclasses,
        nvariables)
    weights : ndarray 1D
        The weight given to each variable.
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

    Returns
    -------
    hydro : dict
        hydrometeor classification field

    References
    ----------
    Besic, N., Figueras i Ventura, J., Grazioli, J., Gabella, M., Germann, U.,
    and Berne, A.: Hydrometeor classification through statistical clustering
    of polarimetric radar measurements: a semi-supervised approach,
    Atmos. Meas. Tech., 9, 4425-4445, doi:10.5194/amt-9-4425-2016, 2016

    """
    lapse_rate = -6.5

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

    # parse the field parameters
    if refl_field is None:
        refl_field = get_field_name('reflectivity')
    if zdr_field is None:
        zdr_field = get_field_name('differential_reflectivity')
    if rhv_field is None:
        rhv_field = get_field_name('cross_correlation_ratio')
    if kdp_field is None:
        kdp_field = get_field_name('specific_differential_phase')
    if hydro_field is None:
        hydro_field = get_field_name('radar_echo_classification')

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

    refl = radar.fields[refl_field]['data']
    zdr = radar.fields[zdr_field]['data']
    rhohv = radar.fields[rhv_field]['data']
    kdp = radar.fields[kdp_field]['data']

    if temp_ref == 'temperature':
        # convert temp in relative height respect to iso0
        temp = radar.fields[temp_field]['data']
        relh = temp*(1000./lapse_rate)
    else:
        relh = radar.fields[iso0_field]['data']

    # standardize data
    refl_std = _standardize(refl, 'Zh')
    zdr_std = _standardize(zdr, 'ZDR')
    kdp_std = _standardize(kdp, 'KDP')
    rhohv_std = _standardize(rhohv, 'RhoHV')
    relh_std = _standardize(relh, 'relH')

    # standardize centroids
    mc_std = np.zeros(np.shape(mass_centers))
    mc_std[:, 0] = _standardize(mass_centers[:, 0], 'Zh')
    mc_std[:, 1] = _standardize(mass_centers[:, 1], 'ZDR')
    mc_std[:, 2] = _standardize(mass_centers[:, 2], 'KDP')
    mc_std[:, 3] = _standardize(mass_centers[:, 3], 'RhoHV')
    mc_std[:, 4] = _standardize(mass_centers[:, 4], 'relH')

    # assign to class
    hydroclass_data, min_dist = _assign_to_class(
        refl_std, zdr_std, kdp_std, rhohv_std, relh_std, mc_std,
        weights=weights)

    # prepare output fields
    hydro = get_metadata(hydro_field)
    hydro['data'] = hydroclass_data

    return hydro


def _standardize(data, field_name, mx=None, mn=None):
    """
    Streches the radar data to -1 to 1 interval

    Parameters
    ----------
    data : array
        radar field

    field_name : str
        type of field (relH, Zh, ZDR, KDP or RhoHV)

    Returns
    -------
    field_std : dict
        standardized radar data
    """
    if field_name == 'relH':
        field_std = 2./(1.+np.ma.exp(-0.005*data))-1.
        return field_std

    if (mx is None) or (mn is None):
        dlimits_dict = _data_limits_table()
        if field_name not in dlimits_dict:
            raise ValueError(
                'Field '+field_name+' unknown. ' +
                'Valid field names for standardizing are: ' +
                'relH, Zh, ZDR, KDP and RhoHV')
        mx, mn = dlimits_dict[field_name]

    if field_name == 'KDP':
        data[data < -0.5] = -0.5
        data = 10.*np.ma.log10(data+0.6)
    elif field_name == 'RhoHV':
        data = 10.*np.ma.log10(1.-data)

    mask = np.ma.getmaskarray(data)
    field_std = 2.*(data-mn)/(mx-mn)-1.
    field_std[data < mn] = -1.
    field_std[data > mx] = 1.
    field_std[mask] = np.ma.masked

    return field_std


def _assign_to_class(zh, zdr, kdp, rhohv, relh, mass_centers,
                     weights=np.array([1., 1., 1., 0.75, 0.5])):
    """
    assigns an hydrometeor class to a radar range bin computing
    the distance between the radar variables an a centroid

    Parameters
    ----------
    zh,zdr,kdp,rhohv,relh : radar field
        variables used for assigment normalized to [-1, 1] values

    mass_centers : matrix
        centroids normalized to [-1, 1] values

    weights : array
        optional. The weight given to each variable

    Returns
    -------
    hydroclass : int array
        the index corresponding to the assigned class
    mind_dist : float array
        the minimum distance to the centroids
    """
    # prepare data
    nrays = zh.shape[0]
    nbins = zdr.shape[1]
    nclasses = mass_centers.shape[0]
    nvariables = mass_centers.shape[1]

    min_dist = np.ma.empty((nrays, nbins))
    hydroclass = np.ma.empty((nrays, nbins), dtype=int)

    for ray in range(nrays):
        data = np.ma.array([zh[ray, :], zdr[ray, :], kdp[ray, :],
                            rhohv[ray, :], relh[ray, :]])
        weights_mat = np.broadcast_to(
            weights.reshape(nvariables, 1), (nvariables, nbins))
        dist = np.ma.zeros((nclasses, nbins), dtype='float64')

        # compute distance: masked entries will not contribute to the distance
        for i in range(nclasses):
            centroids_class = mass_centers[i, :]
            centroids_class = np.broadcast_to(
                centroids_class.reshape(nvariables, 1), (nvariables, nbins))
            dist[i, :] = np.ma.sqrt(np.ma.sum(
                ((centroids_class-data)**2.)*weights_mat, axis=0))

        # use very large fill_value so that masked entries will be sorted at
        # the end. There should not be any masked entry anyway
        class_vec = dist.argsort(axis=0, fill_value=10e40)

        # get minimum distance. Acts as a confidence value
        dist_sorted = dist.sort(axis=0, fill_value=10e40)
        min_dist[ray, :] = dist[0, :]

        # Entries with non-valid reflectivity values are set to 0 (No class)
        mask = np.ma.getmaskarray(zh[ray, :])
        hydroclass_ray = class_vec[0, :]+1
        hydroclass_ray[mask] = 0
        hydroclass[ray, :] = hydroclass_ray

    return hydroclass, min_dist


def _assign_to_class_scan(zh, zdr, kdp, rhohv, relh, mass_centers,
                          weights=np.array([1., 1., 1., 0.75, 0.5])):
    """
    assigns an hydrometeor class to a radar range bin computing
    the distance between the radar variables an a centroid.
    Computes the entire radar volume at once

    Parameters
    ----------
    zh,zdr,kdp,rhohv,relh : radar field
        variables used for assigment normalized to [-1, 1] values

    mass_centers : matrix
        centroids normalized to [-1, 1] values

    weights : array
        optional. The weight given to each variable

    Returns
    -------
    hydroclass : int array
        the index corresponding to the assigned class
    mind_dist : float array
        the minimum distance to the centroids
    """
    # prepare data
    nrays = zh.shape[0]
    nbins = zdr.shape[1]
    nclasses = mass_centers.shape[0]
    nvariables = mass_centers.shape[1]

    data = np.ma.array([zh, zdr, kdp, rhohv, relh])
    weights_mat = np.broadcast_to(
        weights.reshape(nvariables, 1, 1),
        (nvariables, nrays, nbins))
    dist = np.ma.zeros((nclasses, nrays, nbins), dtype='float64')

    # compute distance: masked entries will not contribute to the distance
    for i in range(nclasses):
        centroids_class = mass_centers[i, :]
        centroids_class = np.broadcast_to(
            centroids_class.reshape(nvariables, 1, 1),
            (nvariables, nrays, nbins))
        dist[i, :, :] = np.ma.sqrt(np.ma.sum(
            ((centroids_class-data)**2.)*weights_mat, axis=0))

    # use very large fill_value so that masked entries will be sorted at the
    # end. There should not be any masked entry anyway
    class_vec = dist.argsort(axis=0, fill_value=10e40)

    # get minimum distance. Acts as a confidence value
    dist_sorted = dist.sort(axis=0, fill_value=10e40)
    min_dist = dist[0, :, :]

    # Entries with non-valid reflectivity values are set to 0 (No class)
    mask = np.ma.getmaskarray(zh)
    hydroclass = class_vec[0, :, :]+1
    hydroclass[mask] = 0

    return hydroclass, min_dist


def _get_mass_centers(freq):
    """
    get mass centers for a particular frequency

    Parameters
    ----------
    freq : float
        radar frequency [Hz]

    Returns
    -------
    mass_centers : ndarray 2D
        The centroids for each variable and hydrometeor class in (nclasses,
        nvariables)

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
    defines the mass centers look up table for each frequency band.

    Returns
    -------
    mass_centers_dict : dict
        A dictionary with the mass centers for each frequency band

    """
    nclasses = 9
    nvariables = 5
    mass_centers = np.zeros((nclasses, nvariables))

    mass_centers_dict = dict()
    # C-band centroids derived for MeteoSwiss Albis radar
    #                       Zh        ZDR     kdp   RhoHV    delta_Z
    mass_centers[0, :] = [13.5829,  0.4063, 0.0497, 0.9868,  1330.3]  # DS
    mass_centers[1, :] = [02.8453,  0.2457, 0.0000, 0.9798,  0653.8]  # CR
    mass_centers[2, :] = [07.6597,  0.2180, 0.0019, 0.9799, -1426.5]  # LR
    mass_centers[3, :] = [31.6815,  0.3926, 0.0828, 0.9978,  0535.3]  # GR
    mass_centers[4, :] = [39.4703,  1.0734, 0.4919, 0.9876, -1036.3]  # RN
    mass_centers[5, :] = [04.8267, -0.5690, 0.0000, 0.9691,  0869.8]  # VI
    mass_centers[6, :] = [30.8613,  0.9819, 0.1998, 0.9845, -0066.1]  # WS
    mass_centers[7, :] = [52.3969,  2.1094, 2.4675, 0.9730, -1550.2]  # MH
    mass_centers[8, :] = [50.6186, -0.0649, 0.0946, 0.9904,  1179.9]  # IH/HDG

    mass_centers_dict.update({'C': mass_centers})

    # X-band centroids derived for MeteoSwiss DX50 radar
    #                       Zh        ZDR     kdp    RhoHV   delta_Z
    mass_centers[0, :] = [19.0770,  0.4139, 0.0099, 0.9841,  1061.7]  # DS
    mass_centers[1, :] = [03.9877,  0.5040, 0.0000, 0.9642,  0856.6]  # CR
    mass_centers[2, :] = [20.7982,  0.3177, 0.0004, 0.9858, -1375.1]  # LR
    mass_centers[3, :] = [34.7124, -0.3748, 0.0988, 0.9828,  1224.2]  # GR
    mass_centers[4, :] = [33.0134,  0.6614, 0.0819, 0.9802, -1169.8]  # RN
    mass_centers[5, :] = [08.2610, -0.4681, 0.0000, 0.9722,  1100.7]  # VI
    mass_centers[6, :] = [35.1801,  1.2830, 0.1322, 0.9162, -0159.8]  # WS
    mass_centers[7, :] = [52.4539,  2.3714, 1.1120, 0.9382, -1618.5]  # MH
    mass_centers[8, :] = [44.2216, -0.3419, 0.0687, 0.9683,  1272.7]  # IH/HDG

    mass_centers_dict.update({'X': mass_centers})

    return mass_centers_dict


def _data_limits_table():
    """
    defines the data limits used in the standardization.

    Returns
    -------
    dlimits_dict : dict
        A dictionary with the limits for each variable

    """
    dlimits_dict = dict()
    dlimits_dict.update({'Zh': (60., -10.)})
    dlimits_dict.update({'ZDR': (5., -1.5)})
    dlimits_dict.update({'KDP': (7., -10.)})
    dlimits_dict.update({'RhoHV': (-5.23, -50.)})

    return dlimits_dict


def get_freq_band(freq):
    """
    returns the frequency band name (S, C, X, ...)

    Parameters
    ----------
    freq : float
        radar frequency [Hz]

    Returns
    -------
    freq_band : str
        frequency band name

    """
    if freq >= 2e9 and freq < 4e9:
        return 'S'
    if freq >= 4e9 and freq < 8e9:
        return 'C'
    if freq >= 8e9 and freq <= 12e9:
        return 'X'

    warn('Unknown frequency band')

    return None
