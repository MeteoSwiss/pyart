"""
pyart.retrieve.gecsx
==================

Functions for visibility and ground echoes estimation from a DEM.

"""
from copy import deepcopy
import logging
import numpy as np
from scipy.ndimage import sobel
from pyproj import Transformer, Proj

from ..config import get_field_name, get_metadata, get_fillvalue
from ..core import antenna_vectors_to_cartesian, antenna_to_cartesian
from . import _gecsx_functions as gf

MANDATORY_RADAR_SPECS = ['tau', 'loss', 'power', 'frequency', 'beamwidth',
                         'gain']

def gecsx(radar, radar_specs, dem_grid,
                         fill_value = None,
                         terrain_altitude_field = None,
                         bent_terrain_altitude_field = None,
                         terrain_slope_field = None,
                         terrain_aspect_field = None,
                         elevation_angle_field = None,
                         visibility_field = None,
                         min_vis_elevation_field = None,
                         min_vis_altitude_field = None,
                         incident_angle_field = None,
                         effective_area_field = None,
                         sigma_0_field = None,
                         rcs_clutter_field = None,
                         dBm_clutter_field = None,
                         dBZ_clutter_field = None,
                         visibility_polar_field = None,
                         az_conv = 0,
                         dr = 100,
                         daz = 0.012,
                         ke = 4/3.,
                         atm_att = 0.2,
                         mosotti_kw = 0.9644,
                         raster_oversampling = 1,
                         sigma0_method = 'Gabella',
                         clip = True,
                         return_pyart_objects = True,
                         verbose = True):
    """
    Estimate the radar visibility and ground clutter echoes from a digital
    elevation model

    Parameters
    ----------
    radar : radar
        Radar object which contains the coordinates on which the visibility
        will be computed
    radar_specs : dict
        Dictionary containing the radar specifications. Must have the following
        keys:
            "frequency" : radar frequenty in GHz
            "tau"       : radar pulsewidth in m
            "beamwidth" : antenna 3dB beamwidth in deg
            "loss"      : total loss (antenna + radar system) in dB
            "gain"      : total gain (antenna + radar system) in dB
            "power"     : radar total power in W
    terrain_altitude_field : str, optional
        Field name which represents the terrain altitude class field.
        A value of None will use the default field name as defined in the
        Py-ART configuration file.
    bent_terrain_altitude_field : str, optional
        Field name which represents the terrain altitude after radar
        refractivity height correction class field.
        A value of None will use the default field name as defined in the
        Py-ART configuration file.
    terrain_slope_field : str, optional
        Field name which represents the terrain slope class field.
        A value of None will use the default field name as defined in the
        Py-ART configuration file.
    terrain_aspect_field : str, optional
        Field name which represents the terrain aspect class field.
        A value of None will use the default field name as defined in the
        Py-ART configuration file.
    elevation_angle_field : str, optional
        Field name which represents the elevation angle class field.
        A value of None will use the default field name as defined in the
        Py-ART configuration file.
    visibility_field : str, optional
        Field name which represents the Cartesian visibility field.
        A value of None will use the default field name as defined in the
        Py-ART configuration file.
    min_vis_elevation_field : str, optional
        Field name which represents the minimum visible elevation angle field.
        A value of None will use the default field name as defined in the
        Py-ART configuration file.
    min_vis_altitude_field : str, optional
        Field name which represents the minimum visible altitude field.
        A value of None will use the default field name as defined in the
        Py-ART configuration file.
    sigma_0_field : str, optional
        Field name which represents the ratio between rcs and backscattering
        area field. A value of None will use the default field name as defined
        in the Py-ART configuration file.
    incident_angle_field : str, optional
        Field name which represents the incidence angle at the topography
        field. A value of None will use the default field name as defined
        in the Py-ART configuration file.
    effective_area_field : str, optional
        Field name which represents the effective backscattering area. A
        value of None will use the default field name as defined
        in the Py-ART configuration file.
    rcs_clutter_field : str, optional
        Field name which represents the backscattering cross section.
        A value of None will use the default field name as defined
        in the Py-ART configuration file.
    dBm_clutter_field : str, optional
        Field name which represents the ground clutter power signal
        field. A value of None will use the default field name as defined
        in the Py-ART configuration file.
    dBZ_clutter_field : str, optional
        Field name which represents the ground clutter reflectivity field.
        A value of None will use the default field name as defined
        in the Py-ART configuration file.
    visibility_polar_field : str, optional
        Field name which represents the polar visibility field.
        A value of None will use the default field name as defined
        in the Py-ART configuration file.
    az_conv : float, optional
        If larger than zero assumes that the antenna moves in azimuth direction
        (PPI) and averages the rx signal over the angle given by this keyword
    dr : float, optional
        Range discretization used when computing the Cartesian visibility field
        the larger the better but the slower the processing will be
    daz : float, optional
        Azimuth discretization used when computing the Cartesian visibility
        field, the larger the better but the slower the processing will be
    ke : float(optional)
        Equivalent earth-radius factor used in the computation of the radar
        beam refraction
    atm_att : float, optional
        One-way atmospheric refraction in db / km
    mosotti_kw : float, optional
        Clausius-Mosotti factor K, depends on material (water) and wavelength
        for water = sqrt(0.93)
    raster_oversampling : int, optional
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
    sigma0_method : string, optional
        Which estimation method to use, either 'Gabella' or 'Delrieu'
    clip : bool, optional
        If set to true, the provided DEM will be clipped to the extent
        of the polar radar domain. Increases computation speed a lot but
        Cartesian output fields will be available only over radar domain
    return_pyart_objects : bool, optional
        If set to true, the generated products will be directly added to the
        input DEM Grid object (for Cartesian products) and the input Radar
        object (for polar products). Otherwise all products will be output
        separately
    verbose : bool, optional
        If set to true, the program will display info about the current
        progress

    Returns
    -------
    bent_terrain_altitude_dic : dict
        Terrain altitude corrected for radar beam height data and metadata
    terrain_slope_dic : dict
        Terrain slope data and metadata
    terrain_aspect_dic,: dict
        Terrain aspect data and metadata
    elevation_dic,: dict
        Elevation angle at topography data and metadata
    min_vis_elevation_dic,: dict
        Minimum visible elevation data and metadata
    min_vis_altitude_dic,: dict
        Minimum visible altitude data and metadata
    visibility_dic,: dict
        Visibility over a Cartesian domain data and metadata
    incident_angle_dic,: dict
        Incidence angle at topography data and metadata
    effective_area_dic,: dict
        Effective backscattering area data and metadata
    sigma_0_dic,: dict
        Ratio between RCS and backscattering area data and metadata
    rcs_clutter_dic,: dict
        Radar cross-section data and metadata
    dBm_clutter_dic,: dict
        Ground clutter power data and metadata
    dBZ_clutter_dic,: dict
        Ground clutter reflectivity data and metadata
    visibility_polar_dic,: dict
        Visibility over the polar radar domain data and metadata

    References
    ----------
    Gabella, M., & Perona, G. (1998). Simulation of the Orographic Influence
    on Weather Radar Using a Geometric–Optics Approach, Journal of Atmospheric
    and Oceanic Technology, 15(6), 1485-1494.
    """


    if verbose:
        logging.basicConfig(level = logging.INFO)
    else:
        logging.basicConfig(level = logging.WARNING)

    if radar.scan_type != 'ppi':
        raise NotImplementedError('Currently only ppi scans are supported!')

    for spec in MANDATORY_RADAR_SPECS:
        if spec not in radar_specs.keys():
            raise ValueError('Key {:s} is missing from '.format(spec) +
                             'radar_specs argument!')

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
    if elevation_angle_field is None:
        elevation_angle_field = get_field_name('elevation_angle')
    if visibility_field is None:
        visibility_field = get_field_name('visibility')
    if min_vis_elevation_field is None:
        min_vis_elevation_field = get_field_name('min_vis_elevation')
    if min_vis_altitude_field is None:
        min_vis_altitude_field = get_field_name('min_vis_altitude')
    if incident_angle_field is None:
        incident_angle_field = get_field_name('incident_angle')
    if sigma_0_field is None:
        sigma_0_field = get_field_name('sigma_0')
    if effective_area_field is None:
        effective_area_field = get_field_name('effective_area')
    if rcs_clutter_field is None:
        rcs_clutter_field = get_field_name('rcs_clutter')
    if dBm_clutter_field is None:
        dBm_clutter_field = get_field_name('dBm_clutter')
    if dBZ_clutter_field is None:
        dBZ_clutter_field = get_field_name('dBZ_clutter')
    if visibility_polar_field is None:
        visibility_polar_field = get_field_name('visibility_polar')

    elevations   = radar.fixed_angle['data']
    azimuths_pol = radar.get_azimuth(0)
    range_pol    = radar.range['data']

    # Define aeqd projection for radar local Cartesian coords
    pargs = Proj(proj="aeqd", lat_0 = radar.latitude['data'][0],
                 lon_0 = radar.longitude['data'][0], datum = "WGS84",
                 units="m")

    # Define coordinate transform: (local radar Cart coords) -> (DEM coords)
    transformer = Transformer.from_proj(pargs, dem_grid.projection)

    # Get local radar coordinates at elevaiton = 0
    xr, yr, _ = antenna_vectors_to_cartesian(radar.range['data'],
                          radar.get_azimuth(0), 0, ke = ke)

    # Project them in DEM proj
    xr_proj, yr_proj = transformer.transform(xr,yr)
    rad_x = xr_proj[0,0] # radar x coord in DEM proj
    rad_y = yr_proj[0,0] # radar y coord in DEM proj


    # Clip DEM outside radar domain
    if clip:
        dem_grid = gf.clip_grid(dem_grid, xr_proj, yr_proj)

    res_dem = dem_grid.metadata['resolution']
    xmin_dem = np.min(dem_grid.x['data'])
    ymin_dem = np.min(dem_grid.y['data'])

    # Processing starts here...
    ###########################################################################
    # 1) Compute range map
    logging.info('1) computing radar range map...')
    X_dem, Y_dem = np.meshgrid(dem_grid.x['data'], dem_grid.y['data'])
    range_map = np.sqrt((X_dem - rad_x) ** 2 + (Y_dem - rad_y) ** 2)

    # 2) Compute azimuth map
    logging.info('2) computing radar azimuth map...')

    az_map = (np.arctan2((X_dem - rad_x), (Y_dem - rad_y)) + 2*np.pi) % (2*np.pi)
    az_map *= 180 / np.pi

    # 3) Compute bent DEM map
    logging.info('3) computing bent DEM...')
    # Index first level
    dem = np.ma.filled(dem_grid.fields['terrain_altitude']['data'], np.nan)[0]
    _, _, zgrid = antenna_to_cartesian(range_map / 1000., az_map, 0, ke = ke)
    bent_map = dem - (zgrid + radar.altitude['data'])

    # 4) Compute slope and aspect
    logging.info('4) computing DEM slope and aspect...')
    gx = sobel(dem, axis = 1) / (8 * res_dem) # gradient w-e direction
    gy = sobel(dem, axis = 0) /  (8 * res_dem) # gradient s-n direction
    slope_map = np.arctan(np.sqrt(gy**2 + gx**2)) * 180 / np.pi
    aspect_map = (np.arctan2(gy, -gx) + np.pi) * 180 / np.pi

    # 5) Compute theta (elevation) angle at topography
    logging.info('5) computing radar elevation angle at topography...')
    elev_map = (np.arctan2(bent_map, range_map) * 180 / np.pi)

    # 6) COmpute visibility map and minimum visible elevation angle map
    logging.info('6) computing radar visibility map...')
    visib_map, minviselev_map = gf.visibility(az_map, range_map, elev_map,
                res_dem, xmin_dem, ymin_dem, rad_x, rad_y, dr, daz, verbose)

    # 7) Compute min visible altitude
    logging.info('7) computing min visible altitude...')
    R = 6371.0 * 1000.0 * ke     # effective radius of earth in meters.
    minvisalt_map = ((range_map ** 2 + R ** 2 + 2.0 * range_map * R *
           np.sin((minviselev_map + radar_specs['beamwidth'] / 2.) *
                  np.pi / 180.)) ** 0.5 - R)

    # 8) Compute effective area
    logging.info('8) computing effective area...')
    effarea_map = res_dem ** 2 / np.cos(slope_map * np.pi / 180.0)

    # 9) Compute local incidence angle
    logging.info('9) computing local incidence angle...')
    slope  = slope_map* np.pi / 180.0
    aspect = aspect_map * np.pi / 180.0
    zenith = (90. - elev_map * np.pi / 180.0 )
    az     = az_map* np.pi / 180.0

    incang_map = np.arccos(-( np.sin(slope) * np.sin(zenith) *
          (np.sin(aspect) * np.sin(az) + np.cos(aspect) * np.cos(az)) +
                  np.cos(slope) * np.cos(zenith))) * 180 / np.pi

    # 10) Compute sigma 0
    logging.info('10) computing sigma0...')
    sigma0_map = gf.sigma0(incang_map, radar_specs['frequency'],
                           sigma0_method)

    # Processing for every elevation angle starts here...
    ###########################################################################
    # 11) Compute rcs
    strelevs = ','.join([str(e) for e in elevations])
    logging.info('13) computing polar RCS at elevations {:s}...'
                     .format(strelevs))

    rcs_pol =  gf.rcs(az_map, range_map, elev_map, effarea_map, sigma0_map,
                         visib_map, range_pol, azimuths_pol, elevations,
                         res_dem, xmin_dem, ymin_dem, rad_x, rad_y,
                         radar_specs['beamwidth'], radar_specs['tau'],
                         raster_oversampling, verbose)
    rcs_pol = np.ma.array(rcs_pol, mask = np.isnan(rcs_pol),
                         fill_value = fill_value)

    # 12) Clutter power map in dBm
    logging.info('12) computing clutter power in dBm...')
    range_pol_e = np.tile(range_pol,(rcs_pol.shape[0], 1))
    range_log = 10* np.log10(range_pol_e)
    sigma_map = 10 * np.log10(rcs_pol)

    lambd = 3. / (radar_specs['frequency'] * 10.)
    pconst = (10 * np.log10(radar_specs['power']) + 2 * radar_specs['gain'] +
              20 * np.log10(lambd) -  radar_specs['loss']
              - 30 * np.log10(4 * np.pi))

    clutter_dBm_pol = (pconst - 4 * range_log -
                   2 * atm_att * range_pol_e / 1000. + sigma_map)

    # 13) Clutter reflectivity map in dBZ
    logging.info('13) computing clutter reflectivity in dBZ...')
    lambd = 3. / (radar_specs['frequency'] * 10.)
    dbzconst = (10 * np.log10(16 * np.log(2)) + 40 * np.log10(lambd) -
                10 * np.log10(radar_specs['tau'] * 3e8) -
                20 * np.log10(radar_specs['beamwidth'] + np.pi / 180.) -
                60 * np.log10(np.pi) - 20 * np.log10(mosotti_kw))

    convert_dbzm_to_dbz = 180. # 10*log10(1 m^6 / 1 mm^6) = 180
    clutter_dBZ_pol = (sigma_map - 2 * range_log + dbzconst +
                       convert_dbzm_to_dbz)

    # 14) Visibility map by angle
    logging.info('13) computing polar visibility at elevations {:s}...'.
                     format(strelevs))

    vispol =  gf.visibility_angle(minviselev_map, az_map, range_map,
                                  range_pol, azimuths_pol, elevations,
                                  res_dem, xmin_dem, ymin_dem, rad_x, rad_y,
                                  radar_specs['beamwidth'], radar_specs['tau'],
                                  az_conv, raster_oversampling, verbose)
    vispol = np.ma.array(vispol, mask = np.isnan(vispol),
                         fill_value = fill_value)

    logging.info('All done, creating outputs...')

    # Note that the [None,:,:]  indexing is for compatibility with
    # pyart grid.add_field() function
    bent_terrain_altitude_dic = get_metadata(bent_terrain_altitude_field)
    bent_terrain_altitude_dic['data'] = bent_map[None,:,:]

    elevation_dic = get_metadata(elevation_angle_field)
    elevation_dic['data'] = elev_map[None,:,:]

    terrain_slope_dic = get_metadata(terrain_slope_field)
    terrain_slope_dic['data'] = slope_map[None,:,:]

    terrain_aspect_dic = get_metadata(terrain_aspect_field)
    terrain_aspect_dic['data'] = aspect_map[None,:,:]

    visibility_dic = get_metadata(visibility_field)
    visibility_dic['data'] = visib_map[None,:,:]

    min_vis_elevation_dic = get_metadata(min_vis_elevation_field)
    min_vis_elevation_dic['data'] = minviselev_map[None,:,:]

    min_vis_altitude_dic = get_metadata(min_vis_altitude_field)
    min_vis_altitude_dic['data'] = minvisalt_map[None,:,:]

    incident_angle_dic = get_metadata(incident_angle_field)
    incident_angle_dic['data'] = incang_map[None,:,:]

    effective_area_dic = get_metadata(effective_area_field)
    effective_area_dic['data'] = effarea_map[None,:,:]

    sigma_0_dic = get_metadata(sigma_0_field)
    sigma_0_dic['data'] = sigma0_map[None,:,:]

    rcs_clutter_dic = get_metadata(rcs_clutter_field)
    rcs_clutter_dic['data'] = rcs_pol

    dBm_clutter_dic = get_metadata(dBm_clutter_field)
    dBm_clutter_dic['data'] = clutter_dBm_pol

    dBZ_clutter_dic = get_metadata(dBZ_clutter_field)
    dBZ_clutter_dic['data'] = clutter_dBZ_pol

    visibility_polar_dic = get_metadata(visibility_polar_field)
    visibility_polar_dic['data'] = vispol

    if not return_pyart_objects:
        return (bent_terrain_altitude_dic, terrain_slope_dic,
                terrain_aspect_dic, elevation_dic, min_vis_elevation_dic,
                min_vis_altitude_dic, visibility_dic, incident_angle_dic,
                effective_area_dic, sigma_0_dic, rcs_clutter_dic,
                dBm_clutter_dic, dBZ_clutter_dic, visibility_polar_dic)

    logging.info('Adding Cartesian output fields to input Grid (DEM) object...')
    dem_grid.add_field(bent_terrain_altitude_field,
                       bent_terrain_altitude_dic)
    dem_grid.add_field(elevation_angle_field,
                       elevation_dic)
    dem_grid.add_field(terrain_slope_field,
                       terrain_slope_dic)
    dem_grid.add_field(terrain_aspect_field,
                       terrain_aspect_dic)
    dem_grid.add_field(visibility_field,
                       visibility_dic)
    dem_grid.add_field(min_vis_elevation_field,
                       min_vis_elevation_dic)
    dem_grid.add_field(min_vis_altitude_field,
                       min_vis_altitude_dic)
    dem_grid.add_field(incident_angle_field,
                       incident_angle_dic)
    dem_grid.add_field(effective_area_field,
                       effective_area_dic)
    dem_grid.add_field(sigma_0_field,
                       sigma_0_dic)

    logging.info('Creating Radar (DEM) object...')
    new_radar = deepcopy(radar)
    new_radar.fields = dict()

    radar.add_field(rcs_clutter_field,
                       rcs_clutter_dic)
    radar.add_field(dBm_clutter_field,
                       dBm_clutter_dic)
    radar.add_field(dBZ_clutter_field,
                       dBZ_clutter_dic)
    radar.add_field(visibility_polar_field,
                       visibility_polar_dic)

    return dem_grid, radar