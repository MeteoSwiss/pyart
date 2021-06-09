"""
pyart.io.cf1_cartesian
======================

Utilities for reading CF1 Cartesian files.

.. autosummary::
    :toctree: generated/
    :template: dev_template.rst

    _NetCDFVariableDataExtractor

.. autosummary::
    :toctree: generated/

    read_cf1_cartesian
    read_cf1_cartesian_mf

"""

from warnings import warn

import numpy as np
import netCDF4

from ..config import FileMetadata, get_metadata
from ..io.common import _test_arguments
from ..core.grid import Grid
from ..io.cfradial import _ncvar_to_dict

SAT_FIELD_NAMES = [
    'IR_016', 'IR_039', 'IR_087', 'IR_097', 'IR_108', 'IR_120', 'IR_134',
    'CTH', 'HRV', 'VIS006', 'VIS008', 'WV_062', 'WV_073']

MF_FIELD_NAMES = ['radar_estimated_rain_rate', 'data_quality']

MF_FIELD_NAMES_DICT = {
    'data_quality': 'QUALITY',
    'radar_estimated_rain_rate': 'ACRR'}


def read_cf1_cartesian(filename, field_names=None, delay_field_loading=False,
                       chy0=255., chx0=-160., **kwargs):
    """
    Read a CF-1 netCDF file.

    Parameters
    ----------
    filename : str
        Name of CF/Radial netCDF file to read data from.
    field_names : list of fields to read from file.
        If None all files will be read.
    delay_field_loading : bool
        True to delay loading of field data from the file until the 'data'
        key in a particular field dictionary is accessed.  In this case
        the field attribute of the returned Radar object will contain
        LazyLoadDict objects not dict objects.  Delayed field loading will not
        provide any speedup in file where the number of gates vary between
        rays (ngates_vary=True) and is not recommended.
    chy0, chx0 : float
        Swiss coordinates position of the south-western point in the grid

    Returns
    -------
    grid : Grid
        Grid object containing data from METRANET file.

    """
    # test for non empty kwargs
    _test_arguments(kwargs)

    reserved_variables = [
        'band0', 'band1', 'band10', 'band11', 'band12', 'band2', 'band3',
        'band4', 'band5', 'band6', 'band7', 'band8', 'band9',
        'grid_mapping_0', 'nominal_wavelength0', 'nominal_wavelength1',
        'nominal_wavelength10', 'nominal_wavelength11',
        'nominal_wavelength12', 'nominal_wavelength2', 'nominal_wavelength3',
        'nominal_wavelength4', 'nominal_wavelength5', 'nominal_wavelength6',
        'nominal_wavelength7', 'nominal_wavelength8', 'nominal_wavelength9',
        'time', 'wl_CTH', 'wl_HRV', 'wl_IR_016', 'wl_IR_039', 'wl_IR_087',
        'wl_IR_097', 'wl_IR_108', 'wl_IR_120', 'wl_IR_134', 'wl_VIS006',
        'wl_VIS008', 'wl_WV_062', 'wl_WV_073', 'x0', 'y0']

    # create metadata retrieval object
    if field_names is None:
        field_names = SAT_FIELD_NAMES
    filemetadata = FileMetadata('satellite')

    # required reserved variables
    time = filemetadata('grid_time')
    origin_latitude = filemetadata('origin_latitude')
    origin_longitude = filemetadata('origin_longitude')
    origin_altitude = filemetadata('origin_altitude')
    x = filemetadata('x')
    y = filemetadata('y')
    z = filemetadata('z')

    # read the data
    ncobj = netCDF4.Dataset(filename)
    ncvars = ncobj.variables
    ncdims = ncobj.dimensions

    nx = ncdims['x0'].size
    ny = ncdims['y0'].size

    # 4.1 Global attribute -> move to metadata dictionary
    metadata = dict([(k, getattr(ncobj, k)) for k in ncobj.ncattrs()])

    grid_mapping = _ncvar_to_dict(ncvars['grid_mapping_0'])

    x['data'] = (
        1000.*(np.arange(nx)+chy0+1/2.)-grid_mapping['false_easting'])

    y['data'] = (
        1000.*(np.arange(ny)+chx0+1/2.)-grid_mapping['false_northing'])

    z['data'] = np.array([0.])

    # Origin of LV03 Swiss coordinates
    origin_latitude['data'] = np.array(
        [grid_mapping['latitude_of_projection_origin']])
    origin_longitude['data'] = np.array(
        [grid_mapping['longitude_of_projection_origin']])
    origin_altitude['data'] = np.array([0.])

    time = _ncvar_to_dict(ncvars['time'])

    # projection (Swiss Oblique Mercator)
    projection = {
        'proj': 'somerc',
        '_include_lon_0_lat_0': True
    }

    # read in the fields
    fields = {}

    # fields in the file has a shape of (ny, nx, 1) with 1
    # indicating frequency band
    # but should shaped (nz, ny, nx) in the Grid object
    field_shape = (ny, nx, 1)

    # check all non-reserved variables, those with the correct shape
    # are added to the field dictionary, if a wrong sized field is
    # detected a warning is raised
    field_keys = [k for k in ncvars if k not in reserved_variables]
    for field in field_names:
        if field not in field_keys:
            warn('Field '+field+' not in file')
            continue
        field_dic_file = _ncvar_to_dict(ncvars[field])
        if field_dic_file['data'].shape == field_shape:
            field_dic_file['data'] = np.expand_dims(
                np.squeeze(field_dic_file['data'], axis=-1),
                axis=0)[:, ::-1, :]
            field_dic = get_metadata(field)  # get field definition from Py-ART
            field_dic['data'] = field_dic_file['data']
            fields[field] = field_dic
        else:
            bad_shape = field_dic_file['data'].shape
            warn(
                'Field %s skipped due to incorrect shape %s'
                % (field, bad_shape))

    # radar_ variables
    if 'radar_latitude' in ncvars:
        radar_latitude = _ncvar_to_dict(ncvars['radar_latitude'])
    else:
        radar_latitude = None

    if 'radar_longitude' in ncvars:
        radar_longitude = _ncvar_to_dict(ncvars['radar_longitude'])
    else:
        radar_longitude = None

    if 'radar_altitude' in ncvars:
        radar_altitude = _ncvar_to_dict(ncvars['radar_altitude'])
    else:
        radar_altitude = None

    if 'radar_name' in ncvars:
        radar_name = _ncvar_to_dict(ncvars['radar_name'])
    else:
        radar_name = None

    if 'radar_time' in ncvars:
        radar_time = _ncvar_to_dict(ncvars['radar_time'])
    else:
        radar_time = None

    # do not close file if field loading is delayed
    if not delay_field_loading:
        ncobj.close()

    return Grid(
        time, fields, metadata,
        origin_latitude, origin_longitude, origin_altitude, x, y, z,
        projection=projection,
        radar_latitude=radar_latitude, radar_longitude=radar_longitude,
        radar_altitude=radar_altitude, radar_name=radar_name,
        radar_time=radar_time)


def read_cf1_cartesian_mf(filename, field_names=None,
                          delay_field_loading=False, **kwargs):
    """
    Read a CF-1 netCDF file.

    Parameters
    ----------
    filename : str
        Name of CF/Radial netCDF file to read data from.
    field_names : list of fields to read from file.
        If None all files will be read.
    delay_field_loading : bool
        True to delay loading of field data from the file until the 'data'
        key in a particular field dictionary is accessed.  In this case
        the field attribute of the returned Radar object will contain
        LazyLoadDict objects not dict objects.  Delayed field loading will not
        provide any speedup in file where the number of gates vary between
        rays (ngates_vary=True) and is not recommended.
    chy0, chx0 : float
        Swiss coordinates position of the south-western point in the grid

    Returns
    -------
    grid : Grid
        Grid object containing data from METRANET file.

    """
    # test for non empty kwargs
    _test_arguments(kwargs)

    reserved_variables = [
        'X', 'Y', 'time', 'grid_mapping', 'time_coverage_start',
        'time_coverage_end', 'data_mf_products', 'radar_image_time',
        'radar_image_latitude', 'radar_image_longitude', 'radar_image_wmoid',
        'radar_image_name']

    # create metadata retrieval object
    if field_names is None:
        field_names = MF_FIELD_NAMES
    filemetadata = FileMetadata('MF')

    # required reserved variables
    time = filemetadata('grid_time')
    origin_latitude = filemetadata('origin_latitude')
    origin_longitude = filemetadata('origin_longitude')
    origin_altitude = filemetadata('origin_altitude')
    x = filemetadata('x')
    y = filemetadata('y')
    z = filemetadata('z')

    # read the data
    ncobj = netCDF4.Dataset(filename)
    ncvars = ncobj.variables
    ncdims = ncobj.dimensions

    nx = ncdims['X'].size
    ny = ncdims['Y'].size

    # 4.1 Global attribute -> move to metadata dictionary
    metadata = dict([(k, getattr(ncobj, k)) for k in ncobj.ncattrs()])

    grid_mapping = _ncvar_to_dict(ncvars['grid_mapping'])

    x['data'] = _ncvar_to_dict(ncvars['X'])['data']
    y['data'] = _ncvar_to_dict(ncvars['Y'])['data']
    z['data'] = np.array([0.])

    # Origin
    origin_latitude['data'] = np.array(
        [grid_mapping['latitude_of_projection_origin']])
    origin_longitude['data'] = np.array(
        [grid_mapping['straight_vertical_longitude_from_pole']])
    origin_altitude['data'] = np.array([0.])

    time = _ncvar_to_dict(ncvars['time'])

    # projection
    projection = metadata['crs_proj4_string']

    # read in the fields
    fields = {}

    field_shape = (1, nx, ny)

    # check all non-reserved variables, those with the correct shape
    # are added to the field dictionary, if a wrong sized field is
    # detected a warning is raised
    field_keys = [k for k in ncvars if k not in reserved_variables]
    for field in field_names:
        if MF_FIELD_NAMES_DICT[field] not in field_keys:
            warn('Field '+field+' not in file')
            continue
        field_dic_file = _ncvar_to_dict(ncvars[MF_FIELD_NAMES_DICT[field]])
        if field_dic_file['data'].shape == field_shape:
            field_dic_file['data'] = np.ma.transpose(
                field_dic_file['data'], axes=[0, 2, 1])
            if field == 'radar_estimated_rain_rate':
                # put 1/100 mm/ 5 min in mm/h
                field_dic_file['data'] = 0.12*field_dic_file['data']
            field_dic = get_metadata(field)  # get field definition from Py-ART
            field_dic['data'] = field_dic_file['data']
            fields[field] = field_dic
        else:
            bad_shape = field_dic_file['data'].shape
            warn(
                'Field %s skipped due to incorrect shape %s'
                % (field, bad_shape))

    # radar_ variables
    if 'radar_image_latitude' in ncvars:
        radar_latitude = _ncvar_to_dict(ncvars['radar_image_latitude'])
    else:
        radar_latitude = None

    if 'radar_image_longitude' in ncvars:
        radar_longitude = _ncvar_to_dict(ncvars['radar_image_longitude'])
    else:
        radar_longitude = None

    if 'radar_altitude' in ncvars:
        radar_altitude = _ncvar_to_dict(ncvars['radar_altitude'])
    else:
        radar_altitude = None

    if 'radar_image_name' in ncvars:
        radar_name = _ncvar_to_dict(ncvars['radar_image_name'])
    else:
        radar_name = None

    if 'radar_image_time' in ncvars:
        radar_time = _ncvar_to_dict(ncvars['radar_image_time'])
    else:
        radar_time = None

    # do not close file if field loading is delayed
    if not delay_field_loading:
        ncobj.close()

    return Grid(
        time, fields, metadata,
        origin_latitude, origin_longitude, origin_altitude, x, y, z,
        projection=projection,
        radar_latitude=radar_latitude, radar_longitude=radar_longitude,
        radar_altitude=radar_altitude, radar_name=radar_name,
        radar_time=radar_time)
