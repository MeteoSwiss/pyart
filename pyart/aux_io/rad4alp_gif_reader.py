"""
pyart.aux_io.rad4alp_gif_reader
======================================

Routines for putting MeteoSwiss operational radar data contained in gif files
into grid object.

.. autosummary::
    :toctree: generated/

    read_gif
    _get_metadata
    _get_datatype_from_file
    _get_physical_data

"""

import os
import datetime
from warnings import warn

import numpy as np

# check existence of imageio
try:
    from imageio import imread
    _IMAGEIO_AVAILABLE = True
except ImportError:
    _IMAGEIO_AVAILABLE = False

from ..config import FileMetadata
from ..io.common import _test_arguments
from ..core.grid import Grid
from ..exceptions import MissingOptionalDependency

GIF_FIELD_NAMES = {
    'CPC0005': 'radar_estimated_rain_rate',
    'CPC0060': 'rainfall_accumulation',
    'CPC0180': 'rainfall_accumulation',
    'CPC0360': 'rainfall_accumulation',
    'CPC0720': 'rainfall_accumulation',
    'CPC1440': 'rainfall_accumulation',
    'CPC2880': 'rainfall_accumulation',
    'CPC4320': 'rainfall_accumulation',
    'CPCH0005': 'radar_estimated_rain_rate',
    'CPCH0060': 'rainfall_accumulation',
    'CPCH0180': 'rainfall_accumulation',
    'CPCH0360': 'rainfall_accumulation',
    'CPCH0720': 'rainfall_accumulation',
    'CPCH1440': 'rainfall_accumulation',
    'CPCH2880': 'rainfall_accumulation',
    'CPCH4320': 'rainfall_accumulation'
}


def read_gif(filename, additional_metadata=None, chy0=255., chx0=-160.,
             xres=1., yres=1., nx=710, ny=640, nz=1, **kwargs):

    """
    Read a MeteoSwiss operational radar data gif file.

    Parameters
    ----------
    filename : str
        Name of the file to read.
    additional_metadata : dict of dicts, optional
        Dictionary of dictionaries to retrieve metadata during this read.
        This metadata is not used during any successive file reads unless
        explicitly included.  A value of None, the default, will not
        introduct any addition metadata and the file specific or default
        metadata as specified by the Py-ART configuration file will be used.
    chy0, chx0 : float
        Swiss coordinates position of the south-western point in the grid
    xres, yres : float
        resolution of each grid point [km]
    nx, ny, nz : int
        dimensions of the grid

    Returns
    -------
    grid : Grid
        Grid object containing the data.

    """
    # check that wradlib is available
    if not _IMAGEIO_AVAILABLE:
        raise MissingOptionalDependency(
            "imageio is required to use read_gif but is not installed")

    # test for non empty kwargs
    _test_arguments(kwargs)

    try:
        ret = imread(filename, format='gif')
    except OSError:
        warn('Unable to read file '+filename)
        return None

    # reserved_variables = [
    #     'time', 'x', 'y', 'z',
    #     'origin_latitude', 'origin_longitude', 'origin_altitude',
    #     'point_x', 'point_y', 'point_z', 'projection',
    #     'point_latitude', 'point_longitude', 'point_altitude',
    #     'radar_latitude', 'radar_longitude', 'radar_altitude',
    #     'radar_name', 'radar_time', 'base_time', 'time_offset',
    #     'ProjectionCoordinateSystem']

    # metadata
    metadata = _get_metadata(ret.meta)

    filemetadata = FileMetadata('GIF', GIF_FIELD_NAMES, additional_metadata)

    # required reserved variables
    time = filemetadata('grid_time')
    origin_latitude = filemetadata('origin_latitude')
    origin_longitude = filemetadata('origin_longitude')
    origin_altitude = filemetadata('origin_altitude')
    x = filemetadata('x')
    y = filemetadata('y')
    z = filemetadata('z')

    x['data'] = 1000.*(np.arange(nx)*xres+chy0+xres/2.-600.)
    y['data'] = 1000.*(np.arange(ny)*yres+chx0+yres/2.-200.)
    z['data'] = np.array([0.])

    # Origin of LV03 Swiss coordinates
    origin_latitude['data'] = np.array([46.951082877])
    origin_longitude['data'] = np.array([7.438632495])
    origin_altitude['data'] = np.array([0.])

    bfile = os.path.basename(filename)
    # Time
    prod_time = datetime.datetime.strptime(bfile[3:12], '%y%j%H%M')
    time['units'] = 'seconds since '+prod_time.strftime('%Y-%m-%d %H:%M:%S')
    time['data'] = np.array([0])

    # projection (Swiss Oblique Mercator)
    projection = {
        'proj': 'somerc',
        '_include_lon_0_lat_0': True
    }

    # read in the fields
    datatype = _get_datatype_from_file(bfile)
    if datatype is None:
        return None

    fields = {}
    field = filemetadata.get_field_name(datatype)
    field_dict = filemetadata(field)
    field_dict['data'] = np.broadcast_to(
        _get_physical_data(ret, datatype, prod_time)[::-1, :], (nz, ny, nx))
    fields[field] = field_dict

    # radar variables
    radar_latitude = None
    radar_longitude = None
    radar_altitude = None
    radar_name = None
    radar_time = None

    return Grid(
        time, fields, metadata,
        origin_latitude, origin_longitude, origin_altitude, x, y, z,
        projection=projection,
        radar_latitude=radar_latitude, radar_longitude=radar_longitude,
        radar_altitude=radar_altitude, radar_name=radar_name,
        radar_time=radar_time)


def _get_metadata(raw_metadata):
    """
    puts metadata in a dictionary

    Parameters
    ----------
    raw_metadata : str
        dictionary

    Returns
    -------
    datatype : str
        Data type contained in the file

    """
    if 'comment' not in raw_metadata:
        return dict()

    raw_comments = str(raw_metadata['comment'], 'utf-8')
    raw_comments = raw_comments.split('"')

    comments = []
    for comment in raw_comments:
        comments.append(comment.lstrip())

    metadata_dict = dict()
    for i, comment in enumerate(comments):
        if '=' in comment:
            comments_aux = comment.split(' ')
            for comment_aux in comments_aux:
                comment_aux2 = comment_aux.split('=')
                if comment_aux2[1] == '':
                    metadata_dict.update({comment_aux2[0]: comments[i+1]})
                else:
                    metadata_dict.update({comment_aux2[0]: comment_aux2[1]})

    return metadata_dict


def _get_datatype_from_file(filename):
    """
    gets data type from file name

    Parameters
    ----------
    filename : str
        base name of the file

    Returns
    -------
    datatype : str
        Data type contained in the file

    """
    if filename.startswith('CPC'):
        acronym = 'CPC'
        if filename.endswith('_00005.801.gif'):
            accu_time = '0005'
        elif filename.endswith('_00060.801.gif'):
            accu_time = '0060'
        elif filename.endswith('_00180.801.gif'):
            accu_time = '0180'
        elif filename.endswith('_00360.801.gif'):
            accu_time = '0360'
        elif filename.endswith('_00720.801.gif'):
            accu_time = '0720'
        elif filename.endswith('_01440.801.gif'):
            accu_time = '1440'
        elif filename.endswith('_02880.801.gif'):
            accu_time = '2880'
        elif filename.endswith('_04320.801.gif'):
            accu_time = '4320'
        else:
            warn('Unknown CPC product')
            return None
        return acronym+accu_time

    warn('Unknown product')
    return None


def _get_physical_data(rgba_data, datatype, prod_time):
    """
    gets data in physical units

    Parameters
    ----------
    rgba_data : uint8 ndarray
        the data as 4 channel rgba
    datatype : str
        The data type
    prod_time : datetime object
        The date at which the product was generated

    Returns
    -------
    data : float ndarray
        the data in physical units

    """
    if datatype.startswith('CPC'):
        scale = np.ma.masked_all(256)
        ind = np.arange(256.)+1
        if prod_time > datetime.datetime(2018, 6, 28):
            scale[2:251] = np.ma.power(
                np.ma.power(10., (ind[2:251]-71.5)/20.)/316., 0.6666667)
        else:
            scale[2:251] = np.ma.power(
                np.ma.power(10., (ind[1:250]-71.5)/20.)/316., 0.6666667)

        ind_vals = 255-rgba_data[:, :, 1]
        data = scale[ind_vals]

        return data
