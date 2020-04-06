"""
pyart.aux_io.rad4alp_bin_reader
===============================

Routines for putting MeteoSwiss operational radar data contained in binary
files into grid object.

.. autosummary::
    :toctree: generated/

    read_bin

"""

import os
import datetime
from warnings import warn

import numpy as np

from ..config import FileMetadata
from ..io.common import _test_arguments
from ..core.grid import Grid
from ..util import ma_broadcast_to

BIN_FIELD_NAMES = {
    'ACC': 'rainfall_accumulation',
    'ARC': 'rainfall_accumulation'
}


def read_bin(filename, additional_metadata=None, chy0=255., chx0=-160.,
             xres=1., yres=1., nx=710, ny=640, nz=1, **kwargs):

    """
    Read a MeteoSwiss operational radar data binary file.

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
        Grid object containing data the data.

    """
    # test for non empty kwargs
    _test_arguments(kwargs)

    try:
        with open(filename, 'rb') as file:
            file.readline()
            data = np.fromfile(file, dtype=np.dtype('float32'), count=nx*ny*4)
            data = np.ma.masked_equal(data, -1.)
            data = np.transpose(np.reshape(data, [nx, ny], order='F'))[::-1, :]
    except EnvironmentError as ee:
        warn(str(ee))
        warn('Unable to read file '+filename)
        return None, None

    # reserved_variables = [
    #     'time', 'x', 'y', 'z',
    #     'origin_latitude', 'origin_longitude', 'origin_altitude',
    #     'point_x', 'point_y', 'point_z', 'projection',
    #     'point_latitude', 'point_longitude', 'point_altitude',
    #     'radar_latitude', 'radar_longitude', 'radar_altitude',
    #     'radar_name', 'radar_time', 'base_time', 'time_offset',
    #     'ProjectionCoordinateSystem']

    # metadata
    metadata = dict()

    filemetadata = FileMetadata('BIN', BIN_FIELD_NAMES, additional_metadata)

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
    prod_time = (
        datetime.datetime.strptime(bfile[3:12], '%y%j%H%M') -
        datetime.timedelta(minutes=1440))
    time['units'] = 'seconds since '+prod_time.strftime('%Y-%m-%d %H:%M:%S')
    time['data'] = np.array([1440.*60.])

    # projection (Swiss Oblique Mercator)
    projection = {
        'proj': 'somerc',
        '_include_lon_0_lat_0': True
    }

    # read in the fields
    datatype = bfile[0:3]

    fields = {}
    field = filemetadata.get_field_name(datatype)
    field_dict = filemetadata(field)
    field_dict['data'] = ma_broadcast_to(data, (nz, ny, nx))
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
