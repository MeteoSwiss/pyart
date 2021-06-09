"""
pyart.aux_io.mf_dat_reader
==========================

Routines for putting MeteoFrance operational radar data contained in text
files into grid object.

.. autosummary::
    :toctree: generated/

    read_dat_mf

"""

import os
import datetime
from warnings import warn

import numpy as np

from ..config import FileMetadata
from ..io.common import _test_arguments
from ..core.grid import Grid
from ..util import ma_broadcast_to

DAT_FIELD_NAMES = {
    'RR': 'radar_estimated_rain_rate'
}


def read_dat_mf(filename, additional_metadata=None, xres=1., yres=1., nx=1536,
                ny=1536, nz=1, nd=65535,
                field_name='radar_estimated_rain_rate', **kwargs):

    """
    Read a MeteoFrance operational radar data binary file. The data is in
    stereopolar projection

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
    xres, yres : float
        resolution of each grid point [km]
    nx, ny, nz : int
        dimensions of the grid
    nd : int
        integer representing No Data
    field_name : str
        name of the field stored in the binary file

    Returns
    -------
    grid : Grid
        Grid object containing data the data.

    """
    # test for non empty kwargs
    _test_arguments(kwargs)

    try:
        with open(filename, 'r') as file:
            data = np.loadtxt(file, comments='#')
            data = np.ma.masked_equal(data, nd)
            # conversion to mm/h
            # original data in 0.1 mm/5 min
            data = 1.2*data
            data = data[::-1, :]
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

    filemetadata = FileMetadata('DAT', DAT_FIELD_NAMES, additional_metadata)

    # required reserved variables
    time = filemetadata('grid_time')
    origin_latitude = filemetadata('origin_latitude')
    origin_longitude = filemetadata('origin_longitude')
    origin_altitude = filemetadata('origin_altitude')
    x = filemetadata('x')
    y = filemetadata('y')
    z = filemetadata('z')

    # Nortwest corner
    # -619652.074056 -3526818.337932 m
    # -9.965 53.670 NW (deg)

    x_vals = 1000.*(np.arange(nx)*xres+xres/2.)-619652.074056
    y_vals = -3526818.337932-1000.*(np.arange(ny)*yres+yres/2.)
    x['data'] = x_vals
    y['data'] = y_vals[::-1]
    z['data'] = np.array([0.])

    # origin of stereo-polar projection
    origin_latitude['data'] = np.array([90.])
    origin_longitude['data'] = np.array([0.])
    origin_altitude['data'] = np.array([0.])

    # projection (stereo-polar)
    projection = {
        'proj': 'stere',
        'lat_ts': 45.,
        'ellps': 'WGS84',
        'datum': 'WGS84',
        'lat_0': 90.,
        'lon_0': 0.
    }

    bfile = os.path.basename(filename)
    # Time
    prod_time = datetime.datetime.strptime(bfile[:12], '%Y%m%d%H%M')
    time['units'] = 'seconds since '+prod_time.strftime('%Y-%m-%d %H:%M:%S')
    time['data'] = np.array([0.])

    # read in the fields
    fields = {}
    field_dict = filemetadata(field_name)
    field_dict['data'] = ma_broadcast_to(data, (nz, ny, nx))
    fields[field_name] = field_dict

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
