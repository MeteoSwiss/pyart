"""
pyart.aux_io.mf_bin_reader
==========================

Routines for putting MeteoFrance operational radar data contained in binary
files into grid object.

.. autosummary::
    :toctree: generated/

    read_bin_mf

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


def read_bin_mf(filename, additional_metadata=None, xres=1., yres=1., nx=1536,
                ny=1536, nz=1, dtype='float32', date_format='%Y%m%d',
                added_time=86400., x_offset=-619652.074056,
                y_offset=-3526818.337932, lat_0=90., lon_0=0., proj='gnom',
                field_name='rainfall_accumulation', **kwargs):

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
    dtype : str
        data type
    date_format : str
        date format in file name
    added_time : float
        seconds to add to the nominal time in the file name. The default will
        add 24h and it is used for the 24h accumulation files at MF
    x_offset, y_offset : x and y offset from origin of coordinates of the
        projection (m). Assumes stereo-polar. The default corresponds to the
        northwest corner of the Metropolitan French radar composite
        -9.965 53.670 (deg) in stereo-polar projection
    lat_0, lon_0 : latitude and longitude of the origin of the projection
        (deg). Default corresponds to polar stereographic
    proj : str
        projection. Can be webmerc, stere or gnom
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
        with open(filename, 'rb') as file:
            data = np.fromfile(file, dtype=np.dtype(dtype), count=nx*ny)
            # print(np.unique(data))
            # print(data)
            data = np.transpose(np.reshape(data, [nx, ny], order='F'))[::-1, :]
            if dtype == 'int32':
                # not illuminated for rain accu:
                data = np.ma.masked_equal(data, 65535)
                # not illuminated for dBZ:
                data = np.ma.masked_equal(data, 2047)
                # not illuminated for height:
                data = np.ma.masked_equal(data, 4095)
                data[data == -9999000] = 0  # 0 value for 5 min rain accu
                data = data.astype('float')
                # if field_name == 'rainfall_accumulation':
                #     # accumulation expressed in mm/100
                #     data /= 100.
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
    metadata = {}

    filemetadata = FileMetadata('BIN', BIN_FIELD_NAMES, additional_metadata)

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

    # origin of projection
    origin_latitude['data'] = np.array([lat_0])
    origin_longitude['data'] = np.array([lon_0])
    origin_altitude['data'] = np.array([0.])

    if proj == 'webmerc':
        x['data'] = 1000.*(np.arange(nx)*xres+xres/2.)+x_offset
        y['data'] = 1000.*(np.arange(ny)*yres+yres/2.)+y_offset
        z['data'] = np.array([0.])

        # projection (web mercator)
        projection = {
            'proj': 'webmerc',
            'ellps': 'WGS84',
            'datum': 'WGS84',
        }
    elif proj == 'stere' or proj =='gnom':
        x_vals = 1000.*(np.arange(nx)*xres+xres/2.)+x_offset
        y_vals = y_offset-1000.*(np.arange(ny)*yres+yres/2.)
        x['data'] = x_vals
        y['data'] = y_vals[::-1]
        z['data'] = np.array([0.])

        # projection (stereo-polar)
        projection = {
            'proj': proj,
            'ellps': 'WGS84',
            'datum': 'WGS84',
            'lat_0': lat_0,
            'lon_0': lon_0
        }
        if proj == 'stere':
            projection.update({'lat_ts': 45.})
    else:
        raise ValueError('Accepted projections: webmerc, stere or gnom')

    # Time
    prod_time = find_date_in_file_name(filename, date_format=date_format)
    time['units'] = 'seconds since '+prod_time.strftime('%Y-%m-%d %H:%M:%S')
    time['data'] = np.array([added_time])

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


def find_date_in_file_name(filename, date_format='%Y%m%d%H%M%S'):
    """
    Find a date with date format defined in date_format in a file name.
    If no date is found returns None

    Parameters
    ----------
    filename : str
        file name
    date_format : str
        The time format

    Returns
    -------
    fdatetime : datetime object
        date and time in file name

    """
    today = datetime.datetime.now()
    len_datestr = len(today.strftime(date_format))
    count = 0
    bfile = os.path.basename(filename)
    while True:
        try:
            fdatetime = datetime.datetime.strptime(
                bfile[count:count+len_datestr], date_format)
        except ValueError:
            count += 1
            if count+len_datestr > len(bfile):
                warn(f'Unable to find date from string name. Date format '
                     f'{date_format}. File name {bfile}')
                return None
        else:
            # No error, stop the loop
            break

    return fdatetime
