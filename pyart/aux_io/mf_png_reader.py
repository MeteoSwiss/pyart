"""
pyart.aux_io.mf_png_reader
==========================

Routines for putting MeteoSwiss operational radar data contained in gif files
into grid object.

.. autosummary::
    :toctree: generated/

    read_png
    _get_datatype_from_file
    _get_physical_data

"""

import os
import datetime
from warnings import warn
from copy import deepcopy

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
from ..util import ma_broadcast_to

PNG_FIELD_NAMES = {
    'hail': 'precipitation_type',
    'snow': 'precipitation_type',
    'rain': 'precipitation_type',
    'ice': 'precipitation_type'
}


def read_png(filename, additional_metadata=None, xres=1.25, yres=1.25, nx=1840,
             ny=1670, nz=1, **kwargs):

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
        ret = imread(filename, format='png', pilmode='I')
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
    filemetadata = FileMetadata('PNG', PNG_FIELD_NAMES, additional_metadata)

    # required reserved variables
    time = filemetadata('grid_time')
    origin_latitude = filemetadata('origin_latitude')
    origin_longitude = filemetadata('origin_longitude')
    origin_altitude = filemetadata('origin_altitude')
    x = filemetadata('x')
    y = filemetadata('y')
    z = filemetadata('z')

    x['data'] = 1000.*(np.arange(nx)*xres+xres/2.)-889375.418
    y['data'] = 1000.*(np.arange(ny)*yres+yres/2.)+4794775.243
    z['data'] = np.array([0.])

    # origin of webmercator
    origin_latitude['data'] = np.array([0.])
    origin_longitude['data'] = np.array([0.])
    origin_altitude['data'] = np.array([0.])

    # projection (web mercator)
    projection = {
        'proj': 'webmerc',
        'ellps': 'WGS84',
        'datum': 'WGS84',
    }

    bfile = os.path.basename(filename).split('.')[0]
    # Time
    prod_time = datetime.datetime.strptime(bfile.split('_')[-1], '%Y%m%d')
    time['units'] = 'seconds since '+prod_time.strftime('%Y-%m-%d 00:00:00')
    time['data'] = np.array([0])

    # read in the fields
    datatype = _get_datatype_from_file(bfile)
    if datatype is None:
        return None

    fields = {}
    field = filemetadata.get_field_name(datatype)
    field_dict = filemetadata(field)
    field_dict['data'] = ma_broadcast_to(
        _get_physical_data(ret, datatype)[::-1, :], (nz, ny, nx))
    fields[field] = field_dict

    # radar variables
    radar_latitude = None
    radar_longitude = None
    radar_altitude = None
    radar_name = None
    radar_time = None

    return Grid(
        time, fields, dict(), origin_latitude, origin_longitude,
        origin_altitude, x, y, z, projection=projection,
        radar_latitude=radar_latitude, radar_longitude=radar_longitude,
        radar_altitude=radar_altitude, radar_name=radar_name,
        radar_time=radar_time)


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
    if 'GRELE' in filename:
        datatype = 'hail'
    elif 'PLUIE' in filename:
        datatype = 'rain'
    elif 'NEIGE' in filename:
        datatype = 'snow'
    elif 'VERGLAS' in filename:
        datatype = 'ice'
    else:
        warn('Unknown HYDRE product')
        return None
    return datatype


def _get_physical_data(data, datatype):
    """
    gets data in physical units

    Parameters
    ----------
    data : int ndarray
        the data as int
    datatype : str
        The data type

    Returns
    -------
    data_ph : float ndarray
        the data in physical units

    """
    data_ph = deepcopy(data)
    # print(np.unique(data))

    #   0 Précipitations indétectables/Absence de neige au sol
    #   1 Précipitations indétectables/Présence de neige au sol
    #   2 Précipitations indétectables/Sol invisible (nuit ou nuages)
    #   3 Bruine
    #   4 Bruine sur sol gelé
    #   5 Bruine verglaçante
    #   6 Pluie
    #   7 Pluie sur sol gelé
    #   8 Pluie verglaçante
    #   9 Pluie et neige mêlées
    #  10 Neige mouillée
    #  11 Neige mouillée tenant au sol
    #  12 Neige humide
    #  13 Neige humide tenant au sol
    #  14 Neige sèche
    #  15 Neige sèche tenant au sol
    #  16 Granules de glace
    #  17 Grésil/Petite grêle
    #  18 Moyenne grêle
    #  19 Grosse grêle
    # 255 Indéterminé

    # Precip not detected
    data_ph[data == 0] = 0

    if datatype == 'rain':
        # rain categories
        data_ph[data == 174] = 3
        data_ph[data == 113] = 6
    elif datatype == 'ice':
        # ice categories
        data_ph[data == 77] = 4
        data_ph[data == 166] = 5

        data_ph[data == 121] = 7
        data_ph[data == 216] = 8
    elif datatype == 'snow':
        # snow categories
        data_ph[data == 105] = 9

        data_ph[data == 15] = 10
        data_ph[data == 26] = 11
        data_ph[data == 101] = 12

        data_ph[data == 113] = 13
        data_ph[data == 143] = 14
        data_ph[data == 191] = 15
    elif datatype == 'hail':
        # hail categories
        data_ph[data == 226] = 17
        data_ph[data == 189] = 18
        data_ph[data == 151] = 19

    # no data
    data_ph[data == 171] = 20

    return data_ph
