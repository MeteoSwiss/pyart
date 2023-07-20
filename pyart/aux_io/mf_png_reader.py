"""
pyart.aux_io.mf_png_reader
==========================

Routines for putting MeteoFrance operational radar data contained in png files
into grid object.

.. autosummary::
    :toctree: generated/

    read_png
    _get_datatype_from_file
    _get_physical_data

"""

import os
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
from .mf_bin_reader import find_date_in_file_name

PNG_FIELD_NAMES = {
    'hail': 'precipitation_type',
    'snow': 'precipitation_type',
    'rain': 'precipitation_type',
    'ice': 'precipitation_type'
}


def read_png(filename, additional_metadata=None, xres=1.25, yres=1.25,
             nz=1, date_format='%Y%m%d', added_time=0.,
             x_offset=-889375.418, y_offset=4794775.243, lat_0=0., lon_0=0.,
             proj='webmerc', field_name='precipitation_type', **kwargs):
    """
    Read a MeteoFrance data png file.

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
    date_format : str
        date format in file name
    added_time : float
        seconds to add to the nominal time in the file name.
    x_offset, y_offset : x and y offset from origin of coordinates of the
        projection (m). The default corresponds to the
        northwest corner of the Metropolitan French radar composite
        -9.965 53.670 (deg) in web mercator projection
    lat_0, lon_0 : latitude and longitude of the origin of the projection
        (deg). Default corresponds to web mercator projection
    proj : str
        projection. Can be webmerc or stere
    field_name : str
        name of the field stored in the png file

    Returns
    -------
    grid : Grid
        Grid object containing the data.

    """
    # check that wradlib is available
    if not _IMAGEIO_AVAILABLE:
        raise MissingOptionalDependency(
            "imageio is required to use read_png but is not installed")

    # test for non empty kwargs
    _test_arguments(kwargs)

    try:
        # pilmode can be:
        # 'L' (8-bit pixels, grayscale)
        # 'P' (8-bit pixels, mapped to any other mode using a color palette)
        # 'RGB' (3x8-bit pixels, true color)
        # 'RGBA' (4x8-bit pixels, true color with transparency mask)
        # 'CMYK' (4x8-bit pixels, color separation)
        # 'YCbCr' (3x8-bit pixels, color video format)
        # 'I' (32-bit signed integer pixels)
        # 'F' (32-bit floating point pixels)
        # 'LA' (L with alpha) limited support
        # 'RGBX' (true color with padding) limited support
        # 'RGBa' (true color with premultiplied alpha) limited support
        # When translating a color image to grayscale (mode 'L', 'I' or 'F'),
        # the library uses the ITU-R 601-2 luma transform:

        #    L = R * 299/1000 + G * 587/1000 + B * 114/1000

        ret = imread(filename, format='png', pilmode='RGBA')
        r_ch = ret[:, :, 0].astype(int)
        g_ch = ret[:, :, 1].astype(int)
        b_ch = ret[:, :, 2].astype(int)
        transparency = ret[:, :, 3].astype(int)

        nx = transparency.shape[1]
        ny = transparency.shape[0]

    except (OSError, SyntaxError):
        warn('Unable to read file ' + filename)
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

    # origin of projection
    origin_latitude['data'] = np.array([lat_0])
    origin_longitude['data'] = np.array([lon_0])
    origin_altitude['data'] = np.array([0.])

    if proj == 'webmerc':
        # daily accumulation of hydre hydrometeor classification on the ground
        # coordinates of the corners (x, y) (lon, lat):
        # Upper Left    ( -889375.418, 6882275.273)  (-7.9893953,52.4520337)
        # Lower Left    ( -889375.418, 4794775.243)  (-7.9893953,39.5085100)
        # Upper Right   ( 1410624.572, 6882275.273)  (12.6718561,52.4520337)
        # Lower Right   ( 1410624.572, 4794775.243)  (12.6718561,39.5085100)

        x['data'] = 1000. * (np.arange(nx) * xres + xres / 2.) + x_offset
        y['data'] = 1000. * (np.arange(ny) * yres + yres / 2.) + y_offset
        z['data'] = np.array([0.])

        # projection (web mercator)
        projection = {
            'proj': 'webmerc',
            'ellps': 'WGS84',
            'datum': 'WGS84',
        }
    elif proj == 'stere' or proj == 'gnom':
        x_vals = 1000. * (np.arange(nx) * xres + xres / 2.) + x_offset
        y_vals = y_offset - 1000. * (np.arange(ny) * yres + yres / 2.)
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
    time['units'] = 'seconds since ' + prod_time.strftime('%Y-%m-%d %H:%M:%S')
    time['data'] = np.array([added_time])

    # read in the fields
    datatype = None
    if field_name == 'precipitation_type':
        # only needed for hydre data
        datatype = _get_datatype_from_file(os.path.basename(filename))

    fields = {}
    field_dict = filemetadata(field_name)
    field_dict['data'] = ma_broadcast_to(
        _get_physical_data(
            r_ch, g_ch, b_ch, transparency, field_name=field_name,
            datatype=datatype)[::-1, :],
        (nz, ny, nx))
    fields[field_name] = field_dict

    # radar variables
    radar_latitude = None
    radar_longitude = None
    radar_altitude = None
    radar_name = None
    radar_time = None

    return Grid(
        time, fields, {}, origin_latitude, origin_longitude,
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


def _get_physical_data(r_ch, g_ch, b_ch, transparency,
                       field_name='precipitation_type', datatype='rain'):
    """
    gets data in physical units

    Parameters
    ----------
    r_ch, g_ch, b_ch : int ndarray
        the red, green and blue channels
    transparency : int ndarray
        the transparency channel
    datatype : str
        The data type
    field_name : str
        The field name

    Returns
    -------
    data_ph : float ndarray
        the data in physical units

    """
    if field_name == 'precipitation_type':
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

        data = np.round(
            r_ch *
            299 /
            1000 +
            g_ch *
            587 /
            1000 +
            b_ch *
            114 /
            1000).astype(int)
        data_ph = deepcopy(data)

        if datatype == 'rain':
            # rain categories
            data_ph[data == 174] = 3
            data_ph[data == 113] = 6

            data_ph[data == 0] = 0  # Precip not detected
            data_ph[data == 171] = 20  # no data
        elif datatype == 'ice':
            # ice categories
            data_ph[data == 77] = 4
            data_ph[data == 166] = 5

            data_ph[data == 121] = 7
            data_ph[data == 216] = 8

            data_ph[data == 0] = 0  # Precip not detected
            data_ph[data == 171] = 20  # no data
        elif datatype == 'snow':
            # # snow categories
            data_ph[data == 105] = 9  # rain/snow

            data_ph[data == 15] = 10  # WS
            data_ph[data == 101] = 11  # WS on ground

            data_ph[data == 26] = 12  # HS
            data_ph[data == 143] = 13  # HS on ground

            data_ph[data == 113] = 14  # DS
            data_ph[data == 191] = 15  # DS on ground

            data_ph[data == 0] = 0  # Precip not detected
            data_ph[data == 171] = 20  # no data
        elif datatype == 'hail':
            # hail categories
            data_ph[data == 226] = 17
            data_ph[data == 189] = 18
            data_ph[data == 151] = 19

            data_ph[data == 0] = 0  # Precip not detected
            data_ph[data == 171] = 20  # no data

    elif field_name == 'radar_rainrate_relation':
        data = np.round(
            r_ch *
            299 /
            1000 +
            g_ch *
            587 /
            1000 +
            b_ch *
            114 /
            1000).astype(int)
        data_ph = deepcopy(data)
        data_ph[data == 170] = 1  # no radar-R relation
        data_ph[data == 120] = 2  # Z-R
        data_ph[data == 150] = 3  # Z-S
        data_ph[data == 94] = 4  # KDP-R
        data_ph = np.ma.masked_where(transparency == 0, data_ph)  # no data

    elif field_name == 'radar_echo_classification_MF':
        # 'labels': [
        #    'ZH_MQT' 1, 'SOL' 2, 'INSECTES' 3, 'OISEAUX' 4, 'MER_CHAFF' 5,
        #    'PARASITES' 6, 'ROND_CENTRAL' 7, 'PRECIP_INDIFFERENCIEE' 8,
        #    'PLUIE' 9, 'NEIGE_MOUILLEE' 10, 'NEIGE_SECHE' 11, 'GLACE' 12,
        #    'PETITE_GRELE' 13, 'MOYENNE_GRELE' 14, 'GROSSE_GRELE' 15]
        # 1 to 15
        data_ph = np.ma.masked_all(transparency.shape, dtype=int)
        data_ph[
            (r_ch == int('DC', 16))
            & (g_ch == int('DC', 16))
            & (b_ch == int('DC', 16))] = 1
        data_ph[
            (r_ch == int('00', 16))
            & (g_ch == int('00', 16))
            & (b_ch == int('00', 16))] = 2
        data_ph[
            (r_ch == int('CD', 16))
            & (g_ch == int('37', 16))
            & (b_ch == int('00', 16))] = 3
        data_ph[
            (r_ch == int('FF', 16))
            & (g_ch == int('45', 16))
            & (b_ch == int('00', 16))] = 4
        data_ph[
            (r_ch == int('FF', 16))
            & (g_ch == int('A5', 16))
            & (b_ch == int('00', 16))] = 5
        data_ph[
            (r_ch == int('8B', 16))
            & (g_ch == int('69', 16))
            & (b_ch == int('14', 16))] = 6
        data_ph[
            (r_ch == int('EE', 16))
            & (g_ch == int('B4', 16))
            & (b_ch == int('22', 16))] = 7
        data_ph[
            (r_ch == int('FF', 16))
            & (g_ch == int('D7', 16))
            & (b_ch == int('00', 16))] = 8
        data_ph[
            (r_ch == int('00', 16))
            & (g_ch == int('8B', 16))
            & (b_ch == int('00', 16))] = 9
        data_ph[
            (r_ch == int('00', 16))
            & (g_ch == int('CD', 16))
            & (b_ch == int('00', 16))] = 10
        data_ph[
            (r_ch == int('00', 16))
            & (g_ch == int('FF', 16))
            & (b_ch == int('00', 16))] = 11
        data_ph[
            (r_ch == int('9A', 16))
            & (g_ch == int('CD', 16))
            & (b_ch == int('32', 16))] = 12
        data_ph[
            (r_ch == int('00', 16))
            & (g_ch == int('FF', 16))
            & (b_ch == int('FF', 16))] = 13
        data_ph[
            (r_ch == int('00', 16))
            & (g_ch == int('CD', 16))
            & (b_ch == int('CD', 16))] = 14
        data_ph[
            (r_ch == int('48', 16))
            & (g_ch == int('76', 16))
            & (b_ch == int('FF', 16))] = 15

        # used in fuzzy logic clutter detection:
        data_ph[
            (r_ch == int('3A', 16))
            & (g_ch == int('5F', 16))
            & (b_ch == int('CD', 16))] = 8

        data_ph = np.ma.masked_where(
            (r_ch == int('FF', 16))
            & (g_ch == int('FF', 16))
            & (b_ch == int('FF', 16)), data_ph)  # noise

        data_ph = np.ma.masked_where(transparency == 0, data_ph)

    elif field_name == 'rainfall_accumulation':
        data_ph = np.ma.masked_all(transparency.shape, dtype=float)

        data_ph[
            (r_ch == int('FF', 16))
            & (g_ch == int('FF', 16))
            & (b_ch == int('FF', 16))] = 0
        data_ph[
            (r_ch == int('28', 16))
            & (g_ch == int('00', 16))
            & (b_ch == int('50', 16))] = 4 / 100
        data_ph[
            (r_ch == int('24', 16))
            & (g_ch == int('00', 16))
            & (b_ch == int('64', 16))] = 8 / 100
        data_ph[
            (r_ch == int('20', 16))
            & (g_ch == int('00', 16))
            & (b_ch == int('78', 16))] = 11 / 100
        data_ph[
            (r_ch == int('1C', 16))
            & (g_ch == int('00', 16))
            & (b_ch == int('8C', 16))] = 15 / 100
        data_ph[
            (r_ch == int('18', 16))
            & (g_ch == int('00', 16))
            & (b_ch == int('A0', 16))] = 19 / 100
        data_ph[
            (r_ch == int('20', 16))
            & (g_ch == int('10', 16))
            & (b_ch == int('B8', 16))] = 23 / 100
        data_ph[
            (r_ch == int('2C', 16))
            & (g_ch == int('20', 16))
            & (b_ch == int('D0', 16))] = 26 / 100
        data_ph[
            (r_ch == int('34', 16))
            & (g_ch == int('30', 16))
            & (b_ch == int('E8', 16))] = 30 / 100
        data_ph[
            (r_ch == int('40', 16))
            & (g_ch == int('40', 16))
            & (b_ch == int('FC', 16))] = 34 / 100
        data_ph[
            (r_ch == int('30', 16))
            & (g_ch == int('5C', 16))
            & (b_ch == int('E8', 16))] = 38 / 100
        data_ph[
            (r_ch == int('20', 16))
            & (g_ch == int('78', 16))
            & (b_ch == int('D4', 16))] = 42 / 100
        data_ph[
            (r_ch == int('10', 16))
            & (g_ch == int('94', 16))
            & (b_ch == int('C0', 16))] = 45 / 100
        data_ph[
            (r_ch == int('00', 16))
            & (g_ch == int('AC', 16))
            & (b_ch == int('AC', 16))] = 49 / 100
        data_ph[
            (r_ch == int('20', 16))
            & (g_ch == int('C0', 16))
            & (b_ch == int('C0', 16))] = 53 / 100
        data_ph[
            (r_ch == int('40', 16))
            & (g_ch == int('D4', 16))
            & (b_ch == int('D4', 16))] = 57 / 100
        data_ph[
            (r_ch == int('60', 16))
            & (g_ch == int('E8', 16))
            & (b_ch == int('E8', 16))] = 60 / 100
        data_ph[
            (r_ch == int('80', 16))
            & (g_ch == int('FC', 16))
            & (b_ch == int('FC', 16))] = 64 / 100
        data_ph[
            (r_ch == int('80', 16))
            & (g_ch == int('FC', 16))
            & (b_ch == int('C0', 16))] = 68 / 100
        data_ph[
            (r_ch == int('80', 16))
            & (g_ch == int('FC', 16))
            & (b_ch == int('80', 16))] = 72 / 100
        data_ph[
            (r_ch == int('80', 16))
            & (g_ch == int('FC', 16))
            & (b_ch == int('40', 16))] = 75 / 100
        data_ph[
            (r_ch == int('80', 16))
            & (g_ch == int('FC', 16))
            & (b_ch == int('00', 16))] = 79 / 100
        data_ph[
            (r_ch == int('94', 16))
            & (g_ch == int('FC', 16))
            & (b_ch == int('14', 16))] = 83 / 100
        data_ph[
            (r_ch == int('A8', 16))
            & (g_ch == int('FC', 16))
            & (b_ch == int('28', 16))] = 87 / 100
        data_ph[
            (r_ch == int('BC', 16))
            & (g_ch == int('FC', 16))
            & (b_ch == int('3C', 16))] = 91 / 100
        data_ph[
            (r_ch == int('CC', 16))
            & (g_ch == int('FC', 16))
            & (b_ch == int('50', 16))] = 94 / 100
        data_ph[
            (r_ch == int('D8', 16))
            & (g_ch == int('FC', 16))
            & (b_ch == int('3C', 16))] = 98 / 100
        data_ph[
            (r_ch == int('E4', 16))
            & (g_ch == int('FC', 16))
            & (b_ch == int('28', 16))] = 102 / 100
        data_ph[
            (r_ch == int('F0', 16))
            & (g_ch == int('FC', 16))
            & (b_ch == int('14', 16))] = 106 / 100
        data_ph[
            (r_ch == int('FC', 16))
            & (g_ch == int('FC', 16))
            & (b_ch == int('00', 16))] = 109 / 100
        data_ph[
            (r_ch == int('FC', 16))
            & (g_ch == int('F0', 16))
            & (b_ch == int('10', 16))] = 113 / 100
        data_ph[
            (r_ch == int('FC', 16))
            & (g_ch == int('DC', 16))
            & (b_ch == int('20', 16))] = 117 / 100
        data_ph[
            (r_ch == int('FC', 16))
            & (g_ch == int('CC', 16))
            & (b_ch == int('30', 16))] = 121 / 100
        data_ph[
            (r_ch == int('FC', 16))
            & (g_ch == int('B4', 16))
            & (b_ch == int('44', 16))] = 125 / 100
        data_ph[
            (r_ch == int('FC', 16))
            & (g_ch == int('A8', 16))
            & (b_ch == int('30', 16))] = 128 / 100
        data_ph[
            (r_ch == int('FC', 16))
            & (g_ch == int('98', 16))
            & (b_ch == int('20', 16))] = 132 / 100
        data_ph[
            (r_ch == int('FC', 16))
            & (g_ch == int('8C', 16))
            & (b_ch == int('10', 16))] = 136 / 100
        data_ph[
            (r_ch == int('FC', 16))
            & (g_ch == int('7C', 16))
            & (b_ch == int('00', 16))] = 140 / 100
        data_ph[
            (r_ch == int('FC', 16))
            & (g_ch == int('74', 16))
            & (b_ch == int('00', 16))] = 143 / 100
        data_ph[
            (r_ch == int('FC', 16))
            & (g_ch == int('6C', 16))
            & (b_ch == int('00', 16))] = 147 / 100
        data_ph[
            (r_ch == int('FC', 16))
            & (g_ch == int('5C', 16))
            & (b_ch == int('00', 16))] = 151 / 100
        data_ph[
            (r_ch == int('FC', 16))
            & (g_ch == int('50', 16))
            & (b_ch == int('00', 16))] = 155 / 100
        data_ph[
            (r_ch == int('FC', 16))
            & (g_ch == int('58', 16))
            & (b_ch == int('1C', 16))] = 158 / 100
        data_ph[
            (r_ch == int('FC', 16))
            & (g_ch == int('60', 16))
            & (b_ch == int('3C', 16))] = 162 / 100
        data_ph[
            (r_ch == int('FC', 16))
            & (g_ch == int('68', 16))
            & (b_ch == int('58', 16))] = 166 / 100
        data_ph[
            (r_ch == int('FC', 16))
            & (g_ch == int('74', 16))
            & (b_ch == int('74', 16))] = 170 / 100
        data_ph[
            (r_ch == int('FC', 16))
            & (g_ch == int('58', 16))
            & (b_ch == int('58', 16))] = 174 / 100
        data_ph[
            (r_ch == int('FC', 16))
            & (g_ch == int('3C', 16))
            & (b_ch == int('3C', 16))] = 177 / 100
        data_ph[
            (r_ch == int('FC', 16))
            & (g_ch == int('1C', 16))
            & (b_ch == int('1C', 16))] = 181 / 100
        data_ph[
            (r_ch == int('FC', 16))
            & (g_ch == int('00', 16))
            & (b_ch == int('00', 16))] = 185 / 100
        data_ph[
            (r_ch == int('EC', 16))
            & (g_ch == int('00', 16))
            & (b_ch == int('2C', 16))] = 189 / 100
        data_ph[
            (r_ch == int('D8', 16))
            & (g_ch == int('00', 16))
            & (b_ch == int('5C', 16))] = 192 / 100
        data_ph[
            (r_ch == int('C8', 16))
            & (g_ch == int('00', 16))
            & (b_ch == int('88', 16))] = 196 / 100
        data_ph[
            (r_ch == int('B4', 16))
            & (g_ch == int('00', 16))
            & (b_ch == int('B4', 16))] = 200 / 100
        data_ph[
            (r_ch == int('00', 16))
            & (g_ch == int('00', 16))
            & (b_ch == int('00', 16))] = 10000 / 100

        data_ph = np.ma.masked_where(
            (r_ch == int('BC', 16))
            & (g_ch == int('BC', 16))
            & (b_ch == int('BC', 16)), data_ph)  # no measurement

        data_ph = np.ma.masked_where(transparency == 0, data_ph)

    else:
        warn(f'Unknown scale for field {field_name}.'
             f' Returning gray scale values')
        data = np.round(
            r_ch *
            299 /
            1000 +
            g_ch *
            587 /
            1000 +
            b_ch *
            114 /
            1000).astype(int)
        data_ph = deepcopy(data)

    return data_ph
