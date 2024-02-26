"""
pyart.aux_io.metranet_cartesian_reader
======================================

Routines for putting METRANET Cartesian data files into grid object.
(Used by ELDES www.eldesradar.it)

.. autosummary::
    :toctree: generated/

    read_cartesian_metranet

"""

import datetime
import platform
from warnings import warn

import numpy as np

from ..config import FileMetadata
from ..core.grid import Grid
from ..io.common import _test_arguments
from .metranet_c import get_library
from .metranet_c import read_product as read_product_c
from .metranet_python import read_product as read_product_python

# check existence of METRANET library
try:
    METRANET_LIB = get_library(momentms=False, momentpm=True)
    if platform.system() == 'Linux':
        METRANET_LIB = get_library(momentms=True)
    _METRANETLIB_AVAILABLE = True
except BaseException:
    # bare exception needed to capture error
    _METRANETLIB_AVAILABLE = False

METRANET_FIELD_NAMES = {
    'NWP_HZEROCL': 'iso0_height',
    'Rain_Rate': 'radar_estimated_rain_rate',  # RZC, R1F, rZC, RZF, dRZC
    'Rainfall_accumulation': 'rainfall_accumulation',  # AZC, aZC, nowpal
    # 'CPC': 'rainfall_accumulation', # Check product name
    # 'CPCH': 'rainfall_accumulation', # Check product name
    # 'dACC': 'rainfall_accumulation', # Check product name
    # 'dACCH': 'rainfall_accumulation', # Check product name
    'POH': 'probability_of_hail',  # BZC, dBZC
    'Hail': 'probability_of_hail',  # GZC, dGZC
    'MESH': 'maximum_expected_severe_hail_size',  # MZC, dMZC
    'Column_Max@swp_1_to_20': 'maximum_echo',  # CZC, dCZC
    'MAXECHO_HEIGHT': 'maximum_echo_height',
    'ECHOTOP_15': 'echo_top_15dBZ',  # EZC, dEZC
    'ECHOTOP_20': 'echo_top_20dBZ',  # EZC
    'ECHOTOP_45': 'echo_top_45dBZ',  # EZC, dEZC
    'ECHOTOP_50': 'echo_top_50dBZ',  # EZC
    'VIL': 'vertically_integrated_liquid',  # LZC, dLZC
    'CAPPI_Zh_1.0km': 'reflectivity',  # OZC
    'CAPPI_Zh_2.0km': 'reflectivity',  # OZC
    'CAPPI_Zh_3.0km': 'reflectivity',  # OZC
    'CAPPI_Zh_4.0km': 'reflectivity',  # OZC
    'CAPPI_Zh_5.0km': 'reflectivity',  # OZC
    'CAPPI_Zh_6.0km': 'reflectivity',  # OZC
    'CAPPI_Zh_7.0km': 'reflectivity',  # OZC
    'CAPPI_Zh_8.0km': 'reflectivity',  # OZC
    'CAPPI_Zh_9.0km': 'reflectivity',  # OZC
    'CAPPI_Zh_10.0km': 'reflectivity',  # OZC
    'CAPPI_Zh_11.0km': 'reflectivity',  # OZC
    'CAPPI_Zh_12.0km': 'reflectivity',  # OZC
    'CAPPI_Zh_13.0km': 'reflectivity',  # OZC
    'CAPPI_Zh_14.0km': 'reflectivity',  # OZC
    'CAPPI_Zh_15.0km': 'reflectivity',  # OZC
    'CAPPI_Zh_16.0km': 'reflectivity',  # OZC
    'CAPPI_Zh_17.0km': 'reflectivity',  # OZC
    'CAPPI_Zh_18.0km': 'reflectivity'  # OZC
}


def read_cartesian_metranet(filename, additional_metadata=None, chy0=255.,
                            chx0=-160., reader='C', **kwargs):
    """
    Read a METRANET product file.

    Parameters
    ----------
    filename : str
        Name of the METRANET file to read.
    additional_metadata : dict of dicts, optional
        Dictionary of dictionaries to retrieve metadata during this read.
        This metadata is not used during any successive file reads unless
        explicitly included.  A value of None, the default, will not
        introduct any addition metadata and the file specific or default
        metadata as specified by the Py-ART configuration file will be used.
    chy0, chx0 : float
        Swiss coordinates position of the south-western point in the grid
    reader : str
        The reader library to use. Can be either 'C' or 'python'

    Returns
    -------
    grid : Grid
        Grid object containing data from METRANET file.

    """
    if reader == 'c':
        reader = 'C'
        
    # test for non empty kwargs
    _test_arguments(kwargs)

    if reader == 'C' and _METRANETLIB_AVAILABLE:
        ret = read_product_c(filename, physic_value=True, masked_array=True)
    elif reader == 'python':
        warn('Python cartesian reader is unstable, use at your own risks!')
        ret = read_product_python(
            filename, physic_value=True, masked_array=True)
    else:
        warn('Invalid reader name or C library not available,' +
             ' using python (default) instead')
        ret = read_product_python(
            filename, physic_value=True, masked_array=True)

    if ret is None:
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
    metadata = ret.header

    filemetadata = FileMetadata(
        'METRANET', METRANET_FIELD_NAMES, additional_metadata)

    nx = int(ret.header['column'])
    ny = int(ret.header['row'])
    nz = 1

    # required reserved variables
    time = filemetadata('grid_time')
    origin_latitude = filemetadata('origin_latitude')
    origin_longitude = filemetadata('origin_longitude')
    origin_altitude = filemetadata('origin_altitude')
    x = filemetadata('x')
    y = filemetadata('y')
    z = filemetadata('z')

    x['data'] = 1000. * (
        np.arange(nx) * float(ret.header['rect_xres']) + chy0 +
        float(ret.header['rect_xres']) / 2. - 600.)

    y['data'] = 1000. * (
        np.arange(ny) * float(ret.header['rect_yres']) + chx0 +
        float(ret.header['rect_yres']) / 2. - 200.)

    if ret.header['product'].startswith('CAPPI_Zh_'):
        alt = ret.header['product'].split('_')[2]
        alt = float(alt.replace('km', '')) * 1000.
        z['data'] = np.array([alt])
    else:
        z['data'] = np.array([0.])

    # Origin of LV03 Swiss coordinates
    origin_latitude['data'] = np.array([46.9524055556])
    origin_longitude['data'] = np.array([7.43958333333])
    origin_altitude['data'] = np.array([0.])

    prod_time = datetime.datetime.strptime(
        ret.header['time'][0:9], '%y%j%H%M')
    time['units'] = 'seconds since ' + prod_time.strftime('%Y-%m-%dT%H:%M:%SZ')
    if 'usr_forecast_hour' in ret.header:
        time['data'] = np.array(
            [float(ret.header['usr_forecast_hour']) * 3600.])
    else:
        time['data'] = np.array([0])

    # projection (Swiss Oblique Mercator)
    projection = {
        'proj': 'somerc',
        '_include_lon_0_lat_0': True
    }

    # read in the fields
    fields = {}
    field = filemetadata.get_field_name(ret.header['product'])
    field_dict = filemetadata(field)
    data = np.broadcast_to(ret.data[::-1, :], (nz, ny, nx))
    mask = np.broadcast_to(ret.data.mask[::-1, :], (nz, ny, nx))
    field_dict['data'] = np.ma.array(data, mask=mask)
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
