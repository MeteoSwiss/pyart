"""
pyart.aux_io.mf_grib_reader
==========================

Routines for putting MeteoFrance operational radar data contained in grib
files into grid object.

.. autosummary::
    :toctree: generated/

    read_png
    _get_physical_data

"""

from copy import deepcopy

import numpy as np

# check existence of pygrib library
try:
    import pygrib
    _PYGRIB_AVAILABLE = True
except ImportError:
    _PYGRIB_AVAILABLE = False

from ..config import FileMetadata
from ..io.common import _test_arguments
from ..core.grid import Grid
from ..core.transforms import geographic_to_cartesian
from ..exceptions import MissingOptionalDependency
from ..util import ma_broadcast_to

GRIB_FIELD_NAMES = {
    'hail': 'precipitation_type',
    'snow': 'precipitation_type',
    'rain': 'precipitation_type',
    'ice': 'precipitation_type'
}


def read_grib(filename, additional_metadata=None,
              field_name='precipitation_type', **kwargs):

    """
    Read a MeteoFrance data GRIB file.

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
   field_name : str
        Py-ART name of the field contained in the file

    Returns
    -------
    grid : Grid
        Grid object containing the data.

    """
    # check that wradlib is available
    if not _PYGRIB_AVAILABLE:
        raise MissingOptionalDependency(
            "pygrib is required to use read_grib but is not installed")

    # test for non empty kwargs
    _test_arguments(kwargs)

    grbs = pygrib.open(filename)
    for grb in grbs:
        data = grb.values
        lats = grb.distinctLatitudes
        lons = grb.distinctLongitudes
        dt_file = grb.validDate
    grbs.close()
    ny, nx = data.shape
    nz = 1

    # reserved_variables = [
    #     'time', 'x', 'y', 'z',
    #     'origin_latitude', 'origin_longitude', 'origin_altitude',
    #     'point_x', 'point_y', 'point_z', 'projection',
    #     'point_latitude', 'point_longitude', 'point_altitude',
    #     'radar_latitude', 'radar_longitude', 'radar_altitude',
    #     'radar_name', 'radar_time', 'base_time', 'time_offset',
    #     'ProjectionCoordinateSystem']

    # metadata
    filemetadata = FileMetadata('GRIB', GRIB_FIELD_NAMES, additional_metadata)

    # required reserved variables
    time = filemetadata('grid_time')
    origin_latitude = filemetadata('origin_latitude')
    origin_longitude = filemetadata('origin_longitude')
    origin_altitude = filemetadata('origin_altitude')
    x = filemetadata('x')
    y = filemetadata('y')
    z = filemetadata('z')

    # origin of eqc
    origin_latitude['data'] = np.array([0.])
    origin_longitude['data'] = np.array([0.])
    origin_altitude['data'] = np.array([0.])

    # projection (eqc)
    projection = {
        'proj': 'eqc',
        'R': '6371229'
    }

    x0, y0 = geographic_to_cartesian(lons.min(), lats.min(), projection)
    x1, y1 = geographic_to_cartesian(lons.max(), lats.max(), projection)
    xres = (x1-x0)/nx
    yres = (y1-y0)/ny
    x['data'] = np.arange(nx)*xres+x0
    y['data'] = np.arange(ny)*yres+y0
    z['data'] = np.array([0.])

    # Time
    time['units'] = 'seconds since '+dt_file.strftime('%Y-%m-%d %H:%M:%S')
    time['data'] = np.array([0])

    # read in the fields
    fields = {}
    field_dict = filemetadata(field_name)
    field_dict['data'] = ma_broadcast_to(
        _get_physical_data(data)[::-1, :], (nz, ny, nx))
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


def _get_physical_data(data):
    """
    gets data in physical units

    Parameters
    ----------
    data : int ndarray
        the data as int

    Returns
    -------
    data_ph : float ndarray
        the data in physical units

    """
    data_ph = deepcopy(data)

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

    # No data
    data_ph[data == 255] = 20

    return data_ph
