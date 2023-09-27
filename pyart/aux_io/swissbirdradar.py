"""
pyart.aux_io.swissbirdradar
========================

Routines for reading H5 files from swissbirdradars.

.. autosummary::
    :toctree: generated/

    read_swissbirdradar_spectra

"""

import datetime

import numpy as np

try:
    import h5py
    _H5PY_AVAILABLE = True
except ImportError:
    _H5PY_AVAILABLE = False

from ..config import FileMetadata
from ..core.radar_spectra import RadarSpectra
from ..exceptions import MissingOptionalDependency

SWISSBIRDRADAR_FIELD_NAMES = {
    'meanRangeDopplerHH': 'complex_spectra_hh_ADU',
    'meanRangeDopplerVV': 'complex_spectra_vv_ADU'
}

def read_swissbirdradar_spectra(filename, field_names = None, additional_metadata=None,
                                file_field_names = None,
                                exclude_fields=None, include_fields=None,
                                latitude = None, longitude = None, altitude = None):

    """
    Read a spectrum file from SwissBirdRadar.

    Parameters
    ----------
    filename : str
        Name of the SwissBirdRadar file to read.
    field_names : dict, optional
        Dictionary mapping SwissBirdRadar field names to radar field names. If a
        data type found in the file does not appear in this dictionary or has
        a value of None it will not be placed in the radar.fields dictionary.
        A value of None, the default, will use the mapping defined in the
        Py-ART configuration file.
    additional_metadata : dict of dicts, optional
        Dictionary of dictionaries to retrieve metadata from during this read.
        This metadata is not used during any successive file reads unless
        explicitly included.  A value of None, the default, will not
        introduct any addition metadata and the file specific or default
        metadata as specified by the Py-ART configuration file will be used.
    file_field_names : bool, optional
        True to use the MDV data type names for the field names. If this
        case the field_names parameter is ignored. The field dictionary will
        likely only have a 'data' key, unless the fields are defined in
        `additional_metadata`.
    exclude_fields : list or None, optional
        List of fields to exclude from the radar object. This is applied
        after the `file_field_names` and `field_names` parameters. Set
        to None to include all fields specified by include_fields.
    include_fields : list or None, optional
        List of fields to include from the radar object. This is applied
        after the `file_field_names` and `field_names` parameters. Set
        to None to include all fields not specified by exclude_fields.
    latitude : float, optional
        Latitude of the Swissbirdradar in degrees
    longitude : float, optional
        Longitude of the Swissbirdradar in degrees
    Altitude : float, optional
        Altitude above mean sea level of the Swissbirdradar in meters

    Returns
    -------
    spectra : Spectra
        Spectra object containing data from the SwissBirdRadar file.

    """

    # check that h5py is available
    if not _H5PY_AVAILABLE:
        raise MissingOptionalDependency(
            "h5py is required to use read_sinarame_h5 but is not installed")

    h5obj = h5py.File(filename)

    # create metadata retrieval object
    if field_names is None:
        field_names = SWISSBIRDRADAR_FIELD_NAMES
    filemetadata = FileMetadata(
        'SWISSBIRDRADAR', field_names, additional_metadata, file_field_names,
        exclude_fields)

    times = {'data':[]}
    all_fields = {}

    # Find time keys
    for k in h5obj.keys():
        try:
            tstamp = datetime.datetime.utcfromtimestamp(int(k)/1000)
            times['data'].append(tstamp)
            for field in h5obj[k].keys():
                if field not in all_fields.keys():
                    all_fields[field] = []
                all_fields[field].append(h5obj[k][field][:])
        except ValueError:
            pass

    fields = {}
    for key in all_fields.keys():
        field_name = filemetadata.get_field_name(key)
        if field_name is None:
            if exclude_fields is not None and key in exclude_fields:
                if key not in include_fields:
                    continue
            if include_fields is None or key in include_fields:
                field_name = key
            else:
                continue
        fields[field_name] = {'data': np.array(all_fields[field])}

    data_shape = fields['complex_spectra_hh_ADU']['data'].shape
    rres = h5obj['RadarParameters'].attrs['rangeResolution']
    rrange = {'data': np.arange(data_shape[1]) * rres}
    nrays = data_shape[0]
    data_shape[1]
    npulses_max = data_shape[2]
    doppler_vel =  (h5obj['RadarParameters'].attrs['dopplerResolution'] *
                        np.arange(-data_shape[2]/2, data_shape[2]/2))
    doppler_vel = np.tile(doppler_vel, (nrays, 1))

    Doppler_velocity = {'data': doppler_vel}
    fixed_angle = {'data':[90]}
    sweep_start_ray_index =  {'data':[0]}
    sweep_end_ray_index =  {'data':[nrays-1]}
    sweep_number = {'data':[0]}
    scan_type = 'ppi'
    sweep_mode =  {'data':['ppi']}
    latitude = {'data':[47.49657]}
    longitude = {'data':[8.64729]}
    altitude = {'data':[642.5]}
    npulses = {'data':nrays*[npulses_max]}
    azimuth = {'data':[0]}
    elevation = {'data':[90]}
    radar_calibration = {'dBADU_to_dBm_hh': {'data': [0]},
                         'dBADU_to_dBm_vv': {'data': [0]},
                         'calibration_constant_vv': {'data': [0]},
                         'calibration_constant_hh': {'data': [0]}}

    times_fmt = {'units': f"seconds since {times['data'][0].strftime('%Y-%m-%dT%H:%M:%SZ')}",
                'calendar': 'gregorian',
                'data': np.array([(t - times['data'][0]).total_seconds() for t in times['data']])}


    radar_spectra = RadarSpectra(time = times_fmt, _range = rrange, fields = fields,
                                        metadata = None, latitude = latitude, longitude = longitude,
                                        altitude = altitude, sweep_number = sweep_number,
                                        sweep_mode = sweep_mode, scan_type = scan_type,
                                        fixed_angle = fixed_angle,
                                        sweep_end_ray_index=sweep_end_ray_index,
                                        sweep_start_ray_index=sweep_start_ray_index,
                                        azimuth=azimuth, Doppler_velocity = Doppler_velocity,
                                        elevation=elevation,
                                        npulses=npulses, radar_calibration = radar_calibration)
    return radar_spectra
