
import pyart
import xarray as xr
import gzip
import io
import numpy as np
import datetime
from scipy.interpolate import interp1d

def _xr_var_to_pyart_dict(xr_var):
    dic = {}
    for k in xr_var.attrs:
        dic[k] = xr_var.attrs[k]
    dic['data'] = xr_var.data
    return dic

def _interpolate_var(data, raw_ranges, ref_ranges):
    data_interp = []
    for i in range(len(data)):
        if np.any(raw_ranges[i] != ref_ranges):
            data_i = interp1d(raw_ranges[i],
                    data[i], bounds_error = False)(ref_ranges)
            data_interp.append(data_i)
        else:
            data_interp.append(data[i])
    return np.array(data_interp)

def read_windcube(filename, field_names=None, additional_metadata=None):
    """
    Reads a NetCDF file in the WindCube format and returns a Py-ART radar
    object.

    Parameters
    ----------
    filename : str
        Name of the input file.
    field_names : list of str or None, optional
        List of variable names to include as radar fields. If None, all 2D
        variables in the file will be included. Default is None.
    additional_metadata : dict or None, optional
        Dictionary of additional metadata to include in the output radar object.
        Default is None.
    
    Returns
    -------
    radar : Py-ART radar object
        Radar object containing the data from the input file.
    
    Notes
    -----
    This function assumes that the input file contains a single sweep group.
    If the file is compressed with gzip, the function expects a '.gz' extension.
    This function is compatible with the WindCube v2.0 format.
    """

    if '.gz' in filename:
        head = xr.open_dataset(gzip.open(filename))
        sweep = head.variables['sweep_group_name'].data[0]
        nc = xr.open_dataset(gzip.open(filename), group = sweep, 
                             decode_times= False)
    else:
        head = xr.open_dataset(filename)
        sweep = head.variables['sweep_group_name'].data[0]
        nc = xr.open_dataset(filename, group = sweep, decode_times= False)
    
    scan_type = nc.scan_file_name.split('_')[0].lower()
    
    # Get dimensions
    n_gates = nc.range.size
    n_time = nc.time.size
        
    # Get time
    time = {}
    t0 = (datetime.datetime.utcfromtimestamp(nc.variables['time'].data[0])
              .replace(microsecond=0).isoformat() + 'Z')
    time['units'] = 'seconds since '+t0
    time['data'] = nc.variables['time'].data - nc.variables['time'].data[0] 
    
    # Get range
    if scan_type == 'dbs':
        shortest_ranges = nc.variables['range'][np.argmin(
                                            nc.variables['range'].data[:,-1])]
        range_array = _xr_var_to_pyart_dict(shortest_ranges)
    else:
        range_array = _xr_var_to_pyart_dict(nc.variables['range'])
    
    # Get azimuth and elevation
    azimuth = _xr_var_to_pyart_dict(nc.variables['azimuth'])
    elevation = _xr_var_to_pyart_dict(nc.variables['elevation'])
    
    # Create fields
    fields = {}
    for var in nc.variables:
        if len(nc.variables[var].shape) == 2:
            if field_names is not None:
                if var not in field_names:
                    continue  # ignore this variable

            fields[var] = _xr_var_to_pyart_dict(nc.variables[var])

            if scan_type == 'dbs':
                fields[var]['data'] = _interpolate_var(
                    fields[var]['data'],
                    nc.variables['range'],
                    range_array['data']
                )

    # Create metadata
    metadata = head.attrs
    if additional_metadata is not None:
        metadata.update(additional_metadata)

    latitude = _xr_var_to_pyart_dict(head.latitude)
    longitude = _xr_var_to_pyart_dict(head.longitude)
    altitude = _xr_var_to_pyart_dict(head.altitude)

    sweep_start_ray_index = {'data': np.array([0])}
    sweep_end_ray_index = {'data': np.array([n_time])}
    sweep_number = {'data': np.array([0])}
    sweep_mode = {'data': [scan_type]}

    if scan_type == 'rhi':
        fixed_angle = {'data': np.array([azimuth['data'][0]])}
    elif scan_type == 'ppi':
        fixed_angle = {'data': np.array(elevation['data'][0])}
    else:
        fixed_angle = {'data': np.array(elevation['data'])}
        
    radar = pyart.core.Radar(
        time, range_array, fields, metadata, scan_type,
        latitude, longitude, altitude, sweep_number, sweep_mode,
        fixed_angle, sweep_start_ray_index, sweep_end_ray_index,
        azimuth, elevation
    )

    return radar