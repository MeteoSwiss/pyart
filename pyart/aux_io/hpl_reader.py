"""
pyart.io.cf1
============

Utilities for reading hpl files from the Halo Photonics  streamline lidar.

.. autosummary::
    :toctree: generated/
    :template: dev_template.rst

.. autosummary::
    :toctree: generated/

    read_hpl

"""

import datetime
import numpy as np
import os

from ..config import FileMetadata
from ..core import Radar
from ..io.common import make_time_unit_str

LOCATION_CAMPAIGNS = {}
LOCATION_CAMPAIGNS['Meiringen2021_a'] = [8.111, 46.741, 575.4]
LOCATION_CAMPAIGNS['Meiringen2021_b'] = [8.121, 46.740, 588]           
       
def read_hpl(filename, additional_metadata=None, location = None):
    """
    Read a hpl ASCII file from the Halo Photonoics Steamline lidar.

    Parameters
    ----------
    filename : str
        Name of CF/Radial netCDF file to read data from.
    additional_metadata : dict of dicts, optional
        This parameter is not used, it is included for uniformity.
    location : str
        Name of the location from which the data comes, if provided will
        assign corresponding values of lat, lon and alt. The named locations
        can be added to the global variable LOCATION_CAMPAIGNS in this script

    Returns
    -------
    radar : Radar
        Radar object.

    Notes
    -----
    This function has not been tested on "stream" Cfradial files.

    """
    
    filemetadata = FileMetadata('hpl')
    latitude = filemetadata('latitude')
    longitude = filemetadata('longitude')
    altitude = filemetadata('altitude')
    azimuth = filemetadata('azimuth')
    elevation = filemetadata('elevation')
    sweep_start_ray_index = filemetadata('sweep_start_ray_index')
    sweep_end_ray_index = filemetadata('sweep_end_ray_index')
    sweep_number = filemetadata('sweep_number')
    sweep_mode = filemetadata('sweep_mode')
    fixed_angle = filemetadata('fixed_angle')
    _range = filemetadata('range')
    _time = filemetadata('time')
    fields = {}
    fields['avg_velocity'] = filemetadata('range')
    fields['avg_reflectivity'] = filemetadata('avg_reflectivity')
    fields['absolute_beta'] = filemetadata('absolute_beta')
            
    day = datetime.datetime.strptime(os.path.basename(filename).split('_')[2],
                                     '%Y%m%d')
    
    # read header
    endheader = False
    metadata = {}
    
    azimuth['data'] = []
    elevation['data'] = []
    _time['data'] = []
    fields['avg_velocity']['data'] = []
    fields['avg_reflectivity']['data'] = []
    fields['absolute_beta']['data'] = []
    
    with open(filename) as fo:
        for i,line in enumerate(fo):
            if ':' in line:
                val = line.split(':')[1].replace('\n','').replace('\t','')
                try:
                    if '.' in val:
                        val = float(val)
                    else:
                        val = int(val)
                except:
                    pass
                metadata[line.split(':')[0]] = val
            if '***' in line:
                endheader = True
                continue
            if endheader:
                lsplit = line.split()
                if len(lsplit[0]) > 4:
                    _time['data'].append(day + datetime.timedelta(
                        hours=float(lsplit[0])))
                    azimuth['data'].append(float(lsplit[1]))
                    elevation['data'].append(float(lsplit[2]))
                    
                else:
                    fields['avg_velocity']['data'].append(float(lsplit[1]))
                    fields['avg_reflectivity']['data'].append(float(lsplit[2]))
                    fields['absolute_beta']['data'].append(float(lsplit[3]))
                    
    for k in fields.keys():
        fields[k]['data'] = np.reshape(np.array(fields[k]['data']),
                                       (len(azimuth['data']), 
                                        metadata['Number of gates']))
    
    _range['data'] = (metadata['Range gate length (m)'] * 
                      np.arange(metadata['Number of gates']))
    azimuth['data'] = np.array(azimuth['data'])
    elevation['data'] = np.array(elevation['data'])
    _time['units'] = make_time_unit_str(_time['data'][0])
    tseconds = [(d - _time['data'][0]).total_seconds() for d in _time['data']]
    _time['data'] = tseconds
    
    if type(additional_metadata) == dict:
        if 'altitude' in additional_metadata.keys():
            altitude = additional_metadata['altitude']
        if 'latitude' in additional_metadata.keys():
            latitude = additional_metadata['latitude']
        if 'longitude' in additional_metadata.keys():
            longitude = additional_metadata['longitude']
    if location in LOCATION_CAMPAIGNS.keys():
        longitude['data'] = np.array([LOCATION_CAMPAIGNS[location][0]])
        latitude['data'] = np.array([LOCATION_CAMPAIGNS[location][1]])
        altitude['data'] =  np.array([LOCATION_CAMPAIGNS[location][2]])
        
    sweep_start_ray_index['data'] = np.array([0])
    sweep_end_ray_index['data'] = np.array(
        [azimuth['data'].size-1], dtype=np.int32)
    sweep_number['data'] = np.array([0])
    
    nazi_unique = np.unique(azimuth['data'])
    nele_unique = np.unique(elevation['data'])
    if nele_unique.size == 1 and nazi_unique.size == 1:
        scan_type = 'other'
        sweep_mode['data'] = np.array(['pointing'])
        fixed_angle['data'] = nele_unique
    elif nele_unique.size == 1 and nazi_unique.size > 1:
        scan_type = 'ppi'
        sweep_mode['data'] = np.array(['azimuth_surveillance'])
        fixed_angle['data'] = nele_unique
    elif nele_unique.size > 1 and nazi_unique.size == 1:
        scan_type = 'rhi'
        sweep_mode['data'] = np.array(['elevation_surveillance'])
        fixed_angle['data'] = nazi_unique
    else:
        raise ValueError(
            'Only single sweeps PPI, RHI or pointing are supported')
   
    return Radar(
        _time, _range, fields, metadata, scan_type,
        latitude, longitude, altitude,
        sweep_number, sweep_mode, fixed_angle, sweep_start_ray_index,
        sweep_end_ray_index,
        azimuth, elevation)
