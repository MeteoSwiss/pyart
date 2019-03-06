#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
metranet python library
================

functions to read METRANET files in pure python, no other library required!

.. autosummary::
    :toctree: generated/

    _float_mapping
    _nyquist_vel
    _selex2deg
    _get_chunk
    _get_moment_info
    read_polar


.. autosummary::
    :toctree: generated/
    :template: dev_template.rst

    RadarData

History

      V0.1 20190306 wod first prototype

"""


import sys
import numpy as np
import struct
import os

from .mfile_structure_info import SWEEP_HEADER, MOMENT_INFO_STRUCTURE 
from .mfile_structure_info import RAY_HEADER, MOMENT_HEADER, BYTE_SIZES

# fix for python3
if sys.version_info[0] == 3:
    def xrange(i):
        return range(i)

# Test endianness
if sys.byteorder == 'little':
    STRUC_PREFIX = '<'
else:
    STRUC_PREFIX = '>'

NP_TYPES = {'f':np.float,'d':np.float64}

# For some reasons, the radar name is not encoded in a consistent way in the
# binary files, this maps all names in files to a single character
RENAME_RADARS = {'Weissfluhgipfel':'W',
                 'Albis':'A',
                 'L':'L',
                 'Dole':'D',
                 'P':'P'}

# Following dic maps moment names hardcoded in M files to terminology used
# in metranet C library
MOM_NAME_MAPPING = {'Z_V_CLUT':'ZVC',
                  'Z_CLUT':'ZHC',
                  'W':'WID',
                  'V':'VEL',
                  'PHIDP':'PHI',
                  'STAT1':'ST1',
                  'STAT2':'ST2',
                  'CLUT':'CLT',
                  'UZ':'ZH',
                  'UZ_V':'ZV',
                  'APH':'MPH'}

# Other way round, f.ex CLT --> CLUT
MOM_NAME_MAPPING_INV = dict(zip(MOM_NAME_MAPPING.values(), 
                                MOM_NAME_MAPPING.keys()))

class RadarData(object):
    def __init__(self, header, pol_header, data):
        self.header = header
        self.pol_header = pol_header
        self.data = data
        
def _float_mapping(moment, time, radar, nyquist_vel):
    if moment in ('ZH', 'ZV', 'ZHC', 'ZVC'):
        prd_data_level = np.fromiter(xrange(256), dtype=np.float32)/2.-32.
        prd_data_level[0] = np.nan
    elif moment == 'ZDR':
        prd_data_level = (
            (np.fromiter(xrange(256), dtype=np.float32)+1) /
            16.1259842-7.9375)
        prd_data_level[0] = np.nan
    elif moment == 'RHO':
        if ((time > 1341619200) or
                (time > 1335484800 and
                 (radar == ord('D') or
                  radar == ord('L')))):
            # logaritmic scale
            prd_data_level = (
                1.003-10.**(-np.fromiter(xrange(256), dtype=np.float32)*0.01))
        else:
            # linear scale (old data)
            prd_data_level = (
                np.fromiter(xrange(256), dtype=np.float32)/255.)
    
        prd_data_level[0] = np.nan
    elif moment == 'PHI':
        prd_data_level = ((np.fromiter(
            xrange(256*256), dtype=np.float32)-32768)/32767.*180.)
        prd_data_level[0] = np.nan
    elif moment == 'VEL':
        prd_data_level = (
            (np.fromiter(xrange(256), dtype=np.float32)-128)/127. *
            nyquist_vel)
        prd_data_level[0] = np.nan
    elif moment == 'WID':
        prd_data_level = (np.fromiter(
            xrange(256), dtype=np.float32)/255.*nyquist_vel)
        prd_data_level[0] = np.nan
    elif moment == 'MPH':
        prd_data_level = ((np.fromiter(
            xrange(256), dtype=np.float32)-128)/127.*180.)
    elif moment in ('ST1', 'ST2', 'WBN'):
        prd_data_level = (np.fromiter(
            xrange(256), dtype=np.float32)/10.)
    elif moment == "CLT":
        prd_data_level = np.fromiter(xrange(256), dtype=np.float32)
    return prd_data_level


# Nyquist velocity (+-nv_value)
def _nyquist_vel(sweep_number):
    nv_value = 20.55
    if (sweep_number == 11 or
            sweep_number == 10 or
            sweep_number == 9):
        nv_value = 16.44
    elif (sweep_number == 8 or
          sweep_number == 6):
        nv_value = 13.7
    elif (sweep_number == 7 or
          sweep_number == 5 or
          sweep_number == 3):
        nv_value = 12.33
    elif sweep_number == 4:
        nv_value = 10.96
    elif sweep_number == 1:
        nv_value = 9.59
    elif (sweep_number== 2 or
          sweep_number == 0):
        nv_value = 8.22
    return nv_value

def _selex2deg(angle):
     # Convert angles from SELEX format to degree
     conv = angle * 360. / 65535.
     return conv
 
        
def _get_chunk(ba, file_info):
    # Read the bytearray byte by byte
    dic_values = {} # output dictionary
    
    read_pos = 0
    
    for i in range(len(file_info['names'])):
        len_val = file_info['len'][i]
        name_val = file_info['names'][i]

        if type(len_val) is list: 
            len_val = len_val[0] * dic_values[len_val[1]]
            
        type_var = file_info['type'][i]
        ffmt = STRUC_PREFIX + "{}".format(int(len_val)) # file format
        ffmt += type_var
        
        offset = len_val * BYTE_SIZES[type_var]

        if name_val == 'moments':
            # For the moments structure some additional processing is needed
            val = _get_moment_info(ba[read_pos:read_pos + offset], MOMENT_INFO_STRUCTURE, 
                                   dic_values['nummoments'])
        else:
            val  = struct.unpack_from(ffmt, ba[read_pos:read_pos + offset])
            
    
            if len(val) == 1:
                val = val[0]
            else:
                val = np.array(val)
            
            if type_var == 's':
                # Strip null hexadecimal codes
                val = val.decode('utf-8').replace('\x00','')
            
        if 'spare' not in name_val:
            dic_values[name_val] = val
        
        read_pos += offset
    
    return dic_values, ba[read_pos:]

def _get_moment_info(ba, file_info, num_moments):
    moments_info = []
    for i in range(num_moments):
        dic, rem = _get_chunk(ba, MOMENT_INFO_STRUCTURE)
        moments_info.append(dic)
        ba = rem
        
    return moments_info


def read_polar(filename, moments = None, physic_value = True, 
               masked_array = True, reorder_angles = True): 
    """
    Reads a METRANET polar data file

    Parameters
    ----------
    radar_file : str
        file name
    moment : list
        List of moments to read, by default all are used
    physic_value : boolean
        If true returns the physical value. Otherwise the digital value
    masked_array : boolean
        If true returns a numpy masked array with NaN values masked. Otherwise
        returns a regular masked array with NaN values
    reorder_anges: boolean
        If true all recorded rays are sorted by ascending order of their angles
        In addition the scan is truncated to a maximum of 360 rays
        

    Returns
    -------
    ret_data : RadarData object
        An object containing the information read from the file.

    """
    
    # check if it is the right file. Open it and read it
    bfile = os.path.basename(filename)

    supported_file = (bfile.startswith('MH') or
                      bfile.startswith('MS') or
                      bfile.startswith('ML'))


    if not supported_file:
        raise ValueError(
            'Only polar data files starting by MS, MH or ML are supported')

    if moments != None and np.isscalar(moments):
        moments = [moments]
        # Map from usual names (e.g. ZH) to moments names in file (e.g. UZ)
        for i in range(len(moments)):
            moments[i] = MOM_NAME_MAPPING_INV[moments[i]]
        
    # Open binary file
    with open(filename, 'rb') as f:      
        ba = memoryview(bytearray(f.read())) # bytearray 

 
    head, rem = _get_chunk(ba, SWEEP_HEADER) 
    head['radarname'] = RENAME_RADARS[head['radarname']] # make consistent names
    nummoments = head['nummoments']
    
    pol_header=[]
    
    # Initialize dictionary that contains the moments
    all_moments = [a['name'] for a in head['moments']]
    
    if moments == None:
        moments = all_moments

    moments_data = {}
    for m in moments:
        moments_data[m] = []
    
    while len(rem):
        pol,rem = _get_chunk(rem, RAY_HEADER)
        
        # Convert DN angles to float angles
        pol['startangle_az'] = _selex2deg(pol['startangle_az'])
        pol['startangle_el'] = _selex2deg(pol['startangle_el'])
        pol['endangle_az'] = _selex2deg(pol['endangle_az'])
        pol['endangle_el'] = _selex2deg(pol['endangle_el'])
        
        # Convert datetime to UTC + residue
        pol['datatime_residue'] = int(100 * ((pol['datatime']* 0.01) % 1))
        pol['datatime'] = int(0.01 * pol['datatime'])        
        pol_header.append(pol)
        
        for i in range(nummoments):
            mom_header, rem = _get_chunk(rem, MOMENT_HEADER)
            len_mom = mom_header['datasize']
            
            if all_moments[i] in moments:
                size_moment_bytes = head['moments'][i]['num_bytes']

                len_array = int(len_mom/size_moment_bytes) # file format
                ffmt = STRUC_PREFIX
                if size_moment_bytes == 1:
                    ffmt += 'B'
                else:
                    ffmt += 'H'

                mom = np.ndarray((len_array,),ffmt, rem[0:len_mom])
                rem = rem[len_mom:]
          
                moments_data[all_moments[i]].append(mom)
            else:
                rem = rem[len_mom:]
                

    for m in moments:
        # Find index of moment in the moments structure of the header
        names = [h['name'] for h in head['moments']]
        idx = names.index(m)
        if head['moments'][idx]['num_bytes']==1:
            dtype = np.uint8
        else:
            dtype = np.uint16
                    
        moments_data[m] = np.array(moments_data[m], dtype = dtype)
        
        if masked_array:
            if bfile.startswith('ML'):
                mask = np.logical_or( moments_data[m] == 0, 
                                     moments_data[m] == 1)
            else:
                mask = moments_data[m] == 0
                
                
            # This is the ELDES way to do it...but it does not agree perfectly
            # with the original C-library outputs and takes more time
            '''           
            moment_info = head['moments'][idx]
            a = moment_info['a']
            b = moment_info['b']
            if moment_info['scale_type'] == 1:
                moments_data[m] = a *  moments_data[m] + b
            elif moment_info['scale_type'] == 2:
                c = moment_info['c']
                moments_data[m] = a + c * 10**((1 -  moments_data[m]) / b)
            
            if m in ['V','W']:
               moments_data[m] *= NYQUIST(head['currentsweep'] - 1)
            if m == 'PHIDP':
                moments_data[m] *= 180. 
            '''
            
        if m in MOM_NAME_MAPPING.keys():
            # Rename moment if needed
            moments_data[MOM_NAME_MAPPING[m]] = moments_data.pop(m)
            m = MOM_NAME_MAPPING[m]
            
        if physic_value:
            moments_data[m] = (_float_mapping(m, 
                                    pol_header[0]['datatime'], 
                                    head['radarname'],
                                    _nyquist_vel(head['currentsweep'] - 1))
                                    [moments_data[m]].astype(np.float32))
        
        if masked_array:
            moments_data[m] = np.ma.array(moments_data[m], 
                                            mask = mask)
        else:
            moments_data[m][mask] = np.nan
            
    if reorder_angles:
        # Reorder dependent angle in ascending order
        if head['antmode'] in [0, 2]:
            angles = np.array([ray['startangle_az'] for ray in pol_header])
        else:
            angles = np.array([ray['startangle_el'] for ray in pol_header])

        nray = len(angles)

        angles = angles[-360:]
        order = np.argsort(angles)
        
        tmp = [pol_header[i + nray - 360] for i in order]
        pol_header = tmp
   
        for m in moments_data.keys():
            moments_data[m] = moments_data[m][-360:][order]
            
    ret_data = RadarData(head, pol_header, moments_data)
    return ret_data

    
