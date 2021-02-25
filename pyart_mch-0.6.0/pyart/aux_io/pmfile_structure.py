# -*- coding: utf-8 -*-
"""
metranet python library
================

Information about the variable names, types and length that are stored in
the metranet files

History

      V0.1 20190306 wod first prototype

"""


#List of abbreviations
# i = signed integer
# I = unsigned integer
# B = byte
# c = character
# d = double
# f = float (single)
# s = string
# H = unsigned short
# h = short
# L = unsigned long
# Q = unsigned long long


# Here are the byte sizes for all types
BYTE_SIZES = {'i':4, 'f':4, 'B':1, 'I':4, 'H':2, 'd': 8, 's': 1, 'L':4, 'c':1,
              'b':1, 'h':2, 'Q':8, 'p': 1}


'''
===============================================================================
M-files
===============================================================================
'''


###############################################################################
# SWEEP_HEADER INFO

MSWEEP_HEADER = {}

MSWEEP_HEADER['names'] = [
    'field',
    'version',
    'spare',
    'length',
    'radarname',
    'scanname',
    'radarlat',
    'radarlon',
    'radarheight',
    'sequencesweep',
    'currentsweep',
    'totalsweep',
    'antmode',
    'priority',
    'quality',
    'spare',
    'repeattime',
    'nummoments',
    'gatewidth',
    'frequency',
    'pulsewidth',
    'startrange',
    'metadatasize',
    'moments',
    'metadata']

# Sweep header entries variable types (see https://docs.python.org/3/library/struct.html)
MSWEEP_HEADER['type'] = [
    's',
    'b',
    'b',
    'L',
    's',
    's',
    'f',
    'f',
    'f',
    'b',
    'b',
    'b',
    'b',
    'b',
    'b',
    'b',
    'H',
    'H',
    'f',
    'f',
    'f',
    'f',
    'L',
    'b',
    's']

'''
Note: length of sweep header entries in multiples of type byte sizes. ex if
type = 'd' and len = 3, the total size in bytes will be
8 (bytes/double) * 3 = 18 bytes. When the length is a list ex.
[4, 'variable'], it means that the size of this field will be 4 x the value of
the 'variable' key in the SWEEP_HEADER x the size in bytes of the type.
'''

MSWEEP_HEADER['len'] = [
    4,
    1,
    3,
    1,
    16,
    16,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    2,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    [48, 'nummoments'],
    [1, 'metadatasize']]


##############################################################################
### MOMENT_INFO STRUCTURE INFO

MMOMENT_INFO_STRUCTURE = {}

MMOMENT_INFO_STRUCTURE['names'] = [
    'moment_uuid',
    'data_format',
    'num_bytes',
    'spare1',
    'name',
    'unit',
    'a',
    'b',
    'c',
    'scale_type',
    'spare2']

MMOMENT_INFO_STRUCTURE['type'] = [
    'L',
    'b',
    'b',
    'b',
    's',
    's',
    'f',
    'f',
    'f',
    'b',
    'b']

MMOMENT_INFO_STRUCTURE['len'] = [
    1,
    1,
    1,
    2,
    12,
    12,
    1,
    1,
    1,
    1,
    3]

##############################################################################
### RAY HEADER INFO

MRAY_HEADER = {}

MRAY_HEADER['names'] = [
    'length',
    'startangle_az',
    'startangle_el',
    'endangle_az',
    'endangle_el',
    'sequence',
    'numpulses',
    'databytes',
    'prf',
    'datatime',
    'dataflags',
    'metadatasize',
    'metadata']

MRAY_HEADER['type'] = [
    'I',
    'H',
    'h',
    'H',
    'h',
    'H',
    'H',
    'I',
    'f',
    'Q',
    'I',
    'I',
    's']

MRAY_HEADER['len'] = [
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    [1, 'metadatasize']]

##############################################################################
### DATA MOMENT HEADER INFO

MMOMENT_HEADER = {}

MMOMENT_HEADER['names'] = [
    'momentuuid',
    'datasize']

MMOMENT_HEADER['type'] = [
    'L',
    'I']

MMOMENT_HEADER['len'] = [
    1,
    1]

'''
==============================================================================
P-files
==============================================================================
'''

##############################################################################
### RAY HEADER INFO
PRAY_HEADER = {}

PRAY_HEADER = {}

PRAY_HEADER['names'] = [
    'dataformat',
    'bitmask',
    'scanid',
    'hostid',
    'startangle_el',
    'startangle_az',
    'endangle_el',
    'endangle_az',
    'antmode',
    'totalsweep',
    'currentsweep',
    'endofsweep',
    'sequence',
    'totalrecord',
    'pulses',
    'numgates',
    'databytes',
    'dataflag',
    'datatime_residue',
    'datatime',
    'repeattime',
    'compressed',
    'priority',
    'nyquest',
    'gatewidth',
    'wnyquest',
    'startrange']

PRAY_HEADER['type'] = [
    'b',
    'b',
    's',
    'I',
    'h',
    'H',
    'h',
    'H',
    'b',
    'b',
    'b',
    'b',
    'h',
    'h',
    'h',
    'h',
    'i',
    'H',
    'h',
    'I',
    'h',
    'b',
    'b',
    'f',
    'f',
    'f',
    'f']

PRAY_HEADER['len'] = [
    1,
    3,
    4,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1]

##############################################################################
### DATA MOMENT SIZES
PMOMENTS = {}
PMOMENTS['names'] = {}
PMOMENTS['names']['M'] = [
    'VEL', 'ZH', 'ZV', 'ZDR', 'RHO', 'PHI', 'ST1', 'ST2', 'WBN', 'MPH']
PMOMENTS['names']['L'] = [
    'WID', 'VEL', 'ZH', 'ZV', 'ZDR', 'RHO', 'PHI', 'ZHC', 'ZVC']
PMOMENTS['names']['H'] = [
    'WID', 'VEL', 'ZH', 'ZV', 'ZDR', 'RHO', 'PHI', 'ST1', 'ST2', 'WBN', 'MPH',
    'CLT']
PMOMENTS['types'] = {'PHI':'h'} # ALl others are bytes
