"""Creation: 4 April 2018, D. Wolfensberger"""

# Here are the required
BYTE_SIZES = {
    'i': 4,
    'f': 4,
    'B': 1,
    'I': 4,
    'H': 2,
    'd': 8,
    's': 1,
    'L': 4,
    'c': 1,
    'b': 1,
    'p': 1,
    'h': 2,
    'Q': 8}

###############################################################################
# SWEEP_HEADER INFO

SWEEP_HEADER = {}

SWEEP_HEADER['names'] = ['field',
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

# Sweep header entries variable types (see
# https://docs.python.org/3/library/struct.html)

SWEEP_HEADER['type'] = ['s',
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

# Length of sweep header entries in multiples of type byte sizes. ex if type = 'd'and
# len = 3, the total size in bytes will be 8 (bytes/double) * 3 = 18 bytes.
# When the length is a list ex. [4, 'variable'], it means that the size of
#  this field will be 4 x the value of the 'variable' key
# in the SWEEP_HEADER x the size in bytes of the type.

SWEEP_HEADER['len'] = [4,
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


###############################################################################
# MOMENT_INFO STRUCTURE INFO

MOMENT_INFO_STRUCTURE = {}

MOMENT_INFO_STRUCTURE['names'] = ['moment_uuid',
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

MOMENT_INFO_STRUCTURE['type'] = ['L',
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

MOMENT_INFO_STRUCTURE['len'] = [1,
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

###############################################################################
# RAY HEADER INFO

RAY_HEADER = {}

RAY_HEADER['names'] = ['length',
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

RAY_HEADER['type'] = ['I',
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

RAY_HEADER['len'] = [1,
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

###############################################################################
# DATA MOMENT HEADER INFO

MOMENT_HEADER = {}

MOMENT_HEADER['names'] = ['momentuuid',
                          'datasize']

MOMENT_HEADER['type'] = ['L',
                         'I']

MOMENT_HEADER['len'] = [1,
                        1]
