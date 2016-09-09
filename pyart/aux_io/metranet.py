"""
pyart.aux_io.metranet
====================

Routines for reading METRANET files. (Used by ELDES www.eldesradar.it)

.. autosummary::
    :toctree: generated/

    metranet_read_polar
    read_metranet

"""

import ctypes
import os
import sys
import platform
import argparse
import datetime

import numpy as np

from ..config import FileMetadata, get_fillvalue
from ..io.common import make_time_unit_str, _test_arguments
from ..core.radar import Radar
from ..exceptions import MissingOptionalDependency
import string

# define metranet reading C library
metranetlib_path = os.environ.get('METRANETLIB_PATH')
if metranetlib_path is None:
    print('METRANET library path not specified. Using default')
    metranetlib_path = '/proj/lom/idl/lib/radlib4/'

library_metranet_linux64 = 'srn_idl_py_lib.x86_64.so'
library_metranet_sparc32 = 'srn_idl_py_lib.sparc32.so'
library_metranet_sparc64 = 'srn_idl_py_lib.sparc64.so'

# load the right reading library
library_metranet = 'x'
if platform.system() == 'SunOS':
    if platform.architecture()[0] == '64bit':
        # 32 bit has to be used even if it is 64 bit architecture
        library_metranet = metranetlib_path+library_metranet_sparc32
    else:
        library_metranet = metranetlib_path+library_metranet_sparc32

if platform.system() == 'Linux':
    library_metranet = metranetlib_path+library_metranet_linux64

if library_metranet == 'x':
    raise MissingOptionalDependency("METRANET library not found")

metranet_lib = ctypes.cdll.LoadLibrary(library_metranet)


METRANET_FIELD_NAMES = {
    'WID': 'spectrum_width',
    'VEL': 'velocity',
    'ZH': 'reflectivity',
    'ZV': 'reflectivity_vv',  # non standard name
    'ZDR': 'differential_reflectivity',
    'RHO': 'uncorrected_cross_correlation_ratio',
    'PHI': 'uncorrected_differential_phase',
    'ST1': 'stat_test_lag1',  # statistical test on lag 1 (non standard name)
    'ST2': 'stat_test_lag2',  # statistical test on lag 2 (non standard name)
    'WBN': 'wide_band_noise',  # (non standard name)
    'MPH': 'mean_phase',  # (non standard name)
    'CLT': 'clutter_exit_code',  # (non standard name)
    'ZHC': 'reflectivity_hh_clut',  # cluttered horizontal reflectivity
    'ZVC': 'reflectivity_hh_clut',  # cluttered vertical reflectivity
}

PM_MOM = ["ZH", "ZV", "ZDR", "RHO", "PHI", "VEL", "WID", "ST1", "ST2", "WBN",
          "MPH"]
PH_MOM = ["ZH", "ZV", "ZDR", "RHO", "PHI", "VEL", "WID", "ST1", "ST2", "WBN",
          "MPH", "CLT"]
PL_MOM = ["ZH", "ZV", "ZDR", "RHO", "PHI", "VEL", "WID", "ZHC", "ZVC"]

NPM_MOM = 11
NPH_MOM = 12
NPL_MOM = 9


def read_metranet(filename, field_names=None, additional_metadata=None,
                  file_field_names=False, exclude_fields=None, **kwargs):
    """
    Read a METRANET file.

    Parameters
    ----------
    filename : str
        Name of the METRANET file to read.
    field_names : dict, optional
        Dictionary mapping METRANET field names to radar field names. If a
        data type found in the file does not appear in this dictionary or has
        a value of None it will not be placed in the radar.fields dictionary.
        A value of None, the default, will use the mapping defined in the
        Py-ART configuration file.
    additional_metadata : dict of dicts, optional
        Dictionary of dictionaries to retrieve metadata during this read.
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
        after the `file_field_names` and `field_names` parameters.

    Returns
    -------
    radar : Radar
        Radar object containing data from METRANET file.

    """

    # test for non empty kwargs
    _test_arguments(kwargs)

    # check if it is the right file. Open it and read it
    bfile = os.path.basename(filename)

    supported_file = (bfile.startswith('PM') or bfile.startswith('PH') or
                      bfile.startswith('PL'))
    if not supported_file:
        raise ValueError(
            'Only polar data files starting by PM, PH or PL are supported')

    # create metadata retrieval object
    if field_names is None:
        field_names = METRANET_FIELD_NAMES
    filemetadata = FileMetadata(
        'METRANET', field_names, additional_metadata, file_field_names,
        exclude_fields)

    # get definitions from filemetadata class
    latitude = filemetadata('latitude')
    longitude = filemetadata('longitude')
    altitude = filemetadata('altitude')
    metadata = filemetadata('metadata')
    sweep_start_ray_index = filemetadata('sweep_start_ray_index')
    sweep_end_ray_index = filemetadata('sweep_end_ray_index')
    sweep_number = filemetadata('sweep_number')
    sweep_mode = filemetadata('sweep_mode')
    fixed_angle = filemetadata('fixed_angle')
    elevation = filemetadata('elevation')
    _range = filemetadata('range')
    azimuth = filemetadata('azimuth')
    _time = filemetadata('time')

    # other metadata
    frequency = filemetadata('frequency')

    ret = metranet_read_polar(filename, 'ZH', physic_value=True)

    # total number of rays composing the sweep
    total_record = ret.pol_header[0].total_record
    if total_record == 0:
        raise ValueError('Number of rays in file=0.')

    # number of gates in a ray
    num_gates = ret.pol_header[0].num_gates

    # sweep_number (is the sweep index)
    # current sweep number (from 0 to 19)
    sweep_number['data'] = np.array([ret.pol_header[0].current_sweep-1])

    az_data = np.empty(total_record, dtype='float64')
    el_data = np.empty(total_record, dtype='float64')
    time_data = np.empty(total_record, dtype='float64')
    ray_index_data = np.empty(total_record, dtype='float64')

    # get radar id
    radar_id = "".join(map(chr, ret.pol_header[0].scan_id))
    radar_id = radar_id.strip()

    ant_mode = ret.pol_header[0].ant_mode  # scanning mode code
    if ant_mode == 0:
        scan_type = 'ppi'
        sweep_mode['data'] = np.array(['azimuth_surveillance'])
        # ray starting elevation angle information
        fixed_angle['data'] = np.array(
            [Selex_Angle(ret.pol_header[0].start_angle).el],
            dtype='float64')

        # azimuth
        for i in range(total_record):
            # ray starting azimuth angle information
            start_angle = Selex_Angle(ret.pol_header[i].start_angle).az
            # ray ending azimuth angle information
            end_angle = Selex_Angle(ret.pol_header[i].end_angle).az
            if end_angle > start_angle:
                az_data[i] = start_angle + (end_angle-start_angle)/2.
            else:
                az_data[i] = start_angle + (end_angle+360.-start_angle)/2.
        azimuth['data'] = az_data

        # elevation
        elevation['data'] = np.repeat(fixed_angle['data'], total_record)
    elif ant_mode == 1:
        scan_type = 'rhi'
        sweep_mode['data'] = np.array(['elevation_surveillance'])
        # ray starting azimuth angle information
        fixed_angle['data'] = np.array(
            [Selex_Angle(ret.pol_header[0].start_angle).az],
            dtype='float64')

        # azimuth
        azimuth['data'] = np.repeat(fixed_angle['data'], total_record)

        # elevation
        for i in range(total_record):
            # ray starting elevation angle information
            start_angle = Selex_Angle(ret.pol_header[i].start_angle).el
            # ray ending elevation angle information
            end_angle = Selex_Angle(ret.pol_header[i].end_angle).el
            el_data[i] = start_angle + (end_angle-start_angle)/2.
        elevation['data'] = el_data
    elif ant_mode == 2:
        scan_type = 'sector'  # sector scan
        sweep_mode['data'] = np.array(['sector'])
        # ray starting elevation angle information
        fixed_angle['data'] = np.array(
            [Selex_Angle(ret.pol_header[0].start_angle).el],
            dtype='float64')

        # azimuth
        for i in range(total_record):
            # ray starting azimuth angle information
            start_angle = Selex_Angle(ret.pol_header[i].start_angle).az
            # ray ending azimuth angle information
            end_angle = Selex_Angle(ret.pol_header[i].end_angle).az
            if end_angle > start_angle:
                az_data[i] = start_angle + (end_angle-start_angle)/2.
            else:
                az_data[i] = start_angle + (end_angle+360.-start_angle)/2.
        azimuth['data'] = az_data

        # elevation
        elevation['data'] = np.repeat(fixed_angle['data'], total_record)
    elif ant_mode == 3:
        scan_type = 'other'  # point of interest scan
        sweep_mode['data'] = np.array(['pointing'])
        # ray starting elevation angle information
        fixed_angle['data'] = np.array(
            [Selex_Angle(ret.pol_header[0].start_angle).el],
            dtype='float64')

        # azimuth
        # ray starting elevation angle information
        azimuth['data'] = Selex_Angle(ret.pol_header[0].start_angle).az

        # elevation
        elevation['data'] = fixed_angle['data']
    elif ant_mode == 4:
        scan_type = 'other'  # off
        sweep_mode['data'] = np.array(['idle'])
    else:
        raise ValueError('Unknown scan type')

    # range (to center of beam [m])
    # distance to start of first range gate [usually 0 m]
    start_range = float(ret.pol_header[0].start_range)
    # range resolution [m]
    gate_width = float(ret.pol_header[0].gate_width)*1000.
    _range['data'] = np.linspace(
        start_range+gate_width/2., float(num_gates-1.) *
        gate_width+gate_width/2., num_gates, dtype='float32')

    # time (according to default_config this is the Time at the center of
    # each ray, in fractional seconds since the volume started)
    # here we find the time of end of ray since the first ray in the sweep
    for i in range(total_record):
        # time when the ray was created [s from 1.1.1970].
        # (last received pulse+processing time)
        data_time = float(ret.pol_header[i].data_time)
        # the hundreths of seconds to add to the data_time
        data_time_residue = float(ret.pol_header[i].data_time_residue)
        time_data[i] = data_time+data_time_residue/100.
        ray_index_data[i] = ret.pol_header[i].sequence

    sweep_start = min(time_data)
    start_time = datetime.datetime.utcfromtimestamp(sweep_start)
    _time['data'] = time_data-sweep_start
    _time['units'] = make_time_unit_str(start_time)

    # sweep_start_ray_index, sweep_end_ray_index
    # should be specified since start of volume but we do not have this
    # information so we specify it since start of sweep instead.
    sweep_start_ray_index['data'] = np.array(
        [min(ray_index_data)], dtype='int32')  # ray index of first ray
    sweep_end_ray_index['data'] = np.array(
        [max(ray_index_data)], dtype='int32')   # ray index of last ray

    # ----  other information that can be obtained from metadata in file
    #       sweep information:
    #       total number of sweeps compositing the volume (i.e. 20):
    #       total_sweep=ret.pol_header[0].total_sweep
    #       total number of data bytes in the ray
    #   (num_gates*number_of_moments*(number_of_bytes in each moment)):
    #       data_bytes=ret.pol_header[0].data_bytes
    #       # time period of repetition of the volume scan:
    #       repeat_time=ret.pol_header[0].repeat_time
    #       # Nyquist velocity [m/s]:
    #       ny_quest=ret.pol_header[0].ny_quest
    #       Maximum Doppler spectrum width [m/s]:
    #       w_ny_quest=ret.pol_header[0].w_ny_quest
    #
    #       # ray specific information
    #       0 no end of sweep, 1 end of sweep, 2 end of volume scan:
    #       end_of_sweep=ret.pol_header[0].end_of_sweep
    #       number of pulses used in data integration:
    #       pulses=ret.pol_header[0].pulses
    # ------------------------------------------------------------------

    # metadata
    metadata['instrument_name'] = radar_id

    # hardcoded radar dependent metadata
    print('radar name', radar_id)
    if radar_id.startswith('ALB'):
        latitude['data'] = np.array([47.284333], dtype='float64')
        longitude['data'] = np.array([8.512000], dtype='float64')
        altitude['data'] = np.array([938.], dtype='float64')
        frequency['data'] = np.array([5450e6], dtype='float64')
    elif radar_id.startswith('DOL'):
        latitude['data'] = np.array([46.425113], dtype='float64')
        longitude['data'] = np.array([6.099415], dtype='float64')
        altitude['data'] = np.array([1682.], dtype='float64')
        frequency['data'] = np.array([5430e6], dtype='float64')
    elif radar_id.startswith('LEM'):
        latitude['data'] = np.array([46.040761], dtype='float64')
        longitude['data'] = np.array([8.833217], dtype='float64')
        altitude['data'] = np.array([1626.], dtype='float64')
        frequency['data'] = np.array([5455e6], dtype='float64')
    elif radar_id.startswith('PLA'):
        latitude['data'] = np.array([46.370646], dtype='float64')
        longitude['data'] = np.array([7.486552], dtype='float64')
        altitude['data'] = np.array([2937.], dtype='float64')
        frequency['data'] = np.array([5468e6], dtype='float64')
    elif radar_id.startswith('WEI'):
        latitude['data'] = np.array([46.834974], dtype='float64')
        longitude['data'] = np.array([9.794458], dtype='float64')
        altitude['data'] = np.array([2850.], dtype='float64')
        frequency['data'] = np.array([5433e6], dtype='float64')
    else:
        print('Unknown radar. Radar position cannot be specified')

    # fields
    fields = {}

    # ZH field
    field_name = filemetadata.get_field_name('ZH')
    if field_name is not None:
        # create field dictionary
        field_dic = filemetadata(field_name)
        field_dic['data'] = ret.data
        field_dic['_FillValue'] = get_fillvalue()
        fields[field_name] = field_dic

    # rest of fields
    if bfile.startswith('PM'):
        for i in range(1, NPM_MOM):
            field_name = filemetadata.get_field_name(PM_MOM[i])
            if field_name is not None:
                ret = metranet_read_polar(
                    filename, PM_MOM[i], physic_value=True)
                # create field dictionary
                field_dic = filemetadata(field_name)
                field_dic['data'] = ret.data
                field_dic['_FillValue'] = get_fillvalue()
                fields[field_name] = field_dic
    elif bfile.startswith('PH'):
        for i in range(1, NPH_MOM):
            field_name = filemetadata.get_field_name(PH_MOM[i])
            if field_name is not None:
                ret = metranet_read_polar(
                    filename, PH_MOM[i], physic_value=True)
                # create field dictionary
                field_dic = filemetadata(field_name)
                field_dic['data'] = ret.data
                field_dic['_FillValue'] = get_fillvalue()
                fields[field_name] = field_dic
    else:
        for i in range(1, NPL_MOM):
            field_name = filemetadata.get_field_name(PL_MOM[i])
            if field_name is not None:
                ret = metranet_read_polar(
                    filename, PL_MOM[i], physic_value=True)
                # create field dictionary
                field_dic = filemetadata(field_name)
                field_dic['data'] = ret.data
                field_dic['_FillValue'] = get_fillvalue()
                fields[field_name] = field_dic

    # instrument_parameters
    instrument_parameters = dict()
    instrument_parameters.update({'frequency': frequency})

    return Radar(_time, _range, fields, metadata, scan_type, latitude,
                 longitude, altitude, sweep_number, sweep_mode, fixed_angle,
                 sweep_start_ray_index, sweep_end_ray_index, azimuth,
                 elevation, instrument_parameters=instrument_parameters)


def metranet_read_polar(radar_file, moment="ZH", physic_value=True):
    """
    Reads a METRANET polar data file

    Parameters
    ----------
    radar_file : str
        file name
    moment : str
        moment name
    physica_value : boolean
        If true returns the physical value. Otherwise the digital value.

    Returns
    -------
    ret_data : Radar_Metranet object
        An object containing the information read from the file

    """
    ret_data = Radar_Metranet(moment=moment)
    prd_header = {}
    prd_data_level = np.zeros(256)

    # uppercase moment
    moment = moment.upper()

    # as big as possible
    max_bins = 3000
    max_azimuths = 500

    # expected number of azimuths
    naz = 360

    # read BINARY data
    prdt_size = max_azimuths * max_bins
    if moment == "PHI":
        prdt_size *= 2
        prd_data = np.zeros(prdt_size, np.ushort)
    else:
        prd_data = np.zeros(prdt_size, np.ubyte)

    t_pol_header = (Header_stru * max_azimuths)()

    ret = metranet_lib.py_decoder_p2(
        ctypes.c_char_p(radar_file.encode("utf-8")),
        np.ctypeslib.as_ctypes(prd_data), ctypes.c_int(prdt_size),
        ctypes.c_char_p(moment.encode("utf-8")), ctypes.byref(t_pol_header))

    if ret > max_azimuths:
        if moment == "PHI":
            ret *= 0.5

        # reshape
        bins = int(t_pol_header[0].num_gates)
        if bins < 1:
            # if num_gates is less than 1 (exception)
            bins = ret/naz
        prd_data = prd_data[0:naz*bins]
        prd_data = np.reshape(prd_data, (naz, bins))

        # reorder pol_header
        pol_header = (Header_stru * naz)()
        for i in range(0, naz):
            angle_start = Selex_Angle(t_pol_header[i].start_angle)
            pol_header[int(angle_start.az)] = t_pol_header[i]

        # select scale
        if (moment == "ZH" or moment == "ZV" or moment == "ZHC" or
                moment == "ZVC"):
            prd_data_level = np.fromiter(range(256), dtype=np.float32)/2.-32.
            prd_data_level[0] = get_fillvalue()
        elif moment == "ZDR":
            prd_data_level = (
                (np.fromiter(range(256), dtype=np.float32)+1) /
                16.1259842-7.9375)
            prd_data_level[0] = get_fillvalue()
        elif moment == "RHO":
            if ((pol_header[0].data_time > 1341619200) or
                    (pol_header[0].data_time > 1335484800 and
                     (pol_header[0].scan_id[0] == ord('D') or
                      pol_header[0].scan_id[0] == ord('L')))):
                # logaritmic scale
                prd_data_level = (
                    1.003-10.**(-np.fromiter(range(256), dtype=np.float32) *
                                0.01))
            else:
                # linear scale (old data)
                prd_data_level = (
                    np.fromiter(range(256), dtype=np.float32)/255.)

            prd_data_level[0] = get_fillvalue()
        elif moment == "PHI":
            prd_data_level = (
                (np.fromiter(range(256*256), dtype=np.float32)-32768) /
                32767.*180.)
        elif moment == "VEL":
            prd_data_level = (
                (np.fromiter(range(256), dtype=np.float32)-128)/127. *
                pol_header[0].ny_quest)
            prd_data_level[0] = get_fillvalue()
        elif moment == "WID":
            prd_data_level = (
                (np.fromiter(range(256), dtype=np.float32)/255. *
                 pol_header[0].ny_quest))
            prd_data_level[0] = get_fillvalue()
        elif moment == "MPH":
            prd_data_level = (
                (np.fromiter(range(256), dtype=np.float32)-128)/127.*180.)
        elif moment == "ST1" or moment == "ST2" or moment == "WBN":
            prd_data_level = np.fromiter(range(256), dtype=np.float32)/10.
        elif moment == "CLT":
            prd_data_level = np.fromiter(range(256), dtype=np.float32)

        ret_data = Radar_Metranet(
            data=prd_data, scale=prd_data_level, pol_header=pol_header,
            moment=moment)
        mask = prd_data == 0

        if (physic_value):
            masked_data = np.ma.array(
                prd_data_level[prd_data], mask=mask,
                fill_value=get_fillvalue())
            ret_data.data = masked_data
        else:
            masked_data = np.ma.array(prd_data, mask=mask, fill_value=0)
            ret_data.data = masked_data
        ret_data.pol_header = pol_header
        ret_data.scale = prd_data_level

        # change parameters in header
        for i in range(naz):
            ret_data.pol_header[i].total_record = naz

    return ret_data


class Selex_Angle:
    """
    Class used to convert from digital number to angle

    Attributes
    ----------
    az : float
        azimuth angle value (degrees or radiants)

    el : float
        elevation angle value (degrees or radiants)
    """
    def __init__(self, angle=0, radiant=False):
        if (radiant):
            reform = 2 * 3.1415926
        else:
            reform = 360.
        self.az = (angle & 0xFFFF) / 65535. * reform
        self.el = (angle >> 16) / 65535. * reform


class Radar_Metranet:
    """
    Class containing the information read from the METRANET file

    Attributes
    ----------
    type : str
        Information type

    data : matrix
        The digital number values

    scale : array
        The scale used to convert from digital units to physical units

    pol_header : Header_stru object
        Object containing the polar data header

    moment : str
        moment name

    """
    type = "Radar"

    def __init__(self, data=(-1), scale=np.zeros(256), header=(),
                 pol_header=(), moment='ZH'):
        self.data = data
        self.scale = scale
        self.header = header
        self.pol_header = pol_header
        self.moment = moment


class Header_stru(ctypes.Structure):
    """
    Class that stores the header of a METRANET POLAR data file contained in a
    structure used by the C-library reader

    C-Structure of METRANET POLAR data:

    struct moment_header_struct
    {
        unsigned int record_type;    /* data format (moment1) + moment mask */
        unsigned int scan_id;
        unsigned int host_id;
        unsigned int start_angle;
        unsigned int end_angle;

        unsigned char ant_mode;
        unsigned char total_sweep;
        unsigned char current_sweep;  /* 1-any number up to 99            */
        unsigned char end_of_sweep;   /* 0=not end, 1=end sweep, 2=end volume
                                      */
        short sequence;               /* ray sequence  number in a sweep  */
        short total_record;           /* total ray number in sweep     */
        short pulses;
        short num_gates;

        int data_bytes;
        unsigned short data_flag;
        short data_time_residue;      /* data time residue in 0.01 sec    */
        unsigned int data_time;       /* data time in second              */
        short repeat_time;
        char compressed;              /* flag for compression of data     */
        char priority;                /* for file name use                */

        float   ny_quest;
        float   gate_width;
        float   w_ny_quest;           /* may be used for other variable   */
        float   start_range;
    };
    """
    _fields_ = [
        ("record_type", ctypes.c_uint),
        ("scan_id", ctypes.c_ubyte*4),
        ("host_id", ctypes.c_ubyte*4),
        ("start_angle", ctypes.c_int),
        ("end_angle", ctypes.c_int),

        ("ant_mode", ctypes.c_ubyte),
        ("total_sweep", ctypes.c_ubyte),
        ("current_sweep", ctypes.c_ubyte),
        ("end_of_sweep", ctypes.c_ubyte),

        ("sequence", ctypes.c_short),
        ("total_record", ctypes.c_short),
        ("pulses", ctypes.c_short),
        ("num_gates", ctypes.c_short),

        ("data_bytes", ctypes.c_int),
        ("data_flag", ctypes.c_ushort),
        ("data_time_residue", ctypes.c_short),
        ("data_time", ctypes.c_uint),
        ("repeat_time", ctypes.c_short),
        ("compressed", ctypes.c_ubyte),
        ("priority", ctypes.c_ubyte),

        ("ny_quest", ctypes.c_float),
        ("gate_width", ctypes.c_float),
        ("w_ny_quest", ctypes.c_float),
        ("start_range", ctypes.c_float),
    ]
