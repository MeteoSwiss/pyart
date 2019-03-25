#!/opt/local/opkg/bin/python
"""
metranet library
================

functions to read METRANET files, require external shared library
srn_idl_py_lib.<ARCH>.so

.. autosummary::
    :toctree: generated/

    get_radar_site_info
    get_library_path
    get_library
    read_polar
    read_product
    read_file


.. autosummary::
    :toctree: generated/
    :template: dev_template.rst

    RadarData
    Header_struPM
    Header_struMS
    Selex_Angle


History

      V0.1 20160512 mbc first prototype
      V0.2 20160519 mbc changed in module
      V0.3 20171020 mbc changed for MS/MH/ML files
      V0.4 20171104 mbc fill some basic metadata in header of polar files
      V0.5 20171108 mbc added msheader with new info's
      V0.6 20171205 mbc fix for python3

"""

from __future__ import print_function

import ctypes
import os
import sys
import platform
import string
import time
from warnings import warn
import traceback

import numpy as np

# some values valid for all sites
NPM_MOM = 11
NPH_MOM = 12
NPL_MOM = 9

# fix for python3
if sys.version_info[0] == 3:
    def xrange(i):
        return range(i)


class RadarData:
    """
    A class for storing radar data.

    Attributes
    ----------
    type : str
        type of data
    data : numpy array or numpy masked array
        array containing the data
    scale : numpy array
        array containing the scale used to transform the data from digital
        to physical units
    header : dict
        dictionary containing metadata
    pol_header : dict
        dictionary containing metadata of the polar files
    moment : str
        moment name

    """
    type = "RadarData"

    def __init__(self, data=np.zeros(0),
                 scale=np.fromiter(xrange(256), dtype=np.uint), header=(),
                 pol_header=(), moment='ZH'):
        self.data = data
        self.scale = scale
        self.header = header
        self.pol_header = pol_header
        self.moment = moment


class Header_struPM(ctypes.Structure):
    """
    A class containing the data from the header of the polar PM files

    Attributes
    ----------
    _fields_: dict
        A dictionary containing the metadata contained in the PM file

    C-Structure of METRANET POLAR data PM format

    struct moment_header_struct
    {
            unsigned int record_type;  data format (moment1) + moment mask
            unsigned int scan_id;
            unsigned int host_id;
            unsigned int start_angle;
            unsigned int end_angle;

            unsigned char ant_mode;
            unsigned char total_sweep;
            unsigned char current_sweep;  1-any number up to 99
            unsigned char end_of_sweep;  0=not end, 1=end sweep, 2=end volume

            short sequence;  ray sequence number in a sweep
            short total_record;  total ray number in sweep
            short pulses;
            short num_gates;

            int data_bytes;
            unsigned short data_flag;
            short data_time_residue;  data time residue in 0.01 sec
            unsigned int data_time;  data time in second
            short repeat_time;
            char compressed;  flag for compression of data
            char priority;  for file name use

            float   ny_quest;
            float   gate_width;
            float   w_ny_quest;  may be used for other variable
            float   start_range;
    };
    """
    _fields_ = [
        ("record_type", ctypes.c_uint),
        ("scan_id", ctypes.c_int),
        ("host_id", ctypes.c_int),
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


class Header_struMS(ctypes.Structure):
    """
    A class containing the data from the header of the polar MS files

    Attributes
    ----------
    _fields_: dict
        A dictionary containing the metadata contained in the MS file

    C-Structure of METRANET POLAR data MS format

    C-code from METRANET2/share/include/sweep_file.h

    struct sweep_header
        {
            int8_t FileId[4];           //4:4
            uint8_t Version;            //1:5
            uint8_t Spare1[3];          //3:8
            uint32_t Length;            //4:12
            int8_t RadarName[16];       //16:28
            int8_t ScanName[16];        //16:44
            float RadarLat;             //4:48
            float RadarLon;             //4:52
            float RadarHeight;          //4:56
            uint8_t SequenceSweep;      //1:57
            uint8_t CurrentSweep;       //1:58
            uint8_t TotalSweep;         //1:59
            uint8_t AntMode;            //1:60
            uint8_t Priority;           //1:61
            uint8_t Quality;            //1:62
            uint8_t Spare2[2];          //2:64
            uint16_t RepeatTime;        //2:66
            uint16_t NumMoments;        //2:68
            float GateWidth;            //4:72
            float WaveLength;           //4:76
            float PulseWidth;           //4:80
            float StartRange;           //4:84
            uint32_t MetaDataSize;      //4:88
        };
    """
    _fields_ = [
        ("FileId", ctypes.c_int32),
        ("Version", ctypes.c_uint8),
        ("Spare1", ctypes.c_uint8*3),
        ("Length", ctypes.c_uint32),
        ("RadarName", ctypes.c_int8*16),
        ("ScanName", ctypes.c_int8*16),
        ("RadarLat", ctypes.c_float),
        ("RadarLon", ctypes.c_float),
        ("RadarHeight", ctypes.c_float),
        ("SequenceSweep", ctypes.c_uint8),
        ("CurrentSweep", ctypes.c_uint8),
        ("TotalSweep", ctypes.c_uint8),
        ("AntMode", ctypes.c_uint8),
        ("Priority", ctypes.c_uint8),
        ("Quality", ctypes.c_uint8),
        ("Spare2", ctypes.c_uint8*2),
        ("RepeatTime", ctypes.c_uint16),
        ("NumMoments", ctypes.c_uint16),
        ("GateWidth", ctypes.c_float),
        ("WaveLength", ctypes.c_float),
        ("PulseWidth", ctypes.c_float),
        ("StartRange", ctypes.c_float),
        ("MetaDataSize", ctypes.c_uint32),
    ]


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
        if radiant:
            reform = 2 * 3.1415926
        else:
            reform = 360.
        self.az = (angle & 0xFFFF)/65535.*reform
        self.el = (angle >> 16)/65535.*reform


def get_radar_site_info(verbose=False):
    """
        return dictionary with radar'info

        Returns
        -------
        radar_def : dict
            dictionary containing radar site information
    """
    radar_def_load = False
    try:
        import yaml
        yaml_module = True
    except ImportError:
        yaml_module = False

    if yaml_module:
        metranet_yaml_file = "metranet.yaml"
        path_yaml_file = [
            "/store/msrad/python/library/radar/io/", "/opt/ccs4/python",
            "/proj/lom/python/library/radar/io"]
        for p in path_yaml_file:
            full_file = p + "/" + metranet_yaml_file
            if os.path.isfile(full_file):
                print("look for Radar_Site_info from %s" % full_file)
                try:
                    with open(full_file) as f:
                        radar_def = yaml.load(f)
                        radar_def_load = True
                        if verbose:
                            print("Read Radar_Site_info from %s" % full_file)
                except Exception as ee:
                    warn(str(ee))
                    traceback.print_exc()

    if not radar_def_load:
        # HardCoded definition
        print("**** HARD-CODED Radar site Value, may be not updated!!")
        # some values valid for all sites
        c_speed = 299792458  # m/s

        radar_default = {}

        radar_default['dwhname'] = 'undef'
        radar_default['RadarCHY'] = float('nan')
        radar_default['RadarCHX'] = float('nan')
        radar_default['FileId'] = 1179665477
        radar_default['Version'] = 0
        radar_default['RadarName'] = "undef"
        radar_default['ScanName'] = "undef"
        radar_default['RadarLat'] = float('nan')
        radar_default['RadarLon'] = float('nan')
        radar_default['RadarHeight'] = float('nan')
        radar_default['Frequency'] = float('nan')
        radar_default['WaveLength'] = float('nan')
        radar_default['PulseWidth'] = 0.5
        radar_default['SweepsOrder'] = (
            9, 7, 5, 3, 1, 19, 17, 15, 13, 11, 10, 8, 6, 4, 2, 20, 18, 16, 14,
            12)
        radar_default['NumMomentsPM'] = NPM_MOM
        radar_default['NumMomentsPH'] = NPH_MOM
        radar_default['NumMomentsPL'] = NPL_MOM

        radar_def = {}

        rname = 'A'
        radar_def[rname] = radar_default.copy()
        radar_def[rname]['dwhname'] = "ALB"
        radar_def[rname]['RadarName'] = "Albis"
        radar_def[rname]['RadarCHY'] = 681201
        radar_def[rname]['RadarCHX'] = 237604
        radar_def[rname]['RadarLat'] = 47.284333
        radar_def[rname]['RadarLon'] = 8.512000
        radar_def[rname]['RadarHeight'] = 938.0
        radar_def[rname]['ScanName'] = "1095516672"
        radar_def[rname]['Frequency'] = 5450e6
        radar_def[rname]['WaveLength'] = (
            c_speed/radar_def[rname]['Frequency']*1e2)

        rname = 'D'
        radar_def[rname] = radar_default.copy()
        radar_def[rname]['dwhname'] = "DOL"
        radar_def[rname]['RadarName'] = "Dole"
        radar_def[rname]['RadarCHY'] = 497057
        radar_def[rname]['RadarCHX'] = 142408
        radar_def[rname]['RadarLat'] = 46.425113
        radar_def[rname]['RadarLon'] = 6.099415
        radar_def[rname]['RadarHeight'] = 1682.0
        radar_def[rname]['ScanName'] = "1146047488"
        radar_def[rname]['Frequency'] = 5430e6
        radar_def[rname]['WaveLength'] = (
            c_speed/radar_def[rname]['Frequency']*1e2)

        rname = 'L'
        radar_def[rname] = radar_default.copy()
        radar_def[rname]['dwhname'] = "MLE"
        radar_def[rname]['RadarName'] = "Lema"
        radar_def[rname]['RadarCHY'] = 707957
        radar_def[rname]['RadarCHX'] = 99762
        radar_def[rname]['RadarLat'] = 46.040761
        radar_def[rname]['RadarLon'] = 8.833217
        radar_def[rname]['RadarHeight'] = 1626.0
        radar_def[rname]['ScanName'] = "1279610112"
        radar_def[rname]['Frequency'] = 5455e6
        radar_def[rname]['WaveLength'] = (
            c_speed/radar_def[rname]['Frequency']*1e2)

        rname = 'P'
        radar_def[rname] = radar_default.copy()
        radar_def[rname]['dwhname'] = "PPM"
        radar_def[rname]['RadarName'] = "PlaineMorte"
        radar_def[rname]['RadarCHY'] = 603687
        radar_def[rname]['RadarCHX'] = 135476
        radar_def[rname]['RadarLat'] = 46.370646
        radar_def[rname]['RadarLon'] = 7.486552
        radar_def[rname]['RadarHeight'] = 2937.0
        radar_def[rname]['ScanName'] = "0"
        radar_def[rname]['Frequency'] = 5468e6
        radar_def[rname]['WaveLength'] = (
            c_speed/radar_def[rname]['Frequency']*1e2)

        rname = 'W'
        radar_def[rname] = radar_default.copy()
        radar_def[rname]['dwhname'] = "WEI"
        radar_def[rname]['RadarName'] = "WeissFluhGipfel"
        radar_def[rname]['RadarCHY'] = 779700
        radar_def[rname]['RadarCHX'] = 189790
        radar_def[rname]['RadarLat'] = 47.284333
        radar_def[rname]['RadarLon'] = 9.794458
        radar_def[rname]['RadarHeight'] = 2850.0
        radar_def[rname]['ScanName'] = "0"
        radar_def[rname]['Frequency'] = 5433e6
        radar_def[rname]['WaveLength'] = (
            c_speed/radar_def[rname]['Frequency']*1e2)

    return radar_def


def get_library_path():
    """
        look for valid library path

        Returns
        -------
        library_metranet_path : str
            METRANET library path
    """
    libray_paths = [
        os.environ.get('METRANETLIB_PATH'),
        "/store/msrad/idl/lib/radlib4", "/opt/ccs4/lib",
        "/opt/local/opkg/share/ccs4/lib",
        "/proj/lom/idl/lib/radlib4"]

    library_metranet_path = ''
    for p in libray_paths:
        if p is not None:
            if os.path.isdir(p):
                library_metranet_path = p
                break

    if not library_metranet_path:
        sys.exit(" ENV Variable library_metranet_path NOT defined")


    return library_metranet_path


def get_library(verbose=False, momentms=True):
    """
        return the link to C-shared library

        Parameters
        ----------
        verbose : Boolean
            If true print out extra information
        momentsms : Boolean
            If true returns the link to the MS library

        Returns
        -------
        metranet_lib : link
            loaded METRANET C-library

    """
    # default path
    try:
        library_metranet_path = get_library_path()
    except SystemExit:
        sys.exit('ERROR: METRANET library path not found')

    # library system
    library_metranet_linux = 'srn_idl_py_lib.x86_64'
    library_metranet_sparc32 = 'srn_idl_py_lib.sparc32.so'
    library_metranet_sparc64 = 'srn_idl_py_lib.sparc64.so'


    library_metranet = 'x'
    if platform.system() == 'SunOS':
        if platform.architecture()[0] == '64bit':
            library_metranet = library_metranet_sparc32
        else:
            library_metranet = library_metranet_sparc32

    if platform.system() == 'Linux':
        if momentms:
            library_metranet = library_metranet_linux + '.MS.so'
        else:
            library_metranet = library_metranet_linux + '.so'

    if verbose:
        print("library %s/%s:" % (library_metranet_path, library_metranet))

    if library_metranet == 'x':
        sys.exit("ERROR: Platform not found")

    metranet_lib = ctypes.cdll.LoadLibrary(
        library_metranet_path + '/' + library_metranet)

    return metranet_lib


def read_polar(radar_file, moment="ZH", physic_value=False,
               masked_array=False, verbose=False):
    """
    Reads a METRANET polar data file

    Parameters
    ----------
    radar_file : str
        file name
    moment : str
        moment name
    physic_value : boolean
        If true returns the physical value. Otherwise the digital value
    masked_array : boolean
        If true returns a numpy masked array with NaN values masked. Otherwise
        returns a regular masked array with NaN values
    verbose : boolean
        If true prints out extra information

    Returns
    -------
    ret_data : RadarData object
        An object containing the information read from the file. None if
        the file has not been properly read

    """
    c_speed = 299792458  # m/s

    ret_data = RadarData(moment=moment)
    prd_header = {}
    prd_data_level = ret_data.scale

    # uppercase moment
    moment = moment.upper()

    # as big as possible
    max_bins = 3000
    max_azimuths = 500

    if verbose:
        print("Read POLAR file %s " % radar_file)

        # read BINARY data
    prdt_size = max_bins * max_azimuths
    if moment == 'PHI':
        prdt_size *= 2
        prd_data = np.zeros(prdt_size, np.ushort)
    else:
        prd_data = np.zeros(prdt_size, np.ubyte)

    bfile = os.path.basename(radar_file)
    if (bfile.startswith('MS') or bfile.startswith('MH') or
            bfile.startswith('ML')):
        momentms = True
        Header_stru = Header_struPM
    else:
        momentms = False
        Header_stru = Header_struPM

    t_pol_header = (Header_stru * max_azimuths)()
    t_rad_header = (Header_struMS * 1)()

    metranet_lib = get_library(momentms=momentms, verbose=verbose)

    ret = metranet_lib.py_decoder_p2(
        ctypes.c_char_p(radar_file.encode('utf-8')),
        np.ctypeslib.as_ctypes(prd_data), ctypes.c_int(prdt_size),
        ctypes.c_char_p(moment.encode('utf-8')), ctypes.byref(t_pol_header),
        ctypes.byref(t_rad_header), ctypes.c_int(verbose))

    if ret <= max_azimuths:
        return None

    if moment == 'PHI':
        ret *= 0.5

    # reshape matrix data
    bins = t_pol_header[0].num_gates
    nr_az = int(ret / bins)
    if bins < 1:
        # if num_gates is less than 1 (exception)
        bins = ret/360
    if nr_az > 360:
        nr_az = 360

    if nr_az < 360:
        print("WARNING incomplete sweep")

    if verbose:
        print("ret=%d" % ret)
        print("bins=%d" % bins)
        print("nr_az=%d" % nr_az)
        print("len(prd_data)=%d" % len(prd_data))
    prd_data = prd_data[0: nr_az * bins]
    prd_data = np.reshape(prd_data, (nr_az, bins))

    # reorder pol_header
    pol_header = (Header_stru * nr_az)()
    for i in range(0, nr_az):
        angle_start = Selex_Angle(t_pol_header[i].start_angle)
        pol_header[int(angle_start.az)] = t_pol_header[i]

    # select scale
    if moment in ('ZH', 'ZV', 'ZHC', 'ZVC'):
        prd_data_level = np.fromiter(xrange(256), dtype=np.float32)/2.-32.
        prd_data_level[0] = np.nan
    elif moment == 'ZDR':
        prd_data_level = (
            (np.fromiter(xrange(256), dtype=np.float32)+1) /
            16.1259842-7.9375)
        prd_data_level[0] = np.nan
    elif moment == 'RHO':
        if ((pol_header[0].data_time > 1341619200) or
                (pol_header[0].data_time > 1335484800 and
                 (pol_header[0].scan_id[0] == ord('D') or
                  pol_header[0].scan_id[0] == ord('L')))):
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
            pol_header[0].ny_quest)
        prd_data_level[0] = np.nan
    elif moment == 'WID':
        prd_data_level = (np.fromiter(
            xrange(256), dtype=np.float32)/255.*pol_header[0].ny_quest)
        prd_data_level[0] = np.nan
    elif moment == 'MPH':
        prd_data_level = ((np.fromiter(
            xrange(256), dtype=np.float32)-128)/127.*180.)
    elif moment in ('ST1', 'ST2', 'WBN'):
        prd_data_level = (np.fromiter(
            xrange(256), dtype=np.float32)/10.)
    elif moment == "CLT":
        prd_data_level = np.fromiter(xrange(256), dtype=np.float32)

    if verbose:
        print("prd_data shape ", prd_data.shape)
        print("min/max prd_data: ", prd_data.min(), prd_data.max())
        set(string.printable)

        print("prd_data scan_id ", pol_header[0].scan_id)
        print("prd_data host_id ", pol_header[0].host_id)
        print("data level ", prd_data_level[0:10])
        for i in range(0, nr_az, 10):
            angle_start = Selex_Angle(pol_header[i].start_angle)
            angle_end = Selex_Angle(pol_header[i].end_angle)
            x = pol_header[i].num_gates / 2
            print(("pol_header[%3d].num_gates: %d, time %d.%03d start_az/el:"
                   " %6.1f/%4.1f  end_az/el: %6.1f/%4.1f real[%d]=%6.2f"
                   " ( raw=%5d)") %
                  (i, pol_header[i].num_gates, pol_header[i].data_time,
                   pol_header[i].data_time_residue, angle_start.az,
                   angle_start.el, angle_end.az, angle_end.el, x,
                   prd_data_level[prd_data[i, x]], prd_data[i, x]))

    if physic_value:
        ret_data.data = prd_data_level[prd_data]
        if masked_array:
            ret_data.data = np.ma.array(
                ret_data.data, mask=np.isnan(ret_data.data))
            if bfile[1] == 'L':
                if moment in ('ZH', 'ZV', 'ZHC', 'ZVC'):
                    prd_data_level[1] = np.nan
                    ret_data.data = np.ma.masked_where(
                        prd_data == 1, ret_data.data)
    else:
        ret_data.data = prd_data
        if masked_array:
            ret_data.data = np.ma.array(
                ret_data.data, mask=prd_data == 0)
            if bfile[1] == 'L':
                if moment in ('ZH', 'ZV', 'ZHC', 'ZVC'):
                    prd_data_level[1] = np.nan
                    ret_data.data = np.ma.masked_where(
                        prd_data == 1, ret_data.data)

    # header
    prd_header['pid'] = bfile[0:3]
    prd_header['radar'] = bfile[2]
    prd_header['moment'] = moment
    prd_header['column'] = bins
    prd_header['row'] = nr_az
    # prd_header['elevation'] = "%03d" % pol_header[0].current_sweep
    prd_header['volume_time'] = int(
        pol_header[0].data_time - pol_header[0].data_time % 300 + 300)
    prd_header['time'] = time.strftime(
        '%y%j%H%M', time.gmtime(int(prd_header['volume_time'])))
    prd_header["quality"] = bfile[12]

    # if exists extended header, fill header
    if t_rad_header[0].FileId > 0 and t_rad_header[0].GateWidth > 0.:
        prd_header["FileId"] = t_rad_header[0].FileId
        prd_header["Version"] = t_rad_header[0].Version
        # prd_header["Spare1"] = t_rad_header[0].Spare1
        # prd_header["Length"] = t_rad_header[0].Length
        prd_header["RadarName"] = ctypes.string_at(
            t_rad_header[0].RadarName)
        prd_header["ScanName"] = ctypes.string_at(t_rad_header[0].ScanName)
        prd_header["RadarLat"] = t_rad_header[0].RadarLat
        prd_header["RadarLon"] = t_rad_header[0].RadarLon
        prd_header["RadarHeight"] = t_rad_header[0].RadarHeight
        prd_header["SequenceSweep"] = t_rad_header[0].SequenceSweep
        prd_header["CurrentSweep"] = t_rad_header[0].CurrentSweep
        prd_header["TotalSweep"] = t_rad_header[0].TotalSweep
        prd_header["AntMode"] = t_rad_header[0].AntMode
        prd_header["Priority"] = t_rad_header[0].Priority
        # prd_header["Quality"] = t_rad_header[0].Quality
        prd_header["quality"] = t_rad_header[0].Quality
        # prd_header["Spare2"] = t_rad_header[0].Spare2
        prd_header["RepeatTime"] = t_rad_header[0].RepeatTime
        prd_header["NumMoments"] = t_rad_header[0].NumMoments
        prd_header["GateWidth"] = t_rad_header[0].GateWidth
        prd_header["WaveLength"] = t_rad_header[0].WaveLength
        prd_header["Frequency"] = c_speed/(prd_header["WaveLength"]*1e-2)
        prd_header["PulseWidth"] = t_rad_header[0].PulseWidth
        prd_header["StartRange"] = t_rad_header[0].StartRange
        prd_header["MetaDataSize"] = t_rad_header[0].MetaDataSize
        # update with radar sweep radar
        prd_header['radar'] = prd_header["RadarName"][0]
    else:
        # specific radar metadata not available (old PM/PH/PL format)
        # from polar header
        prd_header["CurrentSweep"] = pol_header[0].current_sweep
        prd_header["TotalSweep"] = pol_header[0].total_sweep
        prd_header["AntMode"] = pol_header[0].ant_mode
        prd_header["RepeatTime"] = pol_header[0].repeat_time
        prd_header["GateWidth"] = pol_header[0].gate_width
        prd_header["StartRange"] = pol_header[0].start_range
        prd_header["Priority"] = pol_header[0].priority

        # from hard-coded table
        rname = prd_header['radar']
        radar_def = get_radar_site_info()
        if rname in radar_def.keys():
            prd_header["FileId"] = radar_def[rname]["FileId"]
            prd_header["Version"] = radar_def[rname]["Version"]
            prd_header["RadarName"] = radar_def[rname]["RadarName"]
            prd_header["ScanName"] = radar_def[rname]["ScanName"]
            prd_header["RadarLat"] = radar_def[rname]["RadarLat"]
            prd_header["RadarLon"] = radar_def[rname]["RadarLon"]
            prd_header["RadarHeight"] = radar_def[rname]["RadarHeight"]
            prd_header["Frequency"] = radar_def[rname]["Frequency"]
            prd_header["WaveLength"] = radar_def[rname]["WaveLength"]
            prd_header["PulseWidth"] = radar_def[rname]["PulseWidth"]
            prd_header["SequenceSweep"] = (
                radar_def[rname]['SweepsOrder'].index(
                    pol_header[0].current_sweep))

            if bfile[1] == 'M':
                prd_header["NumMoments"] = radar_def[rname]['NumMomentsPM']
            if bfile[1] == 'H':
                prd_header["NumMoments"] = radar_def[rname]['NumMomentsPH']
            if bfile[1] == 'L':
                prd_header["NumMoments"] = radar_def[rname]['NumMomentsPL']

    ret_data.header = prd_header
    ret_data.scale = prd_data_level
    ret_data.pol_header = pol_header

    # change parameters in header
    for i in range(nr_az):
        ret_data.pol_header[i].total_record = nr_az

    return ret_data


def read_product(radar_file, physic_value=False, masked_array=False,
                 verbose=False):
    """
    Reads a METRANET cartesian data file

    Parameters
    ----------
    radar_file : str
        file name
    physic_value : boolean
        If true returns the physical value. Otherwise the digital value
    masked_array : boolean
        If true returns a numpy masked array with NaN values masked. Otherwise
        returns a regular masked array with NaN values
    verbose : boolean
        If true prints out extra information

    Returns
    -------
    ret_data : RadarData object
        An object containing the information read from the file. None if
        the file has not been properly read

    """
    ret_data = RadarData()
    prd_header = {'row': 0, 'column': 0}

    # read ASCII data
    if verbose:
        print("physic_value: ", physic_value)
        print("File %s: read ASCII" % radar_file)

    try:
        with open(radar_file, 'rb') as data_file:
            lines = data_file.readlines()
    except Exception as ee:
        warn(str(ee))
        print("Unable to read file '%s'" % radar_file)
        return None

    for t_line in lines:
        line = t_line.decode("utf-8").strip('\n')
        if line.find('end_header') == -1:
            data = line.split('=')
            prd_header[data[0]] = data[1]
        else:
            break

    # read BINARY data
    prdt_size = int(prd_header['column']) * int(prd_header['row'])
    if prdt_size < 1:
        print("Error, no size found row=%3d column=%3d" %
              (prd_header['row'], prd_header['column']))
        return None

    if verbose:
        print("File %s: read BINARY data: expected %s bytes, " %
              (radar_file, prdt_size), end='')

    prd_data = np.zeros(
        [int(prd_header['row']), int(prd_header['column'])], np.ubyte)
    prd_data_level = np.zeros(256, np.float32)
    metranet_lib = get_library(verbose=verbose)

    ret = metranet_lib.py_decoder(
        ctypes.c_char_p(radar_file.encode('utf-8')),
        np.ctypeslib.as_ctypes(prd_data), ctypes.c_int(prdt_size),
        np.ctypeslib.as_ctypes(prd_data_level), ctypes.c_int(verbose))

    # convert 0 at end of array with NAN
    conv_zero2nan = True

    i = len(prd_data_level)
    if prd_data_level.max() == prd_data_level.min():
        prd_data_level = np.fromiter(xrange(256), dtype=np.uint32)
    else:
        while conv_zero2nan:
            i -= 1
            if i < 0:
                conv_zero2nan = False
            elif prd_data_level[i] == 0.0:
                prd_data_level[i] = np.nan
            else:
                conv_zero2nan = False

    if verbose:
        print("Found %d bytes" % ret)
        print("prd_data_level[10] = %f" % prd_data_level[10])
        print("min/max prd_data: %d/%d" % (prd_data.min(), prd_data.max()))
        print("first 100 bytes", prd_data[0:100, 0])
        print("data level ", prd_data_level[0:10])

    # ret_data = RadarData(
    #    data=prd_data, header=prd_header, scale=prd_data_level)
    if physic_value:
        ret_data.data = prd_data_level[prd_data]
        if masked_array:
            ret_data.data = np.ma.array(
                ret_data.data, mask=np.isnan(ret_data.data))
            ret_data.data = np.ma.masked_where(prd_data == 0, ret_data.data)
    else:
        ret_data.data = prd_data
        if masked_array:
            ret_data.data = np.ma.array(
                ret_data.data, mask=prd_data == 0)
    ret_data.header = prd_header
    ret_data.scale = prd_data_level

    return ret_data


def read_file(file, moment="ZH", physic_value=False, masked_array=False,
              verbose=False):
    """
    Reads a METRANET data file

    Parameters
    ----------
    file : str
        file name
    moment : str
        moment name
    physic_value : boolean
        If true returns the physical value. Otherwise the digital value
    masked_array : boolean
        If true returns a numpy masked array with NaN values masked. Otherwise
        returns a regular masked array with NaN values
    verbose : boolean
        If true prints out extra information

    Returns
    -------
    ret_data : RadarData object
        An object containing the information read from the file

    """
    bfile = os.path.basename(file)

    if (bfile.startswith('PM') or bfile.startswith('PH') or
            bfile.startswith('PL') or bfile.startswith('MS') or
            bfile.startswith('MH') or bfile.startswith('ML')):
        # polar data from SITE (PH/PM/PL)
        if verbose:
            print("calling read_polar")
        ret = read_polar(
            file, moment=moment, physic_value=physic_value,
            masked_array=masked_array, verbose=verbose)
    else:
        # cartesian / CCS4 products
        if verbose:
            print("calling read_product")
        ret = read_product(
            file, physic_value=physic_value, masked_array=masked_array,
            verbose=verbose)

    return ret
