#!/usr/bin/env python3
"""
metranet python library
================

functions to read METRANET files in pure python, no other library required!

.. autosummary::
    :toctree: generated/

    read_polar
    read_product
    read_file
    _get_radar_site_info
    _nyquist_vel
    _selex2deg

.. autosummary::
    :toctree: generated/
    :template: dev_template.rst

    RadarData
    PolarParser

"""


import copy
import os
import struct
import sys
import traceback
from warnings import warn

import numpy as np

from . import metranet_reader
from .dn_to_float import float_mapping_m, float_mapping_p, nyquist_vel
from .lzw15 import decompress
from .pmfile_structure import (
    BYTE_SIZES,
    MMOMENT_HEADER,
    MMOMENT_INFO_STRUCTURE,
    MRAY_HEADER,
    MSWEEP_HEADER,
    PMOMENTS,
    PRAY_HEADER,
)

###############################################################################
#        Constants
###############################################################################


# For some reasons, the radar name is not encoded in a consistent way in the
# M binary files, this maps all names in files to a single character
RENAME_RADARS_M = {
    "Weissfluhgipfel": "W",
    "Albis": "A ",
    "L": "L",
    "Dole": "D",
    "P": "P",
}
# In P files, radar name is three characters, map to single character
RENAME_RADARS_P = {"WEI": "W", "ALB": "A", "LEM": "L", "DOL": "D", "PPM": "P"}

# Dict to map moment names hardcoded in M files to terminology used
# in metranet C library
MOM_NAME_MAPPING = {
    "Z_V_CLUT": "ZVC",
    "Z_CLUT": "ZHC",
    "W": "WID",
    "V": "VEL",
    "PHIDP": "PHI",
    "STAT1": "ST1",
    "STAT2": "ST2",
    "CLUT": "CLT",
    "UZ": "ZH",
    "UZ_V": "ZV",
    "APH": "MPH",
}

# Other way round, f.ex CLT --> CLUT
MOM_NAME_MAPPING_INV = dict(zip(MOM_NAME_MAPPING.values(), MOM_NAME_MAPPING.keys()))


class RadarData:
    """
    A class for storing radar data.

    Attributes
    ----------
    header : dictionary
        file header (metadara valid for the whole sweep)
    pol_header : list
        list of ray metadata dictionaries
    data : dictionary
        dic of arrays containing data
    """

    def __init__(self, header=None, pol_header=None, data=None, scale=None):
        self.header = header
        self.pol_header = pol_header
        self.data = data
        self.scale = scale


class PolarParser:
    """
    A class for parsing metranet polar data

    Parameters
    ----------
    filename : str
        complete file path of the file to read

    Attributes
    ----------
    file_format : char
        P or M depending on the file format that was used
    file_type : char
        file type (H, M or L)
    bname : str
        basename (without path) of the file to read
    bytarray: memoryview of a bytearray
        binary data contained in the file

    """

    def __init__(self, filename):
        # check if it is the right file. Open it and read it
        bfile = os.path.basename(filename)
        supported_file = (
            bfile.startswith("MH")
            or bfile.startswith("PH")
            or bfile.startswith("MS")
            or bfile.startswith("PM")
            or bfile.startswith("ML")
            or bfile.startswith("PL")
        )

        if not supported_file:
            raise ValueError(
                """Only polar data files starting by MS, MH or ML or PM, PH, PL
                   are supported"""
            )

        self.file_format = bfile[0]  # M or P
        self.file_type = bfile[1]  # H, M or L
        self.bname = bfile

        # Open binary file
        with open(filename, "rb") as f:
            self.bytearray = memoryview(bytearray(f.read()))  # bytearray

        self.read_pos = 0  # Current "cursor" position in file

        # Get endianness
        if self.file_format == "P":
            self.endian_prefix = ">"
        elif self.file_format == "M":
            if sys.byteorder == "little":
                self.endian_prefix = "<"
            else:
                self.endian_prefix = ">"

    def parse(self, moments=None):
        """
        Parses a metranet binary file by calling the appropriate parser
        depending on the file type (M or P)

        Parameters
        ----------
        moments : list (optional)
            list of moments to get from the file, possible moments
            are ZH, ZV, ZDR, ST1, ST2, VEL, WID, PHI, RHO, CLT, ZVC, ZHC,
            MPH, default = ALL moments available in the file are read


        Returns
        -------
        out : RadarClass instance
            RadarClass instance containing the file data and metadata

        """
        self.read_pos = 0  # reset read position if needed

        out = None
        if self.file_format == "P":
            out = self._parse_p(moments)
        elif self.file_format == "M":
            out = self._parse_m(moments)

        return out

    def _parse_p(self, moments):
        """
        Parses a P format metranet binary file

        Parameters
        ----------
        moments : list
            list of moments to get from the file, possible moments
            are ZH, ZV, ZDR, ST1, ST2, VEL, WID, PHI, RHO, CLT, ZVC, ZHC,
            MPH, if none ALL moments available in the file are read


        Returns
        -------
        head : dict
            file metadata
        pol_header: list
            list of metadata dictionaries for every ray
        moments_data: dictionary
            dictionary containing the data in DN for every moment

        """

        # For P files we get available moment from pmfile_structure.py
        moments_avail = np.array(PMOMENTS["names"][self.file_type])
        nummoments = len(moments_avail)

        if moments is None:
            moments = moments_avail

        # Initialize outputs
        pol_header = []
        moments_data = {}
        for m in moments:
            moments_data[m] = []

        while self.read_pos != len(self.bytearray):
            # Get ray metadata
            ray = self._get_chunk(PRAY_HEADER)

            # Convert DN angles to float angles
            ray["startangle_az"] = _selex2deg(ray["startangle_az"])
            ray["startangle_el"] = _selex2deg(ray["startangle_el"])
            ray["endangle_az"] = _selex2deg(ray["endangle_az"])
            ray["endangle_el"] = _selex2deg(ray["endangle_el"])
            pol_header.append(ray)

            for i in range(nummoments):  # After ray metadata get all moments
                ngates = ray["numgates"]

                if moments_avail[i] in moments:  # Check if this moment is required
                    # Check type of this particular moment (byte or short)
                    if moments_avail[i] in PMOMENTS["types"].keys():
                        data_type = PMOMENTS["types"][moments_avail[i]]
                    else:
                        data_type = "B"

                    ffmt = self.endian_prefix + data_type

                    len_mom = ngates * BYTE_SIZES[data_type]
                    mom = np.ndarray(
                        (ngates,),
                        ffmt,
                        self.bytearray[self.read_pos : self.read_pos + len_mom],
                    )
                    self.read_pos += len_mom
                    moments_data[moments_avail[i]].append(mom)
                else:
                    # Even if moment is not required, we need to update cursor
                    self.read_pos += len_mom

        # Create global file header to be consistent with M files
        head = {}
        head["antmode"] = pol_header[0]["antmode"]
        head["radarname"] = pol_header[0]["scanid"]
        head["moments"] = moments
        # Rename radar with appropriate dict if needed, otherwise get name from
        # file
        if head["radarname"] in RENAME_RADARS_P.keys():
            head["radarname"] = RENAME_RADARS_P[head["radarname"]]
        else:
            head["radarname"] = self.bname[2]
        head["startrange"] = pol_header[0]["startrange"]
        head["currentsweep"] = pol_header[0]["currentsweep"]
        head["gatewidth"] = pol_header[0]["gatewidth"]

        radmetadata = _get_radar_site_info()
        radname = head["radarname"]
        head["frequency"] = radmetadata[radname]["Frequency"]
        head["radarheight"] = radmetadata[radname]["RadarHeight"]
        head["radarlat"] = radmetadata[radname]["RadarLat"]
        head["radarlon"] = radmetadata[radname]["RadarLon"]

        return head, pol_header, moments_data

    def _parse_m(self, moments):
        """
        Parses a M format metranet binary file

        Parameters
        ----------
        moments : list
            list of moments to get from the file, possible moments
            are ZH, ZV, ZDR, ST1, ST2, VEL, WID, PHI, RHO, CLT, ZVC, ZHC,
            MPH, if none ALL moments available in the file are read


        Returns
        -------
        head : dict
            file metadata
        pol_header: list
            list of metadata dictionaries for every ray
        moments_data: dictionary
            dictionary containing the data in DN for every moment

        """

        head = self._get_chunk(MSWEEP_HEADER)
        if head["radarname"] in RENAME_RADARS_M.keys():
            # make consistent names
            head["radarname"] = RENAME_RADARS_M[head["radarname"]]
        else:  # get radar name from filename
            head["radarname"] = self.bname[2]

        # Be consistent with C-library where it is in hZ
        head["frequency"] *= 10**9
        nummoments = head["nummoments"]

        pol_header = []

        # Get all available moments
        moments_avail = [a["name"] for a in head["moments"]]

        if moments is None:
            moments = moments_avail

        moments_data = {}
        for m in moments:
            moments_data[m] = []

        while self.read_pos != len(self.bytearray):
            pol = self._get_chunk(MRAY_HEADER)
            # Convert DN angles to float angles
            pol["startangle_az"] = _selex2deg(pol["startangle_az"])
            pol["startangle_el"] = _selex2deg(pol["startangle_el"])
            pol["endangle_az"] = _selex2deg(pol["endangle_az"])
            pol["endangle_el"] = _selex2deg(pol["endangle_el"])

            # Convert datetime to UTC + residue
            pol["datatime_residue"] = int(100 * ((pol["datatime"] * 0.01) % 1))
            pol["datatime"] = int(0.01 * pol["datatime"])
            pol_header.append(pol)

            for i in range(nummoments):
                mom_header = self._get_chunk(MMOMENT_HEADER)

                len_mom = mom_header["datasize"]

                if moments_avail[i] in moments:
                    size_moment_bytes = head["moments"][i]["num_bytes"]

                    ngates = int(len_mom / size_moment_bytes)  # nb of gates
                    ffmt = self.endian_prefix
                    if size_moment_bytes == 1:
                        ffmt += "B"
                    else:
                        ffmt += "H"

                    mom = np.ndarray(
                        (ngates,),
                        ffmt,
                        self.bytearray[self.read_pos : self.read_pos + len_mom],
                    )
                    moments_data[moments_avail[i]].append(mom)
                    self.read_pos += len_mom
                else:
                    self.read_pos += len_mom

        return head, pol_header, moments_data

    def _get_chunk(self, file_info):
        """
        Parses part of the bytearray given the info about what data to
        expect

        Parameters
        ----------
        file_info : dict
            dictionary with three keys: 'names', 'len', 'type'
            containing the names, length and types of the variables that
            should be read from the binary file. These dictionaries are
            defined in the pmfile_structure.py file


        Returns
        -------
        dic_values : dict
            dictionary containing all the variables that were read from
            the file, with keys corresponding to the 'names' key in the
            file_info dict

        """

        # Read the bytearray byte by byte
        dic_values = {}  # output dictionary

        for i in range(len(file_info["names"])):
            len_val = file_info["len"][i]
            name_val = file_info["names"][i]

            if isinstance(len_val, list):
                # If length in pmfile_structure.py is a list, generate
                # real length with previously read key
                len_val = len_val[0] * dic_values[len_val[1]]

            type_var = file_info["type"][i]
            ffmt = self.endian_prefix + f"{int(len_val)}"
            ffmt += type_var

            offset = len_val * BYTE_SIZES[type_var]

            if name_val == "moments":
                # M files only
                # For the moments structure some additional processing is
                # needed
                val = []
                for j in range(dic_values["nummoments"]):
                    # recursion!
                    val.append(self._get_chunk(MMOMENT_INFO_STRUCTURE))

            else:
                val = struct.unpack_from(
                    ffmt, self.bytearray[self.read_pos : self.read_pos + offset]
                )

                # Convert to array if list
                if len(val) == 1:
                    val = val[0]
                else:
                    val = np.array(val)

                if type_var == "s":  # For strings
                    # Strip null hexadecimal codes
                    val = val.decode("utf-8").replace("\x00", "")
                self.read_pos += offset

            dic_values[name_val] = val

        return dic_values


def read_polar(
    filename, moments=None, physic_value=True, masked_array=True, reorder_angles=True
):
    """
    Reads a METRANET polar data file

    Parameters
    ----------
    radar_file : str
        file name
    moments : list
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

    parser = PolarParser(filename)
    head, pol_header, moments_data = parser.parse(moments)

    # deep-copy needed since we alter the keys during loop
    moments = copy.deepcopy(list(moments_data.keys()))

    for m in moments:
        # Find index of moment in the moments structure of the header
        moments_data[m] = np.array(moments_data[m], dtype=moments_data[m][0].dtype)

        if masked_array:
            if parser.file_type == "L" and m in ["UZ", "UZ_V", "Z_V_CLUT", "Z_CLUT"]:
                mask = np.logical_or(moments_data[m] == 0, moments_data[m] == 1)
            else:
                mask = moments_data[m] == 0

        if parser.file_format == "M":
            idx_mom_head = [h["name"] == m for h in head["moments"]].index(True)
            if m in MOM_NAME_MAPPING.keys():
                moments_data[MOM_NAME_MAPPING[m]] = moments_data.pop(m)
                m = MOM_NAME_MAPPING[m]

        if physic_value:
            if parser.file_format == "M":
                moments_data[m] = float_mapping_m(
                    m,
                    head["moments"][idx_mom_head],
                    pol_header[0]["datatime"],
                    head["radarname"],
                    nyquist_vel(head["currentsweep"] - 1),
                )[moments_data[m]].astype(np.float32)
            else:
                moments_data[m] = float_mapping_p(
                    m,
                    pol_header[0]["datatime"],
                    head["radarname"],
                    nyquist_vel(head["currentsweep"] - 1),
                )[moments_data[m]].astype(np.float32)
        # Rename moment if needed

        if masked_array:
            moments_data[m] = np.ma.array(moments_data[m], mask=mask)
        else:
            moments_data[m][mask] = np.nan

    if reorder_angles:
        # Reorder dependent angle in ascending order
        if head["antmode"] in [0, 2]:
            angles = np.array([ray["startangle_az"] for ray in pol_header])
        else:
            angles = np.array([ray["startangle_el"] for ray in pol_header])

        nray = len(angles)

        maxrays = min([nray, 360])
        angles = angles[-maxrays:]
        order = np.argsort(angles)

        tmp = [pol_header[i + nray - maxrays] for i in order]
        pol_header = tmp

        for m in moments_data.keys():
            moments_data[m] = moments_data[m][-360:][order]

    ret_data = RadarData(header=head, pol_header=pol_header, data=moments_data)
    return ret_data


def read_product(radar_file, physic_value=False, masked_array=False, verbose=False):
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
    prd_header = {"row": 0, "column": 0}

    # read ASCII data
    if verbose:
        print("physic_value: ", physic_value)
        print(f"File {radar_file}: read ASCII")

    try:
        with open(radar_file, "rb") as data_file:
            for t_line in data_file:
                line = t_line.decode("utf-8").strip("\n")
                if line.find("end_header") == -1:
                    data = line.split("=")
                    prd_header[data[0]] = data[1]
                else:
                    break

            # read BINARY data
            prdt_size = int(prd_header["column"]) * int(prd_header["row"])
            if prdt_size < 1:
                print(
                    f"Error, no size found row={int(prd_header['row']):3} column={int(prd_header['column']):3}"
                )
                return None

            if verbose:
                print(
                    f"File {radar_file}: read BINARY data: expected {prdt_size} bytes, "
                )
                print(prd_header)

            if int(prd_header["table_size"]) != 0:
                prd_data_level = np.fromfile(
                    data_file,
                    dtype=np.dtype(">f4"),
                    count=int(int(prd_header["table_size"]) / 4),
                )
                current_pos = data_file.tell()
                if verbose:
                    print(
                        f"Current file position before reading compressed data: {current_pos}"
                    )
            else:
                prd_data_level = []

            data_type = prd_header.get("data_type", "BYTE").strip()
            if verbose:
                print("Data type:", prd_header.get("data_type"))
                print("header:", prd_header)

            compressed_bytes_len = int(prd_header["compressed_bytes"])
            uncompressed_bytes_len = int(prd_header["uncompressed_bytes"])
            compressed_bytes = data_file.read(compressed_bytes_len)
            if len(compressed_bytes) != compressed_bytes_len:
                raise OSError(
                    f"Incomplete read: expected {compressed_bytes_len}, got {len(compressed_bytes)}"
                )

            # prd_data_iter = decompress(compressed_bytes)
            prd_data_iter = decompress(
                np.frombuffer(compressed_bytes, dtype=np.uint8),
                compressed_bytes_len,
                uncompressed_bytes_len,
            )

            if data_type == "FLOAT":
                prd_data = np.frombuffer(prd_data_iter.tobytes(), dtype="<f4")
            elif data_type == "BYTE":
                prd_data = np.frombuffer(prd_data_iter.tobytes(), dtype=np.uint8)
                if prd_data_level is None or prd_data_level.size == 0:
                    return "[FAIL] BYTE data but no LUT found"

                if np.max(prd_data) >= prd_data_level.size:
                    return f"[FAIL] LUT size {prd_data_level.size} too small for max index {np.max(prd_data)}"

                # convert 0 at end of array with NAN
                conv_zero2nan = True

                nlevels = np.size(prd_data_level)
                if nlevels == 0:
                    prd_data_level = np.arange(256, dtype=np.uint32)
                else:
                    while conv_zero2nan:
                        nlevels -= 1
                        if nlevels == 0:
                            conv_zero2nan = False
                        elif prd_data_level[nlevels] == 0.0:
                            prd_data_level[nlevels] = np.nan
                        else:
                            conv_zero2nan = False

                prd_data = prd_data_level[prd_data]
            else:
                return f"[FAIL] Unknown data_type: {prd_header['data_type']}"

            # Reshape and match orientation
            prd_data = np.reshape(
                prd_data, (int(prd_header["row"]), int(prd_header["column"]))
            )

        data_file.close()

    except OSError as ee:
        warn(str(ee))
        print(f"Unable to read file '{radar_file}'")
        return None

    if verbose:
        print(f"Found {int(np.size(prd_data))} bytes")
        print(f"prd_data_level[10] = {prd_data_level[10]:f}")
        print("first 10 bytes", prd_data[0:10, 0])
        print("data level ", prd_data_level[0:10])

    if physic_value:
        ret_data.data = prd_data
        if masked_array:
            ret_data.data = np.ma.array(ret_data.data, mask=np.isnan(ret_data.data))
            ret_data.data = np.ma.masked_where(prd_data == 0, ret_data.data)
    else:
        ret_data.data = prd_data
        if masked_array:
            ret_data.data = np.ma.array(ret_data.data, mask=prd_data == 0)

    ret_data.header = prd_header
    ret_data.scale = prd_data_level

    return ret_data


def read_file(
    file,
    moment="ZH",
    physic_value=False,
    masked_array=False,
    reorder_angles=True,
    verbose=False,
):
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
    reorder_angles : boolean
        If true angles are reordered
    verbose : boolean
        If true prints out extra information

    Returns
    -------
    ret_data : RadarData object
        An object containing the information read from the file

    """
    bfile = os.path.basename(file)

    if (
        bfile.startswith("PM")
        or bfile.startswith("PH")
        or bfile.startswith("PL")
        or bfile.startswith("MS")
        or bfile.startswith("MH")
        or bfile.startswith("ML")
    ):
        # polar data from SITE (PH/PM/PL)
        if verbose:
            print("calling read_polar")
        ret = read_polar(
            file,
            moments=moment,
            physic_value=physic_value,
            masked_array=masked_array,
            reorder_angles=reorder_angles,
        )
    else:
        # cartesian / CCS4 products
        if verbose:
            print("calling read_product")
        ret = read_product(
            file, physic_value=physic_value, masked_array=masked_array, verbose=verbose
        )

    return ret


def _get_radar_site_info(verbose=False):
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
            "/store/msrad/python/library/radar/io/",
            "/opt/ccs4/python",
            "/proj/lom/python/library/radar/io",
        ]
        for p in path_yaml_file:
            full_file = p + "/" + metranet_yaml_file
            if os.path.isfile(full_file):
                print(f"look for Radar_Site_info from {full_file}")
                try:
                    with open(full_file) as f:
                        radar_def = yaml.load(f)
                        radar_def_load = True
                        if verbose:
                            print(f"Read Radar_Site_info from {full_file}")
                except OSError as ee:
                    warn(str(ee))
                    traceback.print_exc()

    if not radar_def_load:
        # HardCoded definition
        print("**** HARD-CODED Radar site Value, may be not updated!!")
        # some values valid for all sites
        c_speed = 299792458  # m/s

        radar_default = {}

        radar_default["dwhname"] = "undef"
        radar_default["RadarCHY"] = float("nan")
        radar_default["RadarCHX"] = float("nan")
        radar_default["FileId"] = 1179665477
        radar_default["Version"] = 0
        radar_default["RadarName"] = "undef"
        radar_default["ScanName"] = "undef"
        radar_default["RadarLat"] = float("nan")
        radar_default["RadarLon"] = float("nan")
        radar_default["RadarHeight"] = float("nan")
        radar_default["Frequency"] = float("nan")
        radar_default["WaveLength"] = float("nan")
        radar_default["PulseWidth"] = 0.5
        radar_default["SweepsOrder"] = (
            9,
            7,
            5,
            3,
            1,
            19,
            17,
            15,
            13,
            11,
            10,
            8,
            6,
            4,
            2,
            20,
            18,
            16,
            14,
            12,
        )
        radar_default["NumMomentsPM"] = metranet_reader.NPM_MOM
        radar_default["NumMomentsPH"] = metranet_reader.NPH_MOM
        radar_default["NumMomentsPL"] = metranet_reader.NPL_MOM

        radar_def = {}

        rname = "A"
        radar_def[rname] = radar_default.copy()
        radar_def[rname]["dwhname"] = "ALB"
        radar_def[rname]["RadarName"] = "Albis"
        radar_def[rname]["RadarCHY"] = 681201
        radar_def[rname]["RadarCHX"] = 237604
        radar_def[rname]["RadarLat"] = 47.284333
        radar_def[rname]["RadarLon"] = 8.512000
        radar_def[rname]["RadarHeight"] = 938.0
        radar_def[rname]["ScanName"] = "1095516672"
        radar_def[rname]["Frequency"] = 5450e6
        radar_def[rname]["WaveLength"] = c_speed / radar_def[rname]["Frequency"] * 1e2

        rname = "D"
        radar_def[rname] = radar_default.copy()
        radar_def[rname]["dwhname"] = "DOL"
        radar_def[rname]["RadarName"] = "Dole"
        radar_def[rname]["RadarCHY"] = 497057
        radar_def[rname]["RadarCHX"] = 142408
        radar_def[rname]["RadarLat"] = 46.425113
        radar_def[rname]["RadarLon"] = 6.099415
        radar_def[rname]["RadarHeight"] = 1682.0
        radar_def[rname]["ScanName"] = "1146047488"
        radar_def[rname]["Frequency"] = 5430e6
        radar_def[rname]["WaveLength"] = c_speed / radar_def[rname]["Frequency"] * 1e2

        rname = "L"
        radar_def[rname] = radar_default.copy()
        radar_def[rname]["dwhname"] = "MLE"
        radar_def[rname]["RadarName"] = "Lema"
        radar_def[rname]["RadarCHY"] = 707957
        radar_def[rname]["RadarCHX"] = 99762
        radar_def[rname]["RadarLat"] = 46.040761
        radar_def[rname]["RadarLon"] = 8.833217
        radar_def[rname]["RadarHeight"] = 1626.0
        radar_def[rname]["ScanName"] = "1279610112"
        radar_def[rname]["Frequency"] = 5455e6
        radar_def[rname]["WaveLength"] = c_speed / radar_def[rname]["Frequency"] * 1e2

        rname = "P"
        radar_def[rname] = radar_default.copy()
        radar_def[rname]["dwhname"] = "PPM"
        radar_def[rname]["RadarName"] = "PlaineMorte"
        radar_def[rname]["RadarCHY"] = 603687
        radar_def[rname]["RadarCHX"] = 135476
        radar_def[rname]["RadarLat"] = 46.370646
        radar_def[rname]["RadarLon"] = 7.486552
        radar_def[rname]["RadarHeight"] = 2937.0
        radar_def[rname]["ScanName"] = "0"
        radar_def[rname]["Frequency"] = 5468e6
        radar_def[rname]["WaveLength"] = c_speed / radar_def[rname]["Frequency"] * 1e2

        rname = "W"
        radar_def[rname] = radar_default.copy()
        radar_def[rname]["dwhname"] = "WEI"
        radar_def[rname]["RadarName"] = "WeissFluhGipfel"
        radar_def[rname]["RadarCHY"] = 779700
        radar_def[rname]["RadarCHX"] = 189790
        radar_def[rname]["RadarLat"] = 47.284333
        radar_def[rname]["RadarLon"] = 9.794458
        radar_def[rname]["RadarHeight"] = 2850.0
        radar_def[rname]["ScanName"] = "0"
        radar_def[rname]["Frequency"] = 5433e6
        radar_def[rname]["WaveLength"] = c_speed / radar_def[rname]["Frequency"] * 1e2

    return radar_def


def _selex2deg(angle):
    """
    Convert angles from SELEX format to degree

    Parameters
    ----------
    angle : float
        angle in DN


    Returns
    -------
    conv : float
        angle in degrees

    """
    conv = angle * 360.0 / 65535.0
    return conv
