#! /usr/bin/env python
"""
Make a small MDV file containing a single RHI scan.

Single field and scan is taken from full sized file 110041.mdv
"""

import gzip
import struct

import StringIO

import pyart

# MDV file parameters
MASTER_HEADER_SIZE = 1024  # size of master header in bytes
FIELD_HEADER_SIZE = 416  # size of field header in bytes
VLEVEL_HEADER_SIZE = 1024  # size of vlevel header in bytes
CHUNK_HEADER_SIZE = 512  # size of chunk header in bytes
COMPRESSION_INFO_SIZE = 24  # size of compression infomation header in bytes
SWEEP_INFO_SIZE = 8  # size of sweep info header in bytes

# parameters
FIELD_NUMBER = 3  # field to extract in new file (3 is reflectivity)
NGATES = 125
NFIELDS = 1
INFILE = "110041.mdv"
OUTFILE = "example_mdv_rhi.mdv"

# read and compress the reflectivity field data
mdvfile = pyart.io.mdv.MdvFile(INFILE)
number_of_fields = int(mdvfile.master_header["nfields"])
bias = mdvfile.field_headers[FIELD_NUMBER]["bias"]
scale = mdvfile.field_headers[FIELD_NUMBER]["scale"]
in_field_offset = mdvfile.field_headers[FIELD_NUMBER]["field_data_offset"]

# bais, scale and slice to extract the first NGATES
fdata = ((mdvfile.read_a_field(FIELD_NUMBER) - bias) / scale)[0, :, :NGATES]
mdvfile.close()

fdata_str = fdata.astype("uint16").byteswap().tostring()
uncompressed_data_size = len(fdata_str)
fileobj = StringIO.StringIO()
gzipfile = gzip.GzipFile(fileobj=fileobj, mode="w")
gzipfile.write(fdata_str)
gzipfile.close()
compressed_field_data = fileobj.getvalue()
compressed_data_size = len(compressed_field_data)

# prepare input and output files for reading/writing
f = open(INFILE, "rb")
out = open(OUTFILE, "wb")

# read the master header, update, and write
fmt = ">28i 8i i 5i 6f 3f 12f 512c 128c 128c i"
ll = list(struct.unpack(fmt, f.read(struct.calcsize(fmt))))
ll[19] = NFIELDS  # nfields
ll[20] = NGATES  # ngates
ll[24] = MASTER_HEADER_SIZE  # field_hdr_offset
ll[25] = MASTER_HEADER_SIZE + FIELD_HEADER_SIZE  # vlevel_hdr_offset
# chunk_hdr _offset
ll[26] = MASTER_HEADER_SIZE + FIELD_HEADER_SIZE + VLEVEL_HEADER_SIZE
out.write(struct.pack(fmt, *ll))

# read the reflectivity field header, update and write
f.seek(MASTER_HEADER_SIZE + FIELD_NUMBER * FIELD_HEADER_SIZE)
fmt = ">17i 10i 9i 4i f f 8f 12f 4f 5f 64c 16c 16c 16c 16c i"
ll = list(struct.unpack(fmt, f.read(struct.calcsize(fmt))))
ll[9] = NGATES  # ngates
field_data_offset = (
    MASTER_HEADER_SIZE
    + FIELD_HEADER_SIZE
    + VLEVEL_HEADER_SIZE
    + CHUNK_HEADER_SIZE * FIELD_NUMBER
)
ll[15] = field_data_offset  # field_data_offset
# volume_size
ll[16] = compressed_data_size + COMPRESSION_INFO_SIZE + SWEEP_INFO_SIZE
out.write(struct.pack(fmt, *ll))

# read the reflectivity vlevel header, update and write
f.seek(
    MASTER_HEADER_SIZE
    + FIELD_HEADER_SIZE * number_of_fields
    + FIELD_NUMBER * VLEVEL_HEADER_SIZE
)
fmt = ">i i 122i 4i 122f 5f i"
ll = list(struct.unpack(fmt, f.read(struct.calcsize(fmt))))
out.write(struct.pack(fmt, *ll))

# read the three chunk headers, update and write
f.seek(
    MASTER_HEADER_SIZE
    + FIELD_HEADER_SIZE * number_of_fields
    + VLEVEL_HEADER_SIZE * number_of_fields
)
fmt = ">5i 2i 480c i"

chunk_data_offset = (
    compressed_data_size + field_data_offset + SWEEP_INFO_SIZE + COMPRESSION_INFO_SIZE
)

# radar_info chunk header
ll = list(struct.unpack(fmt, f.read(struct.calcsize(fmt))))
info_chunk_offset = int(ll[3])
ll[3] = chunk_data_offset
out.write(struct.pack(fmt, *ll))

# calib chunk header
ll = list(struct.unpack(fmt, f.read(struct.calcsize(fmt))))
calib_chunk_offset = int(ll[3])
ll[3] = chunk_data_offset + 240
out.write(struct.pack(fmt, *ll))

# elevs chunk header
ll = list(struct.unpack(fmt, f.read(struct.calcsize(fmt))))
elevs_chunk_offset = int(ll[3])
ll[3] = chunk_data_offset + 240 + 300
out.write(struct.pack(fmt, *ll))

# read the sweep_info and write
f.seek(in_field_offset)
fmt = "I I"
ll = list(struct.unpack(fmt, f.read(struct.calcsize(fmt))))
out.write(struct.pack(fmt, *ll))

# read the compression header, update and write
fmt = ">I I I I 2I"
ll = list(struct.unpack(fmt, f.read(struct.calcsize(fmt))))
ll[1] = uncompressed_data_size
ll[2] = compressed_data_size + COMPRESSION_INFO_SIZE  # nbytes_compressed
ll[3] = compressed_data_size  # nytes_coded
out.write(struct.pack(fmt, *ll))

# write the compressed field data
out.write(compressed_field_data)

# read/write radar_info chunk
f.seek(info_chunk_offset)
fmt = ">12i 2i 22f 4f 40c 40c"
ll = list(struct.unpack(fmt, f.read(struct.calcsize(fmt))))
ll[2] = NFIELDS  # nfield
ll[3] = NGATES  # ngates
out.write(struct.pack(fmt, *ll))

# read/write calib chunk
f.seek(calib_chunk_offset)
fmt = ">16c 6i 51f 14f"
ll = list(struct.unpack(fmt, f.read(struct.calcsize(fmt))))
out.write(struct.pack(fmt, *ll))

# read/write elevs chunk
f.seek(elevs_chunk_offset)
fmt = "%df" % (2)
ll = list(struct.unpack(fmt, f.read(struct.calcsize(fmt))))
out.write(struct.pack(fmt, *ll))

# close the output file
out.close()
