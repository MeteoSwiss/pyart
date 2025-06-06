#! /usr/bin/env python
"""
Make a small sigmet (IRIS) file containing a single RHI scan.

Reflectivity data is set to 0 dBZ, other parameters are taken from the
full sized file XSW110520113537.RAW7HHL
"""

import struct

import numpy as np

# parameters
INFILE = "XSW110520113537.RAW7HHL"
OUTFILE = "example_sigmet_rhi.sigmet"

# open the input and output files
f = open(INFILE, "rb")
out = open(OUTFILE, "wb")

#############################
# First Record (6144 bytes) #
# product_hdr (640 bytes)   #
#############################

# structure_header (12 bytes)
out.write(f.read(12))  # no change

# product_configuration (320 bytes)
# -> structure_header
# -> generation_time
# -> sweep_ingest_time
# -> file_ingest_time
# -> color_scale_def
out.write(f.read(320))  # no change

# product_end (308 bytes)
# -> ingest_time
fmt = (
    "16s8s8s12s28sh16s16shIIhhiiHHh12sHiiiiiHhHhhhHhhhhHH18sIIIIIIHHIIh"
    + "32shhhBB2s8sI4s"
)
ll = list(struct.unpack(fmt, f.read(308)))
ll[23] = 154000  # last_bin_range
ll[24] = 25  # number_bins
out.write(struct.pack(fmt, *ll))

# padding to next record
out.write(f.read(6144 - 640))  # padding to next record

##############################
# Second Record (6144 bytes) #
# ingest_header (4884 bytes) #
##############################
# structure_header
out.write(f.read(12))  # no change

# ingest_configuration (480 bytes)
# -> volume_scan_start_time
# ->
fmt = "80shhi12s12shhhh4s8s16sh16shIIhhHHHhiiiiiiiIh2s8sI16s228s"
ll = list(struct.unpack(fmt, f.read(480)))
ll[2] = 1  # number_sweeps_completed
ll[22] = 20  # number_rays_sweep
out.write(struct.pack(fmt, *ll))

# task_configuration (2612 bytes)
# -> structure_header (12 byes)
out.write(f.read(12))  # no change

# -> task_sched_info (120 bytes)
out.write(f.read(120))  # no change

# -------------------------------------------------
# -> task_dsp_info (320 bytes)
# ---> major_mode, dsp_type (4 bytes)
out.write(f.read(4))

# ---> current_data_type_mask (24 bytes)
fmt = "IIIIII"
ll = list(struct.unpack(fmt, f.read(24)))
ll[0] = 512  # mask_word_0 indicating field 9 (DBZ2), 2 ** 9
ll[2] = 0  # mask_word_1 indicating no fields
out.write(struct.pack(fmt, *ll))

# ---> original_data_type_mask, task_dsp_mode, etc (292 bytes)
out.write(f.read(292))
# ---------------------------------------------------

# -> task_calib_info (320 bytes)
out.write(f.read(320))  # no change

# -> task_range_info (160 bytes) - file at 7408 seek
fmt = "iihhiiHh136s"
ll = list(struct.unpack(fmt, f.read(160)))
ll[1] = 154000  # last_bin_range
ll[2] = 25  # number_input_bins
ll[3] = 25  # number_output_bins
out.write(struct.pack(fmt, *ll))

# -> task_scan_info (320 bytes)
fmt = "Hh2sh200s112s"
ll = list(struct.unpack(fmt, f.read(320)))
ll[3] = 1  # number_sweeps
out.write(struct.pack(fmt, *ll))

# -> task_misc_info (320 bytes)
out.write(f.read(320))  # no change

# -> task_end_info (320 bytes)
out.write(f.read(320))  # no change

# -> comments (720 bytes)
out.write(f.read(720))  # no change


# spare_0 (732 bytes)
out.write(f.read(732))  # no change

# gparm (128 bytes)
out.write(f.read(128))  # no change

# reserved (920 bytes)
out.write(f.read(920))  # no change

# padding to next record
out.write(f.read(6144 - 4884))

###################
# Data            #
###################
# f is at 12288
# raw_prod_bhdr (12 bytes)
out.write(f.read(12))

# skip over ingest_data_header for field 8
f.read(76)

# ingest_data_header for field 9
fmt = "12s12shhhhhHhH36s"
ll = list(struct.unpack(fmt, f.read(76)))
ll[3] = 20  # number_rays_sweep
ll[5] = 20  # number_rays_file_expected
ll[6] = 20  # number_rays_file_actual
out.write(struct.pack(fmt, *ll))

# compressed ray data
code = np.ones((33), dtype="int16")
code[0] = -32737  # indicates 31 values follow (-32737 + 32768 = 31)
code[1] = 6790  # az0
code[3] = 6790  # az1
code[5] = 25  # nbins
code[6:-1] = -32768  # 0 dBZ
code[-1] = 1  # end of ray marker

for i in range(20):
    code[6] = i  # time
    code.view("uint16")[2] = 91 * 18 * i  # el0 0 -> 180 in steps of 18
    code.view("uint16")[4] = 91 * 18 * i  # el1
    out.write(code.tostring())

out.write("\x00" * 4736)

# close the files
f.close()
out.close()
