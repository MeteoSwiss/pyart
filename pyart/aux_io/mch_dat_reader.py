"""
pyart.aux_io.mch_dat_reader
==========================

Routines for reading MeteoSwiss operational radar data contained in text
files.

.. autosummary::
    :toctree: generated/

    read_mch_vad

"""

import xml.etree.ElementTree as ET

import pyart


def read_mch_vad(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    header_element = root.find('HEADER')
    data_element = root.find('DATA')

    header_data = {}
    for child in header_element:
        header_data[child.tag] = child.text

    data_slices = []
    for slice_element in data_element.findall('slice'):
        slice_data = {}
        for child in slice_element:
            slice_data[child.tag] = child.text
        data_slices.append(slice_data)

    speed = [float(slic['speed']) for slic in data_slices]
    direction = [float(slic['direction']) for slic in data_slices]
    height = [float(slic['height']) for slic in data_slices]

    return pyart.core.HorizontalWindProfile(height, speed, direction)
