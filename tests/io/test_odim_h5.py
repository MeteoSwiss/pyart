""" Unit Tests for Py-ART's io/cfradial.py module. """

import h5py
import numpy
from numpy.testing import assert_almost_equal

import pyart

#################################################
# read_cfradial tests (verify radar attributes) #
#################################################

# read in the sample file and create a a Radar object
radar = pyart.aux_io.read_odim_h5(pyart.testing.ODIM_H5_FILE)


# time attribute
def test_time():
    assert "comment" in radar.time.keys()
    assert "long_name" in radar.time.keys()
    assert "standard_name" in radar.time.keys()
    assert "units" in radar.time.keys()
    assert "calendar" in radar.time.keys()
    assert "data" in radar.time.keys()
    assert radar.time["units"] == "seconds since 2023-08-07T16:10:08Z"
    assert radar.time["data"].shape == (720,)
    assert_almost_equal(radar.time["data"][38], 2, 0)

# range attribute
def test_range():
    assert "long_name" in radar.range
    assert "standard_name" in radar.range
    assert "meters_to_center_of_first_gate" in radar.range
    assert "meters_between_gates" in radar.range
    assert "units" in radar.range
    assert "data" in radar.range
    assert "spacing_is_constant" in radar.range
    assert radar.range["data"].shape == (500,)
    assert_almost_equal(radar.range["data"][0], 250, 0)


# metadata attribute
def test_metadata():
    assert "Conventions" in radar.metadata
    assert "source" in radar.metadata
    assert "version" in radar.metadata
    assert "odim_conventions" in radar.metadata

# scan_type attribute
def test_scan_type():
    assert radar.scan_type == "ppi"


########################################################################
# write_odim_h5 tests (verify data in written netCDF matches original) #
########################################################################

def assert_hdf5_files_equal(file1, file2):
    # Open the HDF5 files

    def check_group(group1, group2):
        # Check attributes of the group
        assert group1.attrs.keys() == group2.attrs.keys()
        for attr_name in group1.attrs:
            if type(group1.attrs[attr_name]) in [list, numpy.ndarray]:
                    assert_almost_equal(group1.attrs[attr_name], 
                        group1.attrs[attr_name], 3)
            else:
                assert (group1.attrs[attr_name] == 
                        group2.attrs[attr_name])

        # Check variables in the group
        assert set(group1.keys()) == set(group2.keys())
        for var_name in group1.keys():
            # Check attributes of the variable
            if var_name == 'data':
                assert_almost_equal(group1[var_name][:], group2[var_name][:])
            else:
                check_group(group1[var_name], group2[var_name])

    # Recursively check groups
    check_group(file1, file2)

def test_write_ppi():
    # CF/Radial example file -> Radar object -> netCDF file
    with pyart.testing.InTemporaryDirectory():
        tmpfile = "tmp_ppi.h5"
        radar = pyart.aux_io.read_odim_h5(pyart.testing.ODIM_H5_FILE)
        pyart.aux_io.write_odim_h5(tmpfile, radar)
        ref = h5py.File(pyart.testing.ODIM_H5_FILE)
        dset = h5py.File(tmpfile)
        assert_hdf5_files_equal(dset, ref)
        dset.close()

########################################################################
# other tests #
########################################################################

def test_title():
    title = pyart.graph.common.generate_title(radar, 'reflectivity', 0)
    assert title == ' 0.5 Deg. 2023-08-07T16:10:08Z \nHorizontal reflectivity'
