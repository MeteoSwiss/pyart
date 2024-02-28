""" Unit Tests for Py-ART's aux_io/odim_h5.py module. """

import h5py
import numpy
from numpy.testing import assert_almost_equal
import numpy as np

import pyart

#################################################
# read_odim_h5 tests (verify radar attributes) #
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
# write_odim_h5 tests (verify data in written hdf5 matches original)   #
########################################################################

def compare_hdf5_files(ref, dset):
    from numpy import nan
    """Compare two HDF5 files for equality."""

    def compare_attrs(dict1, dict2):
        # Check if the values for each key are equal
        for key in dict1:
            val1 = dict1[key]
            val2 = dict2[key]
            # Special case for np.nan since np.nan != np.nan
            if isinstance(val1, float) and np.isnan(val1):
                if not (isinstance(val2, float) and np.isnan(val2)):
                    return False
            elif isinstance(val2, float) and np.isnan(val2):
                if not (isinstance(val1, float) and np.isnan(val1)):
                    return False
            elif isinstance(val1, float) and isinstance(val2, float):
                return np.allclose(val1, val2, 1e-3)
            elif type(val1) in [list, np.ndarray]:
                return np.allclose(val1, val2, 1e-3)
            elif val1 != val2:
                return False
        
        return True

    def compare_datasets(ds1, ds2):
        """Recursively compare datasets."""
        if isinstance(ds1, h5py.Group):
            if not isinstance(ds2, h5py.Group):
                return False
            if not compare_attrs(ds1.attrs, ds2.attrs):
                return False
            for name in ds1.keys():
                if name not in ds2:
                    return False
                if not compare_datasets(ds1[name], ds2[name]):
                    return False
            return True
        else:
            if not isinstance(ds2, h5py.Dataset):
                return False
            if not compare_attrs(ds1.attrs, ds2.attrs):
                return False
            d1 = ds1[:]
            d2 = ds2[:]
            return np.allclose(d1[np.isfinite(d1)], 
                d2[np.isfinite(d2)], 2)

    return compare_datasets(ref, dset)

def test_write_ppi():
    # CF/Radial example file -> Radar object -> netCDF file
    with pyart.testing.InTemporaryDirectory():
        tmpfile = "tmp_ppi.h5"
        radar = pyart.aux_io.read_odim_h5(pyart.testing.ODIM_H5_FILE)
        pyart.aux_io.write_odim_h5(tmpfile, radar)
        ref = h5py.File(pyart.testing.ODIM_H5_FILE)
        dset = h5py.File(tmpfile)
        assert compare_hdf5_files(ref, dset)
        dset.close()
