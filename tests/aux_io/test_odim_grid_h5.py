""" Unit Tests for Py-ART's aux_io/odim_h5.py module. """

import pytest
import h5py
import numpy as np
from numpy.ma.core import MaskedArray
from numpy.testing import assert_almost_equal

import pyart

#####################################################
# read_odim_grid_h5 tests (verify grid attributes) #
#####################################################

# read in the sample file and create a a grid object
grid = pyart.aux_io.read_odim_grid_h5(pyart.testing.ODIM_H5_GRID_FILE)


# time attribute
def test_time():
    assert "comment" in grid.time.keys()
    assert "long_name" in grid.time.keys()
    assert "standard_name" in grid.time.keys()
    assert "units" in grid.time.keys()
    assert "calendar" in grid.time.keys()
    assert "data" in grid.time.keys()
    assert grid.time["units"] == "seconds since 2024-02-27T09:45:00Z"
    assert_almost_equal(grid.time["data"][0], 0, 0)

# x attribute
def test_x():
    assert "long_name" in grid.x
    assert "standard_name" in grid.x
    assert "units" in grid.x
    assert "data" in grid.x
    assert grid.x["data"].shape == (710,)
    assert_almost_equal(grid.x["data"][0], 255500, 0)

# y attribute
def test_y():
    assert "long_name" in grid.y
    assert "standard_name" in grid.y
    assert "units" in grid.y
    assert "data" in grid.y
    assert grid.y["data"].shape == (640,)
    assert_almost_equal(grid.y["data"][0], -159500, 0)

# fields attribute is tested later

# metadata attribute
def test_metadata():
    assert "institution" in grid.metadata
    assert "title" in grid.metadata
    assert "source" in grid.metadata
    assert "comment" in grid.metadata
    assert "instrument_name" in grid.metadata
    assert "radar" in grid.metadata

# origin_latitude attribute
def test_latitude():
    assert "data" in grid.origin_latitude
    assert "standard_name" in grid.origin_latitude
    assert "units" in grid.origin_latitude
    assert grid.origin_latitude["data"].shape == (1,)
    assert_almost_equal(grid.origin_latitude["data"], 46.95240556, 0)


# point_longitude attribute
def test_point_longitude():
    assert "data" in grid.point_longitude
    assert "long_name" in grid.point_longitude
    assert "units" in grid.point_longitude
    assert grid.point_longitude["data"].shape == (1,640,710)
    assert_almost_equal(grid.point_longitude["data"][0,0,0],3.1753, 0)

# point_latitude attribute
def test_point_latitude():
    assert "data" in grid.point_latitude
    assert "long_name" in grid.point_latitude
    assert "units" in grid.point_latitude
    assert grid.point_latitude["data"].shape == (1,640,710)
    assert_almost_equal(grid.point_latitude["data"][0,0,0],43.63508830, 0)

####################
# fields attribute #
####################


@pytest.mark.parametrize(
    "field",
    [
        "radar_estimated_rain_rate",
    ],
)
def test_field_dics(field):
    description = "field : %s, dictionary" % field
    check_field_dic.description = description
    check_field_dic(field)


def check_field_dic(field):
    """Check that the required keys are present in a field dictionary."""
    assert "standard_name" in grid.fields[field]
    assert "units" in grid.fields[field]
    assert "coordinates" in grid.fields[field]


@pytest.mark.parametrize(
    "field",
    [
        "radar_estimated_rain_rate",
    ],
)
def test_field_shapes(field):
    description = "field : %s, shape" % field
    check_field_shape.description = description
    check_field_shape(field)


def check_field_shape(field):
    assert grid.fields[field]["data"].shape == (1, 640, 710)


fields = {
    "radar_estimated_rain_rate": MaskedArray,
}


@pytest.mark.parametrize("field, field_type", fields.items(), ids=list(fields.keys()))
def test_field_types(field, field_type):
    description = "field : %s, type" % field
    check_field_type.description = description
    check_field_type(field, field_type)


def check_field_type(field, field_type):
    assert type(grid.fields[field]["data"]) is field_type


fields = {
    "radar_estimated_rain_rate": 13.43,
}


@pytest.mark.parametrize("field, field_value", fields.items(), ids=list(fields.keys()))
def test_field_points(field, field_value):
    # these values can be found using:
    # [round(grid.fields[f]['data'][0,0]) for f in grid.fields]
    description = "field : %s, first point" % field
    check_field_point.description = description
    check_field_point(field, field_value)


def check_field_point(field, value):
    assert_almost_equal(grid.fields[field]["data"][0,49,501], value, 0)


########################################################################
# write_odim_grid_h5 tests (verify data in ODIM matches original)      #
########################################################################


def test_write_ppi():
    # CF/Radial example file -> Radar object -> netCDF file
    with pyart.testing.InTemporaryDirectory():
        tmpfile = "grid_odim.h5"
        pyart.aux_io.write_odim_grid_h5(tmpfile, grid, 
            odim_convention='ODIM_H5/V2_3', time_ref = 'end')
        ref = h5py.File(pyart.testing.ODIM_H5_GRID_FILE)
        dset = h5py.File(tmpfile)
        assert compare_hdf5_files(ref, dset)
        dset.close()

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
