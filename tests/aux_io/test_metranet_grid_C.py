""" Unit Tests for Py-ART's io/read_metranet.py module. (C reader) """


import platform

import pytest
from numpy.ma.core import MaskedArray
from numpy.testing import assert_almost_equal

import pyart

#################################################
# Cartesian metranet C tests (verify grid attributes)
#################################################

# Skip the entire test module if the operating system is not Linux
if platform.system() != 'Linux':
    pytest.skip(allow_module_level=True)

# read in the sample file and create a a grid object
grid = pyart.aux_io.read_cartesian_metranet(pyart.testing.METRANET_GRID_FILE)

# time attribute
def test_time():
    assert "long_name" in grid.time.keys()
    assert "standard_name" in grid.time.keys()
    assert "units" in grid.time.keys()
    assert "calendar" in grid.time.keys()
    assert "data" in grid.time.keys()
    assert grid.time["units"] == "seconds since 2024-02-22T07:40:00Z"
    assert grid.time["data"].shape == (1,)
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
    assert "product" in grid.metadata
    assert "quality" in grid.metadata
    assert "table_size" in grid.metadata
    assert "total_sweep" in grid.metadata
    assert "volume_time" in grid.metadata
    assert "table_name" in grid.metadata
    assert "moment" in grid.metadata


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
    description = f"field : {field}, dictionary"
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
    description = f"field : {field}, shape"
    check_field_shape.description = description
    check_field_shape(field)


def check_field_shape(field):
    assert grid.fields[field]["data"].shape == (1, 640, 710)


fields = {
    "radar_estimated_rain_rate": MaskedArray,
}


@pytest.mark.parametrize("field, field_type", fields.items(), ids=list(fields.keys()))
def test_field_types(field, field_type):
    description = f"field : {field}, type"
    check_field_type.description = description
    check_field_type(field, field_type)


def check_field_type(field, field_type):
    assert type(grid.fields[field]["data"]) is field_type


fields = {
    "radar_estimated_rain_rate": 16.8,
}


@pytest.mark.parametrize("field, field_value", fields.items(), ids=list(fields.keys()))
def test_field_points(field, field_value):
    # these values can be found using:
    # [round(grid.fields[f]['data'][0,0]) for f in grid.fields]
    description = f"field : {field}, first point"
    check_field_point.description = description
    check_field_point(field, field_value)


def check_field_point(field, value):
    assert_almost_equal(grid.fields[field]["data"][0,472,437], value, 0)
