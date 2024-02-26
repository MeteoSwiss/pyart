""" Unit Tests for Py-ART's io/read_metranet.py module. (C reader) """


import pytest
from numpy.ma.core import MaskedArray
from numpy.testing import assert_almost_equal

import pyart

#################################################
# read_cfradial tests (verify radar attributes) #
#################################################

# read in the sample file and create a a Radar object
radar = pyart.aux_io.read_cartesian_metranet(pyart.testing.METRANET_GRID_FILE)


# time attribute
def test_time():
    assert "long_name" in radar.time.keys()
    assert "standard_name" in radar.time.keys()
    assert "units" in radar.time.keys()
    assert "calendar" in radar.time.keys()
    assert "data" in radar.time.keys()
    assert radar.time["units"] == "seconds since 2024-02-22T07:40:00Z"
    assert radar.time["data"].shape == (1,)
    assert_almost_equal(radar.time["data"][0], 0, 0)


# x attribute
def test_x():
    assert "long_name" in radar.x
    assert "standard_name" in radar.x
    assert "units" in radar.x
    assert "data" in radar.x
    assert radar.x["data"].shape == (710,)
    assert_almost_equal(radar.x["data"][0], -344500.0, 0)

# y attribute
def test_y():
    assert "long_name" in radar.y
    assert "standard_name" in radar.y
    assert "units" in radar.y
    assert "data" in radar.y
    assert radar.y["data"].shape == (640,)
    assert_almost_equal(radar.y["data"][0], -359500.0, 0)

# fields attribute is tested later


# metadata attribute
def test_metadata():
    assert "product" in radar.metadata
    assert "quality" in radar.metadata
    assert "table_size" in radar.metadata
    assert "total_sweep" in radar.metadata
    assert "volume_time" in radar.metadata
    assert "table_name" in radar.metadata
    assert "moment" in radar.metadata


# origin_latitude attribute
def test_latitude():
    assert "data" in radar.origin_latitude
    assert "standard_name" in radar.origin_latitude
    assert "units" in radar.origin_latitude
    assert radar.origin_latitude["data"].shape == (1,)
    assert_almost_equal(radar.origin_latitude["data"], 46.95240556, 0)


# point_longitude attribute
def test_point_longitude():
    assert "data" in radar.point_longitude
    assert "long_name" in radar.point_longitude
    assert "units" in radar.point_longitude
    assert radar.point_longitude["data"].shape == (1,640,710)
    assert_almost_equal(radar.point_longitude["data"][0,0,0],3.1753, 0)

# point_latitude attribute
def test_point_latitude():
    assert "data" in radar.point_latitude
    assert "long_name" in radar.point_latitude
    assert "units" in radar.point_latitude
    assert radar.point_latitude["data"].shape == (1,640,710)
    assert_almost_equal(radar.point_latitude["data"][0,0,0],43.63508830, 0)

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
    assert "standard_name" in radar.fields[field]
    assert "units" in radar.fields[field]
    assert "coordinates" in radar.fields[field]


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
    assert radar.fields[field]["data"].shape == (1, 640, 710)


fields = {
    "radar_estimated_rain_rate": MaskedArray,
}


@pytest.mark.parametrize("field, field_type", fields.items(), ids=list(fields.keys()))
def test_field_types(field, field_type):
    description = "field : %s, type" % field
    check_field_type.description = description
    check_field_type(field, field_type)


def check_field_type(field, field_type):
    assert type(radar.fields[field]["data"]) is field_type


fields = {
    "radar_estimated_rain_rate": 16.8,
}


@pytest.mark.parametrize("field, field_value", fields.items(), ids=list(fields.keys()))
def test_field_first_points(field, field_value):
    # these values can be found using:
    # [round(radar.fields[f]['data'][0,0]) for f in radar.fields]
    description = "field : %s, first point" % field
    check_field_first_point.description = description
    check_field_first_point(field, field_value)


def check_field_first_point(field, value):
    assert_almost_equal(radar.fields[field]["data"][0,472,437], value, 0)
