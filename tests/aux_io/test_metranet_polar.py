""" Unit Tests for Py-ART's io/read_metranet.py module. (C reader) """

import numpy as np
import pytest
from numpy.ma.core import MaskedArray
from numpy.testing import assert_almost_equal

import pyart

#################################################
# metranet python tests (verify radar attributes)
#################################################


# read in the sample file and create a a Radar object
@pytest.fixture(params=["C", "python"])
def radar(request):
    """Return a grid using different readers."""
    reader = request.param
    return pyart.aux_io.read_metranet(pyart.testing.METRANET_FILE, reader=reader)


# time attribute
def test_time(radar):
    assert "comment" in radar.time.keys()
    assert "long_name" in radar.time.keys()
    assert "standard_name" in radar.time.keys()
    assert "units" in radar.time.keys()
    assert "calendar" in radar.time.keys()
    assert "data" in radar.time.keys()
    assert radar.time["units"] == "seconds since 2024-02-13T23:58:47Z"
    assert radar.time["data"].shape == (360,)
    assert_almost_equal(radar.time["data"][38], 3.799999952316284, 0)


# range attribute
def test_range(radar):
    assert "long_name" in radar.range
    assert "standard_name" in radar.range
    assert "units" in radar.range
    assert "data" in radar.range
    assert "spacing_is_constant" in radar.range
    assert radar.range["data"].shape == (54,)
    assert_almost_equal(radar.range["data"][0], 249.999, 0)


# fields attribute is tested later


# metadata attribute
def test_metadata(radar):
    assert "Conventions" in radar.metadata
    assert "comment" in radar.metadata
    assert "history" in radar.metadata
    assert "institution" in radar.metadata
    assert "instrument_name" in radar.metadata
    assert "references" in radar.metadata
    assert "source" in radar.metadata
    assert "title" in radar.metadata
    assert "version" in radar.metadata


# scan_type attribute
def test_scan_type(radar):
    assert radar.scan_type == "ppi"


# latitude attribute
def test_latitude(radar):
    assert "data" in radar.latitude
    assert "standard_name" in radar.latitude
    assert "units" in radar.latitude
    assert radar.latitude["data"].shape == (1,)
    assert_almost_equal(radar.latitude["data"], 46.3706, 0)


# longitude attribute
def test_longitude(radar):
    assert "data" in radar.longitude
    assert "standard_name" in radar.longitude
    assert "units" in radar.longitude
    assert radar.longitude["data"].shape == (1,)
    assert_almost_equal(radar.longitude["data"], 7.4865, 0)


# altitude attribute
def test_altitude(radar):
    assert "data" in radar.altitude
    assert "standard_name" in radar.altitude
    assert "units" in radar.altitude
    assert "positive" in radar.altitude
    assert radar.altitude["data"].shape == (1,)
    assert_almost_equal(radar.altitude["data"], 2937, 0)


# altitude_agl attribute
def test_altitude_agl(radar):
    assert radar.altitude_agl is None


# sweep_number attribute
def test_sweep_number(radar):
    assert "standard_name" in radar.sweep_number
    assert np.all(radar.sweep_number["data"] == 19)


# sweep_mode attribute
def test_sweep_mode(radar):
    assert "standard_name" in radar.sweep_mode
    assert radar.sweep_mode["data"].shape == (1,)
    assert radar.sweep_mode["data"].dtype.char == "U"
    str_array = radar.sweep_mode["data"]
    try:
        assert np.all(str_array == ["azimuth_surveillance"])
    except AssertionError:
        assert np.all(str_array == [b"azimuth_surveillance"])


# fixed_angle attribute
def test_fixed_angle(radar):
    assert "standard_name" in radar.fixed_angle
    assert "units" in radar.fixed_angle
    assert radar.fixed_angle["data"].shape == (1,)
    assert_almost_equal(radar.fixed_angle["data"][0], 39.985, 2)


# sweep_start_ray_index attribute
def test_sweep_start_ray_index(radar):
    assert "long_name" in radar.sweep_start_ray_index
    assert radar.sweep_start_ray_index["data"].shape == (1,)
    assert_almost_equal(radar.sweep_start_ray_index["data"][0], 0, 0)


# sweep_end_ray_index attribute
def test_sweep_end_ray_index(radar):
    assert "long_name" in radar.sweep_end_ray_index
    assert radar.sweep_end_ray_index["data"].shape == (1,)
    assert_almost_equal(radar.sweep_end_ray_index["data"][0], 359, 0)


# target_scan_rate attribute
def test_target_scan_rate(radar):
    assert radar.target_scan_rate is None


# azimuth attribute
def test_azimuth(radar):
    assert "standard_name" in radar.azimuth
    assert "units" in radar.azimuth
    assert_almost_equal(radar.azimuth["data"][0], 0.519, 0)
    assert_almost_equal(radar.azimuth["data"][10], 10.514, 0)


# elevation attribute
def test_elevation(radar):
    assert "standard_name" in radar.elevation
    assert "units" in radar.elevation
    assert radar.elevation["data"].shape == (360,)
    assert_almost_equal(radar.elevation["data"][0], 39.9853, 2)


# scan_rate attribute
def test_scan_rate(radar):
    assert radar.scan_rate is None


# antenna_transition attribute
def test_antenna_transition(radar):
    assert radar.antenna_transition is None


# instrument_parameters attribute
@pytest.mark.parametrize(
    "keys", ["pulse_width", "frequency", "nyquist_velocity", "number_of_pulses"]
)
def test_instument_parameters(radar, keys):
    # instrument_parameter sub-convention
    description = f"instrument_parameters: {keys}"
    check_instrument_parameter.description = description
    check_instrument_parameter(radar, keys)


def check_instrument_parameter(radar, param):
    assert param in radar.instrument_parameters
    radar.instrument_parameters[param]


# ngates attribute
def test_ngates(radar):
    assert radar.ngates == 54


# nrays attribute
def test_nrays(radar):
    assert radar.nrays == 360


# nsweeps attribute
def test_nsweeps(radar):
    assert radar.nsweeps == 1


####################
# fields attribute #
####################


@pytest.mark.parametrize(
    "field",
    [
        "reflectivity",
    ],
)
def test_field_dics(radar, field):
    description = f"field : {field}, dictionary"
    check_field_dic.description = description
    check_field_dic(radar, field)


def check_field_dic(radar, field):
    """Check that the required keys are present in a field dictionary."""
    assert "standard_name" in radar.fields[field]
    assert "units" in radar.fields[field]
    assert "_FillValue" in radar.fields[field]
    assert "coordinates" in radar.fields[field]


@pytest.mark.parametrize(
    "field",
    [
        "reflectivity",
    ],
)
def test_field_shapes(radar, field):
    description = f"field : {field}, shape"
    check_field_shape.description = description
    check_field_shape(radar, field)


def check_field_shape(radar, field):
    assert radar.fields[field]["data"].shape == (360, 54)


fields = {
    "reflectivity": MaskedArray,
}


@pytest.mark.parametrize("field, field_type", fields.items(), ids=list(fields.keys()))
def test_field_types(radar, field, field_type):
    description = f"field : {field}, type"
    check_field_type.description = description
    check_field_type(radar, field, field_type)


def check_field_type(radar, field, field_type):
    assert type(radar.fields[field]["data"]) is field_type


fields = {
    "reflectivity": -12.5,
}


@pytest.mark.parametrize("field, field_value", fields.items(), ids=list(fields.keys()))
def test_field_first_points(radar, field, field_value):
    # these values can be found using:
    # [round(radar.fields[f]['data'][0,0]) for f in radar.fields]
    description = f"field : {field}, first point"
    check_field_first_point.description = description
    check_field_first_point(radar, field, field_value)


def check_field_first_point(radar, field, value):
    assert_almost_equal(radar.fields[field]["data"][317, 39], value, 0)
