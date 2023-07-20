""" Unit Tests for Py-ART's core/radar_spectra.py module. """

import inspect

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_almost_equal

import pyart


def test_gate_longitude_latitude():
    radar = pyart.testing.make_empty_spectra_radar(5, 4, 2)
    radar.azimuth['data'] = np.array([0, 90, 180, 270, 0])
    radar.elevation['data'] = np.array([0, 0, 0, 0, 10])
    radar.range['data'] = np.array([5, 15, 25, 35])
    assert radar.gate_longitude['data'].shape == (5, 4)
    assert radar.gate_latitude['data'].shape == (5, 4)
    assert_almost_equal(radar.gate_longitude['data'][0, 0], -97.5, 1)
    assert_almost_equal(radar.gate_latitude['data'][0, 0], 36.5, 1)
    # reset and try again with a non-default lat_0/lon_0
    radar.projection.pop("_include_lon_0_lat_0")
    radar.projection["lat_0"] = 20.0
    radar.projection["lon_0"] = 60.0
    radar.init_gate_longitude_latitude()
    assert_almost_equal(radar.gate_longitude['data'][0, 0], 60.0, 1)
    assert_almost_equal(radar.gate_latitude['data'][0, 0], 20.0, 1)


def test_gate_altitude():
    radar = pyart.testing.make_empty_spectra_radar(5, 4, 2)
    radar.azimuth['data'] = np.array([0, 90, 180, 270, 0])
    radar.elevation['data'] = np.array([0, 0, 0, 0, 10])
    radar.range['data'] = np.array([5, 15, 25, 35])

    assert radar.gate_altitude['data'].shape == (5, 4)
    assert_almost_equal(radar.gate_altitude['data'][0, 0], 200.0, 1)

    radar.altitude['data'][0] = 150.0
    radar.init_gate_altitude()
    assert_almost_equal(radar.gate_altitude['data'][0, 0], 150.0, 1)


def test_gate_x_y_z():
    radar = pyart.testing.make_empty_spectra_radar(5, 5, 2)
    radar.azimuth['data'] = np.array([0, 90, 180, 270, 0])
    radar.elevation['data'] = np.array([0, 0, 0, 0, 10])
    radar.range['data'] = np.array([5, 15, 25, 35, 45])
    radar.init_gate_x_y_z()
    assert radar.gate_x['data'].shape == (5, 5)
    assert_allclose(radar.gate_x['data'][0], [0, 0, 0, 0, 0], atol=1e-14)
    assert_allclose(radar.gate_x['data'][1], [5, 15, 25, 35, 45], atol=1e-14)
    assert_allclose(radar.gate_x['data'][2], [0, 0, 0, 0, 0], atol=1e-5)
    assert_allclose(radar.gate_x['data'][3], [-5, -15, -25, -35, -45], atol=1e-14)

    assert radar.gate_y['data'].shape == (5, 5)
    assert_allclose(radar.gate_y['data'][0], [5, 15, 25, 35, 45], atol=1e-14)
    assert_allclose(radar.gate_y['data'][1], [0, 0, 0, 0, 0], atol=1e-5)
    assert_allclose(radar.gate_y['data'][2], [-5, -15, -25, -35, -45], atol=1e-14)
    assert_allclose(radar.gate_y['data'][3], [0, 0, 0, 0, 0], atol=1e-6)

    assert radar.gate_z['data'].shape == (5, 5)
    z_sweep0 = np.array([1.47e-6, 1.324e-5, 3.679e-5, 7.210e-5, 1.1919e-4])
    assert_allclose(radar.gate_z['data'][0], z_sweep0, atol=1e-3)
    assert_allclose(radar.gate_z['data'][1], z_sweep0, atol=1e-3)
    assert_allclose(radar.gate_z['data'][2], z_sweep0, atol=1e-3)
    assert_allclose(radar.gate_z['data'][3], z_sweep0, atol=1e-3)

def test_rays_per_sweep_attribute():
    radar = pyart.testing.make_target_spectra_radar()
    rays_per_sweep = radar.rays_per_sweep
    assert rays_per_sweep['data'].shape == (1,)
    assert rays_per_sweep['data'][0] == 10


def test_radar_creation():
    radar = pyart.testing.make_target_spectra_radar()
    assert isinstance(radar, pyart.core.RadarSpectra)


def test_iterators():
    radar = pyart.testing.make_empty_spectra_radar(30, 20, 5)

    starts = [0]
    ends = [29]
    starts_ends = [(s, e) for s, e in zip(starts, ends)]

    assert inspect.isgenerator(radar.iter_start())
    assert [s for s in radar.iter_start()] == starts

    assert inspect.isgenerator(radar.iter_end())
    assert [s for s in radar.iter_end()] == ends

    assert inspect.isgenerator(radar.iter_start_end())
    assert [s for s in radar.iter_start_end()] == starts_ends

    assert inspect.isgenerator(radar.iter_slice())
    for s, start, end in zip(radar.iter_slice(), starts, ends):
        assert s.start == start
        assert s.stop == end + 1
        assert s.step is None

    assert inspect.isgenerator(radar.iter_field("spectra"))
    for d in radar.iter_field("spectra"):
        assert d.shape == (30, 20, 5)
        assert d.dtype == np.float64
    pytest.raises(KeyError, radar.iter_field, "foobar")

    assert inspect.isgenerator(radar.iter_azimuth())
    for d in radar.iter_azimuth():
        assert d.shape == (30,)

    assert inspect.isgenerator(radar.iter_elevation())
    for d in radar.iter_elevation():
        assert d.shape == (30,)
