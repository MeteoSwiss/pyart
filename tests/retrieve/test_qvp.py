""" Unit Tests for Py-ART's retrieve/qvp.py module. """

import numpy as np
from numpy.testing import assert_almost_equal

import pyart


def test_quasi_vertical_profile():
    test_radar = pyart.testing.make_target_radar()
    height = np.arange(0, 1000, 200)
    speed = np.ones_like(height) * 5
    direction = np.array([0, 90, 180, 270, 45])
    profile = pyart.core.HorizontalWindProfile(height, speed, direction)
    sim_vel = pyart.util.simulated_vel_from_profile(test_radar, profile)
    test_radar.add_field("velocity", sim_vel, replace_existing=True)

    qvp = pyart.retrieve.quasi_vertical_profile(test_radar)

    qvp_height = [
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
        1.0,
        2.0,
        2.0,
        2.0,
        2.0,
        3.0,
        3.0,
        3.0,
        3.0,
        3.0,
        4.0,
        4.0,
        4.0,
        4.0,
        5.0,
        5.0,
        5.0,
        5.0,
        6.0,
        6.0,
        6.0,
        7.0,
        7.0,
        7.0,
        7.0,
        8.0,
        8.0,
        8.0,
        8.0,
        9.0,
        9.0,
        9.0,
        10.0,
        10.0,
        10.0,
        10.0,
        11.0,
        11.0,
        11.0,
        11.0,
        12.0,
        12.0,
        12.0,
        13.0,
    ]

    qvp_range = [
        0.0,
        20.408,
        40.816,
        61.224,
        81.632,
        102.040,
        122.448,
        142.857,
        163.265,
        183.673,
        204.081,
        224.489,
        244.897,
        265.306,
        285.714,
        306.122,
        326.530,
        346.938,
        367.346,
        387.755,
        408.163,
        428.571,
        448.979,
        469.387,
        489.795,
        510.204,
        530.612,
        551.020,
        571.428,
        591.836,
        612.244,
        632.653,
        653.061,
        673.469,
        693.877,
        714.285,
        734.693,
        755.102,
        775.510,
        795.918,
        816.326,
        836.734,
        857.142,
        877.551,
        897.959,
        918.367,
        938.775,
        959.183,
        979.591,
        1000,
    ]

    qvp_reflectivity = [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        20.0,
        20.0,
        20.0,
        20.0,
        20.0,
        20.0,
        20.0,
        20.0,
        20.0,
        20.0,
        30.0,
        30.0,
        30.0,
        30.0,
        30.0,
        30.0,
        30.0,
        30.0,
        30.0,
        30.0,
        40.0,
        40.0,
        40.0,
        40.0,
        40.0,
        40.0,
        40.0,
        40.0,
        40.0,
        40.0,
    ]

    assert_almost_equal(qvp["height"], qvp_height, 3)
    assert_almost_equal(qvp["range"], qvp_range, 3)
    assert_almost_equal(qvp["reflectivity"], qvp_reflectivity, 3)