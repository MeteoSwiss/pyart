""" Unit tests for bias and noise module. """

from numpy.testing import assert_allclose

import pyart

radar = pyart.io.read(pyart.testing.NEXRAD_ARCHIVE_MSG31_FILE)


def test_correct_bias():
    corr_field_expected = [-32.0, -32.0, -32.0]
    corr_test = pyart.correct.correct_bias(radar, field_name=None)

    assert_allclose(corr_test["data"][0][0:3], corr_field_expected, atol=1e-14)
    assert_allclose(corr_test["data"][-1][0:3], corr_field_expected, atol=1e-14)
