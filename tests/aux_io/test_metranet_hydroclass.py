""" Unit Tests for Py-ART's io/read_metranet.py module. (C and Python readers) """

import numpy as np
import pytest
from numpy.ma.core import MaskedArray

import pyart

#################################################
# metranet hydroclass tests (verify attributes)
#################################################


is_linux = platform.system() == "Linux"


@pytest.fixture(
    params=[
        pytest.param(
            "C",
            marks=pytest.mark.skipif(
                not is_linux,
                reason="C reader only available on Linux",
            ),
        ),
        "python",
    ]
)
def hydroclass(request):
    """Return a RadarData object using different readers."""
    reader = request.param
    if reader == "C":
        ret = pyart.aux_io.metranet_c.read_product(
            pyart.testing.METRANET_HYDRO_FILE, physic_value=True, masked_array=True
        )
    elif reader == "python":
        ret = pyart.aux_io.metranet_python.read_product(
            pyart.testing.METRANET_HYDRO_FILE, physic_value=True, masked_array=True
        )
    else:
        raise ValueError(f"Unknown reader: {reader}")
    return ret


# Header tests
def test_header(hydroclass):
    """Check the main header keys and values"""
    header = hydroclass.header
    # Basic product info
    assert header["product"] == "HYDROCLASS_EL1"
    assert header["time"] == "1816221107"
    assert header["pid"] == "YMD"
    assert header["format"] == "POLAR"
    assert header["moment"] == "ZDRP"
    assert header["radar"] == "Dole"

    # Numeric fields
    for key in ["row", "column", "compressed_bytes", "uncompressed_bytes"]:
        assert key in header
        value = int(header[key])
        assert value > 0

    # Check table_size is numeric
    assert "table_size" in header
    assert int(header["table_size"]) >= 0

    # Data type
    assert header["data_type"] in ["BYTE", "FLOAT"]


# Data tests
def test_data_shape(hydroclass):
    """Check that the data shape matches row x column"""
    nrow = int(hydroclass.header["row"])
    ncol = int(hydroclass.header["column"])
    assert hydroclass.data.shape == (nrow, ncol)


def test_data_type(hydroclass):
    """Check that data is a masked array"""
    assert isinstance(hydroclass.data, MaskedArray)


def test_data_min_max(hydroclass):
    """Check that min and max of data are reasonable"""
    data = hydroclass.data
    # Make sure all non-masked values are finite
    assert np.all(np.isfinite(data[~data.mask]))
    # Check basic min/max
    min_val = np.min(data)
    max_val = np.max(data)
    assert min_val <= max_val
    # For BYTE products, values should match LUT range if available
    if hydroclass.header["data_type"] == "BYTE":
        nlevels = len(hydroclass.scale)
        assert min_val >= 0 or np.isnan(min_val)
        assert max_val < nlevels or np.isnan(max_val)


# Scale / LUT tests
def test_scale(hydroclass):
    """Check that scale / LUT is present and valid"""
    scale = hydroclass.scale
    assert isinstance(scale, np.ndarray)
    if hydroclass.header["data_type"] == "BYTE":
        expected_size = 2 ** int(hydroclass.header["data_bits"])
        assert len(scale) == expected_size
        # Check first few LUT values are finite
        assert np.all(np.isfinite(scale[:5]))
