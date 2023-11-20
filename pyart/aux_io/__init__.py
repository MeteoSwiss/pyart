"""
================================================
Auxiliary input and output (:mod:`pyart.aux_io`)
================================================

.. currentmodule:: pyart.aux_io

Additional classes and functions for reading and writing data from a number
of file formats.

These auxiliary input/output routines are not as well polished as those in
:mod:`pyart.io`.  They may require addition dependencies beyond those required
for a standard Py-ART install, use non-standard function parameter and naming,
are not supported by the :py:func:`pyart.io.read` function and are not fully
tested if tested at all. Please use these at your own risk.

Bugs in these function should be reported but fixing them may not be a
priority.

Reading radar data
==================

.. autosummary::
    :toctree: generated/

    read_d3r_gcpex_nc
    read_gamic
    read_kazr
    read_noxp_iphex_nc
    read_odim_h5
    read_odim_grid_h5
    read_odim_vp_h5
    read_pattern
    read_radx
    read_rainbow_wrl
    read_metranet
    read_cartesian_metranet
    read_gif
    read_bin
    read_iq
    read_bin_mf
    read_png
    read_dat_mf
    read_grib
    read_rainbow_psr
    read_rainbow_psr_spectra
    read_spectra
    read_cf1
    read_cf1_cartesian
    read_cf1_cartesian_mf
    read_hpl
    read_swissbirdradar_spectra
    read_skyecho
    extract_sweeps_skyecho
    get_sweep_time_coverage

Writing radar data
==================

.. autosummary::
    :toctree: generated/

    write_odim_h5
    write_odim_grid_h5
    write_spectra

"""

from .pattern import read_pattern #noqa
from .radx import read_radx #noqa
from .d3r_gcpex_nc import read_d3r_gcpex_nc #noqa
from .noxp_iphex_nc import read_noxp_iphex_nc #noqa
from .arm_vpt import read_kazr #noqa
from .edge_netcdf import read_edge_netcdf #noqa
from .odim_h5 import read_odim_h5, read_odim_grid_h5, read_odim_vp_h5 #noqa
from .odim_h5_writer import write_odim_h5, write_odim_grid_h5 #noqa
from .gamic_hdf5 import read_gamic #noqa
from .sinarame_h5 import read_sinarame_h5 #noqa
from .sinarame_h5 import write_sinarame_cfradial #noqa
from .rainbow_wrl import read_rainbow_wrl #noqa
from .rainbow_psr import read_rainbow_psr, read_rainbow_psr_spectra #noqa
from .rainbow_psr import read_psr_header, read_psr_cpi_header, read_psr_spectra #noqa
from .rainbow_psr import convert_data #noqa
from .spectra import read_spectra, write_spectra #noqa
from .cf1 import read_cf1 #noqa
from .cf1_cartesian import read_cf1_cartesian, read_cf1_cartesian_mf #noqa

from .metranet_reader import read_metranet #noqa
from .metranet_cartesian_reader import read_cartesian_metranet #noqa
from .rad4alp_iq_reader import read_iq #noqa

from .metranet_c import get_library #noqa
from .metranet_c import read_product as read_product_c #noqa
from .metranet_c import read_file as read_file_c #noqa
from .metranet_python import read_product as read_product_py #noqa
from .metranet_python import read_file as read_file_py #noqa

from .rad4alp_gif_reader import read_gif #noqa
from .rad4alp_bin_reader import read_bin #noqa
from .mf_bin_reader import read_bin_mf #noqa
from .mf_png_reader import read_png #noqa
from .mf_grib_reader import read_grib #noqa
from .mf_dat_reader import read_dat_mf #noqa
from .hpl_reader import read_hpl #noqa

from .swissbirdradar import read_swissbirdradar_spectra #noqa

__all__ = [s for s in dir() if not s.startswith('_')]
