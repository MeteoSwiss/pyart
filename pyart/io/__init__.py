"""
==================================
Input and output (:mod:`pyart.io`)
==================================

.. currentmodule:: pyart.io

Functions to read and write radar and grid data to and from a number of file
formats.

Reading radar data
==================

In most cases the :py:func:`pyart.io.read` function should be used to read
in radar data from a file. In certain cases the function the read function
for the format in question should be used.

.. autosummary::
    :toctree: generated/

    read
    read_rsl
    read_mdv
    read_sigmet
    read_cfradial
    read_cfradial2
    read_chl
    read_nexrad_archive
    read_nexrad_cdm
    read_nexrad_level3
    read_uf

Writing radar data
==================

.. autosummary::
    :toctree: generated/

    write_cfradial
    write_uf

Reading grid data
=================

.. autosummary::
    :toctree: generated/

    read_grid
    read_grid_mdv

Writing grid data
=================

.. autosummary::
    :toctree: generated/

    write_grid
    write_grid_mdv
    write_grid_geotiff

Reading Sonde data
==================

.. autosummary::
    :toctree: generated/

    read_arm_sonde
    read_arm_sonde_vap

Special use
===========

.. autosummary::
    :toctree: generated/

    prepare_for_read
    make_time_unit_str

"""

from .rsl import read_rsl  # noqa
from .mdv_radar import read_mdv  # noqa
from .sigmet import read_sigmet  # noqa
from .chl import read_chl  # noqa
from .cfradial import read_cfradial, write_cfradial  # noqa
from .cfradial2 import read_cfradial2  # noqa
from .nexrad_archive import read_nexrad_archive  # noqa
from .nexrad_cdm import read_nexrad_cdm  # noqa
from .nexradl3_read import read_nexrad_level3  # noqa
from .uf import read_uf  # noqa
from .uf_write import write_uf  # noqa
from .grid_io import read_grid, write_grid  # noqa
from .output_to_geotiff import write_grid_geotiff  # noqa
from .auto_read import read  # noqa
from .mdv_grid import write_grid_mdv, read_grid_mdv  # noqa
from .common import prepare_for_read, make_time_unit_str  # noqa
from .arm_sonde import read_arm_sonde_vap, read_arm_sonde  # noqa

__all__ = [s for s in dir() if not s.startswith("_")]
