"""
==========================
Mapping (:mod:`pyart.map`)
==========================

.. current modules:: pyart.map

Py-ART has a robust function for mapping radar data from the collected radar
coordinates to Cartesian coordinates.

Mapping to grid
============

.. autosummary::
    :toctree: generated/

    grid_from_radars
    map_to_grid
    map_gates_to_grid

Examples of ROI (region of interest)
============

.. autosummary::
    :toctree: generated/

    example_roi_func_constant
    example_roi_func_dist
    example_roi_func_dist_beam

"""

from .gate_mapper import GateMapper  # noqa
from .grid_mapper import map_to_grid, grid_from_radars #noqa
from .grid_mapper import example_roi_func_constant #noqa
from .grid_mapper import example_roi_func_dist #noqa
from .grid_mapper import example_roi_func_dist_beam #noqa
from .polar_to_cartesian import polar_to_cartesian #noqa
from .polar_to_cartesian import get_earth_radius #noqa
from .gates_to_grid import map_gates_to_grid #noqa

__all__ = [s for s in dir() if not s.startswith('_')]
