"""
=============================
Graphing (:mod:`pyart.graph`)
=============================

.. currentmodule:: pyart.graph

Creating plots of Radar and Grid fields.

Plotting radar data
===================

.. autosummary::
    :toctree: generated/

    RadarDisplay
    RadarMapDisplay
    AirborneRadarDisplay
    RadarMapDisplayBasemap

Plotting grid data
==================

.. autosummary::
    :toctree: generated/

    GridMapDisplay
    GridMapDisplayBasemap

"""

from .radardisplay import RadarDisplay  # noqa

# Import colormaps from cmweather
import cmweather  # noqa: F401

from .convstrat_scheme_plot import plot_convstrat_scheme  # noqa
from .radardisplay_airborne import AirborneRadarDisplay  # noqa
from .gridmapdisplay import GridMapDisplay  # noqa
from .gridmapdisplay_basemap import GridMapDisplayBasemap  # noqa
from .radarmapdisplay import RadarMapDisplay  # noqa
from .radarmapdisplay_basemap import RadarMapDisplayBasemap  # noqa
from .max_cappi import plot_maxcappi  # noqa
from . import custom_colormaps  # noqa

__all__ = [s for s in dir() if not s.startswith("_")]
