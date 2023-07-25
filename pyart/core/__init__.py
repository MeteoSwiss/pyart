"""
========================
Core (:mod:`pyart.core`)
========================

.. currentmodule:: pyart.core

Core Py-ART classes and function for interacting with weather radar data.

Core classes
============

.. autosummary::
    :toctree: generated/

    Radar
    Grid
    HorizontalWindProfile
    RadarSpectra

Coordinate transformations
==========================

.. autosummary::
    :toctree: generated/

    antenna_to_cartesian
    antenna_vectors_to_cartesian
    cartesian_to_geographic
    cartesian_vectors_to_geographic
    cartesian_to_geographic_aeqd
    cartesian_to_antenna
    geographic_to_cartesian
    geographic_to_cartesian_aeqd
    wgs84_to_swissCH1903

"""

from .radar import Radar #noqa
from .grid import Grid #noqa
from .wind_profile import HorizontalWindProfile #noqa
from .radar_spectra import RadarSpectra #noqa

from .transforms import antenna_to_cartesian #noqa
from .transforms import antenna_vectors_to_cartesian #noqa
from .transforms import cartesian_to_geographic #noqa
from .transforms import cartesian_vectors_to_geographic #noqa
from .transforms import cartesian_to_geographic_aeqd #noqa
from .transforms import cartesian_to_antenna #noqa
from .transforms import geographic_to_cartesian #noqa
from .transforms import geographic_to_cartesian_aeqd #noqa
from .transforms import wgs84_to_swissCH1903 #noqa

__all__ = [s for s in dir() if not s.startswith('_')]
