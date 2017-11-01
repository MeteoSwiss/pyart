"""
==============================
Filters (:mod:`pyart.filters`)
==============================

.. currentmodule:: pyart.filters

Classes for specifying what gates are included and excluded from routines.

Filtering radar data
====================

.. autosummary::
    :toctree: generated/

    GateFilter
    moment_based_gate_filter
    moment_and_texture_based_gate_filter
    snr_based_gate_filter
    visibility_based_gate_filter
    class_based_gate_filter
    temp_based_gate_filter
    iso0_based_gate_filter
    calculate_velocity_texture

"""

from .gatefilter import GateFilter, moment_based_gate_filter
from .gatefilter import moment_and_texture_based_gate_filter
from .gatefilter import snr_based_gate_filter, visibility_based_gate_filter
from .gatefilter import temp_based_gate_filter, class_based_gate_filter
from .gatefilter import calculate_velocity_texture, iso0_based_gate_filter

__all__ = [s for s in dir() if not s.startswith('_')]
