"""
========================================
Radar Retrievals (:mod:`pyart.retrieve`)
========================================

.. currentmodule:: pyart.retrieve

Radar retrievals.

Radar retrievals
================

.. autosummary::
    :toctree: generated/

    kdp_maesaka
    kdp_leastsquare_single_window
    kdp_leastsquare_double_window
    calculate_snr_from_reflectivity
    compute_snr
    compute_l
    compute_cdr
    compute_noisedBZ
    fetch_radar_time_profile
    map_profile_to_gates
    steiner_conv_strat
    texture_of_complex_phase
    grid_displacement_pc
    grid_shift

"""

from .kdp_proc import kdp_maesaka, kdp_leastsquare_single_window
from .kdp_proc import kdp_leastsquare_double_window
from .echo_class import steiner_conv_strat, hydroclass_semisupervised
from .gate_id import map_profile_to_gates, fetch_radar_time_profile
from .simple_moment_calculations import calculate_snr_from_reflectivity
from .simple_moment_calculations import compute_snr, compute_l, compute_cdr
from .simple_moment_calculations import compute_noisedBZ
from .qpe import rr_z, rr_zpoly, rr_kdp, rr_a, rr_zkdp, rr_za, rr_hydro
from .advection import grid_displacement_pc, grid_shift

__all__ = [s for s in dir() if not s.startswith('_')]
