"""
========================================
Radar Corrections (:mod:`pyart.correct`)
========================================

.. currentmodule:: pyart.correct

Correct radar fields.

Velocity unfolding
==================

.. autosummary::
    :toctree: generated/

    dealias_fourdd
    dealias_unwrap_phase
    dealias_region_based

Other corrections
=================

.. autosummary::
    :toctree: generated/

    calculate_attenuation_zphi
    calculate_attenuation_philinear
    phase_proc_lp
    det_sys_phase_ray
    correct_sys_phase
    smooth_phidp_single_window
    smooth_phidp_double_window
    despeckle_field
    correct_noise_rhohv
    correct_bias
    correct_visibility
    est_rhohv_rain
    est_zdr_precip
    selfconsistency_bias
    selfconsistency_kdp_phidp
    get_sun_hits
    sun_retrieval

Helper functions
================

.. autosummary::
    :toctree: generated/

    find_objects
    get_mask_fzl
    sun_power

"""


from .dealias import dealias_fourdd
from .attenuation import calculate_attenuation_zphi
from .attenuation import calculate_attenuation_philinear, get_mask_fzl
from .phase_proc import phase_proc_lp, det_sys_phase_ray, correct_sys_phase
from .phase_proc import smooth_phidp_single_window, smooth_phidp_double_window
# for backwards compatibility GateFilter available in the correct namespace
from ..filters.gatefilter import GateFilter, moment_based_gate_filter
from .unwrap import dealias_unwrap_phase
from .region_dealias import dealias_region_based
from .despeckle import find_objects, despeckle_field
from .bias_and_noise import correct_noise_rhohv, correct_bias
from .bias_and_noise import correct_visibility
from .bias_and_noise import est_rhohv_rain, est_zdr_precip
from .bias_and_noise import get_sun_hits, sun_retrieval
from .bias_and_noise import selfconsistency_bias, selfconsistency_kdp_phidp
from .sunlib import sun_power

__all__ = [s for s in dir() if not s.startswith('_')]
