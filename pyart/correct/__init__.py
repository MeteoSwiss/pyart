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

Attenuation
=================

.. autosummary::
    :toctree: generated/
    
    calculate_attenuation
    calculate_attenuation_zphi
    calculate_attenuation_philinear

Phase processing
=================

.. autosummary::
    :toctree: generated/

    phase_proc_lp
    phase_proc_lp_gf
    det_sys_phase_ray
    correct_sys_phase
    smooth_phidp_single_window
    smooth_phidp_double_window

Other corrections
=================

.. autosummary::
    :toctree: generated/
    
    sun_retrieval
    get_sun_hits
    get_sun_hits_psr
    get_sun_hits_ivic
    correct_vpr
    correct_vpr_spatialised
    despeckle_field
    correct_noise_rhohv
    correct_bias
    correct_visibility
    est_rhohv_rain
    est_zdr_precip
    est_zdr_snow
    selfconsistency_bias
    selfconsistency_bias2
    selfconsistency_kdp_phidp

Helper functions
================

.. autosummary::
    :toctree: generated/

    find_objects
    get_mask_fzl
    sun_power
    ptoa_to_sf
    solar_flux_lookup
    scanning_losses
    smooth_masked
    sun_position_pysolar
    sun_position_mfr
    compute_apparent_vpr
    compute_theoretical_vpr

"""

from .dealias import dealias_fourdd  # noqa
from .attenuation import calculate_attenuation, calculate_attenuation_zphi  # noqa
from .attenuation import calculate_attenuation_philinear, get_mask_fzl  # noqa
from .phase_proc import det_sys_phase_ray, correct_sys_phase  # noqa
from .phase_proc import smooth_phidp_single_window, smooth_phidp_double_window  # noqa
from .phase_proc import smooth_masked  # noqa
from .phase_proc import phase_proc_lp, phase_proc_lp_gf  # noqa

# for backwards compatibility GateFilter available in the correct namespace
from ..filters.gatefilter import GateFilter, moment_based_gate_filter  # noqa
from .unwrap import dealias_unwrap_phase  # noqa
from .region_dealias import dealias_region_based  # noqa
from .despeckle import find_objects, despeckle_field  # noqa
from .bias_and_noise import correct_noise_rhohv, correct_bias  # noqa
from .bias_and_noise import correct_visibility  # noqa
from .bias_and_noise import est_rhohv_rain, est_zdr_precip, est_zdr_snow  # noqa
from .bias_and_noise import get_sun_hits, get_sun_hits_ivic, sun_retrieval  # noqa
from .bias_and_noise import selfconsistency_bias, selfconsistency_bias2  # noqa
from .bias_and_noise import selfconsistency_kdp_phidp, get_sun_hits_psr  # noqa
from .sunlib import sun_power, solar_flux_lookup, ptoa_to_sf, scanning_losses  # noqa
from .sunlib import sun_position_pysolar, sun_position_mfr  # noqa
from .sunlib import gauss_fit, retrieval_result  # noqa
from .vpr import correct_vpr, compute_apparent_vpr, compute_theoretical_vpr  # noqa
from .vpr import correct_vpr_spatialised  # noqa

__all__ = [s for s in dir() if not s.startswith("_")]
