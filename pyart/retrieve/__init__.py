"""
========================================
Radar Retrievals (:mod:`pyart.retrieve`)
========================================

.. currentmodule:: pyart.retrieve

Radar retrievals.

Composite Reflectivity
===========================

.. autosummary::
    :toctree: generated/

    composite_reflectivity

Melting Layer (ML) Detection
===========================

.. autosummary::
    :toctree: generated/

    detect_ml
    melting_layer_giangrande
    melting_layer_hydroclass
    _get_res_vol_sides
    compute_apparent_profile
    melting_layer_mf
    get_ml_rng_limits
    get_iso0_val

KDP Processing
===========================

.. autosummary::
    :toctree: generated/

    kdp_maesaka
    kdp_schneebeli
    kdp_vulpiani
    kdp_leastsquare_single_window
    kdp_leastsquare_double_window
    filter_psidp

Echo Classification
===========================

.. autosummary::
    :toctree: generated/

    conv_strat_yuter
    steiner_conv_strat
    hydroclass_semisupervised
    get_freq_band
    data_for_centroids
    compute_centroids
    select_samples
    determine_medoids
    _destandardize
    synthetic_obs_distribution
    feature_detection
    conv_strat_raut

Gate ID
===========================

.. autosummary::
    :toctree: generated/

    map_profile_to_gates
    fetch_radar_time_profile

Simple Moment Calculations
===========================

.. autosummary::
    :toctree: generated/

    calculate_snr_from_reflectivity
    calculate_velocity_texture
    compute_snr
    compute_l
    compute_cdr
    compute_refl_from_zdr
    compute_noisedBZ
    compute_signal_power
    get_coeff_attg
    compute_vol_refl
    compute_bird_density
    atmospheric_gas_att
    compute_ccor
    compute_rcs
    compute_rcs_from_pr
    compute_radial_noise_hs
    compute_radial_noise_ivic

QPE (Quantitative Precipitation Estimation)
===========================

.. autosummary::
    :toctree: generated/

    est_rain_rate_z
    est_rain_rate_zpoly
    est_rain_rate_kdp
    est_rain_rate_a
    est_rain_rate_zkdp
    est_rain_rate_za
    est_rain_rate_hydro

Advection
===========================

.. autosummary::
    :toctree: generated/

    grid_displacement_pc
    grid_shift

Wind Estimation
===========================

.. autosummary::
    :toctree: generated/

    est_wind_vel
    est_vertical_windshear
    est_wind_profile
    est_vertical_windshear_lidar

VAD (Velocity Azimuth Display)
===========================

.. autosummary::
    :toctree: generated/

    vad_browning
    vad_michelson

QVP (Quasi Vertical Profile) Retrievals
===========================

.. autosummary::
    :toctree: generated/

    quasi_vertical_profile
    compute_qvp
    compute_rqvp
    compute_evp
    compute_svp
    compute_vp
    compute_ts_along_coord

Spectra Processing
===========================

.. autosummary::
    :toctree: generated/

    compute_spectral_power
    compute_spectral_phase
    compute_spectral_noise
    compute_spectral_reflectivity
    compute_spectral_differential_reflectivity
    compute_spectral_differential_phase
    compute_spectral_rhohv
    compute_reflectivity
    compute_differential_reflectivity
    compute_differential_phase
    compute_rhohv
    compute_Doppler_velocity
    compute_Doppler_width
    compute_pol_variables
    compute_iq
    compute_noise_power
    dealias_spectra

IQ Processing
===========================

.. autosummary::
    :toctree: generated/

    compute_reflectivity_iq
    compute_differential_reflectivity_iq
    compute_differential_phase_iq
    compute_rhohv_iq
    compute_Doppler_velocity_iq
    compute_Doppler_width_iq
    compute_pol_variables_iq
    compute_spectra
    compute_mean_phase_iq
    compute_st1_iq
    compute_st2_iq
    compute_wbn_iq

Visibility estimation
===========================

.. autosummary::
    :toctree: generated/

    gecsx

"""

from .comp_z import composite_reflectivity  # noqa
from .ml import detect_ml, melting_layer_giangrande, melting_layer_hydroclass  # noqa
from .ml import _get_res_vol_sides, compute_apparent_profile, melting_layer_mf  # noqa
from .ml import get_ml_rng_limits, get_iso0_val  # noqa
from .kdp_proc import kdp_maesaka, kdp_schneebeli, kdp_vulpiani  # noqa
from .kdp_proc import kdp_leastsquare_single_window  # noqa
from .kdp_proc import kdp_leastsquare_double_window  # noqa
from .kdp_proc import kdp_operational_mch  # noqa
from .echo_class import conv_strat_yuter  # noqa
from .echo_class import steiner_conv_strat, hydroclass_semisupervised  # noqa
from .echo_class import get_freq_band, data_for_centroids, compute_centroids  # noqa
from .echo_class import select_samples, determine_medoids, _destandardize  # noqa
from .echo_class import synthetic_obs_distribution  # noqa
from .echo_class import feature_detection  # noqa
from .echo_class import conv_strat_raut  # noqa
from .gate_id import map_profile_to_gates, fetch_radar_time_profile  # noqa
from .simple_moment_calculations import calculate_snr_from_reflectivity  # noqa
from .simple_moment_calculations import calculate_velocity_texture  # noqa
from .simple_moment_calculations import compute_snr, compute_l, compute_cdr  # noqa
from .simple_moment_calculations import compute_noisedBZ, compute_signal_power  # noqa
from .simple_moment_calculations import get_coeff_attg, compute_vol_refl  # noqa
from .simple_moment_calculations import compute_bird_density  # noqa
from .simple_moment_calculations import atmospheric_gas_att, compute_ccor  # noqa
from .simple_moment_calculations import compute_rcs, compute_rcs_from_pr  # noqa
from .simple_moment_calculations import compute_radial_noise_hs  # noqa
from .simple_moment_calculations import compute_radial_noise_ivic  # noqa
from .simple_moment_calculations import compute_refl_from_zdr  # noqa
from .qpe import est_rain_rate_z, est_rain_rate_zpoly, est_rain_rate_kdp  # noqa
from .qpe import est_rain_rate_a, est_rain_rate_zkdp, est_rain_rate_za  # noqa
from .qpe import est_rain_rate_hydro  # noqa
from .advection import grid_displacement_pc, grid_shift  # noqa
from .wind import est_wind_vel, est_vertical_windshear, est_wind_profile  # noqa
from .wind import est_vertical_windshear_lidar  # noqa
from .vad import vad_browning, vad_michelson  # noqa
from .qvp import quasi_vertical_profile, compute_qvp, compute_rqvp  # noqa
from .qvp import compute_evp, compute_svp, compute_vp, compute_ts_along_coord  # noqa
from .spectra import compute_spectral_power, compute_spectral_phase  # noqa
from .spectra import compute_spectral_noise, compute_spectral_reflectivity  # noqa
from .spectra import compute_spectral_differential_reflectivity  # noqa
from .spectra import compute_spectral_differential_phase  # noqa
from .spectra import compute_spectral_rhohv, compute_reflectivity  # noqa
from .spectra import compute_differential_reflectivity  # noqa
from .spectra import compute_differential_phase, compute_rhohv  # noqa
from .spectra import compute_Doppler_velocity, compute_Doppler_width  # noqa
from .spectra import compute_pol_variables, compute_iq, compute_noise_power  # noqa
from .spectra import dealias_spectra  # noqa
from .iq import compute_reflectivity_iq, compute_differential_reflectivity_iq  # noqa
from .iq import compute_differential_phase_iq, compute_rhohv_iq  # noqa
from .iq import compute_Doppler_velocity_iq, compute_Doppler_width_iq  # noqa
from .iq import compute_pol_variables_iq, compute_spectra  # noqa
from .iq import compute_mean_phase_iq, compute_st1_iq, compute_st2_iq  # noqa
from .iq import compute_wbn_iq  # noqa
from .gecsx import gecsx  # noqa

__all__ = [s for s in dir() if not s.startswith("_")]
