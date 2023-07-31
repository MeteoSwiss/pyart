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
    kdp_schneebeli
    kdp_vulpiani
    kdp_leastsquare_single_window
    kdp_leastsquare_double_window
    calculate_snr_from_reflectivity
    calculate_velocity_texture
    get_ml_rng_limits
    get_iso0_val
    compute_ccor
    compute_snr
    compute_l
    compute_cdr
    compute_noisedBZ
    compute_radial_noise_hs
    compute_radial_noise_ivic
    compute_signal_power
    compute_rcs
    compute_rcs_from_pr
    compute_vol_refl
    compute_bird_density
    fetch_radar_time_profile
    map_profile_to_gates
    steiner_conv_strat
    conv_strat_yuter
    hydroclass_semisupervised
    data_for_centroids
    select_samples
    compute_centroids
    determine_medoids
    synthetic_obs_distribution
    _destandardize
    get_freq_band
    texture_of_complex_phase
    grid_displacement_pc
    grid_shift
    est_rain_rate_zpoly
    est_rain_rate_z
    est_rain_rate_kdp
    est_rain_rate_a
    est_rain_rate_zkdp
    est_rain_rate_za
    est_rain_rate_hydro
    est_wind_vel
    est_vertical_windshear
    atmospheric_gas_att
    get_coeff_attg
    est_wind_profile
    detect_ml
    melting_layer_giangrande
    melting_layer_hydroclass
    melting_layer_mf
    compute_apparent_profile
    _get_res_vol_sides
    vad_michelson
    vad_browning
    quasi_vertical_profile
    compute_qvp
    compute_rqvp
    compute_evp
    compute_svp
    compute_vp
    compute_ts_along_coord
    compute_iq
    compute_spectral_power
    compute_spectral_noise
    compute_spectral_phase
    compute_spectral_reflectivity
    compute_spectral_differential_reflectivity
    compute_spectral_differential_phase
    compute_spectral_rhohv
    compute_pol_variables
    compute_noise_power
    compute_reflectivity
    compute_differential_reflectivity
    compute_differential_phase
    compute_rhohv
    compute_Doppler_velocity
    compute_Doppler_width
    compute_reflectivity_iq
    compute_st1_iq
    compute_st2_iq
    compute_wbn_iq
    compute_differential_reflectivity_iq
    compute_mean_phase_iq
    compute_differential_phase_iq
    compute_rhohv_iq
    compute_Doppler_velocity_iq
    compute_Doppler_width_iq
    compute_pol_variables_iq
    compute_spectra
    gecsx

"""
from .comp_z import composite_reflectivity # noqa
from .ml import detect_ml, melting_layer_giangrande, melting_layer_hydroclass #noqa
from .ml import _get_res_vol_sides, compute_apparent_profile, melting_layer_mf #noqa
from .ml import get_ml_rng_limits, get_iso0_val #noqa
from .kdp_proc import kdp_maesaka, kdp_schneebeli, kdp_vulpiani #noqa
from .kdp_proc import kdp_leastsquare_single_window #noqa
from .kdp_proc import kdp_leastsquare_double_window #noqa
from .echo_class import conv_strat_yuter  # noqa
from .echo_class import steiner_conv_strat, hydroclass_semisupervised #noqa
from .echo_class import get_freq_band, data_for_centroids, compute_centroids #noqa
from .echo_class import select_samples, determine_medoids, _destandardize #noqa
from .echo_class import synthetic_obs_distribution #noqa
from .gate_id import map_profile_to_gates, fetch_radar_time_profile #noqa
from .simple_moment_calculations import calculate_snr_from_reflectivity #noqa
from .simple_moment_calculations import calculate_velocity_texture #noqa
from .simple_moment_calculations import compute_snr, compute_l, compute_cdr #noqa
from .simple_moment_calculations import compute_noisedBZ, compute_signal_power #noqa
from .simple_moment_calculations import get_coeff_attg, compute_vol_refl #noqa
from .simple_moment_calculations import compute_bird_density #noqa
from .simple_moment_calculations import atmospheric_gas_att, compute_ccor #noqa
from .simple_moment_calculations import compute_rcs, compute_rcs_from_pr #noqa
from .simple_moment_calculations import compute_radial_noise_hs #noqa
from .simple_moment_calculations import compute_radial_noise_ivic #noqa
from .qpe import est_rain_rate_z, est_rain_rate_zpoly, est_rain_rate_kdp #noqa
from .qpe import est_rain_rate_a, est_rain_rate_zkdp, est_rain_rate_za #noqa
from .qpe import est_rain_rate_hydro #noqa
from .advection import grid_displacement_pc, grid_shift #noqa
from .wind import est_wind_vel, est_vertical_windshear, est_wind_profile #noqa
from .vad import vad_browning, vad_michelson #noqa
from .qvp import quasi_vertical_profile, compute_qvp, compute_rqvp #noqa
from .qvp import compute_evp, compute_svp, compute_vp, compute_ts_along_coord #noqa
from .spectra import compute_spectral_power, compute_spectral_phase #noqa
from .spectra import compute_spectral_noise, compute_spectral_reflectivity #noqa
from .spectra import compute_spectral_differential_reflectivity #noqa
from .spectra import compute_spectral_differential_phase #noqa
from .spectra import compute_spectral_rhohv, compute_reflectivity #noqa
from .spectra import compute_differential_reflectivity #noqa
from .spectra import compute_differential_phase, compute_rhohv #noqa
from .spectra import compute_Doppler_velocity, compute_Doppler_width #noqa
from .spectra import compute_pol_variables, compute_iq, compute_noise_power #noqa
from .iq import compute_reflectivity_iq, compute_differential_reflectivity_iq #noqa
from .iq import compute_differential_phase_iq, compute_rhohv_iq #noqa
from .iq import compute_Doppler_velocity_iq, compute_Doppler_width_iq #noqa
from .iq import compute_pol_variables_iq, compute_spectra #noqa
from .iq import compute_mean_phase_iq, compute_st1_iq, compute_st2_iq #noqa
from .iq import compute_wbn_iq #noqa
from .gecsx import gecsx #noqa

__all__ = [s for s in dir() if not s.startswith('_')]
