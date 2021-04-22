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
    hydroclass_semisupervised
    data_for_centroids
    select_samples
    compute_centroids
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
    _get_res_vol_sides
    velocity_azimuth_display
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
from .ml import detect_ml, melting_layer_giangrande, melting_layer_hydroclass
from .ml import _get_res_vol_sides
from .kdp_proc import kdp_maesaka, kdp_schneebeli, kdp_vulpiani
from .kdp_proc import kdp_leastsquare_single_window
from .kdp_proc import kdp_leastsquare_double_window
from .echo_class import steiner_conv_strat, hydroclass_semisupervised
from .echo_class import get_freq_band, data_for_centroids, compute_centroids
from .echo_class import select_samples
from .gate_id import map_profile_to_gates, fetch_radar_time_profile
from .simple_moment_calculations import calculate_snr_from_reflectivity
from .simple_moment_calculations import calculate_velocity_texture
from .simple_moment_calculations import compute_snr, compute_l, compute_cdr
from .simple_moment_calculations import compute_noisedBZ, compute_signal_power
from .simple_moment_calculations import get_coeff_attg, compute_vol_refl
from .simple_moment_calculations import compute_bird_density
from .simple_moment_calculations import atmospheric_gas_att, compute_ccor
from .simple_moment_calculations import compute_rcs, compute_rcs_from_pr
from .simple_moment_calculations import compute_radial_noise_hs
from .simple_moment_calculations import compute_radial_noise_ivic
from .qpe import est_rain_rate_z, est_rain_rate_zpoly, est_rain_rate_kdp
from .qpe import est_rain_rate_a, est_rain_rate_zkdp, est_rain_rate_za
from .qpe import est_rain_rate_hydro
from .advection import grid_displacement_pc, grid_shift
from .wind import est_wind_vel, est_vertical_windshear, est_wind_profile
from .vad import velocity_azimuth_display
from .qvp import quasi_vertical_profile, compute_qvp, compute_rqvp
from .qvp import compute_evp, compute_svp, compute_vp, compute_ts_along_coord
from .spectra import compute_spectral_power, compute_spectral_phase
from .spectra import compute_spectral_noise, compute_spectral_reflectivity
from .spectra import compute_spectral_differential_reflectivity
from .spectra import compute_spectral_differential_phase
from .spectra import compute_spectral_rhohv, compute_reflectivity
from .spectra import compute_differential_reflectivity
from .spectra import compute_differential_phase, compute_rhohv
from .spectra import compute_Doppler_velocity, compute_Doppler_width
from .spectra import compute_pol_variables, compute_iq, compute_noise_power
from .iq import compute_reflectivity_iq, compute_differential_reflectivity_iq
from .iq import compute_differential_phase_iq, compute_rhohv_iq
from .iq import compute_Doppler_velocity_iq, compute_Doppler_width_iq
from .iq import compute_pol_variables_iq, compute_spectra
from .iq import compute_mean_phase_iq, compute_st1_iq, compute_st2_iq
from .iq import compute_wbn_iq
from .gecsx import gecsx

__all__ = [s for s in dir() if not s.startswith('_')]
