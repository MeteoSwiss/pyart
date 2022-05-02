"""
=============================
Utilities (:mod:`pyart.util`)
=============================

Miscellaneous utility functions.

The location and names of these functions within Py-ART may change between
versions without depreciation, use with caution.
.. currentmodule:: pyart.util

Direction statistics
====================
.. autosummary::
    :toctree: generated/
    angular_mean
    angular_std
    angular_mean_deg
    angular_std_deg
    interval_mean
    interval_std
    mean_of_two_angles
    mean_of_two_angles_deg

Miscellaneous functions
=======================
.. autosummary::
    :toctree: generated/

    compute_antenna_diagram
    compute_azimuthal_average
    find_neighbour_gates
    get_target_elevations
    compute_directional_stats
    cross_section_ppi
    cross_section_rhi
    colocated_gates
    colocated_gates2
    intersection
    datetime_from_radar
    datetimes_from_radar
    datetime_from_dataset
    datetimes_from_dataset
    datetime_from_grid
    estimate_noise_hs74
    estimate_noise_ivic13
    get_ivic_pct
    get_ivic_flat_reg_var_max
    get_ivic_snr_thr
    ivic_pct_table
    ivic_snr_thr_table
    ivic_flat_reg_var_max_table
    ivic_flat_reg_wind_len_table
    is_vpt
    to_vpt
    join_radar
    join_spectra
    cut_radar
    cut_radar_spectra
    radar_from_spectra
    interpol_spectra
    ma_broadcast_to
    simulated_vel_from_profile
    texture_along_ray
    texture
    rolling_window
    angular_texture_2d
    compute_nse
    compute_corr
    compute_mse

"""

from .circular_stats import angular_mean, angular_std
from .circular_stats import angular_mean_deg, angular_std_deg
from .circular_stats import interval_mean, interval_std
from .circular_stats import mean_of_two_angles, mean_of_two_angles_deg
from .circular_stats import compute_directional_stats
from .datetime_utils import datetime_from_radar, datetimes_from_radar
from .datetime_utils import datetime_from_dataset, datetimes_from_dataset
from .datetime_utils import datetime_from_grid
from .datetime_utils import EPOCH_UNITS
from .xsect import cross_section_ppi, cross_section_rhi, get_target_elevations
from .xsect import colocated_gates, colocated_gates2, intersection
from .hildebrand_sekhon import estimate_noise_hs74
from .ivic import estimate_noise_ivic13, get_ivic_pct
from .ivic import get_ivic_flat_reg_var_max, get_ivic_snr_thr
from .ivic import ivic_pct_table, ivic_snr_thr_table
from .ivic import ivic_flat_reg_var_max_table, ivic_flat_reg_wind_len_table
from .radar_utils import is_vpt, to_vpt, join_radar, join_spectra
from .radar_utils import cut_radar, cut_radar_spectra, radar_from_spectra
from .radar_utils import interpol_spectra, find_neighbour_gates
from .radar_utils import compute_azimuthal_average, ma_broadcast_to
from .simulated_vel import simulated_vel_from_profile
from .sigmath import texture_along_ray, rolling_window
from .sigmath import texture, angular_texture_2d, grid_texture_2d
from .sigmath import compute_nse, compute_corr, compute_mse

__all__ = [s for s in dir() if not s.startswith('_')]
