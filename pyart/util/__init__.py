"""
Miscellaneous utility functions.

The location and names of these functions within Py-ART may change between
versions without depreciation, use with caution.


Circular Statistics
===========================

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
    compute_directional_stats

Datetime Utilities
===========================

.. autosummary::
    :toctree: generated/

    datetime_from_radar
    datetimes_from_radar
    datetime_from_dataset
    datetimes_from_dataset
    datetime_from_grid
    EPOCH_UNITS

Column Section
===========================

.. autosummary::
    :toctree: generated/

    for_azimuth
    get_column_rays
    get_field_location
    sphere_distance

Cross Section
===========================

.. autosummary::
    :toctree: generated/

    cross_section_ppi
    cross_section_rhi
    get_target_elevations
    colocated_gates
    colocated_gates2
    intersection

Noise estimation
===========================

.. autosummary::
    :toctree: generated/

    estimate_noise_hs74
    estimate_noise_ivic13
    get_ivic_pct
    get_ivic_flat_reg_var_max
    get_ivic_snr_thr
    ivic_pct_table
    ivic_snr_thr_table
    ivic_flat_reg_var_max_table
    ivic_flat_reg_wind_len_table

Radar Utilities
===========================

.. autosummary::
    :toctree: generated/

    is_vpt
    to_vpt
    join_radar
    join_spectra
    subset_radar
    subset_radar_spectra
    radar_from_spectra
    interpol_spectra
    find_neighbour_gates
    compute_azimuthal_average
    ma_broadcast_to
    image_mute_radar
    determine_sweeps

Simulated Velocity
===========================

.. autosummary::
    :toctree: generated/

    simulated_vel_from_profile

Signal Mathematics
===========================

.. autosummary::
    :toctree: generated/

    texture_along_ray
    rolling_window
    texture
    angular_texture_2d
    grid_texture_2d
    compute_nse
    compute_corr
    compute_mse

"""

from .circular_stats import angular_mean, angular_std  # noqa
from .circular_stats import angular_mean_deg, angular_std_deg  # noqa
from .circular_stats import interval_mean, interval_std  # noqa
from .circular_stats import mean_of_two_angles, mean_of_two_angles_deg  # noqa
from .circular_stats import compute_directional_stats  # noqa
from .datetime_utils import datetime_from_radar, datetimes_from_radar  # noqa
from .datetime_utils import datetime_from_dataset, datetimes_from_dataset  # noqa
from .datetime_utils import datetime_from_grid  # noqa
from .datetime_utils import EPOCH_UNITS  # noqa
from .columnsect import for_azimuth  # noqa
from .columnsect import get_column_rays  # noqa
from .columnsect import get_field_location  # noqa
from .columnsect import sphere_distance  # noqa
from .xsect import cross_section_ppi, cross_section_rhi, get_target_elevations  # noqa
from .xsect import colocated_gates, colocated_gates2, intersection  # noqa
from .hildebrand_sekhon import estimate_noise_hs74  # noqa
from .ivic import estimate_noise_ivic13, get_ivic_pct  # noqa
from .ivic import get_ivic_flat_reg_var_max, get_ivic_snr_thr  # noqa
from .ivic import ivic_pct_table, ivic_snr_thr_table  # noqa
from .ivic import ivic_flat_reg_var_max_table, ivic_flat_reg_wind_len_table  # noqa
from .radar_utils import is_vpt, to_vpt, join_radar, join_spectra  # noqa
from .radar_utils import subset_radar, subset_radar_spectra, radar_from_spectra  # noqa
from .radar_utils import interpol_spectra, find_neighbour_gates  # noqa
from .radar_utils import compute_azimuthal_average, ma_broadcast_to  # noqa
from .radar_utils import image_mute_radar  # noqa
from .radar_utils import determine_sweeps  # noqa
from .simulated_vel import simulated_vel_from_profile  # noqa
from .sigmath import texture_along_ray, rolling_window  # noqa
from .sigmath import texture, angular_texture_2d, grid_texture_2d  # noqa
from .sigmath import compute_nse, compute_corr, compute_mse  # noqa
from .simulated_vel import simulated_vel_from_profile  # noqa

__all__ = [s for s in dir() if not s.startswith("_")]
