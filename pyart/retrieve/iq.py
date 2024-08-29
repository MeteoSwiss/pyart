"""
pyart.retrieve.radar
======================

Retrievals from spectral data.

.. autosummary::
    :toctree: generated/

    compute_spectra
    compute_pol_variables_iq
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
    _compute_power
    _compute_autocorrelation
    _compute_lag_diff
    _compute_crosscorrelation

"""

from copy import deepcopy
from warnings import warn

import numpy as np
from scipy.constants import speed_of_light
from scipy.signal.windows import get_window

from ..config import get_field_name, get_metadata
from ..util import radar_from_spectra


def compute_spectra(radar, fields_in_list, fields_out_list, window=None):
    """
    Computes the spectra from IQ data through a Fourier transform

    Parameters
    ----------
    radar : radar object
        Object containing the IQ data
    fields_in_list : list of str
        list of input IQ data fields names
    fields_out_list : list of str
        list with the output spectra fields names obtained from the input
        fields
    window : string, tupple or None
        Parameters of the window used to obtain the spectra. The
        parameters are the ones corresponding to function
        scipy.signal.windows.get_window. If None no window will be used

    Returns
    -------
    spectra : spectra radar object
        radar object containing the spectra fields

    """
    spectra = deepcopy(radar)
    spectra.fields = {}
    for field_name_in, field_name_out in zip(fields_in_list, fields_out_list):
        if field_name_in in ("IQ_hh_ADU", "IQ_vv_ADU"):
            spectrum = np.ma.masked_all(
                (radar.nrays, radar.ngates, radar.npulses_max), dtype=np.complex64
            )
            for ray, npuls in enumerate(radar.npulses["data"]):
                wind_data = radar.fields[field_name_in]["data"][ray, :, 0:npuls].filled(
                    0.0
                )
                if window is not None:
                    wind = get_window(window, npuls)
                    wind = wind / np.sqrt(np.sum(np.power(wind, 2.0)) / npuls)
                    wind = np.broadcast_to(np.atleast_2d(wind), (radar.ngates, npuls))
                    wind_data *= wind

                spectrum[ray, :, 0:npuls] = np.fft.fftshift(
                    np.fft.fft(wind_data, axis=-1) / npuls, axes=-1
                )
        else:
            spectrum = np.ma.masked_all(
                (radar.nrays, radar.ngates, radar.npulses_max), dtype=np.float32
            )
            for ray, npuls in enumerate(radar.npulses["data"]):
                spectrum[ray, :, 0:npuls] = (
                    radar.fields[field_name_in]["data"][ray, :, 0:npuls] / npuls
                )

        field_dict = get_metadata(field_name_out)
        field_dict["data"] = spectrum
        spectra.fields.update({field_name_out: field_dict})

    return spectra


def compute_pol_variables_iq(
    radar,
    fields_list,
    subtract_noise=False,
    lag=0,
    direction="negative_away",
    phase_offset=0.0,
    signal_h_field=None,
    signal_v_field=None,
    noise_h_field=None,
    noise_v_field=None,
):
    """
    Computes the polarimetric variables from the IQ signals in ADU

    Parameters
    ----------
    radar : IQ radar object
        Object containing the required fields
    fields_list : list of str
        list of fields to compute
    subtract_noise : Bool
        If True noise will be subtracted from the signals
    lag : int
        The time lag to use in the estimators
    direction : str
        The convention used in the Doppler mean field. Can be negative_away or
        negative_towards
    phase_offset : float. Dataset keyword
        The system differential phase offset to remove
    signal_h_field, signal_v_field, noise_h_field, noise_v_field : str
        Name of the fields in radar which contains the signal and noise.
        None will use the default field name in the Py-ART configuration file.

    Returns
    -------
    radar : radar object
        Object containing the computed fields

    """
    if signal_h_field is None:
        signal_h_field = get_field_name("IQ_hh_ADU")
    if signal_v_field is None:
        signal_v_field = get_field_name("IQ_vv_ADU")
    if noise_h_field is None:
        noise_h_field = get_field_name("IQ_noiseADU_hh")
    if noise_v_field is None:
        noise_v_field = get_field_name("IQ_noiseADU_vv")

    if (
        "reflectivity" in fields_list
        or "differential_reflectivity" in fields_list
        or ("cross_correlation_ratio" in fields_list and lag == 0)
        or ("spectrum_width" in fields_list and lag == 0)
    ):
        noise_h = None
        if noise_h_field in radar.fields:
            noise_h = radar.fields[noise_h_field]["data"]
        pwr_h = _compute_power(
            radar.fields[signal_h_field]["data"],
            noise=noise_h,
            subtract_noise=subtract_noise,
        )

    if (
        "reflectivity_vv" in fields_list
        or "differential_reflectivity" in fields_list
        or ("cross_correlation_ratio" in fields_list and lag == 0)
        or ("spectrum_width_vv" in fields_list and lag == 0)
    ):
        noise_v = None
        if noise_v_field in radar.fields:
            noise_v = radar.fields[noise_v_field]["data"]
        pwr_v = _compute_power(
            radar.fields[signal_v_field]["data"],
            noise=noise_v,
            subtract_noise=subtract_noise,
        )

    if (
        "velocity" in fields_list
        or "spectrum_width" in fields_list
        or ("differential_reflectivity" in fields_list and lag == 1)
        or ("cross_correlation_ratio" in fields_list and lag == 1)
    ):
        rlag1_h = _compute_autocorrelation(radar, signal_h_field, lag=1)
        if (
            "differential_reflectivity" in fields_list
            or "cross_correlation_ratio" in fields_list
        ):
            rlagn_h_abs = np.ma.abs(rlag1_h)

    if (
        "velocity_vv" in fields_list
        or "spectrum_width_vv" in fields_list
        or ("differential_reflectivity" in fields_list and lag == 1)
        or ("cross_correlation_ratio" in fields_list and lag == 1)
    ):
        rlag1_v = _compute_autocorrelation(radar, signal_v_field, lag=1)
        if (
            "differential_reflectivity" in fields_list
            or "cross_correlation_ratio" in fields_list
        ):
            rlagn_v_abs = np.ma.abs(rlag1_v)

    if "spectrum_width" in fields_list and lag >= 1:
        rlag2_h = _compute_autocorrelation(radar, signal_h_field, lag=2)

    if "spectrum_width_vv" in fields_list and lag >= 1:
        rlag2_v = _compute_autocorrelation(radar, signal_v_field, lag=2)

    if lag > 0 and (
        "differential_reflectivity" in fields_list
        or "cross_correlation_ratio" in fields_list
    ):
        rlagn_h_abs = np.ma.abs(_compute_autocorrelation(radar, signal_h_field, lag=1))
        rlagn_v_abs = np.ma.abs(_compute_autocorrelation(radar, signal_v_field, lag=1))

    if "reflectivity" in fields_list or "differential_reflectivity" in fields_list:
        dBADU2dBm_h = radar.radar_calibration["dBADU_to_dBm_hh"]["data"][0]
        radconst_h = radar.radar_calibration["calibration_constant_hh"]["data"][0]
        mfloss_h = radar.radar_calibration["matched_filter_loss_h"]["data"][0]

    if "reflectivity_vv" in fields_list or "differential_reflectivity" in fields_list:
        dBADU2dBm_v = radar.radar_calibration["dBADU_to_dBm_vv"]["data"][0]
        radconst_v = radar.radar_calibration["calibration_constant_vv"]["data"][0]
        mfloss_v = radar.radar_calibration["matched_filter_loss_v"]["data"][0]

    if "reflectivity" in fields_list or "reflectivity_vv" in fields_list:
        pathatt = radar.radar_calibration["path_attenuation"]["data"][0]
        rangeKm = np.broadcast_to(
            np.atleast_2d(radar.range["data"] / 1000.0), (radar.nrays, radar.ngates)
        )

    if (
        "velocity" in fields_list
        or "velocity_vv" in fields_list
        or "spectrum_width" in fields_list
        or "spectrum_width_vv" in fields_list
    ):
        prt = np.broadcast_to(
            np.expand_dims(radar.instrument_parameters["prt"]["data"], axis=1),
            (radar.nrays, radar.ngates),
        )
        wavelength = (
            speed_of_light / radar.instrument_parameters["frequency"]["data"][0]
        )

    fields = {}
    if "reflectivity" in fields_list or (
        "differential_reflectivity" in fields_list and lag == 0
    ):
        dBZ = (
            10.0 * np.ma.log10(pwr_h)
            + dBADU2dBm_h
            + radconst_h
            + mfloss_h
            + pathatt * rangeKm
            + 20.0 * np.log10(rangeKm)
        )

        if "reflectivity" in fields_list:
            dBZ_field = "reflectivity"
            dBZ_dict = get_metadata(dBZ_field)
            dBZ_dict["data"] = dBZ
            fields.update({dBZ_field: dBZ_dict})

    if "reflectivity_vv" in fields_list or (
        "differential_reflectivity" in fields_list and lag == 0
    ):
        dBZv = (
            10.0 * np.ma.log10(pwr_v)
            + dBADU2dBm_v
            + radconst_v
            + mfloss_v
            + pathatt * rangeKm
            + 20.0 * np.log10(rangeKm)
        )
        if "reflectivity_vv" in fields_list:
            dBZv_field = "reflectivity_vv"
            dBZv_dict = get_metadata(dBZv_field)
            dBZv_dict["data"] = dBZv
            fields.update({dBZv_field: dBZv_dict})

    if "differential_reflectivity" in fields_list:
        if lag == 0:
            zdr = dBZ - dBZv
        else:
            zdr = (10.0 * np.ma.log10(rlagn_h_abs) + dBADU2dBm_h + radconst_h) - (
                10.0 * np.ma.log10(rlagn_v_abs) + dBADU2dBm_v + radconst_v
            )
        zdr_field = "differential_reflectivity"
        zdr_dict = get_metadata(zdr_field)
        zdr_dict["data"] = zdr
        fields.update({zdr_field: zdr_dict})

    if "cross_correlation_ratio" in fields_list:
        if lag == 0:
            rhohv = np.ma.abs(
                np.ma.mean(
                    radar.fields[signal_h_field]["data"]
                    * np.ma.conjugate(radar.fields[signal_v_field]["data"]),
                    axis=-1,
                )
            )
            rhohv /= np.ma.sqrt(pwr_h * pwr_v)
        else:
            rhohv = np.ma.abs(
                _compute_crosscorrelation(
                    radar, signal_h_field, signal_v_field, lag=lag
                )
            )
            rhohv /= np.ma.sqrt(rlagn_h_abs * rlagn_v_abs)
        rhohv_field = "cross_correlation_ratio"
        rhohv_dict = get_metadata(rhohv_field)
        rhohv_dict["data"] = rhohv
        fields.update({rhohv_field: rhohv_dict})

    if "velocity" in fields_list:
        mean_vel = np.ma.angle(rlag1_h) / (2 * np.pi * prt) * (wavelength / 2.0)
        if direction == "negative_away":
            mean_vel = -mean_vel
        vel_field = "velocity"
        vel_dict = get_metadata(vel_field)
        vel_dict["data"] = mean_vel
        fields.update({vel_field: vel_dict})

    if "velocity_vv" in fields_list:
        mean_vel = np.ma.angle(rlag1_v) / (2 * np.pi * prt) * (wavelength / 2.0)
        if direction == "negative_away":
            mean_vel = -mean_vel
        vel_field = "velocity_vv"
        vel_dict = get_metadata(vel_field)
        vel_dict["data"] = mean_vel
        fields.update({vel_field: vel_dict})

    if "spectrum_width" in fields_list:
        if lag == 0:
            width = (
                np.ma.sqrt(np.ma.log(np.ma.abs(pwr_h) / np.ma.abs(rlag1_h)))
                / (np.sqrt(2.0) * np.pi * prt)
                * (wavelength / 2.0)
            )
        else:
            width = (
                np.ma.sqrt(np.ma.log(np.ma.abs(rlag1_h) / np.ma.abs(rlag2_h)))
                / (np.sqrt(6.0) * np.pi * prt)
                * (wavelength / 2.0)
            )
        width_field = "spectrum_width"
        width_dict = get_metadata(width_field)
        width_dict["data"] = width
        fields.update({width_field: width_dict})

    if "spectrum_width_vv" in fields_list:
        if lag == 0:
            width = (
                np.ma.sqrt(np.ma.log(np.ma.abs(pwr_v) / np.ma.abs(rlag1_v)))
                / (np.sqrt(2.0) * np.pi * prt)
                * (wavelength / 2.0)
            )
        else:
            width = (
                np.ma.sqrt(np.ma.log(np.ma.abs(rlag1_v) / np.ma.abs(rlag2_v)))
                / (np.sqrt(6.0) * np.pi * prt)
                * (wavelength / 2.0)
            )
        width_field = "spectrum_width_vv"
        width_dict = get_metadata(width_field)
        width_dict["data"] = width
        fields.update({width_field: width_dict})

    if "uncorrected_differential_phase" in fields_list:
        phidp_dict = compute_differential_phase_iq(
            radar,
            phase_offset=phase_offset,
            signal_h_field=signal_h_field,
            signal_v_field=signal_v_field,
        )
        fields.update({"uncorrected_differential_phase": phidp_dict})

    radar = radar_from_spectra(radar)

    for field_name in fields_list:
        radar.add_field(field_name, fields[field_name])

    return radar


def compute_reflectivity_iq(
    radar, subtract_noise=False, signal_field=None, noise_field=None
):
    """
    Computes the reflectivity from the IQ signal data

    Parameters
    ----------
    radar : IQ radar object
        Object containing the required fields
    subtract_noise : Bool
        If true the noise is subtracted from the power
    signal_field, noise_field : str
        Name of the signal and noise fields. None will use the default field
        name in the Py-ART configuration file.

    Returns
    -------
    dBZ_dict : field dictionary
        Field dictionary containing the reflectivity

    """
    if signal_field is None:
        signal_field = get_field_name("IQ_hh_ADU")
    if noise_field is None:
        noise_field = get_field_name("IQ_noiseADU_hh")

    if radar.radar_calibration is None:
        raise ValueError(
            "Unable to compute reflectivity. Calibration parameters unknown"
        )

    pol = "hh"
    if signal_field is not None and "vv" in signal_field:
        pol = "vv"

    if pol == "hh":
        if (
            "dBADU_to_dBm_hh" not in radar.radar_calibration
            or "calibration_constant_hh" not in radar.radar_calibration
        ):
            raise ValueError(
                "Unable to compute reflectivity. " + "Calibration parameters unknown"
            )
        dBADU2dBm = radar.radar_calibration["dBADU_to_dBm_hh"]["data"][0]
        radconst = radar.radar_calibration["calibration_constant_hh"]["data"][0]
    else:
        if (
            "dBADU_to_dBm_vv" not in radar.radar_calibration
            or "calibration_constant_vv" not in radar.radar_calibration
        ):
            raise ValueError(
                "Unable to compute reflectivity. " + "Calibration parameters unknown"
            )
        dBADU2dBm = radar.radar_calibration["dBADU_to_dBm_vv"]["data"][0]
        radconst = radar.radar_calibration["calibration_constant_vv"]["data"][0]

    if pol == "hh" and "matched_filter_loss_h" in radar.radar_calibration:
        mfloss = radar.radar_calibration["matched_filter_loss_h"]["data"][0]
    elif pol == "vv" and "matched_filter_loss_v" in radar.radar_calibration:
        mfloss = radar.radar_calibration["matched_filter_loss_v"]["data"][0]
    else:
        warn("Unknown matched filter losses. Assumed 0 dB")
        mfloss = 0.0

    if "path_attenuation" in radar.radar_calibration:
        pathatt = radar.radar_calibration["path_attenuation"]["data"][0]
    else:
        warn("Unknown gas path attenuation. Assumed 0 dB/km")
        pathatt = 0.0

    rangeKm = np.broadcast_to(
        np.atleast_2d(radar.range["data"] / 1000.0), (radar.nrays, radar.ngates)
    )

    noise = None
    if noise_field in radar.fields:
        noise = radar.fields[noise_field]["data"]

    pwr = _compute_power(
        radar.fields[signal_field]["data"], noise=noise, subtract_noise=subtract_noise
    )

    dBZ = (
        10.0 * np.ma.log10(pwr)
        + dBADU2dBm
        + radconst
        + mfloss
        + pathatt * rangeKm
        + 20.0 * np.log10(rangeKm)
    )

    dBZ_field = "reflectivity_" + pol

    dBZ_dict = get_metadata(dBZ_field)
    dBZ_dict["data"] = dBZ

    return dBZ_dict


def compute_st1_iq(radar, signal_field=None):
    """
    Computes the statistical test one lag fluctuation from the horizontal or
    vertical channel IQ data

    Parameters
    ----------
    radar : IQ radar object
        Object containing the required fields
    signal_field : str
        Name of the field that contain the H or V IQ data. None will use the
        default field name in the Py-ART configuration file.

    Returns
    -------
    st1_dict : field dictionary
        Field dictionary containing the st1

    """
    if signal_field is None:
        signal_field = get_field_name("IQ_hh_ADU")

    st1 = _compute_lag_diff(radar, signal_field, is_log=True, lag=1)

    st1_field = "stat_test_lag1"
    if "vv" in signal_field:
        st1_field += "_vv"
    st1_dict = get_metadata(st1_field)
    st1_dict["data"] = st1

    return st1_dict


def compute_st2_iq(radar, signal_field=None):
    """
    Computes the statistical test two lag fluctuation from the horizontal or
    vertical channel IQ data

    Parameters
    ----------
    radar : IQ radar object
        Object containing the required fields
    signal_field : str
        Name of the field that contain the H or V IQ data. None will use the
        default field name in the Py-ART configuration file.

    Returns
    -------
    st2_dict : field dictionary
        Field dictionary containing the st2

    """
    if signal_field is None:
        signal_field = get_field_name("IQ_hh_ADU")

    st2 = _compute_lag_diff(radar, signal_field, is_log=True, lag=2)

    st2_field = "stat_test_lag2"
    if "vv" in signal_field:
        st2_field += "_vv"
    st2_dict = get_metadata(st2_field)
    st2_dict["data"] = st2

    return st2_dict


def compute_wbn_iq(radar, signal_field=None):
    """
    Computes the wide band noise from the horizontal or vertical channel IQ
    data

    Parameters
    ----------
    radar : IQ radar object
        Object containing the required fields
    signal_field : str
        Name of the field that contain the H or V IQ data. None will use the
        default field name in the Py-ART configuration file.

    Returns
    -------
    wbn_dict : field dictionary
        Field dictionary containing the wide band noise

    """
    if signal_field is None:
        signal_field = get_field_name("IQ_hh_ADU")

    rlag0 = _compute_autocorrelation(radar, signal_field, lag=0)
    rlag1 = _compute_autocorrelation(radar, signal_field, lag=1)
    wbn = 20 * np.ma.log10(np.ma.abs(rlag0) / np.ma.abs(rlag1))

    wbn_field = "wide_band_noise"
    if "vv" in signal_field:
        wbn_field += "_vv"
    wbn_dict = get_metadata(wbn_field)
    wbn_dict["data"] = wbn

    return wbn_dict


def compute_differential_reflectivity_iq(
    radar,
    subtract_noise=False,
    lag=0,
    signal_h_field=None,
    signal_v_field=None,
    noise_h_field=None,
    noise_v_field=None,
):
    """
    Computes the differential reflectivity from the horizontal and vertical
    IQ data

    Parameters
    ----------
    radar : IQ radar object
        Object containing the required fields
    subtract_noise : Bool
        If true the noise is subtracted from the power
    lag : int
        Time lag used to compute the differential reflectivity
    signal_h_field, signal_v_field, noise_h_field, noise_v_field : str
        Name of the signal and noise fields. None will use the default field
        name in the Py-ART configuration file.

    Returns
    -------
    zdr_dict : field dictionary
        Field dictionary containing the differential reflectivity

    """
    if signal_h_field is None:
        signal_h_field = get_field_name("IQ_hh_ADU")
    if signal_v_field is None:
        signal_v_field = get_field_name("IQ_vv_ADU")
    if noise_h_field is None:
        noise_h_field = get_field_name("IQ_noiseADU_hh")
    if noise_v_field is None:
        noise_v_field = get_field_name("IQ_noiseADU_vv")

    if radar.radar_calibration is None:
        raise ValueError(
            "Unable to compute spectral reflectivity. "
            + "Calibration parameters unknown"
        )

    if (
        "dBADU_to_dBm_hh" not in radar.radar_calibration
        or "dBADU_to_dBm_vv" not in radar.radar_calibration
        or "calibration_constant_hh" not in radar.radar_calibration
        or "calibration_constant_vv" not in radar.radar_calibration
    ):
        raise ValueError(
            "Unable to compute spectral reflectivity. "
            + "Calibration parameters unknown"
        )

    dBADU2dBm_h = radar.radar_calibration["dBADU_to_dBm_hh"]["data"][0]
    dBADU2dBm_v = radar.radar_calibration["dBADU_to_dBm_vv"]["data"][0]
    radconst_h = radar.radar_calibration["calibration_constant_hh"]["data"][0]
    radconst_v = radar.radar_calibration["calibration_constant_vv"]["data"][0]

    if lag == 0:
        noise = None
        if noise_h_field in radar.fields:
            noise = radar.fields[noise_h_field]["data"]

        pwr_h = _compute_power(
            radar.fields[signal_h_field]["data"],
            noise=noise,
            subtract_noise=subtract_noise,
        )

        noise = None
        if noise_v_field in radar.fields:
            noise = radar.fields[noise_v_field]["data"]

        pwr_v = _compute_power(
            radar.fields[signal_v_field]["data"],
            noise=noise,
            subtract_noise=subtract_noise,
        )
    else:
        pwr_h = np.ma.abs(_compute_autocorrelation(radar, signal_h_field, lag=lag))
        pwr_v = np.ma.abs(_compute_autocorrelation(radar, signal_v_field, lag=lag))

    zdr = (10.0 * np.ma.log10(pwr_h) + dBADU2dBm_h + radconst_h) - (
        10.0 * np.ma.log10(pwr_v) + dBADU2dBm_v + radconst_v
    )

    zdr_dict = get_metadata("differential_reflectivity")
    zdr_dict["data"] = zdr

    return zdr_dict


def compute_mean_phase_iq(radar, signal_field=None):
    """
    Computes the differential phase from the horizontal or vertical channel
    IQ data

    Parameters
    ----------
    radar : IQ radar object
        Object containing the required fields
    signal_field : str
        Name of the field that contain the H or V IQ data. None will use the
        default field name in the Py-ART configuration file.

    Returns
    -------
    mph_dict : field dictionary
        Field dictionary containing the mean phase

    """
    if signal_field is None:
        signal_field = get_field_name("IQ_hh_ADU")

    mph = np.ma.mean(np.ma.angle(radar.fields[signal_field]["data"], deg=True), axis=-1)

    mph_field = "mean_phase"
    if "vv" in signal_field:
        mph_field += "_vv"
    mph_dict = get_metadata(mph_field)
    mph_dict["data"] = mph

    return mph_dict


def compute_differential_phase_iq(
    radar, phase_offset=0.0, signal_h_field=None, signal_v_field=None
):
    """
    Computes the differential phase from the horizontal and vertical channels
    IQ data

    Parameters
    ----------
    radar : IQ radar object
        Object containing the required fields
    phase_offset : float
        system phase offset to add
    signal_h_field, signal_v_field : str
        Name of the fields that contain the H and V IQ data. None will use the
        default field name in the Py-ART configuration file.

    Returns
    -------
    phidp_dict : field dictionary
        Field dictionary containing the differential phase

    """
    if signal_h_field is None:
        signal_h_field = get_field_name("IQ_hh_ADU")
    if signal_v_field is None:
        signal_v_field = get_field_name("IQ_vv_ADU")

    phidp = (
        np.ma.angle(
            np.ma.mean(
                radar.fields[signal_h_field]["data"]
                * np.ma.conjugate(radar.fields[signal_v_field]["data"]),
                axis=-1,
            ),
            deg=True,
        )
        - phase_offset
    )

    if phase_offset != 0:
        phidp = (phidp + 180.0) % 360.0 - 180.0

    phidp_field = "uncorrected_differential_phase"
    phidp_dict = get_metadata(phidp_field)
    phidp_dict["data"] = phidp

    return phidp_dict


def compute_rhohv_iq(
    radar,
    subtract_noise=False,
    lag=0,
    signal_h_field=None,
    signal_v_field=None,
    noise_h_field=None,
    noise_v_field=None,
):
    """
    Computes RhoHV from the horizontal and vertical channels IQ data

    Parameters
    ----------
    radar : IQ radar object
        Object containing the required fields
    subtract_noise : Bool
        If True noise will be subtracted from the signals
    lag : int
        Time lag used in the computation
    signal_h_field, signal_v_field, noise_h_field, noise_v_field : str
        Name of the fields in radar which contains the signal and noise.
        None will use the default field name in the Py-ART configuration file.

    Returns
    -------
    rhohv_dict : field dictionary
        Field dictionary containing the RhoHV

    """
    if signal_h_field is None:
        signal_h_field = get_field_name("IQ_hh_ADU")
    if signal_v_field is None:
        signal_v_field = get_field_name("IQ_vv_ADU")
    if noise_h_field is None:
        noise_h_field = get_field_name("IQ_noiseADU_hh")
    if noise_v_field is None:
        noise_v_field = get_field_name("IQ_noiseADU_vv")

    if lag == 0:
        rhohv = np.ma.abs(
            np.ma.mean(
                radar.fields[signal_h_field]["data"]
                * np.ma.conjugate(radar.fields[signal_v_field]["data"]),
                axis=-1,
            )
        )

        noise = None
        if noise_h_field in radar.fields:
            noise = radar.fields[noise_h_field]["data"]

        pwr_h = _compute_power(
            radar.fields[signal_h_field]["data"],
            noise=noise,
            subtract_noise=subtract_noise,
        )

        noise = None
        if noise_v_field in radar.fields:
            noise = radar.fields[noise_v_field]["data"]

        pwr_v = _compute_power(
            radar.fields[signal_v_field]["data"],
            noise=noise,
            subtract_noise=subtract_noise,
        )
    else:
        rhohv = np.ma.abs(
            _compute_crosscorrelation(radar, signal_h_field, signal_v_field, lag=lag)
        )
        pwr_h = np.ma.abs(_compute_autocorrelation(radar, signal_h_field, lag=lag))
        pwr_v = np.ma.abs(_compute_autocorrelation(radar, signal_v_field, lag=lag))

    rhohv /= np.ma.sqrt(pwr_h * pwr_v)

    rhohv_dict = get_metadata("cross_correlation_ratio")
    rhohv_dict["data"] = rhohv

    return rhohv_dict


def compute_Doppler_velocity_iq(radar, signal_field=None, direction="negative_away"):
    """
    Computes the Doppler velocity from the IQ data

    Parameters
    ----------
    radar : IQ radar object
        Object containing the required fields
    signal_field : str
        Name of the field in the radar which contains the signal.
        None will use the default field name in the Py-ART configuration file.
    direction : str
        The convention used in the Doppler mean field. Can be negative_away or
        negative_towards

    Returns
    -------
    vel_dict : field dictionary
        Field dictionary containing the Doppler velocity

    """
    if signal_field is None:
        signal_field = get_field_name("IQ_hh_ADU")

    if radar.instrument_parameters is None:
        warn("Unable to compute Doppler velocity. No instrument information")
        return None

    if (
        "prt" not in radar.instrument_parameters
        or "frequency" not in radar.instrument_parameters
    ):
        warn("Unable to compute Doppler velocity. " "Unknown PRT or radar frequency")
        return None

    rlag_1 = _compute_autocorrelation(radar, signal_field, lag=1)
    prt = np.broadcast_to(
        np.expand_dims(radar.instrument_parameters["prt"]["data"], axis=1),
        (radar.nrays, radar.ngates),
    )
    wavelength = speed_of_light / radar.instrument_parameters["frequency"]["data"][0]
    mean_vel = np.ma.angle(rlag_1) / (2 * np.pi * prt) * (wavelength / 2.0)

    if direction == "negative_away":
        mean_vel = -mean_vel

    vel_field = "velocity"
    if "vv" in signal_field:
        vel_field += "_vv"
    vel_dict = get_metadata(vel_field)
    vel_dict["data"] = mean_vel

    return vel_dict


def compute_Doppler_width_iq(
    radar, subtract_noise=True, signal_field=None, noise_field=None, lag=1
):
    """
    Computes the Doppler width from the IQ data

    Parameters
    ----------
    radar : Radar radar object
        Object containing the required fields
    subtract_noise : Bool
        If True noise will be subtracted from the signals
    lag : int
        Time lag used in the denominator of the computation
    signal_field, noise_field : str
        Name of the field in the radar which contains the signal and noise.
        None will use the default field name in the Py-ART configuration file.

    Returns
    -------
    width_dict : field dictionary
        Field dictionary containing the Doppler spectrum width

    Reference
    ---------
    lag 1 algorithm based on:
        Time-Domain Computation of Mean and Variance of Doppler Spectra
        R. C. Srivastava, A. R. Jameson, and P. H. Hildebrand
        Journal of Applied Meteorology February 1979, Vol. 18, No. 2

    """
    if signal_field is None:
        signal_field = get_field_name("IQ_hh_ADU")
    if noise_field is None:
        noise_field = get_field_name("IQ_noiseADU_hh")

    prt = np.broadcast_to(
        np.expand_dims(radar.instrument_parameters["prt"]["data"], axis=1),
        (radar.nrays, radar.ngates),
    )
    wavelength = speed_of_light / radar.instrument_parameters["frequency"]["data"][0]

    if lag == 0:
        noise = None
        if noise_field in radar.fields:
            noise = radar.fields[noise_field]["data"]

        rlag_1 = _compute_autocorrelation(radar, signal_field, lag=1)
        pwr = _compute_power(
            radar.fields[signal_field]["data"],
            noise=noise,
            subtract_noise=subtract_noise,
        )

        width = (
            np.ma.sqrt(np.ma.log(np.ma.abs(pwr) / np.ma.abs(rlag_1)))
            / (np.sqrt(2.0) * np.pi * prt)
            * (wavelength / 2.0)
        )
    else:
        rlag_2 = _compute_autocorrelation(radar, signal_field, lag=2)
        rlag_1 = _compute_autocorrelation(radar, signal_field, lag=1)

        width = (
            np.ma.sqrt(np.ma.log(np.ma.abs(rlag_1) / np.ma.abs(rlag_2)))
            / (np.sqrt(6.0) * np.pi * prt)
            * (wavelength / 2.0)
        )

    width_field = "spectrum_width"
    if "vv" in signal_field:
        width_field += "_vv"
    width_dict = get_metadata(width_field)
    width_dict["data"] = width

    return width_dict


def _compute_power(signal, noise=None, subtract_noise=False):
    """
    Compute the signal power in linear units

    Parameters
    ----------
    signal : float array
        The IQ signal
    noise : float array
        The noise power per pulse
    subtract_noise : Bool
        If True and noise not None the noise power will be subtracted from the
        signal power

    Returns
    -------
    pwr : float array
        The computed signal power

    """
    pwr = np.ma.mean(np.ma.power(np.ma.abs(signal), 2.0), axis=-1)

    if subtract_noise and noise is not None:
        noise_gate = np.ma.mean(noise, axis=-1)
        pwr -= noise_gate
        pwr[pwr < 0.0] = np.ma.masked

    return pwr


def _compute_autocorrelation(radar, signal_field, lag=1):
    """
    Compute the signal autocorrelation in linear units

    Parameters
    ----------
    radar : IQ radar object
        The radar object containing the fields
    signal_field : str
        The IQ signal
    lag : int
        Time lag to compute

    Returns
    -------
    rlag : float array
        The computed autocorrelation lag

    """
    rlag = np.ma.masked_all((radar.nrays, radar.ngates), dtype=np.complex64)
    for ray, npulses in enumerate(radar.npulses["data"]):
        if lag >= npulses:
            warn("lag larger than number of pulses in ray")
            continue
        rlag[ray, :] = np.ma.mean(
            np.ma.conjugate(
                radar.fields[signal_field]["data"][ray, :, 0 : npulses - lag]
            )
            * radar.fields[signal_field]["data"][ray, :, lag:npulses],
            axis=-1,
        )

    return rlag


def _compute_lag_diff(radar, signal_field, is_log=True, lag=1):
    """
    Compute the signal autocorrelation in linear units

    Parameters
    ----------
    radar : IQ radar object
        The radar object containing the fields
    signal_field : str
        The IQ signal
    lag : int
        Time lag to compute

    Returns
    -------
    rlag : float array
        The computed autocorrelation lag

    """
    rlag = np.ma.masked_all((radar.nrays, radar.ngates))
    pwr = np.ma.power(np.ma.abs(radar.fields[signal_field]["data"]), 2.0)
    if is_log:
        pwr = 10.0 * np.ma.log10(pwr)
    for ray, npulses in enumerate(radar.npulses["data"]):
        if lag >= npulses:
            warn("lag larger than number of pulses in ray")
            continue

        rlag[ray, :] = np.ma.mean(
            np.ma.abs(pwr[ray, :, 0 : npulses - lag] - pwr[ray, :, lag:npulses]),
            axis=-1,
        )

    return rlag


def _compute_crosscorrelation(radar, signal_h_field, signal_v_field, lag=1):
    """
    Compute the cross-correlation between H and V in linear units

    Parameters
    ----------
    radar : IQ radar object
        The radar object containing the fields
    signal_h_field, signal_v_field : str
        The IQ H and V signal names
    lag : int
        Time lag to compute

    Returns
    -------
    rlag : float array
        The computed cross-correlation lag

    """
    rlag = np.ma.masked_all((radar.nrays, radar.ngates), dtype=np.complex64)
    for ray, npulses in enumerate(radar.npulses["data"]):
        if lag >= npulses:
            warn("lag larger than number of pulses in ray")
            continue
        rlag[ray, :] = np.ma.mean(
            radar.fields[signal_h_field]["data"][ray, :, 0 : npulses - lag]
            * np.ma.conjugate(
                radar.fields[signal_v_field]["data"][ray, :, lag:npulses]
            ),
            axis=-1,
        )

    return rlag
