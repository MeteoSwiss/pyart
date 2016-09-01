"""
pyart.util.xsect
================

Function for extracting cross sections from radar volumes.

.. autosummary::
    :toctree: generated/

    cross_section_ppi
    cross_section_rhi
    _construct_xsect_radar
    _copy_dic

"""

from copy import copy

import numpy as np

from ..core import Radar


def cross_section_ppi(radar, target_azimuths):
    """
    Extract cross sections from a PPI volume along one or more azimuth angles.

    Parameters
    ----------
    radar : Radar
        Radar volume containing PPI sweeps from which azimuthal
        cross sections will be extracted.
    target_azimuth : list
        Azimuthal angles in degrees where cross sections will be taken.

    Returns
    -------
    radar_rhi : Radar
        Radar volume containing RHI sweeps which contain azimuthal
        cross sections from the original PPI volume.

    """

    # determine which rays from the ppi radar make up the pseudo RHI
    prhi_rays = []
    rhi_nsweeps = len(target_azimuths)

    for target_azimuth in target_azimuths:
        for sweep_slice in radar.iter_slice():
            sweep_azimuths = radar.azimuth['data'][sweep_slice]
            ray_number = np.argmin(np.abs(sweep_azimuths - target_azimuth))
            prhi_rays.append(ray_number + sweep_slice.start)

    radar_rhi = _construct_xsect_radar(radar, 'rhi', prhi_rays, rhi_nsweeps)

    return radar_rhi


def cross_section_rhi(radar, target_elevations):
    """
    Extract cross sections from an RHI volume along one or more elevation
    angles.

    Parameters
    ----------
    radar : Radar
        Radar volume containing RHI sweeps from which azimuthal
        cross sections will be extracted.
    target_elevations : list
        Elevation angles in degrees where cross sections will be taken.

    Returns
    -------
    radar_ppi : Radar
        Radar volume containing PPI sweeps which contain azimuthal
        cross sections from the original RHI volume.

    """

    # determine which rays from the rhi radar make up the pseudo PPI
    pppi_rays = []
    ppi_nsweeps = len(target_elevations)

    for target_elevation in target_elevations:
        for sweep_slice in radar.iter_slice():
            sweep_elevations = radar.elevation['data'][sweep_slice]
            ray_number = np.argmin(np.abs(sweep_elevations - target_elevation))
            pppi_rays.append(ray_number + sweep_slice.start)

    radar_ppi = _construct_xsect_radar(radar, 'ppi', pppi_rays, ppi_nsweeps)

    return radar_ppi


def _construct_xsect_radar(radar, scan_type, pxsect_rays, xsect_nsweeps):
    """
    Constructs a new radar object that contains cross-sections at fixed angles
    of a PPI or RHI volume scan.

    Parameters
    ----------
    radar : Radar
        Radar volume containing RHI/PPI sweeps from which a
        cross sections will be extracted.
    scan_type : str
        Type of cross section scan (ppi or rhi)
    pxsect_rays : list
        list of rays from the radar volume to be copied in the cross-sections
        radar object
    xsect_nsweeps : int
        Number of sweeps in the cross-section radar

    Returns
    -------
    radar_xsect : Radar
        Radar volume containing sweeps which contain cross sections from the
        original volume.

    """
    _range = _copy_dic(radar.range)
    latitude = _copy_dic(radar.latitude)
    longitude = _copy_dic(radar.longitude)
    altitude = _copy_dic(radar.altitude)
    metadata = _copy_dic(radar.metadata)

    time = _copy_dic(radar.time, excluded_keys=['data'])
    time['data'] = radar.time['data'][pxsect_rays].copy()

    azimuth = _copy_dic(radar.azimuth, excluded_keys=['data'])
    azimuth['data'] = radar.azimuth['data'][pxsect_rays].copy()

    elevation = _copy_dic(radar.elevation, excluded_keys=['data'])
    elevation['data'] = radar.elevation['data'][pxsect_rays].copy()

    fields = {}
    for field_name, orig_field_dic in radar.fields.items():
        field_dic = _copy_dic(orig_field_dic, excluded_keys=['data'])
        field_dic['data'] = orig_field_dic['data'][pxsect_rays].copy()
        fields[field_name] = field_dic

    sweep_number = _copy_dic(radar.sweep_number, excluded_keys=['data'])
    sweep_number['data'] = np.arange(xsect_nsweeps, dtype='int32')

    sweep_mode = _copy_dic(radar.sweep_mode, excluded_keys=['data'])
    sweep_mode['data'] = np.array([scan_type]*xsect_nsweeps)

    fixed_angle = _copy_dic(radar.fixed_angle, excluded_keys=['data'])
    fixed_angle['data'] = np.array(target_elevations, dtype='float32')

    sweep_start_ray_index = _copy_dic(
        radar.sweep_start_ray_index, excluded_keys=['data'])
    ssri = np.arange(xsect_nsweeps, dtype='int32') * radar.nsweeps
    sweep_start_ray_index['data'] = ssri

    sweep_end_ray_index = _copy_dic(
        radar.sweep_end_ray_index, excluded_keys=['data'])
    seri = (np.arange(xsect_nsweeps, dtype='int32') *
            radar.nsweeps + radar.nsweeps-1)
    sweep_end_ray_index['data'] = seri

    radar_xsect = Radar(
        time, _range, fields, metadata, scan_type,
        latitude, longitude, altitude,
        sweep_number, sweep_mode, fixed_angle,
        sweep_start_ray_index, sweep_end_ray_index,
        azimuth, elevation)

    return radar_xsect


def _copy_dic(orig_dic, excluded_keys=None):
    """ Return a copy of the original dictionary copying each element. """
    if excluded_keys is None:
        excluded_keys = []
    dic = {}
    for k, v in orig_dic.items():
        if k not in excluded_keys:
            dic[k] = copy(v)
    return dic
