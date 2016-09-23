"""
pyart.correct.bias_and_noise
===================

Corrects polarimetric variables for noise

.. autosummary::
    :toctree: generated/

    correct_noise_rhohv
    correct_bias
    selfconsistency_bias
    selfconsistency_kdp_phidp
    _selfconsistency_kdp_phidp
    get_kdp_selfcons

"""

import numpy as np

from ..config import get_metadata, get_field_name, get_fillvalue
from .attenuation import get_mask_fzl
from .phase_proc import smooth_masked


def correct_noise_rhohv(radar, urhohv_field=None, snr_field=None,
                        zdr_field=None, nh_field=None, nv_field=None,
                        rhohv_field=None):
    """
    Corrects RhoHV for noise according to eq. 6 in Gourley et al. 2006.
    This correction should only be performed if noise has not been subtracted
    from the signal during the moments computation.

    Parameters
    ----------
    radar : Radar
        radar object

    urhohv_field : str
        name of the RhoHV uncorrected for noise field

    snr_field, zdr_field, nh_field, nv_field: str
        names of the SNR, ZDR, horizontal channel noise in dBZ and vertical
        channel noise in dBZ used to correct RhoHV

    rhohv_field: str
        name of the rhohv field to output

    Returns
    -------
    rhohv : dict
        noise corrected RhoHV field

    References
    ----------
    Gourley et al. Data Quality of the Meteo-France C-Band Polarimetric
    Radar, JAOT, 23, 1340-1356

    """
    # parse the field parameters
    if urhohv_field is None:
        urhohv_field = get_field_name('uncorrected_cross_correlation_ratio')
    if snr_field is None:
        snr_field = get_field_name('signal_to_noise_ratio')
    if zdr_field is None:
        zdr_field = get_field_name('differential_reflectivity')
    if nh_field is None:
        nh_field = get_field_name('noisedBZ_hh')
    if nv_field is None:
        nv_field = get_field_name('noisedBZ_vv')
    if rhohv_field is None:
        rhohv_field = get_field_name('cross_correlation_ratio')

    # extract fields from radar
    radar.check_field_exists(urhohv_field)
    radar.check_field_exists(snr_field)
    radar.check_field_exists(zdr_field)
    radar.check_field_exists(nh_field)
    radar.check_field_exists(nv_field)

    urhohv = radar.fields[urhohv_field]['data']
    snrdB_h = radar.fields[snr_field]['data']
    zdrdB = radar.fields[zdr_field]['data']
    nh = radar.fields[nh_field]['data']
    nv = radar.fields[nv_field]['data']

    snr_h = np.ma.power(10., 0.1*snrdB_h)
    zdr = np.ma.power(10., 0.1*zdrdB)
    alpha = np.ma.power(10., 0.1*(nh-nv))

    rhohv_data = urhohv*np.ma.sqrt((1.+1./snr_h)*(1.+zdr/(alpha*snr_h)))
    rhohv_data[rhohv_data > 1.] = 1.

    rhohv = get_metadata(rhohv_field)
    rhohv['data'] = rhohv_data

    return rhohv


def correct_bias(radar, bias=0., field_name=None):
    """
    Corrects a radar data bias. If field name is none the correction is
    applied to horizontal reflectivity by default

    Parameters
    ----------
    radar : Radar
        radar object

    bias : float
        the bias magnitude

    field_name: str
        names of the field to be corrected

    Returns
    -------
    corrected_field : dict
        The corrected field

    """
    # parse the field parameters
    if field_name is None:
        field_name = get_field_name('reflectivity')

    # extract fields from radar
    radar.check_field_exists(field_name)
    field_data = radar.fields[field_name]['data']

    corr_field_data = field_data - bias

    if field_name.startswith('corrected_'):
        corr_field_name = field_name
    else:
        corr_field_name = 'corrected_'+field_name

    corr_field = get_metadata(corr_field_name)
    corr_field['data'] = corr_field_data

    return corr_field


def selfconsistency_bias(
        radar, zdr_kdpzh_table, min_rhohv=0.92, max_phidp=20.,
        smooth_wind_len=5, doc=None, fzl=None, min_rcons=20, dphidp_min=2,
        dphidp_max=16, refl_field=None, phidp_field=None, zdr_field=None,
        temp_field=None, rhohv_field=None):
    """
    Estimates reflectivity bias at each ray using the self-consistency
    algorithm by Gourley

    Parameters
    ----------
    radar : Radar
        radar object
    zdr_kdpzh_table : ndarray 2D
        look up table relating ZDR with KDP/Zh
    min_rhohv : float
        minimum RhoHV value to consider the data valid
    max_phidp : float
        maximum PhiDP value to consider the data valid
    smooth_wind_len : int
        length of the smoothing window
    doc : float
        Number of gates at the end of each ray to to remove from the
        calculation.
    fzl : float
        Freezing layer, gates above this point are not included in the
        correction.
    min_rcons : int
        minimum number of consecutive gates to consider a valid segment of
        PhiDP
    dphidp_min : float
        minimum differential phase shift in a segment
    dphidp_max : float
        maximum differential phase shift in a segment
    refl_field, phidp_field, zdr_field : str
        Field names within the radar object which represent the reflectivity,
        differential phase and differential reflectivity fields. A value of
        None will use the default field name as defined in the Py-ART
        configuration file.
    temp_field, rhohv_field : str
        Field names within the radar object which represent the temperature,
        and co-polar correlation fields. A value of None will use the default
        field name as defined in the Py-ART configuration file. They are going
        to be used only if available.
    kdpsim_field, phidpsim_field : str
        Field names which represent the estimated specific differential phase
        and differential phase. A value of None will use the default
        field name as defined in the Py-ART configuration file.

    Returns
    -------
    refl_bias_dict : dict
        the bias at each ray field and metadata

    """
    # parse the field parameters
    if refl_field is None:
        refl_field = get_field_name('reflectivity')
    if zdr_field is None:
        zdr_field = get_field_name('zdr')
    if phidp_field is None:
        # use corrrected_differential_phase or unfolded_differential_phase
        # fields if they are available, if not use differential_phase field
        phidp_field = get_field_name('corrected_differential_phase')
        if phidp_field not in radar.fields:
            phidp_field = get_field_name('unfolded_differential_phase')
        if phidp_field not in radar.fields:
            phidp_field = get_field_name('differential_phase')
    if temp_field is None:
        temp_field = get_field_name('temperature')
    if rhohv_field is None:
        rhohv_field = get_field_name('cross_correlation_ratio')

    # extract fields from radar, refl, zdr and phidp must exist
    radar.check_field_exists(refl_field)
    refl = radar.fields[refl_field]['data']

    radar.check_field_exists(zdr_field)
    zdr = radar.fields[zdr_field]['data']

    radar.check_field_exists(phidp_field)
    phidp = radar.fields[phidp_field]['data']

    if min_rhohv is not None:
        try:
            radar.check_field_exists(rhohv_field)
            rhohv = radar.fields[rhohv_field]['data']
        except KeyError:
            min_rhohv = None
            rhohv = None

    kdp_sim, phidp_sim = _selfconsistency_kdp_phidp(
        radar, refl, zdr, phidp, zdr_kdpzh_table, max_phidp=max_phidp,
        smooth_wind_len=smooth_wind_len, rhohv=rhohv, min_rhohv=min_rhohv,
        doc=doc, fzl=fzl, temp_field=temp_field)

    refl_bias = np.ma.zeros((radar.nrays, 1))
    refl_bias[:] = np.ma.masked
    for ray in range(radar.nrays):
        # split ray in consecutive valid range bins
        isprec = np.ma.getmaskarray(phidp_sim[ray, :]) == 0
        ind_prec = np.where(isprec)[0]
        cons_list = np.split(ind_prec, np.where(np.diff(ind_prec) != 1)[0]+1)

        # check if there is a cell long enough
        found_cell = False
        for ind_prec_cell in cons_list:
            if len(ind_prec_cell) >= min_rcons:
                found_cell = True
                break
        # check if increase in phidp is within limits and compute reflectivity
        # bias
        if found_cell:
            dphidp_obs = (phidp[ray, ind_prec_cell[-1]] -
                          phidp[ray, ind_prec_cell[0]])
            if dphidp_obs >= dphidp_min:
                for i in range(len(ind_prec_cell)):
                    dphidp_obs = (phidp[ray, ind_prec_cell[-1-i]] -
                                  phidp[ray, ind_prec_cell[0]])
                    if dphidp_obs <= dphidp_max:
                        dphidp_sim = (phidp_sim[ray, ind_prec_cell[-1]] -
                                      phidp_sim[ray, ind_prec_cell[0]])
                        refl_bias[ray] = 10.*np.ma.log10(dphidp_sim/dphidp_obs)
                        break

    refl_bias_dict = get_metadata('reflectivity_bias')
    refl_bias_dict['data'] = refl_bias
    return refl_bias_dict


def selfconsistency_kdp_phidp(
        radar, zdr_kdpzh_table, min_rhohv=0.92, max_phidp=20.,
        smooth_wind_len=5, doc=None, fzl=None, refl_field=None,
        phidp_field=None, zdr_field=None, temp_field=None, rhohv_field=None,
        kdpsim_field=None, phidpsim_field=None):
    """
    Estimates KDP and PhiDP in rain from  Zh and ZDR using a selfconsistency
    relation between ZDR, Zh and KDP. Private method

    Parameters
    ----------
    radar : Radar
        radar object
    zdr_kdpzh_table : ndarray 2D
        look up table relating ZDR with KDP/Zh
    min_rhohv : float
        minimum RhoHV value to consider the data valid
    max_phidp : float
        maximum PhiDP value to consider the data valid
    smooth_wind_len : int
        length of the smoothing window
    doc : float
        Number of gates at the end of each ray to to remove from the
        calculation.
    fzl : float
        Freezing layer, gates above this point are not included in the
        correction.
    refl_field, phidp_field, zdr_field : str
        Field names within the radar object which represent the reflectivity,
        differential phase and differential reflectivity fields. A value of
        None will use the default field name as defined in the Py-ART
        configuration file.
    temp_field, rhohv_field : str
        Field names within the radar object which represent the temperature,
        and co-polar correlation fields. A value of None will use the default
        field name as defined in the Py-ART configuration file. They are going
        to be used only if available.
    kdpsim_field, phidpsim_field : str
        Field names which represent the estimated specific differential phase
        and differential phase. A value of None will use the default
        field name as defined in the Py-ART configuration file.

    Returns
    -------
    kdp_sim_dict, phidp_sim_dict : dict
        the KDP and PhiDP estimated fields and metadata

    """
    # parse the field parameters
    if refl_field is None:
        refl_field = get_field_name('reflectivity')
    if zdr_field is None:
        zdr_field = get_field_name('zdr')
    if phidp_field is None:
        # use corrrected_differential_phase or unfolded_differential_phase
        # fields if they are available, if not use differential_phase field
        phidp_field = get_field_name('corrected_differential_phase')
        if phidp_field not in radar.fields:
            phidp_field = get_field_name('unfolded_differential_phase')
        if phidp_field not in radar.fields:
            phidp_field = get_field_name('differential_phase')
    if temp_field is None:
        temp_field = get_field_name('temperature')
    if rhohv_field is None:
        rhohv_field = get_field_name('cross_correlation_ratio')
    if kdpsim_field is None:
        kdpsim_field = get_field_name('specific_differential_phase')
    if phidpsim_field is None:
        phidpsim_field = get_field_name('differential_phase')

    # extract fields from radar, refl, zdr and phidp must exist
    radar.check_field_exists(refl_field)
    refl = radar.fields[refl_field]['data']

    radar.check_field_exists(zdr_field)
    zdr = radar.fields[zdr_field]['data']

    radar.check_field_exists(phidp_field)
    phidp = radar.fields[phidp_field]['data']

    if min_rhohv is not None:
        try:
            radar.check_field_exists(rhohv_field)
            rhohv = radar.fields[rhohv_field]['data']
        except KeyError:
            min_rhohv = None
            rhohv = None

    kdp_sim, phidp_sim = _selfconsistency_kdp_phidp(
        radar, refl, zdr, phidp, zdr_kdpzh_table, max_phidp=max_phidp,
        smooth_wind_len=smooth_wind_len, rhohv=rhohv, min_rhohv=min_rhohv,
        doc=doc, fzl=fzl, temp_field=temp_field)

    kdp_sim_dict = get_metadata(kdpsim_field)
    kdp_sim_dict['data'] = kdp_sim

    phidp_sim_dict = get_metadata(phidpsim_field)
    phidp_sim_dict['data'] = phidp_sim

    return kdp_sim_dict, phidp_sim_dict


def _selfconsistency_kdp_phidp(
        radar, refl, zdr, phidp, zdr_kdpzh_table, max_phidp=20.,
        smooth_wind_len=5, rhohv=None, min_rhohv=None, doc=None, fzl=None,
        temp_field=None):
    """
    Estimates KDP and PhiDP in rain from  Zh and ZDR using a selfconsistency
    relation between ZDR, Zh and KDP. Private method

    Parameters
    ----------
    radar : Radar
        radar object
    refl, zdr, phidp : ndarray 2D
        reflectivity field, differential reflectivity field and differential
        phase field. They must exist
    zdr_kdpzh_table : ndarray 2D
        look up table relating ZDR with KDP/Zh
    rhohv : ndarray 2D
        copolar correlation field used for masking data. Optional
    max_phidp : float
        maximum PhiDP value to consider the data valid
    smooth_wind_len : int
        length of the smoothing window for Zh and ZDR data
    min_rhohv : float
        minimum RhoHV value to consider the data valid
    doc : float
        Number of gates at the end of each ray to to remove from the
        calculation.
    fzl : float
        Freezing layer, gates above this point are not included in the
        correction.
    temp_field : str
        Field name within the radar object which represent the temperature
        field. A value of None will use the default field name as defined in
        the Py-ART configuration file. It is going to be used only if
        available.

    Returns
    -------
    kdp_sim, phidp_sim : ndarray 2D
        the KDP and PhiDP estimated fields

    """
    # smooth reflectivity and ZDR
    if smooth_wind_len > 0:
        sm_refl = smooth_masked(refl, wind_len=smooth_wind_len, min_valid=1,
                                wind_type='mean')
        sm_zdr = smooth_masked(zdr, wind_len=smooth_wind_len, min_valid=1,
                               wind_type='mean')
    else:
        sm_refl = refl
        sm_zdr = zdr

    # determine the valid data (i.e. data below the melting layer)
    mask = np.ma.getmaskarray(refl)

    mask_fzl, end_gate_arr = get_mask_fzl(
        radar, fzl=fzl, doc=doc, min_temp=0., thickness=500.,
        temp_field=temp_field)
    mask = np.logical_or(mask, mask_fzl)

    mask_zdr = np.logical_or(sm_zdr < 0., np.ma.getmaskarray(sm_zdr))
    mask_zdr = np.logical_or(mask_zdr, sm_zdr > zdr_kdpzh_table[0, -1])
    mask = np.logical_or(mask, mask_zdr)

    mask_phidp = np.logical_or(phidp > max_phidp, np.ma.getmaskarray(phidp))
    mask = np.logical_or(mask, mask_phidp)

    if min_rhohv is not None:
        mask_rhohv = np.logical_or(rhohv < min_rhohv,
                                   np.ma.getmaskarray(rhohv))
        mask = np.logical_or(mask, mask_rhohv)

    corr_refl = np.ma.masked_where(mask, sm_refl)
    corr_zdr = np.ma.masked_where(mask, sm_zdr)

    kdp_sim = get_kdp_selfcons(corr_zdr, corr_refl, zdr_kdpzh_table)
    dr = (radar.range['data'][1] - radar.range['data'][0]) / 1000.0
    phidp_sim = np.ma.cumsum(2*dr*kdp_sim, axis=1)

    return kdp_sim, phidp_sim


def get_kdp_selfcons(zdr, refl, zdr_kdpzh_table):
    """
    Estimates KDP and PhiDP in rain from  Zh and ZDR using a selfconsistency
    relation between ZDR, Zh and KDP

    Parameters
    ----------
    zdr, refl : ndarray 2D
        reflectivity and differential reflectivity fields
    zdr_kdpzh_table : ndarray 2D
        look up table relating ZDR with KDP/Zh

    Returns
    -------
    kdp_sim : ndarray 2D
        the KDP estimated from zdr and refl

    """
    # prepare output
    kdpzh = np.ma.zeros(np.shape(zdr))
    kdpzh[:] = np.ma.masked

    refl_lin = np.ma.power(10., refl/10.)
    zdr_mask = np.ma.getmaskarray(zdr)

    zdr_valid = zdr.compressed()
    if np.size(zdr_valid) < 1:
        warn('No valid data for selfconsistency retrieval')
        return kdpzh

    # sort ZDR
    zdr_sorted = np.sort(zdr_valid)
    ind_zdr_sorted = np.argsort(zdr_valid)

    # get the values of kdp/zh as linear interpolation of the table
    kdpzh_valid = np.interp(
        zdr_sorted, zdr_kdpzh_table[0, :], zdr_kdpzh_table[1, :])

    # reorder according to original order of the flat valid data array
    kdpzh_valid[ind_zdr_sorted] = kdpzh_valid

    kdpzh[~zdr_mask] = kdpzh_valid

    return refl_lin * kdpzh
