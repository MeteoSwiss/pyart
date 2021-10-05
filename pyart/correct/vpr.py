"""
pyart.correct.vpr
=================

Computes and corrects the vertical profile of reflectivity

.. autosummary::
    :toctree: generated/

    correct_vpr


"""
from ..config import get_metadata, get_field_name


def correct_vpr(radar, refl_field=None):
    """
    Correct VPR

    Parameters
    ----------
    radar : Radar
        Radar object.
    refl_field : str
        Name of the reflectivity field to correct

    Returns
    -------
    corr_refl : dict
        The corrected reflectivity

    """
    # parse the field parameters
    if refl_field is None:
        refl_field = get_field_name('reflectivity')

    # extract fields from radar
    radar.check_field_exists(refl_field)
    refl = radar.fields[refl_field]['data']

    refl_corr = get_metadata('corrected_reflectivity')
    refl_corr['data'] = refl

    return refl_corr
    