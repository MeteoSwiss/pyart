"""
pyart.util.stats_utils
=========================

Functions for computing weighted statistical moments.

.. autosummary::
    :toctree: generated/
    weighted_mean
    weighted_variance
    weighted_skew
    weighted_kurtosis
    get_statistic

"""

import re
import warnings
import numpy as np


def weighted_mean(var, wts):
    """Calculates the weighted mean"""
    return np.average(var, weights=wts)


def weighted_variance(var, wts):
    """Calculates the weighted variance"""
    return np.average((var - weighted_mean(var, wts)) ** 2, weights=wts)


def weighted_skew(var, wts):
    """Calculates the weighted skewness"""
    return np.average(
        (var - weighted_mean(var, wts)) ** 3, weights=wts
    ) / weighted_variance(var, wts) ** (1.5)


def weighted_kurtosis(var, wts):
    """Calculates the weighted skewness"""
    return np.average(
        (var - weighted_mean(var, wts)) ** 4, weights=wts
    ) / weighted_variance(var, wts) ** (2)


def get_statistic(stats, weighted):
    """Retrieves the correct function given name of stats and
    whether the stats should be weigted or not"""
    if stats == "mean":
        if weighted:
            myfunc = lambda var, wts: weighted_mean(var, wts)
        else:
            myfunc = lambda var, wts: weighted_mean(var, np.ones(var.shape))
    elif stats == "std":
        if weighted:
            myfunc = lambda var, wts: np.sqrt(weighted_variance(var, wts))
        else:
            myfunc = lambda var, wts: np.sqrt(
                weighted_variance(var, np.ones(var.shape))
            )
    elif stats == "skewness":
        if weighted:
            myfunc = lambda var, wts: np.sqrt(weighted_skew(var, wts))
        else:
            myfunc = lambda var, wts: np.sqrt(weighted_skew(var, np.ones(var.shape)))
    elif stats == "kurtosis":
        if weighted:
            myfunc = lambda var, wts: np.sqrt(weighted_kurtosis(var, wts))
        else:
            myfunc = lambda var, wts: np.sqrt(
                weighted_kurtosis(var, np.ones(var.shape))
            )
    elif stats == "max":
        if weighted:
            warnings.warn("Sorry currently, max and min statistics cannot be weighted!")
        myfunc = lambda var, wts: np.max(var)
    elif stats == "min":
        if weighted:
            warnings.warn("Sorry currently, max and min statistics cannot be weighted!")
        myfunc = lambda var, wts: np.min(var)
    elif qval := re.findall("[Qq](\d{2})", stats)[0]:
        if weighted:
            myfunc = lambda var, wts: np.percentile(
                var, q=int(qval), weights=wts, method="inverted_cdf"
            )
        else:
            myfunc = lambda var, wts: np.percentile(var, q=int(qval))
    return myfunc
