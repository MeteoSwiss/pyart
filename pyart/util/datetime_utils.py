"""
pyart.util.datetime_utils
=========================

Functions for converting date and time between various forms.

.. autosummary::
    :toctree: generated/
    datetime_from_radar
    datetimes_from_radar
    datetime_from_dataset
    datetimes_from_dataset
    datetime_from_grid

"""
from datetime import datetime, timezone

import numpy as np

try:
    from cftime import date2num, num2date
except ImportError:
    from netCDF4 import date2num, num2date

EPOCH_UNITS = "seconds since 1970-01-01T00:00:00Z"


def make_utc_aware(cftime_dt):
    if not hasattr(cftime_dt, "__len__"):
        if len(str(cftime_dt)) == 19:
            fmt = "%Y-%m-%d %H:%M:%S"
        else:
            fmt = "%Y-%m-%d %H:%M:%S.%f"
        out = datetime.strptime(str(cftime_dt), fmt).replace(tzinfo=timezone.utc)
    else:
        out = np.array(
            [
                datetime.strptime(
                    str(cf),
                    "%Y-%m-%d %H:%M:%S"
                    if len(str(cf)) == 19
                    else "%Y-%m-%d %H:%M:%S.%f",
                ).replace(tzinfo=timezone.utc)
                for cf in cftime_dt
            ]
        )
    return out


def datetime_from_radar(radar, epoch=False, **kwargs):
    """Return a datetime for the first ray in a Radar."""
    if epoch:
        dtrad = num2date(radar.time["data"][0], radar.time["units"])
        epnum = date2num(dtrad, EPOCH_UNITS)
        return make_utc_aware(num2date(epnum, EPOCH_UNITS, **kwargs))
    else:
        return make_utc_aware(
            num2date(radar.time["data"][0], radar.time["units"], **kwargs)
        )


def datetimes_from_radar(radar, epoch=False, **kwargs):
    """Return an array of datetimes for the rays in a Radar."""
    if epoch:
        dtrad = num2date(radar.time["data"][:], radar.time["units"])
        epnum = date2num(dtrad, EPOCH_UNITS)
        return make_utc_aware(num2date(epnum, EPOCH_UNITS, **kwargs))
    else:
        return make_utc_aware(
            num2date(radar.time["data"][:], radar.time["units"], **kwargs)
        )


def datetime_from_dataset(dataset, epoch=False, **kwargs):
    """Return a datetime for the first time in a netCDF Dataset."""
    if epoch:
        dtdata = make_utc_aware(
            num2date(dataset.variables["time"][0], dataset.variables["time"].units)
        )
        epnum = date2num(dtdata, EPOCH_UNITS)
        return make_utc_aware(num2date(epnum, EPOCH_UNITS, **kwargs))
    else:
        return make_utc_aware(
            num2date(
                dataset.variables["time"][0], dataset.variables["time"].units, **kwargs
            )
        )


def datetimes_from_dataset(dataset, epoch=False, **kwargs):
    """Return an array of datetimes for the times in a netCDF Dataset."""
    if epoch:
        dtdata = num2date(dataset.variables["time"][:], dataset.variables["time"].units)
        epnum = date2num(dtdata, EPOCH_UNITS)
        return make_utc_aware(num2date(epnum, EPOCH_UNITS, **kwargs))
    else:
        return make_utc_aware(
            num2date(
                dataset.variables["time"][:], dataset.variables["time"].units, **kwargs
            )
        )


def datetime_from_grid(grid, epoch=False, **kwargs):
    """Return a datetime for the volume start in a Grid."""
    if epoch:
        dtrad = num2date(grid.time["data"][0], grid.time["units"])
        epnum = date2num(dtrad, EPOCH_UNITS)
        return make_utc_aware(num2date(epnum, EPOCH_UNITS, **kwargs))
    else:
        return make_utc_aware(
            num2date(grid.time["data"][0], grid.time["units"], **kwargs)
        )
