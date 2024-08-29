"""
pyart.aux_io.knmi_h5
====================

Routines for reading ODIM_H5 files.

.. autosummary::
    :toctree: generated/

    read_knmi_grid_h5
    _get_knmi_data
"""

import datetime

import numpy as np

try:
    import h5py

    _H5PY_AVAILABLE = True
except ImportError:
    _H5PY_AVAILABLE = False

try:
    import pyproj

    _PYPROJ_AVAILABLE = True
except ImportError:
    _PYPROJ_AVAILABLE = False

from ..aux_io.odim_h5 import _to_str, proj4_to_dict
from ..config import FileMetadata, get_fillvalue
from ..core.grid import Grid
from ..exceptions import MissingOptionalDependency
from ..io.common import _test_arguments, make_time_unit_str
from ..util import ma_broadcast_to

KNMI_H5_FIELD_NAMES = {
    "RAINFALL_RATE_[MM/H]": "radar_estimated_rain_rate",
    "ACCUMULATION_[MM]": "rainfall_accumulation",
    "QUALITY_[-]": "signal_quality_index",
    "ADJUSTMENT_FACTOR_[DB]": "adjustment_factor",
}


def read_knmi_grid_h5(
    filename,
    field_names=None,
    additional_metadata=None,
    file_field_names=False,
    exclude_fields=None,
    include_fields=None,
    time_ref="end",
    **kwargs
):
    """
    Read a KNMI_H5 grid file.

    Parameters
    ----------
    filename : str
        Name of the field_name file to read.
    field_names : dict, optional
        Dictionary mapping field_name field names to radar field names. If a
        data type found in the file does not appear in this dictionary or has
        a value of None it will not be placed in the radar.fields dictionary.
        A value of None, the default, will use the mapping defined in the
        Py-ART configuration file.
    additional_metadata : dict of dicts, optional
        Dictionary of dictionaries to retrieve metadata from during this read.
        This metadata is not used during any successive file reads unless
        explicitly included.  A value of None, the default, will not
        introduct any addition metadata and the file specific or default
        metadata as specified by the Py-ART configuration file will be used.
    file_field_names : bool, optional
        True to use the MDV data type names for the field names. If this
        case the field_names parameter is ignored. The field dictionary will
        likely only have a 'data' key, unless the fields are defined in
        `additional_metadata`.
    exclude_fields : list or None, optional
        List of fields to exclude from the radar object. This is applied
        after the `file_field_names` and `field_names` parameters. Set
        to None to include all fields specified by include_fields.
    include_fields : list or None, optional
        List of fields to include from the radar object. This is applied
        after the `file_field_names` and `field_names` parameters. Set
        to None to include all fields not specified by exclude_fields.
    time_ref : str
        Time reference in the /what/time attribute. Can be either 'start',
        'mid' or 'end'. If 'start' the attribute is expected to be the
        starttime of the scan, if 'mid', the middle time, if 'end' the
        endtime.

    Returns
    -------
    grid : Grid
        Grid object containing data from ODIM_H5 file.

    """
    # check that h5py is available
    if not _H5PY_AVAILABLE:
        raise MissingOptionalDependency(
            "h5py is required to use read_odim_h5 but is not installed"
        )

    # test for non empty kwargs
    _test_arguments(kwargs)

    # create metadata retrieval object
    if field_names is None:
        field_names = KNMI_H5_FIELD_NAMES
    filemetadata = FileMetadata(
        "odim_h5",
        field_names,
        additional_metadata,
        file_field_names,
        exclude_fields,
        include_fields,
    )

    with h5py.File(filename, "r") as hfile:
        # determine the number of datasets by the number of groups which
        # begin with image
        datasets = [k for k in hfile if k.startswith("image")]
        datasets.sort(key=lambda x: int(x[5:]))

        # latitude, longitude and altitude
        x = filemetadata("x")
        y = filemetadata("y")
        z = filemetadata("z")

        geo_group = hfile["geographic"].attrs
        map_proj = hfile["geographic"]["map_projection"].attrs

        # projection definition
        projection = map_proj["projection_proj4_params"]

        # 0, 49.3621, 0, 55.9736, 10.8565, 55.389, 9.0093, 48.8953
        geo_product_corners = geo_group["geo_product_corners"]

        if _PYPROJ_AVAILABLE:
            wgs84 = pyproj.CRS.from_epsg(4326)
            try:  # pyproj doens't like bytearrays
                projection = projection.decode("utf-8")
            except Exception:
                pass
            # projection = f'{projection} +units=km'
            projection = (
                "+proj=stere +lat_0=90 +lon_0=0.0 +lat_ts=60.0"
                " +a=6378137 +b=6356752 +x_0=0 +y_0=0"
            )

            coordTrans = pyproj.Transformer.from_crs(wgs84, projection)
            xstart, ystart = coordTrans.transform(
                geo_product_corners[1], geo_product_corners[0]
            )
            xend, yend = coordTrans.transform(
                geo_product_corners[5], geo_product_corners[4]
            )
            xvec = np.linspace(xstart, xend, geo_group["geo_number_columns"][0])
            yvec = np.linspace(ystart, yend, geo_group["geo_number_rows"][0])
        else:
            xvec = np.linspace(
                geo_product_corners[5],
                geo_product_corners[1],
                geo_group["geo_number_columns"][0],
            )
            yvec = np.linspace(
                geo_product_corners[0],
                geo_product_corners[4],
                geo_group["geo_number_rows"][0],
            )

        x["data"] = xvec
        y["data"] = yvec
        z["data"] = np.array([0], dtype="float64")

        # metadata
        metadata = filemetadata("metadata")
        # metadata['source'] = _to_str(hfile['what'].attrs['source'])
        metadata["original_container"] = "knmi_h5"
        # metadata['odim_conventions'] = _to_str(hfile.attrs['Conventions'])

        # h_what = hfile['what'].attrs
        # metadata['version'] = _to_str(h_what['version'])
        # metadata['source'] = _to_str(h_what['source'])

        # Get the MeteoSwiss-specific data
        # try:
        #     h_how2 = hfile['how']['MeteoSwiss'].attrs
        # except KeyError:
        #     # if no how group exists mock it with an empty dictionary
        #     h_how2 = {}
        # if 'radar' in h_how2:
        #     metadata['radar'] = h_how2['radar']

        # try:
        #     ds1_how = hfile[datasets[0]]['how'].attrs
        # except KeyError:
        #     # if no how group exists mock it with an empty dictionary
        #     ds1_how = {}
        # if 'system' in ds1_how:
        #     metadata['system'] = ds1_how['system']
        # if 'software' in ds1_how:
        #     metadata['software'] = ds1_how['software']
        # if 'sw_version' in ds1_how:
        #     metadata['sw_version'] = ds1_how['sw_version']

        dset_list = []
        field_list = []
        knmi_list = []
        for knmi_field in field_names.keys():
            for dset in datasets:
                dset_field = _to_str(hfile[dset].attrs["image_geo_parameter"])

                if knmi_field == dset_field:
                    dset_list.append(dset)
                    field_list.append(field_names[knmi_field])
                    knmi_list.append(knmi_field)
                    break
                if (
                    knmi_field == "RAINFALL_RATE_[MM/H]"
                    and dset_field == "ACCUMULATION_[MM]"
                ):
                    dset_list.append(dset)
                    field_list.append(field_names[knmi_field])
                    knmi_list.append(knmi_field)
                    break

        fields = {}
        for knmi_field, field_name, dset in zip(knmi_list, field_list, dset_list):
            fdata = _get_knmi_data(
                hfile[dset], knmi_field, hfile[dset]["calibration"].attrs
            )

            field_dic = filemetadata(field_name)
            # grid object expects a 3D field
            ny = geo_group["geo_number_rows"][0]
            nx = geo_group["geo_number_columns"][0]
            field_dic["data"] = ma_broadcast_to(fdata[::-1, :], (1, ny, nx))

            field_dic["_FillValue"] = get_fillvalue()

            fields[field_name] = field_dic
        if not fields:
            # warn(f'No fields could be retrieved from file')
            return None

        _time = filemetadata("time")
        origin_latitude = filemetadata("origin_latitude")
        origin_longitude = filemetadata("origin_longitude")
        origin_altitude = filemetadata("origin_altitude")

        # format 12-OCT-2023;08:00:04.000
        start_time = hfile["overview"].attrs["product_datetime_start"]
        start_time = datetime.datetime.strptime(
            _to_str(start_time), "%d-%b-%Y;%H:%M:%S.000"
        )
        end_time = hfile["overview"].attrs["product_datetime_end"]
        end_time = datetime.datetime.strptime(
            _to_str(end_time), "%d-%b-%Y;%H:%M:%S.000"
        )

        _time["data"] = [0]
        if time_ref == "mid":
            mid_delta = (end_time - start_time) / 2
            mid_ts = start_time + mid_delta
            _time["units"] = make_time_unit_str(mid_ts)
        elif time_ref == "end":
            _time["units"] = make_time_unit_str(end_time)
        else:
            _time["units"] = make_time_unit_str(start_time)

        projection = proj4_to_dict(projection)
        if "lat_0" in projection:
            origin_latitude["data"] = np.array([projection["lat_0"]])
        else:
            origin_latitude["data"] = np.array([0.0])
        if "lon_0" in projection:
            origin_longitude["data"] = np.array([projection["lat_0"]])
        else:
            origin_longitude["data"] = np.array([0.0])
        origin_altitude["data"] = np.array([0.0])

        # radar variables
        radar_latitude = None
        radar_longitude = None
        radar_altitude = None
        radar_name = None
        radar_time = None

        return Grid(
            _time,
            fields,
            metadata,
            origin_latitude,
            origin_longitude,
            origin_altitude,
            x,
            y,
            z,
            projection=projection,
            radar_latitude=radar_latitude,
            radar_longitude=radar_longitude,
            radar_altitude=radar_altitude,
            radar_name=radar_name,
            radar_time=radar_time,
        )


def _get_knmi_data(group, knmi_field, calibration):
    """Get KNMI data from an HDF5 group."""
    nodata = calibration["calibration_missing_data"]
    undetect = calibration["calibration_out_of_image"]
    formula = _to_str(calibration["calibration_formulas"]).split("=")[1]
    # b'GEO = 0.500000 * PV + -32.000000'
    gain = float(formula.split("*")[0])
    offset = float(formula.split("+")[1])

    # mask raw data
    raw_data = group["image_data"][:]
    data = np.ma.masked_values(raw_data, nodata)
    data = np.ma.masked_values(data, undetect)
    data = data * gain + offset
    if knmi_field == "RAINFALL_RATE_[MM/H]":
        data *= 12.0

    return data
