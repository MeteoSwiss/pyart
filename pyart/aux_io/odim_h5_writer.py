"""
pyart.io.odim_h5_writer.py
=========================

Utilities for writing ODIM hdf5 files.

.. autosummary::
    :toctree: generated/

    write_odim_grid_h5
    write_odim_h5
    _to_str
    _get_sec_since_epoch
    _tree
    _check_file_exists
    _create_odim_h5_file
    _create_odim_h5_grp
    _create_odim_h5_sub_grp
    _create_odim_h5_attr
    _create_odim_h5_dataset
    _map_radar_quantitym
    _get_data_from_field
    _map_radar_to_how_dict
    proj4_to_str
"""

import datetime
import calendar
import time
from warnings import warn
from collections import defaultdict
from pathlib import Path
import numpy as np

from ..exceptions import MissingOptionalDependency
from ..config import get_fillvalue
from ..core.radar import Radar
from ..core.grid import Grid

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


# from ..config import FileMetadata
# from ..core.radar import Radar
# from ..lazydict import LazyLoadDict

def write_odim_grid_h5(filename, grid, corners=None, field_names=None,
                       physical=True, compression="gzip", compression_opts=6):
    """
    Write a Grid object to a EUMETNET OPERA compliant HDF5 file.

    The files produced by this routine follow the EUMETNET OPERA information
    model:
    http://eumetnet.eu/wp-content/uploads/2017/01/OPERA_hdf_description_2014.pdf

    Not yet supported:
      - Multiple datasets


    Parameters
    ----------
    filename : str
        Filename of file to create.
    grid : Grid
        Grid object to process.
    field_names : list of str
        The list of fields from the radar object to save. If none all fields
        in the radar object will be saved.
    physical : Bool
        If true the physical values are stored. nodata parameter is equal to
        the _FillValue parameter in the field metadata or the default Py-ART
        fill value. If false the data is converted into binary values using a
        linear conversion. The gain and offset are either specified in the
        metadata of the field with keywords 'scale_factor' and 'add_offset' or
        calculated on the fly. keyword '_Write_as_dtype' specifies the
        datatype. It can be either 'uint8' or 'uint16'. The default datatype
        is uint8. The 'undetect' parameter is not used
    compression : str
        The type of compression for the datasets. Typical are "gzip" and "lzf".
    compression_opts : any
        The compression options. In the case of gzip is the level between 0 to
        9 (recomended 1 to 6). In the case of lzf there are not options.

    """

    # Initialize hdf5 file
    hdf_id = _create_odim_h5_file(filename)

    # Determine number of different data types per dataset and list of fields
    grid_field_names = list(grid.fields.keys())
    if field_names is not None:
        # check that all desired fields are in grid object
        aux_field_names = []
        for field_name in field_names:
            if field_name not in grid_field_names:
                warn(field_name+' not in grid object')
            else:
                aux_field_names.append(field_name)
        if np.size(aux_field_names) == 0:
            warn('No matching field names available')
            # return
        field_names = aux_field_names

    else:
        field_names = list(grid.fields.keys())

    n_datasets = 1

    # Check from z value if data is COMP (at ground) or CVOL (CAPPI)
    if grid.z['data'][0] == 0:
        odim_object = 'COMP'
    else:
        odim_object = 'CVOL'

    # Create level 1 group structure
    where1_grp = _create_odim_h5_grp(hdf_id, '/where')
    what1_grp = _create_odim_h5_grp(hdf_id, '/what')
    how1_grp = _create_odim_h5_grp(hdf_id, '/how')

    dataset_grps = []
    for i in range(n_datasets):
        name = 'dataset'+str(i+1)
        grp_id = _create_odim_h5_grp(hdf_id, name)
        dataset_grps.append(grp_id)

    # Write ODIM Conventions attribute
    _create_odim_h5_attr(hdf_id, 'Conventions', 'ODIM_H5/V2_2')

    # where - UL, LL, LR, UR, scales, sizes, projdef
    x = grid.x['data']
    y = grid.y['data']
    X, Y = np.meshgrid(x, y)
    lon, lat = X, Y

    if _PYPROJ_AVAILABLE:
        wgs84 = pyproj.Proj(4326)

        if proj4_to_str(grid.projection) != wgs84.definition.decode('utf-8'):
            inproj = pyproj.Proj(proj4_to_str(grid.projection))
            coordTrans = pyproj.Transformer.from_proj(inproj, wgs84)

            # Convert coordinates
            lat, lon = coordTrans.transform(X, Y)

    _create_odim_h5_attr(where1_grp, 'LL_lat', lat[0, 0])
    _create_odim_h5_attr(where1_grp, 'LR_lat', lat[0, -1])
    _create_odim_h5_attr(where1_grp, 'UL_lat', lat[-1, 0])
    _create_odim_h5_attr(where1_grp, 'UR_lat', lat[-1, -1])
    _create_odim_h5_attr(where1_grp, 'LL_lon', lon[0, 0])
    _create_odim_h5_attr(where1_grp, 'LR_lon', lon[0, -1])
    _create_odim_h5_attr(where1_grp, 'UL_lon', lon[-1, 0])
    _create_odim_h5_attr(where1_grp, 'UR_lon', lon[-1, -1])
    _create_odim_h5_attr(where1_grp, 'projdef', proj4_to_str(grid.projection))
    _create_odim_h5_attr(where1_grp, 'xscale', float(1))
    _create_odim_h5_attr(where1_grp, 'yscale', float(1))
    _create_odim_h5_attr(where1_grp, 'xsize', len(x))
    _create_odim_h5_attr(where1_grp, 'ysize', len(y))

    # what - version, date, time, source, object
    odim_version = _to_str(grid.metadata['version'])
    odim_source = _to_str(grid.metadata['source'])

    # Create subgroup for radar info for MeteoSwiss radars
    how1MCH_grp = _create_odim_h5_grp(hdf_id, '/how/MeteoSwiss')
    odim_radar = _to_str(grid.metadata['radar'])
    _create_odim_h5_attr(how1MCH_grp, 'radar', odim_radar)
    grid.metadata.pop('radar', None)

    # Time
    odim_start = datetime.datetime.fromtimestamp(time.mktime(time.strptime(
        grid.time['units'], "seconds since %Y-%m-%dT%H:%M:%SZ")))
    # due to the python indexing, we need to add +1 in the timedelta
    odim_end = odim_start + datetime.timedelta(
        seconds=int(grid.time['data'][-1]+1))

    odim_starttime = datetime.datetime.strftime(odim_start, "%H%M%S")
    odim_startdate = datetime.datetime.strftime(odim_start, "%Y%m%d")
    odim_endtime = datetime.datetime.strftime(odim_end, "%H%M%S")
    odim_enddate = datetime.datetime.strftime(odim_end, "%Y%m%d")

    # Create and fill what1 group attributes
    _create_odim_h5_attr(what1_grp, 'time', odim_starttime)
    _create_odim_h5_attr(what1_grp, 'date', odim_startdate)
    _create_odim_h5_attr(what1_grp, 'version', odim_version)
    _create_odim_h5_attr(what1_grp, 'source', odim_source)
    _create_odim_h5_attr(what1_grp, 'object', odim_object)

    # Dataset specific
    i = 0  # dataset index

    what2_id = _create_odim_h5_sub_grp(dataset_grps[i], 'what')
    _create_odim_h5_attr(what2_id, 'enddate', odim_enddate)
    _create_odim_h5_attr(what2_id, 'endtime', odim_endtime)
    _create_odim_h5_attr(what2_id, 'startdate', odim_startdate)
    _create_odim_h5_attr(what2_id, 'starttime', odim_starttime)

    field_name = list(grid.fields.keys())[0]
    data_dict = _get_data_from_field(
                        grid, i, field_name,
                        physical=physical)

    _create_odim_h5_attr(what2_id, 'gain', data_dict['gain'])
    _create_odim_h5_attr(what2_id, 'offset', data_dict['offset'])
    _create_odim_h5_attr(what2_id, 'nodata', data_dict['nodata'])

    if 'product' in grid.fields[field_name].keys():
        if odim_object == 'COMP':
            product = grid.fields[field_name]['product']
        else:
            product, prodpar = grid.fields[field_name]['product'].split(b'_')
            product = np.bytes_(product)
            _create_odim_h5_attr(what2_id, 'prodpar', float(prodpar))
    else:
        product = field_name.encode('utf-8')
    _create_odim_h5_attr(what2_id, 'product', product)

    if 'prodname' in grid.fields[field_name].keys():
        prodname = grid.fields[field_name]['prodname']
    else:
        prodname = field_name.encode('utf-8')
    _create_odim_h5_attr(what2_id, 'prodname', prodname)
    _create_odim_h5_attr(what2_id, 'quantity', _map_radar_quantity(field_name))

    # Write data
    datatype_ind = _create_odim_h5_sub_grp(dataset_grps[i], 'data1')
    _create_odim_h5_dataset(
        datatype_ind, 'data', data_dict['data'],
        make_legend=False, compression=compression,
        compression_opts=compression_opts)

    # close HDF file
    hdf_id.close()
    _check_file_exists(filename)


def write_odim_h5(filename, radar, field_names=None, physical=True,
                  compression="gzip", compression_opts=6):
    """
    Write a Radar object to a EUMETNET OPERA compliant HDF5 file.

    The files produced by this routine follow the EUMETNET OPERA information
    model:
    http://eumetnet.eu/wp-content/uploads/2017/01/OPERA_hdf_description_2014.pdf

    Supported features:
      - Writing PPIs: PVOL and SCAN objects
        - Different sweeps are saved in different dataset groups
      - Writing sectorized PPIs and SCANs: AZIM objects
      - Writing RHIs: ELEV objects

    Not yet supported:
      - Mixed datasets (how group always on top level)
      - Single ray data (e.g. from fixed staring mode)
      - Profiles


    Parameters
    ----------
    filename : str
        Filename of file to create.
    radar : Radar
        Radar object to process.
    field_names : list of str
        The list of fields from the radar object to save. If none all fields
        in the radar object will be saved.
    physical : Bool
        If true the physical values are stored. nodata parameter is equal to
        the _FillValue parameter in the field metadata or the default Py-ART
        fill value. If false the data is converted into binary values using a
        linear conversion. The gain and offset are either specified in the
        metadata of the field with keywords 'scale_factor' and 'add_offset' or
        calculated on the fly. keyword '_Write_as_dtype' specifies the
        datatype. It can be either 'uint8' or 'uint16'. The default datatype
        is uint8. The 'undetect' parameter is not used
    compression : str
        The type of compression for the datasets. Typical are "gzip" and "lzf".
    compression_opts : any
        The compression options. In the case of gzip is the level between 0 to
        9 (recomended 1 to 6). In the case of lzf there are not options.

    """

    # Initialize hdf5 file
    hdf_id = _create_odim_h5_file(filename)

    # Determine odim object, number of datasets (PPIs, RHIs) and
    # check if sweep mode does not change between different sweeps
    if radar.scan_type == 'rhi':
        odim_object = 'ELEV'
        n_datasets = np.size(radar.fixed_angle['data'])
    else:
        if radar.scan_type == 'ppi':
            n_datasets = radar.nsweeps
            if len(set(radar.fixed_angle['data'])) <= 1:
                odim_object = 'SCAN'
            else:
                if radar.sweep_mode is not None:
                    if len(set(radar.sweep_mode['data'])) <= 1:
                        nray_in_sweep = radar.rays_per_sweep['data'][0]
                        ray_angle_res = radar.ray_angle_res['data'][0]
                        if nray_in_sweep * ray_angle_res < 360:
                            odim_object = 'AZIM'
                        else:
                            odim_object = 'PVOL'
                    else:
                        warn("Radar sweep mode changes, which is not " +
                             "yet supported to write ODIM HDF5 file.")
                        return
                else:
                    warn("ODIM object could not be identified.")
                    return

    # Determine number of different data types per dataset and list of fields
    radar_field_names = list(radar.fields.keys())
    if field_names is not None:
        # check that all desired fields are in radar object
        aux_field_names = []
        for field_name in field_names:
            if field_name not in radar_field_names:
                warn(field_name+' not in radar object')
            else:
                aux_field_names.append(field_name)
        if np.size(aux_field_names) == 0:
            warn('No matching field names available')
            return
        field_names = aux_field_names

    else:
        field_names = list(radar.fields.keys())
    n_datatypes = len(field_names)

    # Create level 1 group structure
    where1_grp = _create_odim_h5_grp(hdf_id, '/where')
    what1_grp = _create_odim_h5_grp(hdf_id, '/what')
    how1_grp = _create_odim_h5_grp(hdf_id, '/how')

    dataset_grps = []
    for i in range(n_datasets):
        name = 'dataset'+str(i+1)
        grp_id = _create_odim_h5_grp(hdf_id, name)
        dataset_grps.append(grp_id)

    # Write ODIM Conventions attribute
    _create_odim_h5_attr(hdf_id, 'Conventions', 'ODIM_H5/V2_2')

    # where - lon, lat, height
    lon = radar.longitude['data']
    lat = radar.latitude['data']
    height = radar.altitude['data']
    _create_odim_h5_attr(where1_grp, 'lon', lon)
    _create_odim_h5_attr(where1_grp, 'lat', lat)
    _create_odim_h5_attr(where1_grp, 'height', height)

    # what - version, date, time, source, object
    odim_version = _to_str(radar.metadata['version'])
    odim_source = _to_str(radar.metadata['source'])

    # Time
    odim_time_struct = time.strptime(
        radar.time['units'], "seconds since %Y-%m-%dT%H:%M:%SZ")
    odim_datetime_struct = datetime.datetime.utcfromtimestamp(
        calendar.timegm(odim_time_struct))

    # Time relative to center of first gate?
    odim_dt = float(radar.time['data'][0])
    odim_datetime_start = odim_datetime_struct + datetime.timedelta(
        seconds=odim_dt)

    odim_time = datetime.datetime.strftime(odim_datetime_start, "%H%M%S")
    odim_date = datetime.datetime.strftime(odim_datetime_start, "%Y%m%d")

    # Create and fill what1 group attributes
    _create_odim_h5_attr(what1_grp, 'time', odim_time)
    _create_odim_h5_attr(what1_grp, 'date', odim_date)
    _create_odim_h5_attr(what1_grp, 'version', odim_version)
    _create_odim_h5_attr(what1_grp, 'source', odim_source)
    _create_odim_h5_attr(what1_grp, 'object', odim_object)

    # How variables
    # General
    how_var_general = ['system', 'software', 'sw_verison']

    # Individual radar
    how_var_instrument = [
        'beamwidth', 'wavelength', 'rpm', 'elevspeed', 'pulsewidth',
        'RXbandwidth', 'lowprf', 'midprf', 'highprf', 'TXlossH', 'TXlossV'
        'injectlossH', 'injectlossV', 'RXlossH', 'RXlossV',
        'radomelossH', 'radomelossV', 'antgainH', 'antgainV', 'beamwH',
        'beamwV', 'gasattn', 'radconstH', 'radconstV', 'nomTXpower', 'TXpower'
        'powerdiff', 'phasediff', 'NI', 'Vsamples']

    # Map radar.metadata to how1_dict if entries are available
    if any(x in how_var_general for x in radar.metadata):
        how_var_gen = []
        if 'system' in radar.metadata:
            how_var_gen.append('system')
        if 'software' in radar.metadata:
            how_var_gen.append('software')
        if 'sw_verison' in radar.metadata:
            how_var_gen.append('sw_version')
        how1_gen_dict = _map_radar_to_how_dict(radar.metadata)
        for name in how_var_gen:
            _create_odim_h5_attr(how1_grp, name, how1_gen_dict[name])

    # Map radar.instrument_parameters to how1_dict
    if radar.instrument_parameters is not None:
        radar_ins_obj = radar.instrument_parameters
        how1_ins_dict = _map_radar_to_how_dict(radar_ins_obj)
    else:
        how1_ins_dict = None
        warn("Instrument parameters not available in radar object")

    # Map radar.radar_calibration to how1_dict
    if radar.radar_calibration is not None:
        radar_cal_obj = radar.radar_calibration
        how1_cal_dict = _map_radar_to_how_dict(radar_cal_obj)
    else:
        how1_cal_dict = None
        warn("Radar calibration parameters not available in radar object")

    for name in how_var_instrument:
        if how1_ins_dict is not None:
            if name in how1_ins_dict:
                _create_odim_h5_attr(how1_grp, name, how1_ins_dict[name])
        elif how1_cal_dict is not None:
            if name in how1_cal_dict:
                _create_odim_h5_attr(how1_grp, name, how1_cal_dict[name])

    # Create level 2 group structure
    datatype_grps = []  # 2D list datatype group ids
    where2_grps = []  # 1D list of where groups
    what2_grps = []  # 1D list of what groups
    how2_grps = []  # 1D list of how groups

    for i in range(n_datasets):
        where2_id = _create_odim_h5_sub_grp(dataset_grps[i], 'where')
        where2_grps.append(where2_id)
        what2_id = _create_odim_h5_sub_grp(dataset_grps[i], 'what')
        what2_grps.append(what2_id)
        how2_id = _create_odim_h5_sub_grp(dataset_grps[i], 'how')
        how2_grps.append(how2_id)

        datatype_grps.append([])  # empty list for each row

        for j in range(n_datatypes):
            name = 'data'+str(j+1)
            datatype_ind = _create_odim_h5_sub_grp(dataset_grps[i], name)
            datatype_grps[i].append(datatype_ind)

    # Dataset specific what header Attributes
    what_var_dataset = ['product', 'prodpar', 'startdate', 'starttime',
                        'enddate', 'endtime']

    # Data attributes
    what_var_data = ['quantity', 'gain', 'offset', 'nodata', 'undetect']

    # Supported scan types
    odim_product_list = ['SCAN', 'RHI']

    if odim_object in ['PVOL', 'AZIM', 'SCAN', 'ELEV']:
        # where2
        where2_variables = ['nbins', 'rstart', 'rscale', 'nrays']
        where2_dict = {}

        where2_dict['nbins'] = np.int64(np.repeat(radar.ngates, n_datasets))
        where2_dict['rstart'] = np.repeat(
            np.double((radar.range['data'][0])/(1000.)), n_datasets)  # [km]
        where2_dict['nrays'] = np.int64(radar.rays_per_sweep['data'])

        if len(set(radar.range['data'][1:] - radar.range['data'][0:-1])) <= 1:
            range_resolution = np.double(
                radar.range['data'][1]-radar.range['data'][0])
        else:
            range_resolution = np.median(
                radar.range['data'][1:] - radar.range['data'][0:-1])
            warn("Radar range resolution changes between gates. " +
                 "The median resolution is taken as reference.")
        where2_dict['rscale'] = np.repeat(range_resolution, n_datasets)

        if odim_object in ['PVOL', 'AZIM', 'SCAN']:
            where2_variables.extend(['elangle', 'a1gate'])
            where2_dict['elangle'] = np.double(radar.fixed_angle['data'])
            where2_dict['a1gate'] = np.int64(
                radar.sweep_start_ray_index['data'])

        # Sector Specific
        elif odim_object == 'AZIM':
            where2_variables.extend(['startaz', 'stopaz'])
            start_az = []
            stop_az = []
            for i in radar.sweep_start_ray_index['data']:
                start_az_tmp = radar.azimuth['data'][i]
                start_az.append(start_az_tmp)
            for i in radar.sweep_end_ray_index['data']:
                stop_az_tmp = radar.azimuth['data'][i]
                stop_az.append(stop_az_tmp)
            where2_dict['startaz'] = np.double(start_az)
            where2_dict['stopaz'] = np.double(stop_az)

        # RHI specific
        elif odim_object == 'ELEV':
            where2_variables.extend(['az_angle', 'angles', 'range'])
            where2_dict['range'] = np.repeat(
                np.double((np.max(radar.range['data']) +
                           (radar.range['data'][0])) / 1000.),
                n_datasets)  # [km] to end of last gate

            az_angle = []
            for i in range(n_datasets):
                az_angle_tmp = radar.fixed_angle['data'][i]
                az_angle.append(az_angle_tmp)
            where2_dict['az_angle'] = np.double(az_angle)
            angles_mx = [
                [0 for x in range(radar.nrays)] for y in range(n_datasets)]
            for i in range(n_datasets):
                angles_mx[i][:] = radar.elevation['data']
            where2_dict['angles'] = np.double(angles_mx)

        # Write where2 group attributes
        ind = 0
        for i in where2_grps:
            for name in where2_variables:
                if where2_dict[name] is not None:
                    _create_odim_h5_attr(i, name, where2_dict[name][ind])
            ind = ind+1

        # what2
        scan_type = radar.scan_type
        if scan_type.upper() == 'PPI':  # PPI is for Cartesian PPIs in ODIM
            scan_type = 'SCAN'
        what2_dict = {}
        what2_dict['product'] = np.repeat(scan_type.upper(), n_datasets)
        if scan_type.upper() in odim_product_list:
            if scan_type.upper() == 'SCAN':
                # Eleveation angle/s (deg)
                what2_dict['prodpar'] = np.double(radar.fixed_angle['data'])
            elif scan_type.upper() == 'RHI':
                # Correct for Index to scalar error - only 1 azimuth -
                # create array
                if n_datasets <= 1:
                    what2_dict['prodpar'] = np.repeat(
                        np.double(radar.fixed_angle['data']), n_datasets)
                else:
                    what2_dict['prodpar'] = np.double(
                        radar.fixed_angle['data'])
            else:
                warn("Scan type "+scan_type+" not yet supported")
        else:
            warn("Scan type "+scan_type+" not yet supported")

        # time
        _time = radar.time['data']
        start_sweep_ind = radar.sweep_start_ray_index['data']
        end_sweep_ind = radar.sweep_end_ray_index['data']
        startdate = []
        starttime = []
        enddate = []
        endtime = []
        for i, start_ind in enumerate(start_sweep_ind):
            _startdate = odim_datetime_struct + datetime.timedelta(
                seconds=float(_time[start_ind]))
            _startdate_str = datetime.datetime.strftime(_startdate, "%Y%m%d")
            _starttime_str = datetime.datetime.strftime(_startdate, "%H%M%S")
            startdate.append(_startdate_str)
            starttime.append(_starttime_str)
            _enddate = odim_datetime_struct + datetime.timedelta(
                seconds=float(_time[end_sweep_ind[i]]))
            _enddate_str = datetime.datetime.strftime(_enddate, "%Y%m%d")
            _endtime_str = datetime.datetime.strftime(_enddate, "%H%M%S")
            enddate.append(_enddate_str)
            endtime.append(_endtime_str)

        what2_dict['startdate'] = startdate
        what2_dict['starttime'] = starttime
        what2_dict['enddate'] = enddate
        what2_dict['endtime'] = endtime

        # fill the what2 group attributes
        ind = 0
        for i in what2_grps:
            for name in what_var_dataset:
                if what2_dict[name] is not None:
                    _create_odim_h5_attr(i, name, what2_dict[name][ind])
            ind = ind + 1

        # Dataset specific how variables
        # Necessary?
        ssri = radar.sweep_start_ray_index['data']
        seri = radar.sweep_end_ray_index['data']

        if odim_object in ['PVOL', 'AZIM', 'SCAN']:
            how2_dict = {}
            # PPI
            how_var_dataset = ['elangles', 'startazT', 'stopazT']

            el_angle = []
            T_start_az = []
            T_stop_az = []

            for i in range(n_datasets):
                el_angle_tmp = radar.elevation['data'][ssri[i]:seri[i]+1]
                el_angle.append(el_angle_tmp)
                T_start_az_tmp = (
                    radar.time['data'][ssri[i]:seri[i]+1] -
                    radar.time['data'][0])
                T_stop_az_tmp = (
                    radar.time['data'][ssri[i]:seri[i]+1] +
                    radar.time['data'][0])
                t1_epo = []
                for sec in T_start_az_tmp:
                    t_tmp = odim_datetime_struct + datetime.timedelta(
                        seconds=float(sec))
                    t_epo_sec = _get_sec_since_epoch(t_tmp)
                    t1_epo.append(t_epo_sec)
                T_start_az.append(t1_epo)
                t2_epo = []
                for sec in T_stop_az_tmp:
                    t_tmp = odim_datetime_struct + datetime.timedelta(
                        seconds=float(sec))
                    t_epo_sec = _get_sec_since_epoch(t_tmp)
                    t2_epo.append(t_epo_sec)
                T_stop_az.append(t2_epo)

            how2_dict['elangles'] = np.double(el_angle)
            how2_dict['startazT'] = np.double(T_start_az)
            how2_dict['stopazT'] = np.double(T_stop_az)

            if radar.azimuth['data'] is not None:
                how_var_dataset.extend(['startazA', 'stopazA'])
                start_azA = []
                stop_azA = []
                for i in range(n_datasets):
                    start_azA_tmp = (
                        radar.azimuth['data'][ssri[i]:seri[i]+1] -
                        radar.ray_angle_res['data'][i] / 2.)
                    start_azA_tmp[start_azA_tmp < 0.] += 360.  # [0 360]
                    stop_azA_tmp = (
                        radar.azimuth['data'][ssri[i]:seri[i]+1] +
                        radar.ray_angle_res['data'][i] / 2.)
                    stop_azA_tmp[stop_azA_tmp > 360.] -= 360.  # [0 360]
                    start_azA.append(start_azA_tmp)
                    stop_azA.append(stop_azA_tmp)

                how2_dict['startazA'] = np.double(start_azA)
                how2_dict['stopazA'] = np.double(stop_azA)

        # ELEV - RHI
        # how_var_dataset = ['startelA', 'stopelA', 'startelT', 'stopelT']
        # Not supported by ODIM reader yet

            # fill the how2 group attributes
            ind = 0
            for i in how2_grps:
                for name in how_var_dataset:
                    if how2_dict[name] is not None:
                        _create_odim_h5_attr(i, name, how2_dict[name][ind])
                ind = ind + 1

        # create level 3 data and what group structure and fill data
        what3_grps = []
        what3_dict = _tree()

        for i in range(n_datasets):
            what3_grps.append([])
            for j, field_name in enumerate(field_names):
                what3_id = _create_odim_h5_sub_grp(
                    datatype_grps[i][j], 'what')
                what3_grps[i].append(what3_id)

                # ODIM field name
                radar_quantity = _map_radar_quantity(field_name)
                what3_dict[i][j]['quantity'] = radar_quantity

                # get data
                data_dict = _get_data_from_field(
                    radar, i, field_name, physical=physical)

                # write data
                what3_dict[i][j]['gain'] = data_dict['gain']
                what3_dict[i][j]['offset'] = data_dict['offset']

                if data_dict['nodata'] is not None:
                    what3_dict[i][j]['nodata'] = data_dict['nodata']
                if data_dict['undetect'] is not None:
                    what3_dict[i][j]['undetect'] = data_dict['undetect']

                _create_odim_h5_dataset(
                    datatype_grps[i][j], 'data', data_dict['data'],
                    make_legend=False, compression=compression,
                    compression_opts=compression_opts)

                # Add legend data if 'label' key is present in radar quantity
                # dictionary
                if ('labels' in radar.fields[field_name] and
                        'ticks' in radar.fields[field_name]):
                    labs = radar.fields[field_name]['labels']
                    tics = radar.fields[field_name]['ticks']
                    data_legend = []
                    for lab_tic in zip(labs, tics):
                        data_legend.append(lab_tic)
                    _create_odim_h5_dataset(
                        datatype_grps[i][j], 'legend', data_legend,
                        make_legend=True, compression=compression,
                        compression_opts=compression_opts)

        # fill the what3 group attributes of data
        for i in range(n_datasets):
            for j in range(n_datatypes):
                for name in what_var_data:
                    if name in what3_dict[i][j]:
                        rq_name = what3_dict[i][j][name]
                        _create_odim_h5_attr(what3_grps[i][j], name, rq_name)
                    # else:
                    #    warn("Attribute "+name+" not specified")

    # close HDF file
    hdf_id.close()
    _check_file_exists(filename)


def _to_str(text):
    """
    Converter: From byte arrays to string if necessary.

    Parameters:
    -----------
    text: byte array
        Input for conversion

    Returns:
    -------
    text : str
        String of input array

    """
    if hasattr(text, 'decode'):
        return text.decode('utf-8')

    return text


def _check_file_exists(filename):
    """
    Check for ODIM h5 file existence

    Parameters:
    -----------
    filename : str
        Filename string pointing to ODIM h5 file that should be present.

    """
    myfile = Path(filename)
    if myfile.is_file():
        print('ODIM h5 file '+filename+' written.')
    else:
        print('Warning: ODIM h5 file '+filename+' not written.')


def _get_sec_since_epoch(time_f):
    """
    Calculate seconds since 1970-01-01 epoch with microseconds precision

    Parameters:
    -----------
    time_f : datetime
        Datetime object to process

    Returns:
    --------
    sec_since_epoch : double
        Seconds since 1970 epoch

    """
    if hasattr(time_f, 'microsecond'):
        sec_since_epoch = (
            calendar.timegm(time_f.timetuple()) +
            time_f.microsecond/1000000.0)
    else:
        sec_since_epoch = calendar.timegm(time_f.timetuple())

    return sec_since_epoch


def _tree():
    """
    Initialize a tree structure for a multidimensional dictionary

    """
    return defaultdict(_tree)


def _create_odim_h5_file(filename, access_mode='w', driver=None):
    """
    Initialize HDF5 file with h5py (https://www.h5py.org/) to write ODIM
    compliant data structure.

    Parameters:
    -----------
    filename : str
        Filename of ODIM HDF5 file to create
    access_mode : str
        'w' for going into write model
    driver : str, optional
        Optional driver specification in h5py

    Returns:
    --------
    file_id : object
        h5py file ID object

    """
    # Check that h5py module is available
    if not _H5PY_AVAILABLE:
        raise MissingOptionalDependency(
            "h5py is required to use the ODIM HDF5 writer, '" +
            "'but is not installed")

    file_id = h5py.File(filename, access_mode, driver)

    return file_id


def _create_odim_h5_grp(file_id, grp_name):
    """
    Create HDF5 group for a specified file ID.

    Parameters:
    -----------
    file_id : object
        File ID object pointing to h5py file where group should be created.
    grp_name : str
        Group name to be created

    Returns:
    --------
    grp_id : obejct
        h5py group ID object

    """
    grp_id = file_id.create_group(grp_name)

    return grp_id


def _create_odim_h5_sub_grp(grp_id, sub_grp_name):
    """
    Create HDF5 subgroup within a specified group.

    Parameters:
    -----------
    grp_id : object
        Group ID object pointing to h5py group where subgroup should be
        created.
    sub_grp_name : str
        Subgroup name to be created

    Returns:
    --------
    sub_grp_id : obejct
        h5py subgroup ID object

    """
    sub_grp_id = grp_id.create_group(sub_grp_name)

    return sub_grp_id


def _create_odim_h5_attr(grp_id, name, data):
    """
    Create and fill group (subgroup) attributes with metadata.

    Parameters:
    -----------
    grp_id : object
        h5py object where attributes shoul be created.
    name : str
        Name of the attribute to be created.
    data : data (str, int, double, array)
        Data to be saved in h5py group attribute

    """
    # Check if data is str
    if isinstance(data, str):
        string_dt = h5py.special_dtype(vlen=str)
        grp_id.attrs.create(
            name, np.array(data, dtype=string_dt), dtype=string_dt)
    else:
        grp_id.attrs.create(name, data)


def _create_odim_h5_dataset(ID, name, data_arr, make_legend=False,
                            compression="gzip", compression_opts=6):
    """
    Create and save radar field data array to h5py dataset.

    Parameters:
    -----------
    ID : object
        h5py object ID pointing to dataset to be created and filled.
    name : str
        Name of the dataset to be created
    data_arr : array
        Array containing the data
    make_legend : Bool
        If true a legend is created
    compression : str
        compression type and level. Can be "gzip", "lzf"
    compression_opts : int
        if "gzip" is choosen, the level of compression. ODIM recommends
        levels between 1 and 6
    """
    if make_legend:
        dt_tmp = [
            ('label', h5py.special_dtype(vlen=str)), ('tick', np.float)]
        ds = ID.create_dataset(
            name, (1, len(data_arr)), dtype=dt_tmp, compression=compression,
            compression_opts=compression_opts)
        data_np_arr = np.array(data_arr, dtype=dt_tmp)
        ds[0] = data_np_arr
    else:
        ID.create_dataset(
            name, data=data_arr, compression=compression,
            compression_opts=compression_opts)


def _map_radar_quantity(field_name):
    """
    Map radar field quantities to ODIM compliant quantities.

    Parameters:
    -----------
    field_name : str
        the  radar field names

    Returns:
    --------
    ODIM_field_name : str
        ODIM compliant radar field name

    """
    odim_quantity_dict = {
        'unfiltered_reflectivity': 'TH',
        'unfiltered_reflectivity_vv': 'TV',
        'reflectivity': 'DBZH',
        'reflectivity_vv': 'DBZV',
        'corrected_reflectivity': 'DBZHC',  # Non standard ODIM
        'corrected_reflectivity_vv': 'DBZVC',  # Non standard ODIM
        'corrected_unfiltered_reflectivity': 'THC',  # Non standard ODIM
        'corrected_unfiltered_reflectivity_vv': 'TVC',  # Non standard ODIM
        'reflectivity_bias': 'ZBIAS',  # Non standard ODIM
        'volumetric_reflectivity': 'etah',  # Non standard ODIM
        'volumetric_reflectivity_vv': 'etav',  # Non standard ODIM
        'radar_cross_section_hh': 'RCSH',  # Non standard ODIM
        'radar_cross_section_vv': 'RCSV',  # Non standard ODIM
        'differential_reflectivity': 'ZDR',
        'unfiltered_differential_reflectivity': 'ZDRU',  # Non standard ODIM
        'corrected_differential_reflectivity': 'ZDRC',  # Non standard ODIM
        'corrected_unfiltered_differential_reflectivity': 'ZDRUC',  # Non standard ODIM
        'differential_reflectivity_in_precipitation': 'ZDRPREC',  # Non standard ODIM
        'differential_reflectivity_in_snow': 'ZDRSNOW',  # Non standard ODIM
        'signal_power_hh': 'DBMH',  # Non standard ODIM
        'signal_power_vv': 'DBMV',  # Non standard ODIM
        'noisedBZ_hh': 'NDBZH',  # Non standard ODIM
        'noisedBZ_vv': 'NDBZV',  # Non standard ODIM
        'sun_hit_power_h': 'DBM_SUNHIT',  # Non standard ODIM
        'sun_hit_power_v': 'DBMV_SUNHIT',  # Non standard ODIM
        'sun_hit_differential_reflectivity': 'ZDR_SUNHIT',  # Non standard ODIM
        'sun_est_power_h': 'DBM_SUNEST',  # Non standard ODIM
        'sun_est_power_v': 'DBMV_SUNEST',  # Non standard ODIM
        'sun_est_differential_reflectivity': 'ZDR_SUNEST',  # Non standard ODIM
        'sun_hit_h': 'POSH_SUNHIT',  # Non standard ODIM
        'sun_hit_v': 'POSV_SUNHIT',  # Non standard ODIM
        'sun_hit_zdr': 'POSZDR_SUNHIT',  # Non standard ODIM
        'cross_correlation_ratio': 'RHOHV',
        'uncorrected_cross_correlation_ratio': 'URHOHV',  # Non standard ODIM
        'corrected_cross_correlation_ratio': 'RHOHVC',  # Non standard ODIM
        'cross_correlation_ratio_in_rain': 'RHOHVRAIN',  # Non standard ODIM
        'logarithmic_cross_correlation_ratio': 'LRHOHV',  # Non standard ODIM
        'circular_depolarization_ratio': 'CDR',  # Non standard ODIM
        'linear_polarization_ratio': 'LDR',
        'differential_phase': 'PHIDP',
        'uncorrected_differential_phase': 'UPHIDP',  # Non standard ODIM
        'corrected_differential_phase': 'PHIDPC',  # Non standard ODIM
        'system_differential_phase': 'PHIDP0',  # Non standard ODIM
        'first_gate_differential_phase': 'PHIDP0_BIN',  # Non standard ODIM
        'specific_differential_phase': 'KDP',
        'corrected_specific_differential_phase': 'KDPC',  # Non standard ODIM
        'normalized_coherent_power': 'SQIH',
        'normalized_coherent_power_vv': 'SQIV',
        'signal_to_noise_ratio_hh': 'SNRH',
        'signal_to_noise_ratio_vv': 'SNRV',
        'clutter_correction_hh': 'CCORH',  # Not used in Pyrad
        'clutter_correction_vv': 'CCORV',  # Not used in Pyrad
        'radar_estimated_rain_rate': 'RATE',
        'uncorrected_rain_rate': 'URATE',  # Not used in Pyrad
        'hail_intensity': 'HI',  # Not used in Pyrad
        'hail_probability': 'HP',  # Not used in Pyrad
        'accumulated_precipitation': 'ACRR',  # Not used in Pyrad
        'echotop_height': 'HGHT',  # Not used in Pyrad
        'vertical_integrated_liquid_water': 'VIL',  # Not used in Pyrad
        'velocity': 'VRADH',
        'velocity_vv': 'VRADV',
        'corrected_velocity': 'VRADHC',  # Non standard ODIM
        'dealiased_velocity': 'VRADDH',
        'dealiased_corrected_velocity': 'VRADDHC',  # Non standard ODIM
        'dealiased_velocity_vv': 'VRADDV',
        'retrieved_velocity': 'VRADEST',  # Non standard ODIM
        'retrieved_velocity_std': 'sd_vvp',  # special vol2bird
        'velocity_difference': 'VDIFF',  # Non standard ODIM
        'spectrum_width': 'WRADH',
        'corrected_spectrum_width': 'WRADHC',  # Non standard ODIM
        'spectrum_width_vv': 'WRADV',
        'eastward_wind_component': 'UWND',
        'northward_wind_component': 'VWND',
        'azimuthal_horizontal_wind_component': 'AHWND',  # Non standard ODIM
        'vertical_wind_component': 'w',  # Standard for vertical profile
        'radial_wind_shear': 'RSHR',  # Not used in Pyrad
        'azimuthal_wind_shear': 'ASHR',  # Not used in Pyrad
        'range_azimuthal_wind_shear': 'CSHR',  # Not used in Pyrad
        'elevation_wind_shear': 'ESHR',  # Not used in Pyrad
        'range_elevation_wind_shear': 'OSHR',  # Not used in Pyrad
        'horizontal_wind_shear': 'HSHR',  # Not used in Pyrad
        'vertical_wind_shear': 'VSHR',
        'three_dimensional_shear': 'TSHR',  # Not used in Pyrad
        'wind_speed': 'ff',  # Standard for vertical profile
        'wind_direction': 'dd',  # Standard for vertical profile
        'specific_attenuation': 'AH',  # Non standard ODIM
        'corrected_specific_attenuation': 'AHC',  # Non standard ODIM
        'path_integrated_attenuation': 'PIA',  # Non standard ODIM
        'corrected_path_integrated_attenuation': 'PIAC',  # Non standard ODIM
        'specific_differential_attenuation': 'ADP',  # Non standard ODIM
        'corrected_specific_differential_attenuation': 'ADPC',  # Non standard ODIM
        'path_integrated_differential_attenuation': 'PIDA',  # Non standard ODIM
        'corrected_path_integrated_differential_attenuation': 'PIDAC',  # Non standard ODIM
        'temperature': 'TEMP',  # Non standard ODIM
        'iso0': 'ISO0',  # Non standard ODIM
        'height_over_iso0': 'HISO0',  # Non standard ODIM
        'cosmo_index': 'COSMOIND',  # Non standard ODIM
        'hzt_index': 'HZTIND',  # Non standard ODIM
        'melting_layer': 'ML',  # Non standard ODIM
        'visibility': 'VIS',  # Non standard ODIM
        'radar_echo_id': 'ECHOID',  # Non standard ODIM
        'clutter_exit_code': 'CLT',  # Non standard ODIM
        'occurrence': 'OCC',  # Non standard ODIM
        'frequency_of_occurrence': 'OCCFREQ',  # Non standard ODIM
        'radar_border': 'BRDR',  # Not used in Pyrad
        'signal_quality_index': 'QUIND',
        'radar_echo_classification': 'CLASS',
        'hydroclass_entropy': 'ENTROPY',  # Non standard ODIM
        'proportion_AG': 'propAG',  # Non standard ODIM
        'proportion_CR': 'propCR',  # Non standard ODIM
        'proportion_LR': 'propLR',  # Non standard ODIM
        'proportion_RP': 'propRP',  # Non standard ODIM
        'proportion_RN': 'propRN',  # Non standard ODIM
        'proportion_VI': 'propVI',  # Non standard ODIM
        'proportion_WS': 'propWS',  # Non standard ODIM
        'proportion_MH': 'propMH',  # Non standard ODIM
        'proportion_IH': 'propIH',  # Non standard ODIM
        'time_avg_flag': 'TAFLAG',  # Non standard ODIM
        'colocated_gates': 'COLGATES',  # Non standard ODIM
        'number_of_samples': 'ns',  # standard vertical profile
        'bird_density': 'dens',  # standard vol2bird
        'standard_deviation': 'STD',  # Non standard ODIM
        'sum': 'SUM',  # Non standard ODIM
        'sum_squared': 'SUM2',  # Non standard ODIM
        'height_resolution': 'width',  # Special vol2bird
        'gap': 'gap',  # Special vol2bird
        'bird_reflectivity': 'eta',  # Special vol2bird
        'number_of_samples_velocity': 'n',  # Special vol2bird
        'number_of_samples_reflectivity': 'n_dbz',  # Special vol2bird
        'number_of_samples_velocity_all': 'n_all',  # Special vol2bird
        'number_of_samples_reflectivity_all': 'n_dbz_all'  # Special vol2bird
    }
    if field_name not in odim_quantity_dict:
        warn('Field name '+field_name+' does not have an ODIM quantity ' +
             'equivalent. The same field name will be used.')
        return field_name

    return odim_quantity_dict[field_name]


def _get_data_from_field(radar, sweep_ind, field_name, physical=True):
    """
    Extract data from radar field object with respect to different datasets
    and datatypes.

    Parameters:
    -----------
    radar : object
        Radar object
    sweep_ind : int
        sweep index, equal to ODIM dataset index
    field_name : str
        name of field to convert
    physical : bool
        If true the data will be stored in physical units. Otherwise it will
        be converted in binary

    Returns:
    --------
    data_dict : dict
        Dictionary containing the data, gain, offset, nodata and undetect

    """
    if isinstance(radar, Radar):
        data_ph = np.ma.asarray(
            radar.get_field(sweep_ind, field_name, copy=True))
    elif isinstance(radar, Grid):
        data_ph = np.ma.asarray(radar.fields[field_name]['data'])

    if physical:
        fill_value = radar.fields[field_name].get(
            '_FillValue', np.double(get_fillvalue()))
        if fill_value is not None:
            data = data_ph.filled(fill_value)
            nodata = fill_value
        else:
            data = np.asarray(data_ph)
            nodata = None
        undetect = None
        gain = 1.
        offset = 0.
    else:
        dtype = 'uint8'
        nvals = 256
        if '_Write_as_dtype' in radar.fields[field_name]:
            dtype = radar.fields[field_name]['_Write_as_dtype']
            if dtype not in ('uint8', 'uint16'):
                warn('Type '+dtype+' not supported. uint8 will be used')
                dtype = 'uint8'
                nvals = 256
            elif dtype == 'uint8':
                nvals = 256
            else:
                nvals = 65536

        nodata = radar.fields[field_name].get('nodata', 0)
        undetect = radar.fields[field_name].get('undetect', None)

        reserved = 0
        if nodata is not None:
            reserved = 1
        if undetect is not None:
            reserved += 1

        shift_up = 0
        if nodata == 0 or undetect == 0:
            shift_up = 1
        if nodata == 1 or undetect == 1:
            shift_up = 2

        if (('scale_factor' in radar.fields[field_name]) and
                ('add_offset' in radar.fields[field_name])):
            gain = radar.fields[field_name]['scale_factor']
            offset = radar.fields[field_name]['add_offset']

            shift_down = 0
            if nodata == nvals-1 or undetect == nvals-1:
                shift_down = 1
            if nodata == nvals-2 or undetect == nvals-2:
                shift_down = 2
            val_min = gain*shift_up+offset
            val_max = gain*(nvals-1-shift_down)+offset

            # Check if data within range
            ind = np.ma.where(data_ph < val_min)
            if ind[0].size > 0:
                warn(str(ind[0].size)+' data points below the minimum value ' +
                     str(val_min))
                data_ph[ind] = val_min
            ind = np.ma.where(data_ph > val_max)
            if ind[0].size > 0:
                warn(str(ind[0].size)+' data points above the maximum value ' +
                     str(val_max))
                print(data_ph[ind])
                data_ph[ind] = val_max
        else:
            val_min = np.ma.min(data_ph)
            val_max = np.ma.max(data_ph)
            nsteps = nvals-reserved
            gain = (val_max-val_min)/nsteps
            offset = val_min-shift_up*gain

            warn('gain and offset not specified. Optimal scale with gain ' +
                 str(gain)+' and offset '+str(offset)+' will be used')

        data = (data_ph-offset)/gain
        if nodata is not None:
            data = data.filled(nodata)
        data = np.asarray(data, dtype=eval('np.'+dtype))

    data_dict = {
        'data': data,
        'gain': gain,
        'offset': offset,
        'nodata': nodata,
        'undetect': undetect
    }
    return data_dict


def _map_radar_to_how_dict(radar_obj):
    """
    Tries to map data in a radar sub object (e.g. radar.instrument_parameters)
    to ODIM how attributes.

    Parameters:
    -----------
    radar_obj : Radar
        Containing a radar sub object

    Returns:
    --------
    dict_odim : dictionary
        Dictionary of ODIM how variables

    """
    # Variables in the instrument_parameter convention and
    # radar_parameters sub-convention that will be read from and written to
    # ODIM H5 files using Py-ART.
    # The meta_group attribute cannot be used to identify these parameters as
    # it is often set incorrectly.

    _INSTRUMENT_PARAMS = [
        # instrument parameters sub-convention
        'frequency',
        'pulse_width',
        'prt',
        'nyquist_velocity',
        'n_samples',
        # radar_parameters sub-convention
        'radar_antenna_gain_h',
        'radar_antenna_gain_v',
        'calibration_constant_hh',
        'calibration_constant_vv',
        'radar_beam_width_h',
        'radar_beam_width_v',
        'radar_receiver_bandwidth',
        'radar_rx_bandwidth'    # non-standard
        # 'radar_measured_transmit_power_h', # not found in radar object
        # 'radar_measured_transmit_power_v', # not found in radar object
        # 'measured_transmit_power_v',    # not found in radar object
        # 'measured_transmit_power_h'     # not found in radar object
        ]

    _RADAR_METADATA = [
        'system',
        'software',
        'sw_version'
        ]

    c = 299792458  # Speed of light

    dict_odim = {}

    for key in radar_obj.keys():
        if key in _INSTRUMENT_PARAMS:
            if key == 'frequency':
                dict_odim['wavelength'] = c/np.double(radar_obj[key]['data'])
            if key == 'pulse_width':
                dict_odim['pulsewidth'] = (
                    np.double(radar_obj[key]['data'])*1e6)
            if key == 'prt':
                dict_odim['highprf'] = 1 / np.double(radar_obj[key]['data'])
            if key == 'nyquist_velocity':
                dict_odim['NI'] = np.double(radar_obj[key]['data'])
            if key == 'n_samples':
                dict_odim['Vsamples'] = np.int64(radar_obj[key]['data'])
            if key == 'radar_antenna_gain_h':
                dict_odim['antgainH'] = np.double(radar_obj[key]['data'])
            if key == 'radar_antenna_gain_v':
                dict_odim['antgainV'] = np.double(radar_obj[key]['data'])
            if key == 'calibration_constant_hh':
                dict_odim['radconstH'] = np.double(radar_obj[key]['data'])
            if key == 'calibration_constant_vv':
                dict_odim['radconstV'] = np.double(radar_obj[key]['data'])
            if key == 'radar_beam_width_h':
                dict_odim['beamwH'] = np.double(radar_obj[key]['data'])
            if key == 'radar_beam_width_v':
                dict_odim['beamwV'] = np.double(radar_obj[key]['data'])
            if key in ['radar_receiver_bandwidth', 'radar_rx_bandwidth']:
                dict_odim['RXbandwidth'] = np.double(radar_obj[key]['data'])
        elif key in _RADAR_METADATA:
            dict_odim[key] = _to_str(radar_obj[key])
        else:
            warn("Unknown how parameter: {}, not written to file.".format(key))

    return dict_odim


def proj4_to_str(proj4dict):
    """ Convert proj4 dict to string format"""
    proj4str = ''
    for s in proj4dict.keys():
        if s == 'no_defs':
            proj4str += '+no_defs'
        else:
            proj4str += '+'+s+'='+str(proj4dict[s])
        proj4str += ' '
    proj4str = proj4str.strip()  # remove trailing whitespace
    return proj4str
