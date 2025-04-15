"""
pyart.core.radar
================

A general central radial scanning (or dwelling) spectra instrument class.

.. autosummary::
    :toctree: generated/
    :template: dev_template.rst

    RadarSpectra


"""

import sys

import numpy as np

from .radar import Radar


class RadarSpectra(Radar):
    """
    A class for storing antenna coordinate radar spectra data. Based on the
    radar object class

    The structure of the Radar class is based on the CF/Radial Data file
    format.  Global attributes and variables (section 4.1 and 4.3) are
    represented as a dictionary in the metadata attribute.  Other required and
    optional variables are represented as dictionaries in a attribute with the
    same name as the variable in the CF/Radial standard.  When a optional
    attribute not present the attribute has a value of None.  The data for a
    given variable is stored in the dictionary under the 'data' key.  Moment
    field data is stored as a dictionary of dictionaries in the fields
    attribute.  Sub-convention variables are stored as a dictionary of
    dictionaries under the meta_group attribute.

    Refer to the attribute section for information on the parameters.

    Attributes
    ----------
    time : dict
        Time at the center of each ray.
    range : dict
        Range to the center of each gate (bin).
    npulses : dict
        number of pulses for each ray
    Doppler_velocity : dict or None
        The Doppler velocity of each Doppler bin. The data has dimensions
        nrays x npulses_max
    Doppler_frequency : dict or None
        The Doppler frequency of each Doppler bin. The data has dimensions
        nrays x npulses_max
    fields : dict of dicts
        Moment fields. The data has dimensions nrays x ngates x npulses_max
    metadata : dict
        Metadata describing the instrument and data.
    scan_type : str
        Type of scan, one of 'ppi', 'rhi', 'sector' or 'other'.  If the scan
        volume contains multiple sweep modes this should be 'other'.
    latitude : dict
        Latitude of the instrument.
    longitude : dict
        Longitude of the instrument.
    altitude : dict
        Altitude of the instrument, above sea level.
    altitude_agl : dict or None
        Altitude of the instrument above ground level.  If not provided this
        attribute is set to None, indicating this parameter not available.
    sweep_number : dict
        The number of the sweep in the volume scan, 0-based.
    sweep_mode : dict
        Sweep mode for each mode in the volume scan.
    fixed_angle : dict
        Target angle for thr sweep.  Azimuth angle in RHI modes, elevation
        angle in all other modes.
    sweep_start_ray_index : dict
        Index of the first ray in each sweep relative to the start of the
        volume, 0-based.
    sweep_end_ray_index : dict
        Index of the last ray in each sweep relative to the start of the
        volume, 0-based.
    rays_per_sweep : LazyLoadDict
        Number of rays in each sweep.  The data key of this attribute is
        create upon first access from the data in the sweep_start_ray_index and
        sweep_end_ray_index attributes.  If the sweep locations needs to be
        modified, do this prior to accessing this attribute or use
        :py:func:`init_rays_per_sweep` to reset the attribute.
    target_scan_rate : dict or None
        Intended scan rate for each sweep.  If not provided this attribute is
        set to None, indicating this parameter is not available.
    rays_are_indexed : dict or None
        Indication of whether ray angles are indexed to a regular grid in
        each sweep.  If not provided this attribute is set to None, indicating
        ray angle spacing is not determined.
    ray_angle_res : dict or None
        If rays_are_indexed is not None, this provides the angular resolution
        of the grid.  If not provided or available this attribute is set to
        None.
    azimuth : dict
        Azimuth of antenna, relative to true North. Azimuth angles are
        recommended to be expressed in the range of [0, 360], but other
        representations are not forbidden.
    elevation : dict
        Elevation of antenna, relative to the horizontal plane. Elevation
        angles are recommended to be expressed in the range of [-180, 180],
        but other representations are not forbidden.
    gate_x, gate_y, gate_z : LazyLoadDict
        Location of each gate in a Cartesian coordinate system assuming a
        standard atmosphere with a 4/3 Earth's radius model. The data keys of
        these attributes are create upon first access from the data in the
        range, azimuth and elevation attributes. If these attributes are
        changed use :py:func:`init_gate_x_y_z` to reset.
    gate_longitude, gate_latitude : LazyLoadDict
        Geographic location of each gate.  The projection parameter(s) defined
        in the `projection` attribute are used to perform an inverse map
        projection from the Cartesian gate locations relative to the radar
        location to longitudes and latitudes. If these attributes are changed
        use :py:func:`init_gate_longitude_latitude` to reset the attributes.
    projection : dic or str
        Projection parameters defining the map projection used to transform
        from Cartesian to geographic coordinates.  The default dictionary sets
        the 'proj' key to 'pyart_aeqd' indicating that the native Py-ART
        azimuthal equidistant projection is used. This can be modified to
        specify a valid pyproj.Proj projparams dictionary or string.
        The special key '_include_lon_0_lat_0' is removed when interpreting
        this dictionary. If this key is present and set to True, which is
        required when proj='pyart_aeqd', then the radar longitude and
        latitude will be added to the dictionary as 'lon_0' and 'lat_0'.
    gate_altitude : LazyLoadDict
        The altitude of each radar gate as calculated from the altitude of the
        radar and the Cartesian z location of each gate.  If this attribute
        is changed use :py:func:`init_gate_altitude` to reset the attribute.
    scan_rate : dict or None
        Actual antenna scan rate.  If not provided this attribute is set to
        None, indicating this parameter is not available.
    antenna_transition : dict or None
        Flag indicating if the antenna is in transition, 1 = yes, 0 = no.
        If not provided this attribute is set to None, indicating this
        parameter is not available.
    rotation : dict or None
        The rotation angle of the antenna.  The angle about the aircraft
        longitudinal axis for a vertically scanning radar.
    tilt : dict or None
        The tilt angle with respect to the plane orthogonal (Z-axis) to
        aircraft longitudinal axis.
    roll : dict or None
        The roll angle of platform, for aircraft right wing down is positive.
    drift : dict or None
        Drift angle of antenna, the angle between heading and track.
    heading : dict or None
        Heading (compass) angle, clockwise from north.
    pitch : dict or None
        Pitch angle of antenna, for aircraft nose up is positive.
    georefs_applied : dict or None
        Indicates whether the variables have had georeference calculation
        applied.  Leading to Earth-centric azimuth and elevation angles.
    instrument_parameters : dict of dicts or None
        Instrument parameters, if not provided this attribute is set to None,
        indicating these parameters are not avaiable.  This dictionary also
        includes variables in the radar_parameters CF/Radial subconvention.
    radar_calibration : dict of dicts or None
        Instrument calibration parameters.  If not provided this attribute is
        set to None, indicating these parameters are not available
    ngates : int
        Number of gates (bins) in a ray.
    nrays : int
        Number of rays in the volume.
    npulses_max : int
        Maximum number of pulses per ray in the volume.
    nsweeps : int
        Number of sweep in the volume.

    """

    def __init__(
        self,
        time,
        _range,
        fields,
        metadata,
        scan_type,
        latitude,
        longitude,
        altitude,
        sweep_number,
        sweep_mode,
        fixed_angle,
        sweep_start_ray_index,
        sweep_end_ray_index,
        azimuth,
        elevation,
        npulses,
        Doppler_velocity=None,
        Doppler_frequency=None,
        altitude_agl=None,
        target_scan_rate=None,
        rays_are_indexed=None,
        ray_angle_res=None,
        scan_rate=None,
        antenna_transition=None,
        instrument_parameters=None,
        radar_calibration=None,
        rotation=None,
        tilt=None,
        roll=None,
        drift=None,
        heading=None,
        pitch=None,
        georefs_applied=None,
    ):

        if "calendar" not in time:
            time["calendar"] = "gregorian"
        self.time = time
        self.range = _range

        self.fields = fields
        self.metadata = metadata
        self.scan_type = scan_type

        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        self.altitude_agl = altitude_agl  # optional

        self.sweep_number = sweep_number
        self.sweep_mode = sweep_mode
        self.fixed_angle = fixed_angle
        self.sweep_start_ray_index = sweep_start_ray_index
        self.sweep_end_ray_index = sweep_end_ray_index

        self.target_scan_rate = target_scan_rate  # optional
        self.rays_are_indexed = rays_are_indexed  # optional
        self.ray_angle_res = ray_angle_res  # optional

        self.azimuth = azimuth
        self.elevation = elevation
        self.Doppler_velocity = Doppler_velocity  # optional
        self.Doppler_frequency = Doppler_frequency  # optional
        self.scan_rate = scan_rate  # optional
        self.antenna_transition = antenna_transition  # optional
        self.rotation = rotation  # optional
        self.tilt = tilt  # optional
        self.roll = roll  # optional
        self.drift = drift  # optional
        self.heading = heading  # optional
        self.pitch = pitch  # optional
        self.georefs_applied = georefs_applied  # optional

        self.instrument_parameters = instrument_parameters  # optional
        self.radar_calibration = radar_calibration  # optional

        self.ngates = len(_range["data"])
        self.nrays = len(time["data"])
        self.npulses = npulses
        self.npulses_max = np.max(npulses["data"])
        self.nsweeps = len(sweep_number["data"])
        self.projection = {"proj": "pyart_aeqd", "_include_lon_0_lat_0": True}

        # initalize attributes with lazy load dictionaries
        self.init_rays_per_sweep()
        self.init_gate_x_y_z()
        self.init_gate_longitude_latitude()
        self.init_gate_altitude()

    def add_field(self, field_name, dic, replace_existing=False):
        """
        Add a field to the object.

        Parameters
        ----------
        field_name : str
            Name of the field to add to the dictionary of fields.
        dic : dict
            Dictionary contain field data and metadata.
        replace_existing : bool
            True to replace the existing field with key field_name if it
            exists, loosing any existing data.  False will raise a ValueError
            when the field already exists.

        """
        # check that the field dictionary to add is valid
        if field_name in self.fields and replace_existing is False:
            err = f"A field with name: {field_name} already exists"
            raise ValueError(err)
        if "data" not in dic:
            raise KeyError("dic must contain a 'data' key")
        # Check if field is in spectral shape (self.nrays, self.ngates, self.npulses_max)
        if len(dic["data"]) == 3:
            if dic["data"].shape != (self.nrays, self.ngates, self.npulses_max):
                t = (
                    self.nrays,
                    self.ngates,
                    self.npulses_max,
                    dic["data"].shape[0],
                    dic["data"].shape[1],
                    dic["data"].shape[2],
                )
                err = (
                    "'data' has invalid shape, "
                    + f"should be ({t[0]}, {t[1]}, {t[2]}) but is ({t[3]}, {t[4]}, {t[5]})"
                )
                raise ValueError(err)
        elif len(dic["data"]) == 2:
            if dic["data"].shape != (self.nrays, self.ngates):
                t = (
                    self.nrays,
                    self.ngates,
                    dic["data"].shape[0],
                    dic["data"].shape[1],
                )
                err = (
                    "'data' has invalid shape, "
                    + f"should be ({t[0]}, {t[1]}) but is ({t[2]}, {t[3]})"
                )
                raise ValueError(err)
        # add the field
        self.fields[field_name] = dic

    def add_field_like(
        self, existing_field_name, field_name, data, replace_existing=False
    ):
        """
        Add a field to the object with metadata from a existing field.

        Note that the data parameter is not copied by this method.
        If data refers to a 'data' array from an existing field dictionary, a
        copy should be made within or prior to using this method.  If this is
        not done the 'data' key in both field dictionaries will point to the
        same NumPy array and modification of one will change the second.  To
        copy NumPy arrays use the copy() method.  See the Examples section
        for how to create a copy of the 'reflectivity' field as a field named
        'reflectivity_copy'.

        Parameters
        ----------
        existing_field_name : str
            Name of an existing field to take metadata from when adding
            the new field to the object.
        field_name : str
            Name of the field to add to the dictionary of fields.
        data : array
            Field data. A copy of this data is not made, see the note above.
        replace_existing : bool
            True to replace the existing field with key field_name if it
            exists, loosing any existing data.  False will raise a ValueError
            when the field already exists.

        Examples
        --------
        >>> radar.add_field_like('reflectivity', 'reflectivity_copy',
        ...                      radar.fields['reflectivity']['data'].copy())

        """
        if existing_field_name not in self.fields:
            err = f"field {existing_field_name} does not exist in object"
            raise ValueError(err)
        dic = {}
        for k, v in self.fields[existing_field_name].items():
            if k != "data":
                dic[k] = v
        dic["data"] = data
        return self.add_field(field_name, dic, replace_existing=replace_existing)

    def extract_sweeps(self, sweeps):
        """
        Create a new radar contains only the data from select sweeps.

        Parameters
        ----------
        sweeps : array_like
            Sweeps (0-based) to include in new Radar object.

        Returns
        -------
        radar : Radar
            Radar object which contains a copy of data from the selected
            sweeps.

        """

        # parse and verify parameters
        sweeps = np.array(sweeps, dtype="int32")
        if np.any(sweeps > (self.nsweeps - 1)):
            raise ValueError(
                "invalid sweeps indices in sweeps parameter. "
                + "sweeps: "
                + " ".join(str(sweeps))
                + " nsweeps: "
                + str(self.nsweeps)
            )
        if np.any(sweeps < 0):
            raise ValueError("only positive sweeps can be extracted")

        def mkdic(dic, select):
            """Make a dictionary, selecting out select from data key"""
            if dic is None:
                return None
            d = dic.copy()
            if "data" in d and select is not None:
                d["data"] = d["data"][select].copy()
            return d

        # create array of rays which select the sweeps selected and
        # the number of rays per sweep.
        ray_count = (
            self.sweep_end_ray_index["data"] - self.sweep_start_ray_index["data"] + 1
        )[sweeps]
        ssri = self.sweep_start_ray_index["data"][sweeps]
        rays = np.concatenate(
            [range(s, s + e) for s, e in zip(ssri, ray_count)]
        ).astype("int32")

        # radar location attribute dictionary selector
        if len(self.altitude["data"]) == 1:
            loc_select = None
        else:
            loc_select = sweeps

        # create new dictionaries
        time = mkdic(self.time, rays)
        _range = mkdic(self.range, None)

        fields = {}
        for field_name, dic in self.fields.items():
            fields[field_name] = mkdic(dic, rays)
        metadata = mkdic(self.metadata, None)
        scan_type = str(self.scan_type)

        latitude = mkdic(self.latitude, loc_select)
        longitude = mkdic(self.longitude, loc_select)
        altitude = mkdic(self.altitude, loc_select)
        altitude_agl = mkdic(self.altitude_agl, loc_select)

        sweep_number = mkdic(self.sweep_number, sweeps)
        sweep_mode = mkdic(self.sweep_mode, sweeps)
        fixed_angle = mkdic(self.fixed_angle, sweeps)
        sweep_start_ray_index = mkdic(self.sweep_start_ray_index, None)
        sweep_start_ray_index["data"] = np.cumsum(np.append([0], ray_count[:-1]))
        sweep_end_ray_index = mkdic(self.sweep_end_ray_index, None)
        sweep_end_ray_index["data"] = np.cumsum(ray_count) - 1
        target_scan_rate = mkdic(self.target_scan_rate, sweeps)

        azimuth = mkdic(self.azimuth, rays)
        elevation = mkdic(self.elevation, rays)
        Doppler_velocity = mkdic(self.Doppler_velocity, rays)
        Doppler_frequency = mkdic(self.Doppler_frequency, rays)
        scan_rate = mkdic(self.scan_rate, rays)
        antenna_transition = mkdic(self.antenna_transition, rays)

        # instrument_parameters
        # Filter the instrument_parameter dictionary based size of leading
        # dimension, this might not always be correct.
        if self.instrument_parameters is None:
            instrument_parameters = None
        else:
            instrument_parameters = {}
            for key, dic in self.instrument_parameters.items():
                if dic["data"].ndim != 0:
                    dim0_size = dic["data"].shape[0]
                else:
                    dim0_size = -1
                if dim0_size == self.nsweeps:
                    fdic = mkdic(dic, sweeps)
                elif dim0_size == self.nrays:
                    fdic = mkdic(dic, rays)
                else:  # keep everything
                    fdic = mkdic(dic, None)
                instrument_parameters[key] = fdic

        # radar_calibration
        # copy all field in radar_calibration as is except for
        # r_calib_index which we filter based upon time.  This might
        # leave some indices in the "r_calib" dimension not referenced in
        # the r_calib_index array.
        if self.radar_calibration is None:
            radar_calibration = None
        else:
            radar_calibration = {}
            for key, dic in self.radar_calibration.items():
                if key == "r_calib_index":
                    radar_calibration[key] = mkdic(dic, rays)
                else:
                    radar_calibration[key] = mkdic(dic, None)

        return RadarSpectra(
            time,
            _range,
            fields,
            metadata,
            scan_type,
            latitude,
            longitude,
            altitude,
            sweep_number,
            sweep_mode,
            fixed_angle,
            sweep_start_ray_index,
            sweep_end_ray_index,
            azimuth,
            elevation,
            self.npulses,
            Doppler_velocity=Doppler_velocity,
            Doppler_frequency=Doppler_frequency,
            altitude_agl=altitude_agl,
            target_scan_rate=target_scan_rate,
            scan_rate=scan_rate,
            antenna_transition=antenna_transition,
            instrument_parameters=instrument_parameters,
            radar_calibration=radar_calibration,
        )

    def info(self, level="standard", out=sys.stdout):
        """
        Print information on radar.

        Parameters
        ----------
        level : {'compact', 'standard', 'full', 'c', 's', 'f'}
            Level of information on radar object to print, compact is
            minimal information, standard more and full everything.
        out : file-like
            Stream to direct output to, default is to print information
            to standard out (the screen).

        """
        if level == "c":
            level = "compact"
        elif level == "s":
            level = "standard"
        elif level == "f":
            level = "full"

        if level not in ["standard", "compact", "full"]:
            raise ValueError("invalid level parameter")

        self._dic_info("altitude", level, out)
        self._dic_info("altitude_agl", level, out)
        self._dic_info("antenna_transition", level, out)
        self._dic_info("azimuth", level, out)
        self._dic_info("elevation", level, out)

        print("fields:", file=out)
        for field_name, field_dic in self.fields.items():
            self._dic_info(field_name, level, out, field_dic, 1)

        self._dic_info("fixed_angle", level, out)

        if self.instrument_parameters is None:
            print("instrument_parameters: None", file=out)
        else:
            print("instrument_parameters:", file=out)
            for name, dic in self.instrument_parameters.items():
                self._dic_info(name, level, out, dic, 1)

        self._dic_info("latitude", level, out)
        self._dic_info("longitude", level, out)

        print("nsweeps:", self.nsweeps, file=out)
        print("ngates:", self.ngates, file=out)
        print("nrays:", self.nrays, file=out)
        print("npulses_max:", self.npulses_max, file=out)

        if self.radar_calibration is None:
            print("radar_calibration: None", file=out)
        else:
            print("radar_calibration:", file=out)
            for name, dic in self.radar_calibration.items():
                self._dic_info(name, level, out, dic, 1)

        self._dic_info("range", level, out)
        self._dic_info("scan_rate", level, out)
        print("scan_type:", self.scan_type, file=out)
        self._dic_info("sweep_end_ray_index", level, out)
        self._dic_info("sweep_mode", level, out)
        self._dic_info("sweep_number", level, out)
        self._dic_info("sweep_start_ray_index", level, out)
        self._dic_info("target_scan_rate", level, out)
        self._dic_info("time", level, out)

        # Airborne radar parameters
        if self.rotation is not None:
            self._dic_info("rotation", level, out)
        if self.tilt is not None:
            self._dic_info("tilt", level, out)
        if self.roll is not None:
            self._dic_info("roll", level, out)
        if self.drift is not None:
            self._dic_info("drift", level, out)
        if self.heading is not None:
            self._dic_info("heading", level, out)
        if self.pitch is not None:
            self._dic_info("pitch", level, out)
        if self.georefs_applied is not None:
            self._dic_info("georefs_applied", level, out)

        # always print out all metadata last
        self._dic_info("metadata", "full", out)
