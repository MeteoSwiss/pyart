# Changelog

## v1.8.5

**Bug fixes:**
- [fixes in gecsx for irregular azimuth angles in vol scan](https://github.com/MeteoSwiss/pyart/commit/792050b340b1f3180b19ceee7bc2385446b30777)
- [correct reorder of data in odim](https://github.com/MeteoSwiss/pyart/commit/8378f1ad7ec79e6c9cb699f9977bc268e503b2a1)
- [fix bug in time reading of metranet file](https://github.com/MeteoSwiss/pyart/commit/3809cdc6cdc1eafc237bfc5b3fc99144f5ae7aef)
- [bug corrections in skyecho.py. Added function extract_sweeps_skyecho in skyecho.py](https://github.com/MeteoSwiss/pyart/commit/e7695a509ef370fd8d6089ccab9774742c76561f)

**New additions**
- [added file pyart/aux_io/knmi_h5.py with a reader of the KNMI H5 gridded radar data](https://github.com/MeteoSwiss/pyart/commit/3f612d5df06b56b93ef5c78c30f07179ec49e2f6)
- [added reader for SkyEcho proprietary netcdf data](https://github.com/MeteoSwiss/pyart/commit/da4e205010d8059e241fb9ea03b48bb85122ca67)
- [add reader for SwissBirdRadar spectral data](https://github.com/MeteoSwiss/pyart/commit/592cd85a6c5729328e42473a5b1d538f369e4a58)
- [add dealias routine for Doppler spectrum](https://github.com/MeteoSwiss/pyart/commit/58b81d837e635e5e51a6cf6c6d95f36405bbdf0b)


## v1.8.4

**Bug fixes:**
- [fix in odim h5 reader and writer](https://github.com/MeteoSwiss/pyart/commit/7737a66ba269b992aaba895e221ce2d86d930122)
- [correct reorder of data in odim](https://github.com/MeteoSwiss/pyart/commit/8378f1ad7ec79e6c9cb699f9977bc268e503b2a1)

**New additions**
- [added windshear processing for lidar data](https://github.com/MeteoSwiss/pyart/commit/ffb78f672d4c3e9035e5d4a7cb439612de06b8d2)
- [changed latlon plots to cross_section in GridMapDisplay, to be consistent with Py-ART ARM](https://github.com/MeteoSwiss/pyart/commit/b2c1da3ee26615fb4f4554d18825c7b4997dbb67)

## v1.8.3

**Bug fixes:**
- [fix in _rsl_interface.pyx for Cython 3.0](https://github.com/MeteoSwiss/pyart/commit/84b0834d62a5ffe42750c9e216dd4e5e7334b5ba)

**New additions:**
- [added support for older python versions in pyart/testing](https://github.com/MeteoSwiss/pyart/commit/94fe908281d884399b58f8f73723c6791b73d092)

## v1.8.2

**Bug fixes:**
- [update of metranet read_product code for new cartesian format](https://github.com/MeteoSwiss/pyart/commit/8560fbfd1f36ebaf0eb89bf2ac2cf336e64c9022)
- [fixes for change in cython 3.0 behaviour](https://github.com/MeteoSwiss/pyart/commit/6968109da8c61057dbb8907fd4da84992d7fe385)
- Various deprecation fixes (matplotlib, numpy)

**New additions**
- [added conv_strat_yuter function from ARM Py-ART](https://github.com/MeteoSwiss/pyart/commit/c9a620a23a333110cb1c78e321d60b488b3f83ee)
- [Merge pull request](https://github.com/MeteoSwiss/pyart/commit/d58cb079dff559abfd595002075eadbf3ef1415a) https://github.com/MeteoSwiss/pyart/pull/21 [from juhi24/dataset-patch](https://github.com/MeteoSwiss/pyart/commit/d58cb079dff559abfd595002075eadbf3ef1415a)

## v1.8.1

**Bug fixes:**
- [fix in radar_utils, replaced latitude by altitude as it was duplicated](https://github.com/MeteoSwiss/pyart/commit/af1769d5be122b54038d21fa48b535c1c5b59f5e)
- [fix for deprecated matplotlib register_cmap](https://github.com/MeteoSwiss/pyart/commit/1b167e07ea79863d3a371117063d87be9baff31d)
- [fix of a bug in gecsx that lead to always using raster oversampling](https://github.com/MeteoSwiss/pyart/commit/963b01f9cd54cc044bbd98182491e191ceab4091)
- [fix scipy deprecation warning in ml.py](https://github.com/MeteoSwiss/pyart/commit/43d06bc1cdc0b6192be54a66d2c2d2d96718ec49)

**New additions**
- [added option coerce_angles in join_radars to account for antenna misposition](https://github.com/MeteoSwiss/pyart/commit/ac12ef8f6934bdc044632089578a4c3b9153672b)

## v1.8

**Bug fixes:**
 - [improvement in colobar label of pyart/graph/gridmapdisplay.py](https://github.com/MeteoSwiss/pyart/commit/b46a45114913a81d17eb31be4152524b40d319eb)
- [added _label_axes_latlon in class GridMapDisplay. Minor changes to improve grid plots](https://github.com/MeteoSwiss/pyart/commit/77628002f54c9ace0614dcd4df4ee40193251434) 
- [update rad4alp_gif_reader.py for new imageio version](https://github.com/MeteoSwiss/pyart/commit/0c84b9ef8ec21856b1770aedcb9f99a316f2e22f)
- [Fix in gecsx oversampling, nrows and ncols were flipped](https://github.com/MeteoSwiss/pyart/commit/05683b82a1ca7b1b1642d74f45df4d89a268b88a)

**New additions:**
- [improvements to vad.py, option for sign + multi-sweep Browning VAD](https://github.com/MeteoSwiss/pyart/commit/4a21d7af3602105a4cb6176d5f93e0ec6903441d)
- [changed rad4alp gif lookup to make it agree with one of CPC](https://github.com/MeteoSwiss/pyart/commit/24ac020a009d3c3099666dfd565c7177eeb52b5a)

## v1.7.1

**Bug fixes:**
- [bug correction when masking undetect and nodata in odim_h5 reader](https://github.com/MeteoSwiss/pyart/commit/b3a1b0f6102143be0049490f4f4dc90f5d4bad8c)

## v1.7

**New additions:**
-  New plot function: _plot_xsection_ in _RadarDisplay_ which can be used to display a cross-section of polar data between arbitrary coordinates

## v1.6.4

### New additions

**New additions:**
- Support for new RainForest ODIM variable name

## v1.6.3

**Bug fixes:**
- bug correction in compute_refl_time_avg in vpr.py: it was crashing when the elevation to average was not in the current scan
- bug correction in vpr.py compute_refl_time_avg we set the radar_ou.time['data'] to np.zeros(nrays) to account for elevations to average not in the current radar

**New additions:**
-  VPR retrieval: better care of special case when melting layer peak has a value of 1
- support for multi field Grid in odim h5 grid writer
-  change of datatype for time from float32 to float in read_odim_h5
- better support of OPERA guidelines in odim writer and reader
- Added possibility to specify ODIM convention in ODIM writers

## v1.6.2

**Bug fixes:**
  - Fixed an issue that was prohibiting ODIM grid files written by pyart to be read by pyart again

## v1.6.1

**Bug fixes**

- fixed a bug in write_odim_grid that was using a time format different from the other pyart readers/writers

## v1.6

**Bug fixes:**
- adaptation of the Cython codes of the RSL interface and 4DD to make them run with setuptools
- rounding of the angles to a precision of 6 digits in the interpol_field code so we avoid floating point issues in the interpolation range (leads to northernmost ray being empty in some cases)

**New additions:**
-  new functions to accumulated gridded data and get data at multiple points in a grid
-  addition of a new spatialized VPR function
