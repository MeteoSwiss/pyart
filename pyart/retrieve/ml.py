"""
pyart.retrieve.ml
=========================================

Routines to detect the ML from polarimetric RHI scans.

.. autosummary::
    :toctree: generated/

    detect_ml
    _process_map_ml
    _process_map_ml_only_zh
    _r_to_h
    _remap_to_polar
    _normalize_image
    _gradient_2D
    _convolve_with_nan
    _mean_filter
    _calc_sub_ind

"""

import numpy as np
from scipy.ndimage.filters import convolve
from scipy.interpolate import InterpolatedUnivariateSpline, pchip

from ..config import get_field_name, get_metadata, get_fillvalue
from ..map.polar_to_cartesian import get_earth_radius, polar_to_cartesian

# Parameters
# They shouldn't be changed ideally
MAXTHICKNESS_ML = 1000
LOWMLBOUND = 0.7
UPMLBOUND = 1.3
SIZEFILT_M = 75
ZH_IM_BOUNDS = (10, 60)
RHOHV_IM_BOUNDS = (0.75, 1)
RHOHV_VALID_BOUNDS = (0.6, 1)
KE = 4 / 3.  # Constant in the 4/3 earth radius model


def detect_ml(radar, gatefilter=None, fill_value=None, refl_field=None, 
              rhohv_field=None, melting_layer_field = None, 
              melting_layer_height_field = None, max_range=20000, 
              detect_threshold=0.02, interp_holes=False, max_length_holes=250,
              check_min_length=True):
    '''
    Detects the melting layer (ML) on set of RHI scans of reflectivity and
    copolar correlation coefficient and returns its properties both in the
    original polar radar coordinates and in projected Cartesian coordinates

    Inputs:
        radar : Radar
            A Radar class instance
        gatefilter : GateFilter, optional
            A GateFilter indicating radar gates that should be excluded when
            analysing differential phase measurements.
        fill_value : float, optional
            Value indicating missing or bad data in differential phase 
            field, if  not specified, the default in the Py-ART 
            configuration file will be used
        refl_field : str, optional
            Reflectivity field. If None, the default field name must be
            specified in the Py-ART configuration file.
        rhohv_field : str, optional
            Copolar correlation coefficient field. If None, the default 
            field name must be specified in the Py-ART configuration file.
        melting_layer_field : str, optional
            Melting layer field. If None, the default field name must
            be specified in the Py-ART configuration file.
        melting_layer_height_field : str, optional
            Melting layer height field. If None, the default field name must
            be specified in the Py-ART configuration file.
        max_range : float, optional
            the max. range from the radar to be used in the ML determination
        detect_threshold : float, optional
            the detection threshold (see paper), you can play around and 
            see how it affects the output. Lowering the value makes the 
            algorithm more sensitive but increases the number of 
            erroneous detections.
        interp_holes : bool, optional
            Flag to allow for interpolation of small holes in the detected ML
        max_length_holes : float, optional
            The maximum size of holes in the ML for them to be interpolated
        check_min_length : bool, optional
            If true, the length of the detected ML will
            be compared with the length of the valid data and the
            ML will be kept only if sufficiently long

    Reference:
    ----------
    Wolfensberger, D. , Scipion, D. and Berne, A. (2016), Detection and
    characterization of the melting layer based on polarimetric radar scans.
    Q.J.R. Meteorol. Soc., 142: 108-124. doi:10.1002/qj.2672
    '''

    # Check if all sweeps are RHI
    for sweep_type in radar.sweep_mode['data']:
        if sweep_type not in ['rhi', 'manual_rhi', 'elevation_surveillance']:
            msg = """
            Currently this functions supports only scans where all sweeps are
            RHIs...
            Aborting.
            """
            raise ValueError(msg)
            # TODO add support for sector scans

    # parse fill value
    if fill_value is None:
        fill_value = get_fillvalue()

    # parse field names
    if refl_field is None:
        refl_field = get_field_name('reflectivity')
    if rhohv_field is None:
        rhohv_field = get_field_name('copolar_correlation_coefficient')
    if melting_layer_field is None:
        melting_layer_field = get_field_name('melting_layer')
    if melting_layer_height_field is None:
        melting_layer_height_field = get_field_name('melting_layer_height')
        
     # mask radar gates indicated by the gate filter
    if gatefilter is not None:
        refl_field = np.ma.masked_where(gatefilter.gate_excluded, refl_field)
        rhohv_field = np.ma.masked_where(gatefilter.gate_excluded, rhohv_field)
        
    all_ml = []
    
    for sweep in range(radar.nsweeps):
        radar_sweep = radar.extract_sweeps([sweep])
        
        out = _detect_ml_sweep(
            radar_sweep, fill_value, refl_field, rhohv_field,
            melting_layer_field, melting_layer_height_field, max_range, 
            detect_threshold, interp_holes,  max_length_holes, 
            check_min_length)

        all_ml.append(out)

    return all_ml

def _detect_ml_sweep(radar_sweep, fill_value, refl_field, rhohv_field, 
                     melting_layer_field, melting_layer_height_field,
                     max_range, detect_threshold, interp_holes, 
                     max_length_holes, check_min_length):

    '''
    Detects the melting layer (ML) on an RHI scan of reflectivity and copolar
    correlation coefficient and returns its properties both in the original
    polar radar coordinates and in projected Cartesian coordinates

    Parameters
    ----------
        radar_sweep : Radar
            A Radar class instance of a single sweep
        fill_value : float
            Value indicating missing or bad data in differential phase   
        refl_field : str
            Reflectivity field. If None, the default field name must be
            specified in the Py-ART configuration file.
        rhohv_field : str
            Copolar correlation coefficient field. 
        melting_layer_field : str
            Melting layer field. 
        melting_layer_height_field : str
            Melting layer height field.     
        max_range : float
            the max. range from the radar to be used in the ML determination
        detect_threshold : float
            the detection threshold (see paper), you can play around and 
            see how it affects the output. Lowering the value makes the 
            algorithm more sensitive but increases the number of 
            erroneous detections.
        interp_holes : bool
            Flag to allow for interpolation of small holes in the detected ML
        max_length_holes : float
            The maximum size of holes in the ML for them to be interpolated
        check_min_length : bool
            If true, the length of the detected ML will
            be compared with the length of the valid data and the
            ML will be kept only if sufficiently long

    Returns
    -------
    ml : dict
        ml is a dictionnary with the following fields:
        ​ml_pol (a dict with the following keys):
            theta (list of elevation angles)
            range (list of ranges)
            data (2D map with 1 where detected ML and 0 otherwise)
            bottom_ml (the height above the radar of the ML bottom for
                       every angle theta)
            top_ml (the height above the radar of the ML top for every angle
                    theta)

        ml_cart (a dict with the following keys):
            x : x-coordinates of the Cartesian system (distance at ground)
            z : z-coordinates of the Cartesian system (height above surface)
            data (2D map with 1 where detected ML and 0 otherwise)
            bottom_ml (the height above the radar of the ML bottom for every
                       distance x)
            top_ml (the height above the radar of the ML top for every
                    distance x)
        ​ml_exists (a boolean flag = 1 if a ML was detected)
    '''
    
    # Get the labels that will be used for the melting layer
    mdata_ml = get_metadata(melting_layer_field)
    for label, value in zip(mdata_ml['labels'],mdata_ml['ticks']):
        mdata_ml[label] = value

    # Project to cartesian coordinates
    coords_c, refl_field_c, mapping = polar_to_cartesian(radar_sweep,
                                                         refl_field,
                                                         max_range=max_range)
    coords_c, rhohv_field_c, _ = polar_to_cartesian(radar_sweep,
                                                    rhohv_field,
                                                    mapping=mapping)
    cart_res = mapping['res']

    # Get Zh and Rhohv images
    refl_im = _normalize_image(refl_field_c, *ZH_IM_BOUNDS)
    rhohv_im = _normalize_image(rhohv_field_c, *RHOHV_IM_BOUNDS)

    # Combine images
    comb_im = (1 - rhohv_im) * refl_im
    comb_im[np.isnan(comb_im)] = 0.

    # Get vertical gradient
    size_filt = np.floor(SIZEFILT_M / cart_res).astype(int)
    gradient = _gradient_2D(_mean_filter(comb_im, (size_filt, size_filt)))
    gradient_z = gradient['Gy']
    gradient_z[np.isnan(rhohv_field_c)] = np.nan

    # First part, using ZH and RHOHV for top and bottom
    ###################################################################

    # First guess of ML without geometrical constraints
    bottom_ml, top_ml = _process_map_ml(gradient_z, rhohv_field_c,
                                        detect_threshold, *RHOHV_VALID_BOUNDS)

    # Restrict gradient using conditions on medians
    median_bot_height = np.nanmedian(bottom_ml)
    median_top_height = np.nanmedian(top_ml)

    if not np.isnan(median_bot_height):
        gradient_z[0:np.floor(LOWMLBOUND *
                              median_bot_height).astype(int), :] = np.nan
    if not np.isnan(median_top_height):
        gradient_z[np.floor(UPMLBOUND *
                            median_top_height).astype(int):, :] = np.nan

    # Identify top and bottom of ML with restricted gradient
    bottom_ml, top_ml = _process_map_ml(gradient_z,
                                        rhohv_field_c,
                                        detect_threshold,
                                        *RHOHV_VALID_BOUNDS)


    median_bot_height = np.nanmedian(bottom_ml)
    median_top_height = np.nanmedian(top_ml)


    # Final part - cleanup
    ###################################################################

    # Put NaN at pixels which have a ML thickness larger than max_thickness
    # Also put NaN where either the top or the bottom are undefined

    bad_pixels = np.where((top_ml - bottom_ml) > MAXTHICKNESS_ML / cart_res)
    top_ml[bad_pixels] = np.nan
    bottom_ml[bad_pixels] = np.nan
    top_ml[np.isnan(bottom_ml)] = np.nan
    bottom_ml[np.isnan(top_ml)] = np.nan

    median_bot_height = np.nanmedian(bottom_ml)
    median_top_height = np.nanmedian(top_ml)

    ###################################################################

    # If interpolation of small holes is activated (and if at least 2 valid
    # pts)
    if interp_holes and np.sum(np.isfinite(bottom_ml)) >= 2:
        # Find subsequences
        sub = _calc_sub_ind(bottom_ml)
        # If the first and last subset correspond to missing values we remove
        # them, as we want NO extrapolation
        if sub['values'][0] == 0:
            for k in ['values', 'lengths', 'idx']:
                sub[k] = sub[k][1:]
        if sub['values'][-1] == 0:
            for k in ['values', 'lengths', 'idx']:
                sub[k] = sub[k][:-1]
        # Find subset of subsequences where missing vals and length is
        # at most THRESHLENGTH
        sub2interp = {}
        cond = np.logical_and(sub['values'] == 0,
                              sub['lengths'] <= max_length_holes / cart_res)
        sub2interp['lengths'] = sub['lengths'][cond]
        sub2interp['idx'] = sub['idx'][cond]

        # Get corresponding indexes
        index2interp = []
        for k in range(0, len(sub2interp['idx'])):
            index2interp.extend(
                range(sub2interp['idx'][k],
                      sub2interp['idx'][k] +
                      sub2interp['lengths'][k]))
        # Interpolate all subsequences of less than threshLength [m] using
        # piecewise cubic hermite interpolation
        index2interp = np.array(index2interp)

        # Interpolate
        if index2interp.size > 0:
            idx_valid = np.where(np.isfinite(bottom_ml))[0]
            bottom_ml[index2interp] = pchip(idx_valid,
                                            bottom_ml[idx_valid])(index2interp)
            top_ml[index2interp] = pchip(idx_valid,
                                         top_ml[idx_valid])(index2interp)

    mid_ml = (median_top_height + median_bot_height) / 2

    # Check if ML is valid
    # 1) check if median_bot_height and median_top_height are defined
    if np.isnan(median_bot_height + median_top_height):
        invalid_ml = True
    else:
        invalid_ml = False
        # 2) Check how many values in the data are defined at the height of the
        # ML
        line_val = rhohv_field_c[np.int(mid_ml), :]

        # Check if ML is long enough
        if check_min_length:
            # the condition is that the ml is at least half as
            # long as the length of valid data at the ml height
            if np.logical_and(sum(np.isfinite(top_ml)) < 0.5,
                              sum(np.isfinite(line_val))):
                invalid_ml = True

    map_ml = np.zeros(gradient_z.shape)

    # 1 = below ML, 3 = in ML, 5 =  above ML
    
    # If ML is invalid, just fill top_ml and bottom_ml with NaNs
    if invalid_ml:
        top_ml = np.nan * np.zeros((gradient_z.shape[1]))
        bottom_ml = np.nan * np.zeros((gradient_z.shape[1]))
    else:
        for j in range(0, len(top_ml) - 1):
            if(not np.isnan(top_ml[j]) and not np.isnan(bottom_ml[j])):
                map_ml[np.int(top_ml[j]):, j] = mdata_ml['BELOW']
                map_ml[np.int(bottom_ml[j]):np.int(top_ml[j]), j] = mdata_ml['INSIDE']
                map_ml[0:np.int(bottom_ml[j]), j] = mdata_ml['ABOVE']



    
    # create dictionary of output ml

    # Cartesian coordinates
    ml_cart = {}
    ml_cart['data'] = np.array(map_ml)
    ml_cart['x'] = coords_c[0]
    ml_cart['z'] = coords_c[1]

    ml_cart['bottom_ml'] = np.array((bottom_ml) * cart_res)
    ml_cart['top_ml'] = np.array((top_ml) * cart_res)

    # Polar coordinates

    (t, r), (bot, top), map_ml = _remap_to_polar(radar_sweep,
                                             ml_cart['x'],
                                             ml_cart['bottom_ml'],
                                             ml_cart['top_ml'])
    map_ml = np.ma.array(map_ml, mask = map_ml == 0, fill_value = fill_value)

    # Compute melting layer height field

    alt_gates = radar_sweep.get_gate_x_y_z(0)[2]
    # Get indices of top of ML (last occurence of the value 3)
    idx_top_pol = map_ml.shape[1] - np.argmax(map_ml[:,::-1] == 3,axis=1) 
    h_top_pol = alt_gates[np.arange(len(alt_gates)),idx_top_pol - 1]
    h_top_pol[idx_top_pol == map_ml.shape[1]] = np.nan
    height_above_ml = alt_gates - h_top_pol[:,None]
    
    # Convert to masked arrays
    map_ml = np.ma.array(map_ml, mask = map_ml == 0, fill_value = fill_value)
    height_above_ml = np.ma.array(height_above_ml, mask = map_ml == 0, 
                                  fill_value = fill_value)
    
    ml_pol = {}
    ml_pol[melting_layer_field] = get_metadata(melting_layer_field)
    ml_pol[melting_layer_field]['data'] = map_ml
    ml_pol[melting_layer_height_field] = get_metadata(melting_layer_height_field)
    ml_pol[melting_layer_height_field]['data'] = height_above_ml
    
    output = {}
    output['ml_cart'] = ml_cart
    output['ml_pol'] = ml_pol
    output['ml_exists'] = not invalid_ml
    
    return output


def _process_map_ml(gradient_z, rhohv, threshold, threshold_min_rhohv=0,
                    threshold_max_rhohv=np.Inf):

    n_cols = gradient_z.shape[1]
    bottom_ml = np.zeros((n_cols)) * np.nan
    top_ml = np.zeros((n_cols)) * np.nan

    # Loop on all vertical columns
    for j in range(0, n_cols):
        # Get a vertical slice of gradient
        grad_line = gradient_z[:, j]
        grad_nonan = grad_line

        grad_nonan = grad_nonan[~np.isnan(grad_nonan)]
        ind_bot = np.nan
        ind_top = np.nan

        if grad_nonan.size:
            # Sort by ascending order of gradient
            sortedGrad = np.sort(grad_nonan)

            max_val = sortedGrad[-1]
            min_val = sortedGrad[0]

            # Index of ML top is where gradient is min
            ind_top = np.where(grad_line == min_val)
            ind_top = ind_top[0][0] + 2
            # Index of ML bottom is where gradient is max
            ind_bot = np.where(grad_line == max_val)
            ind_bot = ind_bot[0][0]

            # We also check a condition on rhohv which should not be too small
            if not ind_bot.size or not ind_top.size or ind_top <= ind_bot:
                rhohv_thresh_cond = False
                rhohv_nan_cond = False
            else:
                rhohv_thresh_cond = (np.nanmax(rhohv[ind_bot:ind_top, j])
                                     <= threshold_max_rhohv
                                     and np.nanmin(rhohv[ind_bot:ind_top, j])
                                     >= threshold_min_rhohv)

                rhohv_nan_cond = np.sum(
                    np.isnan(rhohv[ind_bot:ind_top, j])) == 0

            if (min_val <= -threshold and max_val >= threshold and
                    rhohv_nan_cond and rhohv_thresh_cond):
                bottom_ml[j] = ind_bot - 1
                top_ml[j] = ind_top + 1

    return bottom_ml, top_ml


def _process_map_ml_only_zh(gradientZ):
    # Basically the same as previous routine, except only on Zh
    n_cols = gradientZ.shape[1]
    top_ml = np.zeros((n_cols)) * float('nan')

    for j in range(0, n_cols - 1):
        grad_line = gradientZ[:, j]
        grad_no_nan = grad_line
        grad_no_nan = grad_no_nan[~np.isnan(grad_no_nan)]
        if grad_no_nan.size:
            sorted_grad = np.sort(grad_no_nan)
            min_val = sorted_grad[0]
            ind_top = np.where(grad_line == min_val)
            ind_top = ind_top[0][0]
            top_ml[j] = ind_top + 1

    return top_ml


def _r_to_h(earth_radius, gate_range, gate_theta):
    '''
    Computes the height of radar gates knowing the earth radius at the given
    latitude and the range and elevation angle of the radar gate.

    Inputs:
        earth_radius : the radius of the earth for a given latitude in m.

        gate_range : the range of the gate(s) in m.

        gate_theta : elevation angle of the gate(s) in degrees.

    Outputs:
        height : the height above ground of all specified radar gates
    '''

    height = ((gate_range**2 + (KE * earth_radius)**2 +
               2 * gate_range * KE * earth_radius *
               np.sin(np.deg2rad(gate_theta)))**(0.5) - KE * earth_radius)

    return height


def _remap_to_polar(radar_sweep, x, bottom_ml, top_ml, tol=1.5, interp=True):
    '''
    This routine converts the ML in Cartesian coordinates back to polar
    coordinates.

    Inputs:
        radar_sweep : Radar
            A pyart radar instance containing the radar data in polar
            coordinates for a single sweep
        x: array of floats
            The horizontal distance in Cartesian coordinates.

        bottom_ml: array of floats
            Bottom of the ML detected in Cartesian coordinates.

        top_ml: array of floats
            Top of the ML detected on Cartesian coordinates.

        tol : float, optional
            Angular tolerance in degrees that is used when mapping elevation
            angles computed on the Cartesian image to the original angles in
            the polar data.

        interp : bool, optional
            Whether or not to interpolate the ML in polar coordinates (fill holes)

    Outputs:
        (theta, r) : tuple of elevation angle and range corresponding to the
                     polar coordinates
        (bottom_ml, top_ml) : tuple of ml bottom and top ranges for every
                              elevation angle theta
        map_ml_pol : a binary map of the ML in polar coordinates
    '''
    # This routine converts the ML in cartesian coordinates back to polar
    # coordinates

    # Get ranges of radar data
    r = radar_sweep.range['data']

    # Get angles of radar data
    theta = radar_sweep.elevation['data']

    # Vectors to store the heights of the ML top and bottom and matrix for the
    # map
    map_ml_pol = np.zeros((len(theta), len(r)))
    bottom_ml_pol = np.zeros(len(map_ml_pol)) + np.nan
    top_ml_pol = np.zeros(len(map_ml_pol)) + np.nan

    if np.sum(np.isfinite(bottom_ml)) > 0:

         # Convert cartesian to polar
        theta_bottom_ml = np.degrees(-(np.arctan2(x, bottom_ml) - np.pi / 2))

        # Get ranges of all pixels located at the top and bottom of cartesian
        # ML
        E = get_earth_radius(radar_sweep.latitude['data'])  # Earth radius
        r_bottom_ml = (np.sqrt((E * KE * np.sin(np.radians(theta_bottom_ml)))**2 +
                               2 * E * KE * bottom_ml + bottom_ml ** 2)
                       - E * KE * np.sin(np.radians(theta_bottom_ml)))

        theta_top_ml = np.degrees(- (np.arctan2(x, top_ml) - np.pi / 2))
        E = get_earth_radius(radar_sweep.latitude['data'])  # Earth radius
        r_top_ml = (np.sqrt((E * KE * np.sin(np.radians(theta_top_ml))) ** 2 +
                            2 * E * KE * top_ml + top_ml ** 2) -
                    E * KE * np.sin(np.radians(theta_top_ml)))

        idx_r_bottom = np.zeros((len(theta))) * np.nan
        idx_r_top = np.zeros((len(theta))) * np.nan

        for i, t in enumerate(theta):
            # Find the pixel at the bottom of the ML with the closest angle
            # to theta
            idx_bot = np.nanargmin(np.abs(theta_bottom_ml - t))

            if np.abs(theta_bottom_ml[idx_bot] - t) < tol:
                # Same with pixel at top of ml
                idx_top = np.nanargmin(np.abs(theta_top_ml - t))
                if np.abs(theta_top_ml[idx_top] - t) < tol:

                    r_bottom = r_bottom_ml[idx_bot]
                    r_top = r_top_ml[idx_top]

                    idx_r_bottom[i] = np.where(r >= r_bottom)[0][0]
                    idx_r_top[i] = np.where(r >= r_top)[0][0]

        if interp:
            if np.sum(np.isfinite(idx_r_bottom)) >= 2:

                idx_valid = np.where(np.isfinite(idx_r_bottom))[0]
                idx_nan = np.where(np.isnan(idx_r_bottom))[0]

                bottom_ml_fill = \
                    InterpolatedUnivariateSpline(idx_valid,
                                                 idx_r_bottom[idx_valid],
                                                 ext=1)(idx_nan)

                bottom_ml_fill[bottom_ml_fill == 0] = -9999

                idx_r_bottom[idx_nan] = bottom_ml_fill

            if np.sum(np.isfinite(idx_r_top)) >= 2:

                idx_valid = np.where(np.isfinite(idx_r_top))[0]
                idx_nan = np.where(np.isnan(idx_r_top))[0]

                top_ml_fill = \
                    InterpolatedUnivariateSpline(idx_valid,
                                                 idx_r_top[idx_valid],
                                                 ext=1)(idx_nan)

                top_ml_fill[top_ml_fill == 0] = -9999
                idx_r_top[idx_nan] = top_ml_fill

            idx_r_bottom = idx_r_bottom.astype(int)
            idx_r_top = idx_r_top.astype(int)

        for i in range(len(map_ml_pol)):
                       
            if idx_r_bottom[i] != -9999 and idx_r_top[i] != -9999:
                 map_ml_pol[i, 0:idx_r_bottom[i]] = 1
                 map_ml_pol[i, idx_r_bottom[i]:idx_r_top[i]] = 3
                 map_ml_pol[i, idx_r_top[i]:] = 5

    return (theta, r), (bottom_ml_pol, top_ml_pol), map_ml_pol


def _normalize_image(im, min_val, max_val):
    '''
    Uniformly normalizes a radar field to the [0-1] range

    Inputs:
        im : array
            A radar image in native units, ex. dBZ

        min_val : float
            All values smaller or equal to min_val in the original image
            will be set to zero

        max_val : 
            All values larger or equal to min_val in the original image
            will be set to zero
    Outputs:
        out : the normalized radar image, with all values in [0,1]
    '''

    new_max = 1
    new_min = 0

    out = (im - min_val) * (new_max - new_min) / (max_val - min_val) + new_min
    out[im > max_val] = new_max
    out[im < min_val] = new_min

    return out


def _gradient_2D(im):
    '''
    Computes the 2D gradient of a radar image

    Inputs:
        im : array
            A radar image in Cartesian coordinates

    Outputs:
        out : a gradient dictionary containing a field 'Gx' for the gradient
              in the horizontal and a field 'Gy' for the gradient in the
              vertical
    '''
    # Computes the 2D gradient of an image
    # dim = 1 = gradient along the rows (Y)
    # dim = 2 = gradient along the column (X)

    Gx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    Gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    out = {}

    out['Gx'] = convolve(im, Gx, mode='reflect')
    out['Gy'] = convolve(im, Gy, mode='reflect')

    return out


def _convolve_with_nan(input_array, kernel, boundary='mirror'):
    '''
    Convolves an image with a kernel while ignoring missing values

    Inputs:
        input_array : the input array can be 1D or 2D

        kernel : the kernel (filter) with which to convolve the input array

        boundary: how to treat the boundaries, see arguments of scipy's
                  convolve function.

    Outputs:
        conv: the convolved signal
    '''
    # Special 2D convolution procedure for data with holes (NaN)
    if isinstance(input_array, np.ma.masked_array):
        input_array = np.ma.masked(input_array, np.nan)

    # Flat function with NaNs for comparison.
    on = np.ones(input_array.shape)

    # Find all the NaNs in the input.
    n = np.isnan(input_array)

    # Replace NaNs with zero, both in 'a' and 'on'.
    input_array[n] = 0.
    on[n] = 0.

    # Check that the filter does not have NaNs.
    if np.isnan(kernel).any():
        print('Kernel contains NaN values.')
        return None

    # Calculate what a 'flat' function looks like after convolution.
    flat = convolve(on, kernel, mode=boundary)
    # Do the actual convolution
    conv = convolve(input_array, kernel, mode=boundary) / flat
    return conv


def _mean_filter(input_array, shape=(3, 3), boundary='mirror'):
    '''
    Local averaging (smoothing) while ignoring missing values

    Inputs:
        input_array : the input array can be 1D or 2D

        shape : the shape of the averaging (smoothing) filter

        boundary: how to treat the boundaries, see arguments of scipy's
                  convolve function.

    Outputs:
        out: the smoothed signal
    '''
    # 2D mean filter for data with holes (NaN)
    kernel = np.ones(shape)
    kernel = kernel / np.sum(kernel.ravel())
    out = _convolve_with_nan(input_array, kernel, boundary)

    return out


def _calc_sub_ind(inputVec):
    '''
    The code belows finds continuous subsequences of missing values, it fills
    a vector values containing 1 for values and 0 for missing values starting
    a new subsequence, a vector idx containing the index of the first value
    of the subsequence and a vector length containing the length of the
    subsequence.


    Inputs:
        inputVec : a binary input vector

    Outputs:
        sub: a dictionary with the keys:
            values : an array containing 1 for sequences of valid values
                    and 0 for sequences of missing values
            idx : an array containing the first index of the sequences
            length : an array containing the length of every sequence
    '''

    # the vector [1 2 3 NaN NaN 3 NaN 3 5 5 1 NaN NaN NaN]
    # would give for exemple :
    # lengths=[3 2 1 1 4 3], values=[1 0 1 0 1 0] and idx=[0 3 5 6 7 11]

    sub = {}
    sub['values'] = []
    sub['lengths'] = []
    sub['idx'] = []
    l = None # For PEP8...
    if len(inputVec):
        for l in range(0, len(inputVec) - 1):
            if l == 0:
                sub['idx'].append(l)
                sub['values'].append(~np.isnan(inputVec[l]))
            if ~np.isnan(inputVec[l]) != sub['values'][-1]:
                sub['values'].append(~np.isnan(inputVec[l]))
                sub['lengths'].append(l - sub['idx'][-1])
                sub['idx'].append(l)
        sub['lengths'].append(l + 1 - sub['idx'][-1])

        sub['lengths'] = np.array(sub['lengths'])
        sub['idx'] = np.array(sub['idx'])
        sub['values'] = np.array(sub['values'])
    return sub
