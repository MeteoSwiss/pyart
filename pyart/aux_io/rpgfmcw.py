"""
pyart.aux_io.rainbow
====================

Routines for reading binary files from the RPG FMCW cloud radar

.. autosummary::
    :toctree: generated/

    

"""

import struct
import numpy as np
import datetime
import bisect
import netCDF4
import re
import itertools

BYTE_SIZES = {'i':4,'f':4,'s':1,'B':1,'I':4,'h':2,'H':2}
        
# The following variables will not be stored in the final netCDF as they
# are not of any scientific interest 
VARIABLES_TO_DUMP =  ['Min-index-blocks-spectra','Nb-blocks-spectra',
                      'Spectral-bytes','Max-index-blocks-spectra',
                      'Min-velocity-Doppler']

# This provides a mapping for filetypes that are different between struct and 
# netcdf

NETCDF_DATATYPES = {'I':'i'}

# The following variables will not be stored in the final netCDF as they
# are not of any scientific interest
VARIABLES_TO_DUMP =  ['Min-index-blocks-spectra','Nb-blocks-spectra',
                      'Spectral-bytes','Max-index-blocks-spectra',
                      'Min-velocity-Doppler']

# This provides a mapping for filetypes that are different between struct and
# netcdf

NETCDF_DATATYPES = {'I':'i'}

def read_rpgfmcw(self,filename, after_2018 = False, level = None):
    if level == None:
        # Try to infer level from file name
        if 'LV0' in filename:
            level = 0
        else:
            level = 1
    
    if level not in [1,0]:
        raise ValueError('Invalid level, must be 0 or 1!')
        
    if level == 0:
        print('Processing lvl 0 file, this can take quite some time')
    if after_2018:
        from . import rpg_file_structure_old as fs
    else:
        from . import rpg_file_structure as fs

    if level == 1:
        header_struc = fs.lvl1_header
        sample_metadata_struc = fs.lvl1_sample_metadata
        data_struc = fs.lvl1_data
    elif level == 0:
        header_struc = fs.lvl0_header
        sample_metadata_struc = fs.lvl0_sample_metadata
        data_struc = fs.lvl0_data

    with open(filename,'rb') as f:
        ba = bytearray(f.read()) # bytearray

    ''' Read the binary file'''
    read_pos = 0
    head, read_pos = _get_header(ba,header_struc,read_pos)

    n_chirps = len(head['Chirp-seq-start-index'])

    metadata = []
    data = []

    for i in range(head['Nb-time-samples']):
        print('Processing profile '+ str(i) +'/' + str(head['Nb-time-samples']))
        m,read_pos = _get_sample_metadata(ba,head,sample_metadata_struc,
                                          read_pos)

        if level:
            d, read_pos = _get_sample_data_lvl1(ba,head,m, data_struc, 
                                                read_pos)
        else:
            d, read_pos = _get_sample_data_lvl0(ba,head,m, data_struc, 
                                                read_pos)

        data.append(d)
        metadata.append(m)

    # Now we merge the metadata and data through time
    merged_data = {}
    merged_metadata = {}

    all_metadata_keys = list(itertools.chain.from_iterable([list(md.keys())
                        for md in metadata]))

    all_metadata_keys = np.unique(all_metadata_keys)

    # Merge metadata (easy)
    for key in all_metadata_keys:
        merged_metadata[key] = np.array([md[key] for md in metadata])

    # Merge data (tricky for level 0)
    all_data_keys = list(itertools.chain.from_iterable([list(dd.keys())
                        for dd in data]))

    all_data_keys = np.unique(all_data_keys)

    if level == 1:
        # Merging data for level 1 is easy since they are supposed to be all
        # 1D vectors with same length, with reasonnable size
        for key in data[0].keys():
            merged_data[key] = np.array([dd[key] for dd in data])
    else:
        # This where the tricky things begin

        # For level 0 it's a bit more tricky since the spectra above the highest
        # valid range is not kept (for the last chirp)
        for key in all_data_keys:
            # Some timesteps might be empty (i.e. only missing data)
            # So we first query through all timesteps, keep track of the
            # valid ones, create an array with the right dimensions
            # (that includes even the invalid timesteps), and assign
            # the valid data at the right place

            idx_valid_profiles = [] # idx of all valid timesteps
            merged_valid_only = []  # array containing only the valid data
            for i, dd in enumerate(data):
                if key in dd.keys():
                    idx_valid_profiles.append(i)
                    merged_valid_only.append(dd[key])


            if len(merged_valid_only[0].shape) == 1: # simple procedure for 1D variables
                merged_valid_only = np.array(merged_valid_only)

                # Initialize to the right dimensions
                merged_data[key] = np.zeros((len(data), 
                           merged_valid_only.shape[1]))

                # Assign the valid data at the right timesteps
                merged_data[key][idx_valid_profiles] = merged_valid_only

            else: # 2D case (for Doppler variables)
                # First dimension is number of ranges (not constant per timestep)
                # So we find the max length of all Doppler spectra for all timesteps
                # = max number of ranges
                max_n_ranges = np.max([mvo.shape[0] for mvo in 
                                       merged_valid_only])
                # Second dimension is FFT length, constant through time
                n_fft = merged_valid_only[0].shape[1]
                # Initialize array filled with NaN
                merged_data[key] = np.zeros((len(data),max_n_ranges, 
                           n_fft),dtype = np.float32) + np.nan
                for i,dd in enumerate(data):
                    if key in dd.keys(): # If timestep contains data for this var
                        # Fill relevant part of the array for every timestep
                        merged_data[key][i,0:dd[key].shape[0],:] = dd[key]

            # The idea here is to truncate for each chirp, the part that is
            # located on the left and right of the spectrum where there are no observations
            # for any timestep or range , so we start by getting the position
            # of the leftmost and rightmost measurements

            if 'chirp' in key:
                n_fft = merged_data[key].shape[2] # number of fft bins for this var
                left_limit_spec  = np.zeros(n_chirps,dtype=int) # Initialize to initial length
                right_limit_spec  =  np.zeros(n_chirps,dtype=int) + n_fft

                chirp = int(key[-1])
                # Find leftermost and righmost limit of spectra
                valid_bins = np.any(np.any(np.isfinite(merged_data[key]),
                                                       axis=1),axis=0)
                valid_bins = np.where(valid_bins)[0]

                if len(valid_bins):
                    left_limit_spec[chirp] = min([left_limit_spec[chirp], 
                                                 valid_bins[0]])
                    right_limit_spec[chirp] = max([right_limit_spec[chirp], 
                                                  valid_bins[-1]]) + 1

        # Now we truncate the matrices per chirp based on these indices
        available_chirps = [] # This list will contain the chirps that are
        # present in the merged data (it can happen that some chirps are missing)
        # when there is no valid data at all for this chirp)
        for key in all_data_keys:
            if 'chirp' in key:
                if chirp not in available_chirps:
                    available_chirps.append(chirp)
                chirp = int(key[-1])
                merged_data[key] = merged_data[key][:,:,left_limit_spec[chirp]:right_limit_spec[chirp]]

        # For level 0, we also need to compute the velocity arrays and range arrays
        # for each chirp

        # Better to use dict (and not list) for chirp dependent variables
        # as some chirps might be missing from the final merged data
        # (e.g. missing data)
        heights_chirps = {}
        vel_chirps = {}
        idx = 0
        available_chirps = sorted(available_chirps)
        for chirp in available_chirps:
            # First check if merged data has something for this chirp
            # (might happen that no valid data is available at all for this chirp)
            key = 'Doppler-spectrum-vert-chirp'+str(chirp)
            if key in merged_data.keys():
                nb_range_chirp = merged_data['Doppler-spectrum-vert-chirp'+str(chirp)].shape[1]
                h = head['Radar-gates-altitudes-m']
                heights_chirps[chirp] = (h[idx:idx+nb_range_chirp])
                # Compute velocity resolution
                fft_len_chirp = head['Nb-samples-Doppler-per-chirp'][chirp]
                max_vel_chirp = head['Max-vel-chirp-seq'][chirp]
                rs_chirp = 2 * max_vel_chirp/fft_len_chirp
                # Compute velocity array
                v_chirp = -max_vel_chirp + rs_chirp * np.arange(fft_len_chirp)
                vel_chirps[chirp] = (v_chirp[left_limit_spec[chirp]: right_limit_spec[chirp]])
                idx+=nb_range_chirp
        

    '''
    Write all info to netCDF
    '''
    self.data = data
    del metadata
    del data

    # Create diskless dummy file
    dummy = netCDF4.Dataset('dummy','w',diskless=True)

    # Create dimensions
    time = dummy.createDimension('Time',head['Nb-time-samples'])
    rgate = dummy.createDimension('Rgate',head['Nb-radar-gates'])

    if level == 0:
        # Addtional range and velocity dimensions for every chirp
        rgate_chirps = {}
        vfft_chirps = {}
        for chirp in available_chirps:
            rgate_chirps[chirp] = dummy.createDimension('Rgate_chirp'+str(chirp),
                                                      len(heights_chirps[chirp]))
            vfft_chirps[chirp] = dummy.createDimension('Vfft_chirp'+str(chirp),
                                                     len(vel_chirps[chirp]))

    # If included in file create dimensions for temp and humidity profiles
    if head['Nb-temp-gates'] > 0:
        tgate = dummy.createDimension('Tgate',head['Nb-temp-gates'])
    else:
        tgate = None
    if head['Nb-hum-gates'] > 0:
        hgate = dummy.createDimension('Hgate',head['Nb-hum-gates'])
    else:
        hgate = None

    # Write header as global attributes
    for key in head:
        setattr(dummy,key,head[key])

    del head
            
    # Write metadata to dummy
    for key in merged_metadata.keys():
        if key == 'Reserved':
            continue
        
        valid_key = True

        # get shape of variable
        shape = merged_metadata[key].shape
        # The two lines below are not very elegant and should be changed (with dic)
        idx = [sm['name'] for sm in sample_metadata_struc].index(key)
        type_var = sample_metadata_struc[idx]['type']
        if type_var in NETCDF_DATATYPES.keys():
            type_var = NETCDF_DATATYPES[type_var]

        dimensions_var = []
        for s in shape:
            if s == len(time):
                dimensions_var.append('Time')
            elif s == len(rgate):
                dimensions_var.append('Rgate')
            elif hgate:
                if s == len(hgate):
                    dimensions_var.append('Hgate')
            elif tgate:
                if s == len(tgate):
                    dimensions_var.append('Tgate')
            else:
                print('Could not find dimension for variable "'+key+'"')
                print('Skipping it')
                valid_key = False

        if valid_key:
            variable = dummy.createVariable(key,type_var,dimensions_var)
            if key == 'Time':
                # Convert from weird RPG timestamp (seconds from 1.1.2001, 00:00:00)
                #  to UNIX timestamp
                merged_metadata[key] = _RPG_timestamp_to_UNIX(merged_metadata
                               [key])

            variable[:] = merged_metadata[key]
            setattr(variable,'units', sample_metadata_struc[idx]['unit'])
            setattr(variable,'description',
                        sample_metadata_struc[idx]['description'])

    del merged_metadata

    # Write data to dummy, tricky for level 0
    for key in merged_data.keys():
        # get shape of variable
        shape = merged_data[key].shape

        # Here we try to find the info about units and type stored in the file_structure
        # We find the corresponding entry in the data_struc list
        if 'chirp' in key:
            # If 'chirp' is in the key, it means that the key won't be in the list
            # since we created custom variable names during runtime (by separating chirps)
            # Hence we need to find the original name
            # Remove -chirpx from var name (where x = integer)
            key_original = 'Full-'+re.sub("-chirp\d","",key)
        else:
            key_original = key

        idx = [d['name'] for d in data_struc].index(key_original)
        type_var = data_struc[idx]['type']
        if type_var in NETCDF_DATATYPES.keys():
            type_var = NETCDF_DATATYPES[type_var]

        dimensions_var = []
        if 'chirp' in key: # Doppler variable
            chirp = int(key[-1])
            dimensions_var.append('Time')
            dimensions_var.append('Rgate_chirp'+str(chirp))
            dimensions_var.append('Vfft_chirp'+str(chirp))

        else:
            dimensions_var.append('Time')
            if shape[1] == len(rgate):
                dimensions_var.append('Rgate')
            elif shape[1] == len(hgate):
                    dimensions_var.append('Rgate')
            elif shape[1] == len(tgate):
                    dimensions_var.append('Tgate')

        variable = dummy.createVariable(key,type_var,dimensions_var)
        variable[:] = merged_data[key]

        setattr(variable,'units', data_struc[idx]['unit'])
        setattr(variable,'description', data_struc[idx]['description'])
    self.merged_data = merged_data
    del merged_data

    # As a last step we need to assign the variables corresponding to the dimensions


    # Range
    rgate_var = dummy.createVariable('Rgate','f',('Rgate',))
    rgate_var[:] = getattr(dummy,'Radar-gates-altitudes-m')
    setattr(rgate_var,'units', 'm')
    setattr(rgate_var,'description', 'Range (altitude) of radar gates')
    delattr(dummy,'Radar-gates-altitudes-m')

    if hgate:
        hgate_var = dummy.createVariable('Hgate','f',('Hgate',))
        hgate_var[:] = getattr(dummy,'Hum-gates-altitudes-m')
        setattr(hgate_var,'units', 'm')
        setattr(hgate_var,'description', 'Range (altitude) of humidity gates')
        delattr(dummy,'Hum-gates-altitudes-m')
    if tgate:
        tgate_var = dummy.createVariable('Tgate','f',('Tgate',))
        tgate_var[:] = getattr(dummy,'Temp-gates-altitudes-m')
        setattr(tgate_var,'units', 'm')
        setattr(tgate_var,'description', 'Range (altitude) of temperature gates')
        delattr(dummy,'Temp-gates-altitudes-m')

    if level == 0:
        for chirp in available_chirps:
            rgate_chirp_var = dummy.createVariable('Rgate_chirp'+str(chirp),'f',
                                                   ('Rgate_chirp'+str(chirp),))
            rgate_chirp_var[:] = heights_chirps[chirp]
            setattr(rgate_var,'units', 'm')
            setattr(rgate_var,'description', 'Range (altitude) of temperature gates, for chirp '+str(chirp))

            vfft_chirp_var = dummy.createVariable('Vfft_chirp'+str(chirp),'f',
                                                   ('Vfft_chirp'+str(chirp),))
            vfft_chirp_var[:] = vel_chirps[chirp]
            setattr(vfft_chirp_var,'units', 'm/s')
            setattr(vfft_chirp_var,'description', 'Velocity bins of the Doppler spectra for chirp'+str(chirp))

    # Assign class variables
    self._dummy_nc = dummy
    self.variables = dummy.variables
    self.dimensions = dummy.dimensions
    self.global_attributes = dummy.__dict__
    self.level = level
    self.filename = filename



def _get_header(ba, header_struc, read_pos):
        
    # Read the file byte by byte
    values = {}
    for i,h in enumerate(header_struc):
        len_val = h['len']
    
        if len_val == None:
            # If the specified len is 0, we read until we encounter \x00 (null)
            null_pos = ba[read_pos:].find(b'\x00') 
            len_val = null_pos
        elif type(len_val) is list: # len_val is a list
            len_val = values[len_val[1]] * len_val[0]
        if len_val != 0:
            type_var = h['type']
            ffmt = '<' # Little-endian
            ffmt += "{}".format(int(len_val))
            ffmt += type_var

            offset = len_val * BYTE_SIZES[type_var]
            val = struct.unpack(ffmt,ba[read_pos:read_pos+offset])
            if len(val) == 1:
                val = val[0]
            else:
                val = np.array(val)
            values[h['name']] = val
            read_pos += offset
            if h['type'] == 's': # Special case for strings
                read_pos += 1
            
    return values, read_pos

def _get_sample_metadata(ba,head, sample_metadata_struc, read_pos):
    values = {}

    for i,sm in enumerate(sample_metadata_struc):
        name_var = sm['name']
    
        if head['Dual-pol-flag'] == 0:
            if name_var in ['Linear-sensitivity-hor', 'total-IF-power-hor']:
                # Not available when no dual pol, so skip
                continue
        
        len_val = sm['len']
    
        if len_val == None:
            # If the specified len is 0, we read until we encounter \x00 (null)
            null_pos = ba[read_pos:].find(b'\x00') 
            len_val = null_pos
        elif type(len_val) is list: # len_val is a list
            len_val = head[len_val[1]] * len_val[0]
    
        if len_val != 0:
            type_var = sm['type']
            ffmt = '<' # Little-endian
            ffmt += "{}".format(int(len_val))
            ffmt += type_var
    
            offset = len_val * BYTE_SIZES[type_var]
            val = struct.unpack(ffmt,ba[read_pos:read_pos+offset])
            if len(val) == 1:
                val = val[0]
            else:
                val = np.array(val)
            
            values[sm['name']] = val
            read_pos += offset
            if sm['type'] == 's':
                read_pos += 1

    return values, read_pos

def _get_sample_data_lvl1(ba,head,metadata, data_struc,read_pos):
    '''
    Get a data block (one profile), from a level 1 file and separate the 
    variables
    '''

    # Initialize
    values = {} # Empty output dictionary
    available_variables = [] # List of all variables contained in data block
    variable_types = [] # List of all variable types contained in data block
    
    # Loop on all variables in data structure
    for i,d in enumerate(data_struc):
        name_var = d['name']
        if name_var == 'Diff-reflectivity' and head['Dual-pol-flag'] == 0:
            # Further variables are not available when no dual pol
            break
        elif name_var == 'Slanted-Ze' and head['Dual-pol-flag'] != 2:
            # Further variables are not available when no dual pol in STSR mode 
            break
        available_variables.append(name_var)
        variable_types.append(d['type'])
        # Initialize values to NaN
        values[name_var] = np.zeros(len(metadata['Mask-occupied-gate'])) + np.nan
        
    # get mask
    mask = metadata['Mask-occupied-gate'].astype(bool)
    # Num of variables in file
    num_vars = len(available_variables)
     # Get length of sample
    len_sample = np.sum(mask) * \
        np.sum([BYTE_SIZES[v_type] for v_type in variable_types])
    
    ffmt = '<'+np.sum(mask)*''.join([str(v_type) for v_type in variable_types])

    all_values = struct.unpack(ffmt,ba[read_pos:read_pos+len_sample])
    
    for i,var in enumerate(available_variables):
        values[var][mask] = all_values[i::num_vars]
        
    read_pos += len_sample
    return values, read_pos
    
def _get_sample_data_lvl0(ba,head,metadata, data_struc, read_pos):
    '''
    Get a data block (one profile), from a level 0 file and separate the 
    variables
    '''
    
    # Initialize
    values = {} # Empty output dictionary
    available_variables = ['Spectral-bytes'] # List of all variables contained in data block
    if head['Spectral-comp-flag'] == 0:
        available_variables.extend(['Full-Doppler-spectrum-vert'])
        if head['Dual-pol-flag'] > 0:
            available_variables.extend(['Full-Doppler-spectrum-hor',
                                        'Full-covariance-spectrum-real',
                                        'Full-covariance-spectrum-imag'])
    elif head['Spectral-comp-flag'] > 0:
        available_variables.extend(['Nb-blocks-spectra',
                                    'Min-index-blocks-spectra',
                                    'Max-index-blocks-spectra',
                                    'Comp-Doppler-spectrum-vert',
                                    'Integ-Doppler-spectrum-noise-vert'])
        if head['Dual-pol-flag'] > 0:
            available_variables.extend(['Comp-Doppler-spectrum-hor',
                                        'Comp-covariance-spectrum-real',
                                        'Comp-covariance-spectrum-imag',
                                        'Integ-Doppler-spectrum-noise-hor'])
    if head['Spectral-comp-flag'] == 2:
        available_variables.extend(['Comp-spectral-diff-refl',
                                    'Comp-spectral-corr',
                                    'Comp-spectral-diff-phase'])
        if head['Dual-pol-flag' == 2]:
            available_variables.extend(['Comp-spectral-slanted-LDR',
                                        'Comp-spectral-slanted-corr',
                                        'Specific-differential-phase-shift',
                                        'Diff-attenuation'])      

    if head['Spectral-comp-flag'] > 0 and head['Dual-pol-flag'] > 0:
        available_variables.extend(['',''])    
             
    if head['Anti-alias'] and head['Spectral-comp-flag'] > 0:
         available_variables.extend(['Alias-mask',
                                    'Min-velocity-Doppler'])     
                 
    # get mask
    mask = metadata['Mask-occupied-gate'].astype(bool)
    
    if np.sum(mask) == 0: # No valid data at all, so return empty dic
        return {}, read_pos

    # Initialize
    all_values = []
    # chirp_number_per_gate is a list containing the chirp number for every gate
    chirp_number_per_gate = np.zeros((len(mask),),dtype=int)
    # I would like to assign NaN but since it's integer, I assign a NoData value of -99
    chirp_number_per_gate[chirp_number_per_gate == 0] = -99 

    n_chirps = len(head['Chirp-seq-start-index'])
    
    idx_valid = np.where(mask)[0]
    
    for i in idx_valid: # Loop on valid bins
        # Get chirp sequence for this bin
        chirp_number_per_gate[i] = bisect.bisect(head['Chirp-seq-start-index'],i) - 1
        nb_blocks_spectra = head['Nb-samples-Doppler-per-chirp'][chirp_number_per_gate[i]] 
        
        values = {}
        
        for v in available_variables:
            ffmt = '<'
            idx = [d['name'] for d in data_struc].index(v) 
            type_var = data_struc[idx]['type']
            len_val = data_struc[idx]['len']
            if len_val == 'full':
                len_val = nb_blocks_spectra
            elif len_val == 'comp':
                len_val = np.sum(values['Max-index-blocks-spectra']-\
                    values['Min-index-blocks-spectra'] + 1)
            elif len_val == 'blocks':
                len_val = np.array(values['Nb-blocks-spectra'])
            len_val = np.squeeze(len_val ) 
            ffmt += '{}'.format(len_val)+type_var
            ffmt = ffmt.replace('[','').replace(']','') # remove unwanted brackets (might happen)
            if len_val <= 0:
                raise ValueError('Found entry with invalid length, there must be an error')
            
            offset = len_val* BYTE_SIZES[type_var]

            val = struct.unpack(ffmt,ba[read_pos:read_pos+offset])
            
            val = np.array(val)
            values[v] = val
            read_pos += offset
        all_values.append(values)

    # Now compile final output
    output = {}
    
    # Calculate some properties about chirps
    # chirp_sequences_idx contains for every chirp, the indexes of all its gates
    chirp_sequences_idx = []
    len_fft = head['Nb-samples-Doppler-per-chirp']

    for c in range(n_chirps):
        chirp_sequences_idx.append(np.where(chirp_number_per_gate==c)[0])
   
    for var in available_variables:
        if var in VARIABLES_TO_DUMP:
            continue
        # Check if variable is 1D or 2D
        if np.any(np.array([len(av[var]) for av in all_values])>1): # Check if 2D variable
            # 2D variables are Doppler variables, they need to be separated
            # by chirp
            idx_data = 0
            valid_chirps = np.unique(chirp_number_per_gate[chirp_number_per_gate!=-99])
            for c in valid_chirps: # Loop on all chirps with valid data
                vname_chirp = var +'-chirp'+str(c)
                # Remove full and comp keywords from variable name
                vname_chirp = vname_chirp.replace('Full-','')
                vname_chirp = vname_chirp.replace('Comp-','')
                
                output[vname_chirp] = np.zeros((max(chirp_sequences_idx[c]) + 1 - head['Chirp-seq-start-index'][c],
                    len_fft[c])) + np.nan
                # Initialize output variable for a chirp
                if 'Comp' in var:
                    # Compressed variables need to be decompressed first
                    for idx in range(len(chirp_sequences_idx[c]) - 1):
                        idx_0 = chirp_sequences_idx[c][idx] - head['Chirp-seq-start-index'][c] 
                        max_idx = np.squeeze(all_values[idx_data]['Max-index-blocks-spectra'])
                        min_idx = np.squeeze(all_values[idx_data]['Min-index-blocks-spectra'])
                        n_blocks = np.squeeze(all_values[idx_data]['Nb-blocks-spectra'])
    
                        if n_blocks == 1:
                            output[vname_chirp][idx_0,min_idx:max_idx + 1] = all_values[idx_data][var]
                        else:
                            idx_b = 0
                            for block in range(n_blocks):
                                len_block = max_idx[block]-min_idx[block] + 1
                                output[vname_chirp][idx_0,min_idx[block]:max_idx[block] + 1] = \
                                    all_values[idx_data][var][idx_b:idx_b+len_block]
                                idx_b += len_block
                        idx_data += 1
                else: # Full Doppler spectrum (non-compressed)
                    
                    len_chirp = len(chirp_sequences_idx[c])
                    idx_0 = chirp_sequences_idx[c] - head['Chirp-seq-start-index'][c]
                    output[vname_chirp][idx_0] = [a[var] for a in \
                           all_values[idx_data:idx_data + len_chirp ]]
                    idx_data += len_chirp
        else:
            # Initialize output variable
            output[var] = np.zeros(len(mask),)
            output[var][mask] = np.squeeze(np.array([av[var] for av in all_values]))

    return output, read_pos


def _RPG_timestamp_to_UNIX(timestamps):
    date0_RPG = datetime.datetime.strptime('01-01-2001 00:00:00Z','%d-%m-%Y %H:%M:%SZ')
    datetimes = [date0_RPG.replace(tzinfo=datetime.timezone.utc).timestamp() + int(t) \
                 for t in timestamps]
    
    return datetimes