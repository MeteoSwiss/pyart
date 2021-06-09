# -*- coding: utf-8 -*-
"""
rpg fmcw python library
================

Information about the variable names, types and length that are stored in
the binary files

This is valid only for older files, pre 2018

-------------------------------------------------------------------------------
LEVEL 0

-header structure 
-sample_metadata structure
-data structure
-------------------------------------------------------------------------------
"""

###############################################################################   
### HEADER INFO

lvl0_header = {}

# Names/type/len of lvl 0 header entries
lvl0_header = []
lvl0_header.append({'name':'File-code',
                    'type':'i',
                    'len':1})
lvl0_header.append({'name':'Header-length',
                    'type':'i',
                    'len':1})
lvl0_header.append({'name':'Chirp-prog-no',
                    'type':'i',
                    'len':1})
lvl0_header.append({'name':'Model-no',
                    'type':'i',
                    'len':1})
lvl0_header.append({'name':'Prog-name',
                    'type':'s',
                    'len':None})
lvl0_header.append({'name':'Cust-name',
                    'type':'s',
                    'len':None})
lvl0_header.append({'name':'Frequency',
                    'type':'f',
                    'len':1})
lvl0_header.append({'name':'Antenna-separation-m',
                    'type':'f',
                    'len':1})
lvl0_header.append({'name':'Antenna-diameter-m',
                    'type':'f',
                    'len':1})
lvl0_header.append({'name':'Linear-antenna-gain',
                    'type':'f',
                    'len':1})
lvl0_header.append({'name':'Half-power-beam-width-deg',
                    'type':'f',
                    'len':1})
lvl0_header.append({'name':'Radar-constant',
                    'type':'f',
                    'len':1})
lvl0_header.append({'name':'Dual-pol-flag',
                    'type':'B',
                    'len':1})
lvl0_header.append({'name':'Spectral-comp-flag',
                    'type':'B',
                    'len':1})
lvl0_header.append({'name':'Anti-alias',
                    'type':'B',
                    'len':1})
lvl0_header.append({'name':'Sample-duration',
                    'type':'f',
                    'len':1})
lvl0_header.append({'name':'GPS-lat',
                    'type':'f',
                    'len':1})
lvl0_header.append({'name':'GPS-lon',
                    'type':'f',
                    'len':1})
lvl0_header.append({'name':'Calibration-interval',
                    'type':'i',
                    'len':1})
lvl0_header.append({'name':'Nb-radar-gates',
                    'type':'i',
                    'len':1})
lvl0_header.append({'name':'Nb-temp-gates',
                    'type':'i',
                    'len':1})
lvl0_header.append({'name':'Nb-hum-gates',
                    'type':'i',
                    'len':1})
lvl0_header.append({'name':'Nb-chirp-sequences',
                    'type':'i',
                    'len':1})
lvl0_header.append({'name':'Radar-gates-altitudes-m',
                    'type':'f',
                    'len':[1, 'Nb-radar-gates']})
lvl0_header.append({'name':'Temp-gates-altitudes-m',
                    'type':'f',
                    'len':[1, 'Nb-temp-gates']})
lvl0_header.append({'name':'Hum-gates-altitudes-m',
                    'type':'f',
                    'len':[1, 'Nb-hum-gates']})
lvl0_header.append({'name':'Range-factors',
                    'type':'i',
                    'len':[1, 'Nb-radar-gates']})
lvl0_header.append({'name':'Nb-samples-Doppler-per-chirp',
                    'type':'i',
                    'len':[1, 'Nb-chirp-sequences']})
lvl0_header.append({'name':'Chirp-seq-start-index',
                    'type':'i',
                    'len':[1, 'Nb-chirp-sequences']})
lvl0_header.append({'name':'Nb-averaged-chirps',
                    'type':'f',
                    'len':[1, 'Nb-chirp-sequences']})
lvl0_header.append({'name':'Eff-seq-integration-time',
                    'type':'f',
                    'len':[1, 'Nb-chirp-sequences']})
lvl0_header.append({'name':'Range-res-chirp-seq',
                    'type':'f',
                    'len':[1, 'Nb-chirp-sequences']})
lvl0_header.append({'name':'Max-vel-chirp-seq',
                    'type':'f',
                    'len':[1, 'Nb-chirp-sequences']})
lvl0_header.append({'name':'Nb-time-samples',
                    'type':'i',
                    'len':1})

###############################################################################   
### SAMPLE METADATA INFO

lvl0_sample_metadata = []

lvl0_sample_metadata.append({'name':'Sample-length-bytes',
                             'type':'i',
                             'len':1,
                             'unit':'bytes',
                             'description':'Length of samples [bytes]'})
lvl0_sample_metadata.append({'name':'Time',
                             'type':'I',
                             'len':1,
                             'unit':'sec',
                             'description':'Time in seconds(UNIX)'})
lvl0_sample_metadata.append({'name':'Time-ms',
                             'type':'i',
                             'len':1,
                             'unit':'msec',
                             'description':'Time in milliseconds'})
lvl0_sample_metadata.append({'name':'Quality-flag',
                             'type':'B',
                             'len':1,
                             'unit':'-',
                             'description':'Quality flag (0 = ok, 1 = ADC saturation, 2 = spectral width too high)'})
lvl0_sample_metadata.append({'name':'Rain-rate','type':'f',
                             'len':1,
                             'unit':'mm/h',
                             'description':'Rain rate'})
lvl0_sample_metadata.append({'name':'Rel-humidity',
                             'type':'f',
                             'len':1,
                             'unit':'%',
                             'description':'Relative humidity'})
lvl0_sample_metadata.append({'name':'Environment-temp',
                             'type':'f',
                             'len':1,
                             'unit':'K',
                             'description':'Environmental temperature'})
lvl0_sample_metadata.append({'name':'Barometric-pressure',
                             'type':'f',
                             'len':1,
                             'unit':'hPa',
                             'description':'Barometric pressure'})
lvl0_sample_metadata.append({'name':'Wind-speed',
                             'type':'f',
                             'len':1,
                             'unit':'km/h',
                             'description':'Wind speed'})
lvl0_sample_metadata.append({'name':'Wind-direction',
                             'type':'f',
                             'len':1,
                             'unit':'deg',
                             'description':'Wind direction'})
lvl0_sample_metadata.append({'name':'Direct-detection-channel-voltage',
                             'type':'f',
                             'len':1,
                             'unit':'V',
                             'description':'Direct detection channel voltage'})
lvl0_sample_metadata.append({'name':'Direct-detection-brightness-temp',
                             'type':'f',
                             'len':1,
                             'unit':'K',
                             'description': 'Direct detection brightness temperature'})
lvl0_sample_metadata.append({'name':'Liquid-water-path',
                             'type':'f',
                             'len':1,
                             'unit':'g/m3',
                             'description':'liquid water path'})
lvl0_sample_metadata.append({'name':'IF-power-ADC',
                             'type':'f',
                             'len':1,
                             'unit':'microW',
                             'description':'IF power at ADC'})
lvl0_sample_metadata.append({'name':'Elevation-angle',
                             'type':'f',
                             'len':1,
                             'unit':'deg',
                             'description':'Elevation angle'})
lvl0_sample_metadata.append({'name':'Azimuth-angle',
                             'type':'f',
                             'len':1,
                             'unit':'deg',
                             'description':'Azimuth angle'})
lvl0_sample_metadata.append({'name':'Mitigation-status-flag',
                             'type':'f',
                             'len':1,
                             'unit':'-',
                             'description':'Mitigation status flag (0/1: heater switch (ON/OFF), 0/10: blower switch (ON/OFF)'})
lvl0_sample_metadata.append({'name':'Transmitter-power',
                             'type':'f',
                             'len':1,
                             'unit':'W',
                             'description':'Transmitter power'})
lvl0_sample_metadata.append({'name':'Transmitter-temp',
                             'type':'f',
                             'len':1,
                             'unit':'K',
                             'description':'Transmitter temperature'})
lvl0_sample_metadata.append({'name':'Receiver-temp',
                             'type':'f',
                             'len':1,
                             'unit':'K',
                             'description':'Receiver temperature'})
lvl0_sample_metadata.append({'name':'PC-temp',
                             'type':'f',
                             'len':1,
                             'unit':'K',
                             'description':'PC temperature'})
lvl0_sample_metadata.append({'name':'Reserved',
                             'type':'f',
                             'len':3,
                             'unit':'-',
                             'description':'Reserved bytes'})
lvl0_sample_metadata.append({'name':'Temp-profile',
                             'type':'f',
                             'len':[1, 'Nb-temp-gates'],
                             'unit':'K',
                             'description':'Temperature profile'})
lvl0_sample_metadata.append({'name':'Abs-humidity-profile',
                             'type':'f',
                             'len':[1, 'Nb-hum-gates'],
                             'unit':'g/m3',
                             'description':'Absolute humidity profile'})
lvl0_sample_metadata.append({'name':'Rel-humidity-profile',
                             'type':'f',
                             'len':[1, 'Nb-hum-gates'],
                             'unit':'%',
                             'description':'Relative humidity profile'})
lvl0_sample_metadata.append({'name':'total-IF-power-vert',
                             'type':'f',
                             'len':[1, 'Nb-radar-gates'],   
                             'unit':'microW',
                             'description':'Total IF power in vertical polarization, measured at ADC input'})
lvl0_sample_metadata.append({'name':'total-IF-power-hor',
                             'type':'f',
                             'len':[1, 'Nb-radar-gates'],
                             'unit':'microW',
                             'description':'Total IF power in horizontal polarization, measured at ADC input (only if Dual-pol-flag > 0)'})
lvl0_sample_metadata.append({'name':'Linear-sensitivity-vert',
                             'type':'f',
                             'len':[1, 'Nb-radar-gates'],
                             'unit':'mm6/m3',
                             'description':'Linear sensitivity limit in Ze units for vertical polarization'})
lvl0_sample_metadata.append({'name':'Linear-sensitivity-hor',
                             'type':'f',
                             'len':[1, 'Nb-radar-gates'],
                             'unit':'mm6/m3',
                             'description':'Linear sensitivity limit in Ze units for horizontal polarization'})
lvl0_sample_metadata.append({'name':'Mask-occupied-gate',
                             'type':'B',
                             'len':[1, 'Nb-radar-gates'],
                             'unit':'-',
                             'description':'Mask array of occupied radar gates (0: gate not occupied, 1: gate occupied'})
                            


###############################################################################                    
### SAMPLE DATA INFO

'''
When specifying the length, "full" means size(type) x Nb-samples-Doppler (for a given chirp)
and                         "comp" means size(type) x Nb-blocks-spectra x (Max-index-blocks-spectra - Min-index-blocks-spectra + 1)
and                         "blocks" means size(type) x Nb-blocks-spectra
'''

lvl0_data = []
lvl0_data.append({'name':'Spectral-bytes',
                  'type':'i',
                  'description':'Number of bytes of spectral block',
                  'len':1,
                  'unit':'bytes'})
# If Spectral-comp-flag == 0
lvl0_data.append({'name':'Full-Doppler-spectrum-vert',
                  'type':'f',
                  'len': 'full',
                  'description':'Full Doppler spectrum (incl. noise) in linear Ze at vert. pol.',
                  'unit':'mm6/m3/(m/s)'})
# If Spectral-comp-flag == 0 and Dual-pol-flag > 0
lvl0_data.append({'name':'Full-Doppler-spectrum-hor',
                  'type':'f',
                  'len': 'full',
                  'description':'Full Doppler spectrum (incl. noise) in linear Ze at hor. pol.',
                  'unit':'mm6/m3/(m/s)'})
lvl0_data.append({'name':'Full-covariance-spectrum-real',
                  'type':'f',
                  'len': 'a',
                  'description':'Full covariance spectrum, real part, in linear Ze',
                  'unit':'mm6/m3/(m/s)'})
lvl0_data.append({'name':'Full-covariance-spectrum-imag',
                  'type':'f',
                  'len': 'full',
                  'description':'Full covariance spectrum, imaginary part, in linear Ze',
                  'unit':'mm6/m3/(m/s)'})
# If Spectral-comp-flag == 1
lvl0_data.append({'name':'Nb-blocks-spectra',
                  'type':'B',
                  'len': 1,
                  'description':'Number of blocks in spectra',
                  'unit':'-'})
lvl0_data.append({'name':'Min-index-blocks-spectra',
                  'type':'h',
                  'len':'blocks',
                  'description':'Minimum index of blocks in spectra',
                  'unit':'-'})
lvl0_data.append({'name':'Max-index-blocks-spectra',
                  'type':'h',
                  'len':'blocks',
                  'description':'Maximum index of blocks in spectra',
                  'unit':'-'})
lvl0_data.append({'name':'Comp-Doppler-spectrum-vert',
                  'type':'f',
                  'len': 'comp',
                  'description':'Compressed Doppler spectrum (incl. noise) in linear Ze at vert. pol.',
                  'unit':'mm6/m3/(m/s)'})
# If Spectral-comp-flag == 1 and Dual-pol-flag > 0
lvl0_data.append({'name':'Comp-Doppler-spectrum-hor',
                  'type':'f',
                  'len': 'comp',
                  'description':'Compressed Doppler spectrum (incl. noise) in linear Ze at hor. pol.',
                  'unit':'mm6/m3/(m/s)'})
lvl0_data.append({'name':'Comp-covariance-spectrum-real',
                  'type':'f',
                  'len': 'comp',
                  'description':'Compressed covariance spectrum, real part, in linear Ze',
                  'unit':'mm6/m3/(m/s)'})
lvl0_data.append({'name':'Comp-covariance-spectrum-imag',
                  'type':'f',
                  'len': 'comp',
                  'description':'Compressed covariance spectrum, imaginary part, in linear Ze',
                  'unit':'mm6/m3/(m/s)'})
# If Spectral-comp-flag == 2 and Dual-pol-flag > 0
lvl0_data.append({'name':'Comp-spectral-diff-refl',
                  'type':'f',
                  'len': 'comp',
                  'description':'Compressed spectral differential reflectivity',
                  'unit':'dB/(m/s)'})
lvl0_data.append({'name':'Comp-spectral-corr',
                  'type':'f',
                  'len': 'comp',
                  'description':'Compressed spectral correlation ceofficient',
                  'unit':'-'})
lvl0_data.append({'name':'Comp-spectral-diff-phase',
                  'type':'f',
                  'len': 'comp',
                  'description':'Compressed spectral differential phase',
                  'unit':'rad/(m/s)'})
# If Dual-pol-flag == 2 (STSR mode)
lvl0_data.append({'name':'Comp-spectral-slanted-LDR',
                  'type':'f',
                  'len': 'comp',
                  'description':'Compressed spectral slanted LDR',
                  'unit':'dB/(m/s)'})
lvl0_data.append({'name':'Comp-spectral-slanted-corr',
                  'type':'f',
                  'len': 'comp',
                  'description':'Compressed spectral slanted correlation coefficient',
                  'unit':'-'})
lvl0_data.append({'name':'Specific-differential-phase-shift',
                  'type':'f',
                  'len': 1,
                  'description':'Specific differential phase shift',
                  'unit':'rad/km'})
lvl0_data.append({'name':'Diff-attenuation',
                  'type':'f',
                  'len': 1,
                  'description':'Differential attenuation',
                  'unit':'rad/km'})
# If Spectral-comp-flag > 0
lvl0_data.append({'name':'Integ-Doppler-spectrum-noise-vert',
                  'type':'f',
                  'len': 1,
                  'description':'Integrated Doppler spectrum noise power in vert. pol.',
                  'unit':'mm6/m3/(m/s)'})
# If Spectral-comp-flag > 0 and Dual-pol-flag > 0
lvl0_data.append({'name':'Integ-Doppler-spectrum-noise-hor',
                  'type':'f',
                  'len': 1,
                  'description':'Integrated Doppler spectrum noise power in hor. pol.',
                  'unit':'mm6/m3/(m/s)'})
# If Anti-alias == 1 and Spectral-comp-flag > 0
lvl0_data.append({'name':'Alias-mask',
                  'type':'B',
                  'len': 1,
                  'description':'Mask indicating if anti-aliasing has been applied (=1) or not (=0)',
                  'unit':'-'})
lvl0_data.append({'name':'Min-velocity-Doppler',
                  'type':'f',
                  'len': 1,
                  'description':'Minimum velocity in Doppler spectrum',
                  'unit':'m/s'})



'''
-------------------------------------------------------------------------------
LEVEL 1

-header structure 
-sample_metadata structure
-data structure
-------------------------------------------------------------------------------
'''

###############################################################################   
### HEADER INFO

lvl1_header = []

# Names/type/len of lvl 1 header entries
lvl1_header.append({'name':'File-code',
                    'type':'i',
                    'len':1})
lvl1_header.append({'name':'Header-length',
                    'type':'i',
                    'len':1})
lvl1_header.append({'name':'Chirp-prog-no',
                    'type':'i',
                    'len':1})
lvl1_header.append({'name':'Model-no',
                    'type':'i',
                    'len':1})
lvl1_header.append({'name':'Prog-name',
                    'type':'s',
                    'len':None})
lvl1_header.append({'name':'Cust-name',
                    'type':'s',
                    'len':None})
lvl1_header.append({'name':'Frequency',
                    'type':'f',
                    'len':1})
lvl1_header.append({'name':'Antenna-separation-m',
                    'type':'f',
                    'len':1})
lvl1_header.append({'name':'Antenna-diameter-m',
                    'type':'f',
                    'len':1})
lvl1_header.append({'name':'Linear-antenna-gain',
                    'type':'f',
                    'len':1})
lvl1_header.append({'name':'Half-power-beam-width-deg',
                    'type':'f',
                    'len':1})
lvl1_header.append({'name':'Dual-pol-flag',
                    'type':'B',
                    'len':1})
lvl1_header.append({'name':'Sample-duration',
                    'type':'f',
                    'len':1})
lvl1_header.append({'name':'GPS-lat',
                    'type':'f',
                    'len':1})
lvl1_header.append({'name':'GPS-lon',
                    'type':'f',
                    'len':1})
lvl1_header.append({'name':'Calibration-interval',
                    'type':'i',
                    'len':1})
lvl1_header.append({'name':'Nb-radar-gates',
                    'type':'i',
                    'len':1})
lvl1_header.append({'name':'Nb-temp-gates',
                    'type':'i',
                    'len':1})
lvl1_header.append({'name':'Nb-hum-gates',
                    'type':'i',
                    'len':1})
lvl1_header.append({'name':'Nb-chirp-sequences',
                    'type':'i',
                    'len':1})
lvl1_header.append({'name':'Radar-gates-altitudes-m',
                    'type':'f',
                    'len':[1, 'Nb-radar-gates']})
lvl1_header.append({'name':'Temp-gates-altitudes-m',
                    'type':'f',
                    'len':[1, 'Nb-temp-gates']})
lvl1_header.append({'name':'Hum-gates-altitudes-m',
                    'type':'f',
                    'len':[1, 'Nb-hum-gates']})
lvl1_header.append({'name':'Nb-samples-Doppler-per-chirp',
                    'type':'i',
                    'len':[1, 'Nb-chirp-sequences']})
lvl1_header.append({'name':'Chirp-seq-start-index',
                    'type':'i',
                    'len':[1, 'Nb-chirp-sequences']})
lvl1_header.append({'name':'Nb-averaged-chirps',
                    'type':'i',
                    'len':[1, 'Nb-chirp-sequences']})
lvl1_header.append({'name':'Eff-seq-integration-time',
                    'type':'f',
                    'len':[1, 'Nb-chirp-sequences']})
lvl1_header.append({'name':'Range-res-chirp-seq',
                    'type':'f',
                    'len':[1, 'Nb-chirp-sequences']})
lvl1_header.append({'name':'Max-vel-chirp-seq',
                    'type':'f',
                    'len':[1, 'Nb-chirp-sequences']})
lvl1_header.append({'name':'Nb-time-samples',
                    'type':'i',
                    'len':1})


###############################################################################   
### SAMPLE METADATA INFO

lvl1_sample_metadata = []

lvl1_sample_metadata.append({'name':'Sample-length-bytes',
                             'type':'i',
                             'len':1,
                             'unit':'bytes',
                             'description':'Length of samples [bytes]'})
lvl1_sample_metadata.append({'name':'Time',
                             'type':'I',
                             'len':1,
                             'unit':'sec',
                             'description':'Time in seconds(UNIX)'})
lvl1_sample_metadata.append({'name':'Time-ms',
                             'type':'i',
                             'len':1,
                             'unit':'msec',
                             'description':'Time in milliseconds'})
lvl1_sample_metadata.append({'name':'Quality-flag',
                             'type':'B',
                             'len':1,
                             'unit':'-',
                             'description':'Quality flag (0 = ok, 1 = ADC saturation, 2 = spectral width too high)'})
lvl1_sample_metadata.append({'name':'Rain-rate',
                             'type':'f',
                             'len':1,
                             'unit':'mm/h',
                             'description':'Rain rate'})
lvl1_sample_metadata.append({'name':'Rel-humidity',
                             'type':'f',
                             'len':1,
                             'unit':'%',
                             'description':'Relative humidity'})
lvl1_sample_metadata.append({'name':'Environment-temp',
                             'type':'f',
                             'len':1,
                             'unit':'K',
                             'description':'Environmental temperature'})
lvl1_sample_metadata.append({'name':'Barometric-pressure',
                             'type':'f',
                             'len':1,
                             'unit':'hPa',
                             'description':'Barometric pressure'})
lvl1_sample_metadata.append({'name':'Wind-speed',
                             'type':'f',
                             'len':1,
                             'unit':'km/h',
                             'description':'Wind speed'})
lvl1_sample_metadata.append({'name':'Wind-direction',
                             'type':'f',
                             'len':1,
                             'unit':'deg',
                             'description':'Wind direction'})
lvl1_sample_metadata.append({'name':'Direct-detection-channel-voltage',
                             'type':'f',
                             'len':1,
                             'unit':'V',
                             'description':'Direct detection channel voltage'})
lvl1_sample_metadata.append({'name':'Direct-detection-brightness-temp','type':'f','len':1,'unit':'K',
                             'description':'Direct detection brightness temperature'})
lvl1_sample_metadata.append({'name':'Liquid-water-path','type':'f','len':1,'unit':'g/m3',
                             'description':'liquid water path'})
lvl1_sample_metadata.append({'name':'IF-power-ADC','type':'f','len':1,'unit':'microW',
                             'description':'IF power at ADC'})
lvl1_sample_metadata.append({'name':'Elevation-angle','type':'f','len':1,'unit':'deg',
                             'description':'Elevation angle'})
lvl1_sample_metadata.append({'name':'Azimuth-angle','type':'f','len':1,'unit':'deg',
                             'description':'Azimuth angle'})
lvl1_sample_metadata.append({'name':'Mitigation-status-flag','type':'f','len':1,'unit':'-',
                             'description':'Mitigation status flag (0/1: heater switch (ON/OFF), 0/10: blower switch (ON/OFF)'})
lvl1_sample_metadata.append({'name':'Transmitter-power','type':'f','len':1,'unit':'W',
                             'description':'Transmitter power'})
lvl1_sample_metadata.append({'name':'Transmitter-temp','type':'f','len':1,'unit':'K',
                             'description':'Transmitter temperature'})
lvl1_sample_metadata.append({'name':'Receiver-temp','type':'f','len':1,'unit':'K',
                             'description':'Receiver temperature'})
lvl1_sample_metadata.append({'name':'PC-temp','type':'f','len':1,'unit':'K',
                             'description':'PC temperature'})
lvl1_sample_metadata.append({'name':'Reserved','type':'f','len':3,'unit':'-',
                             'description':'Reserved bytes'})
lvl1_sample_metadata.append({'name':'Temp-profile','type':'f','len':[1, 'Nb-temp-gates'],'unit':'K',
                             'description':'Temperature profile'})
lvl1_sample_metadata.append({'name':'Abs-humidity-profile','type':'f','len':[1, 'Nb-hum-gates'],'unit':'g/m3',
                             'description':'Absolute humidity profile'})
lvl1_sample_metadata.append({'name':'Rel-humidity-profile','type':'f','len':[1, 'Nb-hum-gates'],'unit':'%',
                             'description':'Relative humidity profile'})
lvl1_sample_metadata.append({'name':'Linear-sensitivity-vert','type':'f','len':[1, 'Nb-radar-gates'],'unit':'mm6/m3',
                             'description':'Linear sensitivity limit in Ze units for vertical polarization'})
lvl1_sample_metadata.append({'name':'Linear-sensitivity-hor','type':'f','len':[1, 'Nb-radar-gates'],'unit':'mm6/m3',
                             'description':'Linear sensitivity limit in Ze units for horizontal polarization'})
lvl1_sample_metadata.append({'name':'Mask-occupied-gate','type':'B','len':[1, 'Nb-radar-gates'],'unit':'-',
                             'description':'Mask array of occupied radar gates (0: gate not occupied, 1: gate occupied'})


###############################################################################                    
### SAMPLE DATA INFO

lvl1_data = []
lvl1_data.append({'name':'Ze',
                  'type':'f',
                  'description':'Linear reflectivity in Ze units for vert. pol.',
                  'unit':'mm6/m3'})
lvl1_data.append({'name':'Mean-velocity',
                  'type':'f',
                  'description':'Nean velocity for vert. pol.',
                  'unit':'m/s'})
lvl1_data.append({'name':'Spectral-width',
                  'type':'f',
                  'description':'Spectral width [m/s] for vert. pol.',
                  'unit':'m/s'})
lvl1_data.append({'name':'Spectral-skewness',
                  'type':'f',
                  'description':'Spectral skewness for vert. pol.',
                  'unit':'-'})
lvl1_data.append({'name':'Spectral-kurtosis',
                  'type':'f',
                  'description':'Spectral kurtosis for vert. pol.',
                  'unit':'-'})
# If Dual-pol-flag == 1 
lvl1_data.append({'name':'Diff-reflectivity',
                  'type':'f',
                  'description':'Differential reflectivity',
                  'unit':'dB'})
lvl1_data.append({'name':'Corr-coefficient',
                  'type':'f',
                  'description':'Copolar correlation coefficient',
                  'unit':'-'})
lvl1_data.append({'name':'Diff-phase',
                  'type':'f',
                  'description':'Differential phase shift',
                  'unit':'deg'})
# If Dual-pol-flag == 2
lvl1_data.append({'name':'Slanted-Ze',
                  'type':'f',
                  'description':'Slanted reflectivity at 45 deg',
                  'unit':'mm6/m3'})
lvl1_data.append({'name':'Slanted-LDR',
                  'type':'f',
                  'description':'Slanted LDR at 45 deg',
                  'unit':'-'})
lvl1_data.append({'name':'Slanted-corr-coefficient',
                  'type':'f',
                  'description':'Slanted copolar correlation coefficient at 45 deg',
                  'unit':'-'})
lvl1_data.append({'name':'KDP',
                  'type':'f',
                  'description':'Specific differential phase shift',
                  'unit':'-'})
lvl1_data.append({'name':'Diff-attenuation',
                  'type':'f',
                  'description':'Differential attenuation',
                  'unit':'dB/km'})

