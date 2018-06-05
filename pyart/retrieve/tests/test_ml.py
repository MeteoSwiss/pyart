import pyart
import matplotlib.pyplot as plt
import numpy as np
from pyrad.io import read_data_mxpol

rhi = read_data_mxpol.pyrad_MXPOL(pyart.testing.RHI_ML_FILE,
                                  field_names = ['Zh','Rhohv'])

# Remove 

refl_field = 'reflectivity'
rhohv_field = 'uncorrected_cross_correlation_ratio'

ml = pyart.retrieve.detect_ml(rhi, refl_field = refl_field,
                       rhohv_field = rhohv_field, 
                       interp_holes = False, 
                       max_length_holes = 250)

plt.figure()
plt.plot(ml[0]['ml_cart']['x'],ml[0]['ml_cart']['bottom_ml'])
plt.plot(ml[0]['ml_cart']['x'],ml[0]['ml_cart']['top_ml'])
plt.xlabel('Distance from radar [m]')
plt.ylabel('Height above radar [m]')
plt.legend(['ML bottom','ML top'])
plt.show()
