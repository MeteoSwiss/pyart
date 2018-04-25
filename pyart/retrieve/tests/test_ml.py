import pyart
import matplotlib.pyplot as plt

rhi = pyart.io.read_cfradial(pyart.testing.RHI_ML_FILE)

refl_field = 'reflectivity'
rhohv_field = 'uncorrected_cross_correlation_ratio'

ml = pyart.retrieve.detect_ml(rhi, refl_field = refl_field,
                       rhohv_field = rhohv_field, 
                       interp_holes = False, 
                       max_length_holes = 250)

plt.figure()
plt.plot(ml['ml_cart']['x'],ml['ml_cart']['bottom_ml'])
plt.plot(ml['ml_cart']['x'],ml['ml_cart']['top_ml'])
plt.xlabel('Distance from radar [m]')
plt.ylabel('Height above radar [m]')
plt.legend(['ML bottom','ML top'])
plt.show()


