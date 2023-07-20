"""
=================================
Create a RHI plot from a MDV file
=================================

An example which creates a RHI plot of a MDV file using a RadarDisplay object.

"""
import matplotlib.pyplot as plt

import pyart

print(__doc__)

# Author: Jonathan J. Helmus (jhelmus@anl.gov)
# License: BSD 3 clause


filename = '110041.mdv'

# create the plot using RadarDisplay
radar = pyart.io.read_mdv(filename)
display = pyart.graph.RadarDisplay(radar)
fig = plt.figure(figsize=[5, 5])
ax = fig.add_subplot(111)
display.plot('reflectivity', 0, vmin=-16, vmax=64.0)
plt.show()
