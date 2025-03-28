:orphan:

Example Gallery
===============

The files used in these examples are available for download_.

.. _download: https://adc.arm.gov/pyart/example_data/



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. thumbnail-parent-div-close

.. raw:: html

    </div>


Moment correction examples
--------------------------

Performing radar moment corrections in antenna (radial) coordinates.



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In this example the reflectivity attenuation is calculated and then corrected for a polarimetric radar using a Z-PHI method implemented in Py-ART.">

.. only:: html

  .. image:: /examples/correct/images/thumb/sphx_glr_plot_attenuation_thumb.png
    :alt:

  :ref:`sphx_glr_examples_correct_plot_attenuation.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Correct reflectivity attenuation</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In this example doppler velocities are dealiased using the ial condition of the dealiasing, using the region-based dealiasing algorithm in Py-ART.">

.. only:: html

  .. image:: /examples/correct/images/thumb/sphx_glr_plot_dealias_thumb.png
    :alt:

  :ref:`sphx_glr_examples_correct_plot_dealias.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Dealias doppler velocities using the Region Based Algorithm</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


Input/Output Examples
--------------------------

Reading/writing a variety of radar data using Py-ART.



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In this example, we will show how to read in older NEXRAD files prior to 2008 that are missing some coordinate metadata.">

.. only:: html

  .. image:: /examples/io/images/thumb/sphx_glr_plot_older_nexrad_data_aws_thumb.png
    :alt:

  :ref:`sphx_glr_examples_io_plot_older_nexrad_data_aws.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Reading Older NEXRAD Data and Fixing Latitude and Longitude Issues</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Within this example, we show how you can remotely access Next Generation Weather Radar (NEXRAD) Data from Amazon Web Services and plot quick looks of the datasets.">

.. only:: html

  .. image:: /examples/io/images/thumb/sphx_glr_plot_nexrad_data_aws_thumb.png
    :alt:

  :ref:`sphx_glr_examples_io_plot_nexrad_data_aws.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Reading NEXRAD Data from the AWS Cloud</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


Mapping examples
----------------

Mapping one or multiple radars from antenna coordinates to a Cartesian grid.



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Map the reflectivity field of a single radar from Antenna coordinates to a Cartesian grid.">

.. only:: html

  .. image:: /examples/mapping/images/thumb/sphx_glr_plot_map_one_radar_to_grid_thumb.png
    :alt:

  :ref:`sphx_glr_examples_mapping_plot_map_one_radar_to_grid.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Map a single radar to a Cartesian grid</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Map the reflectivity field of two nearby ARM XSARP radars from antenna coordinates to a Cartesian grid.">

.. only:: html

  .. image:: /examples/mapping/images/thumb/sphx_glr_plot_map_two_radars_to_grid_thumb.png
    :alt:

  :ref:`sphx_glr_examples_mapping_plot_map_two_radars_to_grid.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Map two radars to a Cartesian grid</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Map the reflectivity field of a single radar in Antenna coordinates to another radar in Antenna coordinates and compare the fields.">

.. only:: html

  .. image:: /examples/mapping/images/thumb/sphx_glr_plot_compare_two_radars_gatemapper_thumb.png
    :alt:

  :ref:`sphx_glr_examples_mapping_plot_compare_two_radars_gatemapper.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Compare Two Radars Using Gatemapper</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


Plotting examples
-----------------

Plotting real world radar data with Py-ART.



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="An example which creates a RHI plot of a MDV file using a RadarDisplay object.">

.. only:: html

  .. image:: /examples/plotting/images/thumb/sphx_glr_plot_rhi_mdv_thumb.png
    :alt:

  :ref:`sphx_glr_examples_plotting_plot_rhi_mdv.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Create a RHI plot from a MDV file</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="An example which creates a PPI plot of a MDV file using a RadarDisplay object.">

.. only:: html

  .. image:: /examples/plotting/images/thumb/sphx_glr_plot_ppi_mdv_thumb.png
    :alt:

  :ref:`sphx_glr_examples_plotting_plot_ppi_mdv.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Create a PPI plot from a MDV file</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="An example which creates a PPI plot of a Cfradial file.">

.. only:: html

  .. image:: /examples/plotting/images/thumb/sphx_glr_plot_ppi_cfradial_thumb.png
    :alt:

  :ref:`sphx_glr_examples_plotting_plot_ppi_cfradial.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Create a PPI plot from a Cfradial file</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="An example which creates a plot containing the first collected scan from a NEXRAD file.">

.. only:: html

  .. image:: /examples/plotting/images/thumb/sphx_glr_plot_nexrad_reflectivity_thumb.png
    :alt:

  :ref:`sphx_glr_examples_plotting_plot_nexrad_reflectivity.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Create a plot of NEXRAD reflectivity</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="An example which extracts a cross section at two azimuth angles from a volume of PPI scans and plots both cross sections.">

.. only:: html

  .. image:: /examples/plotting/images/thumb/sphx_glr_plot_xsect_thumb.png
    :alt:

  :ref:`sphx_glr_examples_plotting_plot_xsect.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Plot a cross section from  a PPI volume</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="An example which creates a RHI plot of a CF/Radial file using a RadarDisplay object.">

.. only:: html

  .. image:: /examples/plotting/images/thumb/sphx_glr_plot_rhi_cfradial_singlescan_thumb.png
    :alt:

  :ref:`sphx_glr_examples_plotting_plot_rhi_cfradial_singlescan.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Create a multiple panel RHI plot from a CF/Radial file</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="An example which creates a two panel RHI plot of a cfradial file.  The fields included in the two panels are reflectivity and doppler velocity.">

.. only:: html

  .. image:: /examples/plotting/images/thumb/sphx_glr_plot_rhi_two_panel_thumb.png
    :alt:

  :ref:`sphx_glr_examples_plotting_plot_rhi_two_panel.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Create a two panel RHI plot</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="An example which creates a plot containing multiple moments taken from a NEXRAD Archive file.">

.. only:: html

  .. image:: /examples/plotting/images/thumb/sphx_glr_plot_nexrad_multiple_moments_thumb.png
    :alt:

  :ref:`sphx_glr_examples_plotting_plot_nexrad_multiple_moments.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Create a plot of multiple moments from a NEXRAD file</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="An example which creates a multiple panel RHI plot of a CF/Radial file using a RadarDisplay object.">

.. only:: html

  .. image:: /examples/plotting/images/thumb/sphx_glr_plot_rhi_cfradial_thumb.png
    :alt:

  :ref:`sphx_glr_examples_plotting_plot_rhi_cfradial.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Create a multiple panel RHI plot from a CF/Radial file</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="An example that creates a 3 panel plot of a PPI, latitude slice, and longitude slice using xarray and a cartopy background.">

.. only:: html

  .. image:: /examples/plotting/images/thumb/sphx_glr_plot_three_panel_gridmapdisplay_thumb.png
    :alt:

  :ref:`sphx_glr_examples_plotting_plot_three_panel_gridmapdisplay.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Create a 3 panel plot using GridMapDisplay</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="An example which creates a PPI plot of a file with a cartopy background and range rings">

.. only:: html

  .. image:: /examples/plotting/images/thumb/sphx_glr_plot_ppi_with_rings_thumb.png
    :alt:

  :ref:`sphx_glr_examples_plotting_plot_ppi_with_rings.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Create a PPI plot on a cartopy map</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Image muting reduces the visual prominence of the reflectivities within identified melting and mixed precipitation features in winter storms (i.e. regions with low correlation coefficient values). Reflectivities corresponding to melting and mixed precipitation features are deemphasized using a gray scale and the regions with just snow and just rain are depicted in a corresponding full-color scale. The ultimate utility of image muting radar reflectivity is to reduce the misinterpretation of regions of melting or mixed precipitation as opposed to heavy snow or heavy rain.">

.. only:: html

  .. image:: /examples/plotting/images/thumb/sphx_glr_plot_nexrad_image_muted_reflectivity_thumb.png
    :alt:

  :ref:`sphx_glr_examples_plotting_plot_nexrad_image_muted_reflectivity.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Create an image-muted reflectivity plot</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This is an example of how to plot a cross section of your radar grid using the GridMapDisplay">

.. only:: html

  .. image:: /examples/plotting/images/thumb/sphx_glr_plot_cross_section_thumb.png
    :alt:

  :ref:`sphx_glr_examples_plotting_plot_cross_section.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Plot a Cross Section from a Grid</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="An example which creates an RHI plot of velocity using a RadarDisplay object and adding Reflectivity contours from the same MDV file.">

.. only:: html

  .. image:: /examples/plotting/images/thumb/sphx_glr_plot_rhi_data_overlay_thumb.png
    :alt:

  :ref:`sphx_glr_examples_plotting_plot_rhi_data_overlay.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Create an RHI plot with reflectivity contour lines from an MDV file</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This is an example of how to modify a colobar within a Py-ART display object.">

.. only:: html

  .. image:: /examples/plotting/images/thumb/sphx_glr_plot_modify_colorbar_thumb.png
    :alt:

  :ref:`sphx_glr_examples_plotting_plot_modify_colorbar.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Modify a Colorbar for your Plot</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="An example which creates an RHI plot of reflectivity using a RadarDisplay object and adding differnential Reflectivity contours from the same MDV file.">

.. only:: html

  .. image:: /examples/plotting/images/thumb/sphx_glr_plot_rhi_contours_differential_reflectivity_thumb.png
    :alt:

  :ref:`sphx_glr_examples_plotting_plot_rhi_contours_differential_reflectivity.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Create an RHI plot with reflectivity contour lines from an MDV file</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This is an example of what colormaps are available in Py-ART, and how to add them to your own plots.">

.. only:: html

  .. image:: /examples/plotting/images/thumb/sphx_glr_plot_choose_a_colormap_thumb.png
    :alt:

  :ref:`sphx_glr_examples_plotting_plot_choose_a_colormap.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Choose a Colormap for your Plot</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


Retrieval Examples
------------------

Retrievals from various radars, such as additional fields or subsets of the data.



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Given a radar and a point, extract the column of radar data values above a point">

.. only:: html

  .. image:: /examples/retrieve/images/thumb/sphx_glr_plot_column_subset_thumb.png
    :alt:

  :ref:`sphx_glr_examples_retrieve_plot_column_subset.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Extract a radar column above a point</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Calculates and plots the composite reflectivity, or the maximum reflectivity across all of the elevations.">

.. only:: html

  .. image:: /examples/retrieve/images/thumb/sphx_glr_plot_composite_reflectivity_thumb.png
    :alt:

  :ref:`sphx_glr_examples_retrieve_plot_composite_reflectivity.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Calculate and Plot Composite Reflectivity</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Calculates a VAD and plots a vertical profile of wind">

.. only:: html

  .. image:: /examples/retrieve/images/thumb/sphx_glr_plot_vad_thumb.png
    :alt:

  :ref:`sphx_glr_examples_retrieve_plot_vad.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Calculate and Plot VAD profile</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Calculates a hydrometeor classification and displays the results">

.. only:: html

  .. image:: /examples/retrieve/images/thumb/sphx_glr_plot_hydrometeor_thumb.png
    :alt:

  :ref:`sphx_glr_examples_retrieve_plot_hydrometeor.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Calculate and Plot hydrometeor classification</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Convective-Stratiform classification">

.. only:: html

  .. image:: /examples/retrieve/images/thumb/sphx_glr_plot_convective_stratiform_thumb.png
    :alt:

  :ref:`sphx_glr_examples_retrieve_plot_convective_stratiform.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Convective-Stratiform classification</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:
   :includehidden:


   /examples/correct/index.rst
   /examples/io/index.rst
   /examples/mapping/index.rst
   /examples/plotting/index.rst
   /examples/retrieve/index.rst


.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-gallery

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download all examples in Python source code: examples_python.zip </examples/examples_python.zip>`

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download all examples in Jupyter notebooks: examples_jupyter.zip </examples/examples_jupyter.zip>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
