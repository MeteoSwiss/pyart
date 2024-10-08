
.. DO NOT EDIT.
.. THIS FILE WAS AUTOMATICALLY GENERATED BY SPHINX-GALLERY.
.. TO MAKE CHANGES, EDIT THE SOURCE PYTHON FILE:
.. "examples/mapping/plot_map_two_radars_to_grid.py"
.. LINE NUMBERS ARE GIVEN BELOW.

.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        :ref:`Go to the end <sphx_glr_download_examples_mapping_plot_map_two_radars_to_grid.py>`
        to download the full example code.

.. rst-class:: sphx-glr-example-title

.. _sphx_glr_examples_mapping_plot_map_two_radars_to_grid.py:


==================================
Map two radars to a Cartesian grid
==================================

Map the reflectivity field of two nearby ARM XSARP radars from antenna
coordinates to a Cartesian grid.

.. GENERATED FROM PYTHON SOURCE LINES 10-56



.. image-sg:: /examples/mapping/images/sphx_glr_plot_map_two_radars_to_grid_001.png
   :alt: plot map two radars to grid
   :srcset: /examples/mapping/images/sphx_glr_plot_map_two_radars_to_grid_001.png
   :class: sphx-glr-single-img





.. code-block:: Python


    print(__doc__)

    # Author: Jonathan J. Helmus (jhelmus@anl.gov)
    # License: BSD 3 clause

    import matplotlib.pyplot as plt

    import pyart
    from pyart.testing import get_test_data

    # read in the data from both XSAPR radars
    xsapr_sw_file = get_test_data("swx_20120520_0641.nc")
    xsapr_se_file = get_test_data("sex_20120520_0641.nc")
    radar_sw = pyart.io.read_cfradial(xsapr_sw_file)
    radar_se = pyart.io.read_cfradial(xsapr_se_file)

    # filter out gates with reflectivity > 100 from both radars
    gatefilter_se = pyart.filters.GateFilter(radar_se)
    gatefilter_se.exclude_transition()
    gatefilter_se.exclude_above("corrected_reflectivity_horizontal", 100)
    gatefilter_sw = pyart.filters.GateFilter(radar_sw)
    gatefilter_sw.exclude_transition()
    gatefilter_sw.exclude_above("corrected_reflectivity_horizontal", 100)

    # perform Cartesian mapping, limit to the reflectivity field.
    grid = pyart.map.grid_from_radars(
        (radar_se, radar_sw),
        gatefilters=(gatefilter_se, gatefilter_sw),
        grid_shape=(1, 201, 201),
        grid_limits=((1000, 1000), (-50000, 40000), (-60000, 40000)),
        grid_origin=(36.57861, -97.363611),
        fields=["corrected_reflectivity_horizontal"],
    )

    # create the plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(
        grid.fields["corrected_reflectivity_horizontal"]["data"][0],
        origin="lower",
        extent=(-60, 40, -50, 40),
        vmin=0,
        vmax=48,
    )
    plt.show()


.. rst-class:: sphx-glr-timing

   **Total running time of the script:** (0 minutes 12.729 seconds)


.. _sphx_glr_download_examples_mapping_plot_map_two_radars_to_grid.py:

.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-example

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download Jupyter notebook: plot_map_two_radars_to_grid.ipynb <plot_map_two_radars_to_grid.ipynb>`

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download Python source code: plot_map_two_radars_to_grid.py <plot_map_two_radars_to_grid.py>`

    .. container:: sphx-glr-download sphx-glr-download-zip

      :download:`Download zipped: plot_map_two_radars_to_grid.zip <plot_map_two_radars_to_grid.zip>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
