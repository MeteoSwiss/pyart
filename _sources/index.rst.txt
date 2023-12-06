===================================================
The MeteoSwiss Py-ART (Python ARM Radar Toolkit)
==================================================

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: User Guide

   userguide/index.rst

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Reference Guide

   API/index.rst

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Developer's Guide

   dev/index.rst

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Example Gallery

   examples/index.rst

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Notebook Gallery

   notebook-gallery.rst

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Changelog

   changelog.md

.. grid:: 1 2 2 2
    :gutter: 2

    .. grid-item-card:: :octicon:`book;10em`
        :link: https://projectpythia.org/radar-cookbook
        :text-align: center

        **Radar Cookbook**

        The cookbook provides in-depth information on how
        to use Py-ART (and other open radar packages), including how to get started.
        This is where to look for general conceptual descriptions on how
        to use parts of Py-ART, like its support for corrections and gridding.

    .. grid-item-card:: :octicon:`list-unordered;10em`
        :link: API/index
        :link-type: doc
        :text-align: center

        **Reference Guide**

        The reference guide contains detailed descriptions on
        every function and class within Py-ART. This is where to turn to understand
        how to use a particular feature or where to search for a specific tool


    .. grid-item-card:: :octicon:`graph;10em`
        :link: examples/index
        :link-type: doc
        :text-align: center

        **Example Gallery**

        Check out Py-ART's gallery of examples which contains
        sample code demonstrating various parts of Py-ART's functionality.

About the MeteoSwiss fork of Py-ART
===============

This is the MeteoSwiss version of the Python ARM Radar Toolkit, Py-ART. This version contains features developed at MeteoSwiss that have not yet been pulled into the ARM-DOE Py-ART. MeteoSwiss contributes to the ARM-DOE Py-ART on a regular basis.

Users of Pyrad that want to exploit its full functionality should preferably use this version instead of the ARM-DOE one. The MeteoSwiss Py-ART is a submodule of the Pyrad superproject.

The MeteoSwiss for of Py-ART proposes many additional bleeding edge features, among others

* additional readers and writers for many radar data formats used in Europe (OPERA, MeteoSwiss, MeteoFrance, FMI, ...)
* Limited support for lidar data (WindCube Leosphere)
* Routines for VPR processing (vertical profile of reflectivity)
* More routines for Doppler processing
* Routines for raw IQ data processing
* A tool to simulate the static visibility based on a DEM (gecsx)

as well as many others...

Citing Py-ART
=============
Py-ART was originally developed in the context of the ARM Research Facility. If you use the MeteoSwiss version of Py-ART for your work, please cite BOTH these papers:

* Helmus J.J., S.M. Collis, (2016). The Python ARM Radar Toolkit (Py-ART), a Library for Working with Weather Radar Data in the Python Programming Language. Journal of Open Research Software. 4(1), p.e25. DOI: http://doi.org/10.5334/jors.119
* Figueras i Ventura J., M. Lainer, Z. Schauwecker, J. Grazioli, U. Germann, (2020). Pyrad: A Real-Time Weather Radar Data Processing Framework Based on Py-ART. Journal of Open Research Software, 8(1), p.28. DOI: http://doi.org/10.5334/jors.330

What can Py-ART do?
===================
Py-ART has the ability to ingest (read) from a number of common weather radar
formats including Sigmet/IRIS, MDV, CF/Radial, UF, and NEXRAD Level II archive
files. Radar data can be written to NetCDF files which conform to the CF/Radial
convention.

Py-ART also contains routines which can produce common radar plots including
PPIs and RHIs.

|PPI|

|RHI|

.. |PPI| image:: _static/ppi.png

.. |RHI| image:: _static/rhi.png

Algorithms in the module are able to performs a number of corrections on the
radar moment data in antenna coordinate including attenuation correction of
the reflectivity, velocity dealiasing, and correction of the specific (Kdp)
and differential (PhiDP) phases.

A sophisticated mapping routines is able to efficiently create uniform
Cartesian grids of radar fields from one or more radars. Routines exist in
Py-ART for plotting these grids as well as saving them to NetCDF files.

Short Courses
=============

Various short courses on Py-ART and open source radar software have been given
which contain tutorial like materials and additional examples.

* `2021 ERAD, Open Source Radar Software Course <https://github.com/openradar/erad2020>`_
* `2022 AMS radar conference, Open Radar Short Course  <https://github.com/openradar/ams-open-radar-2023>`_

Install
=======

The easiest method for installing Py-ART is to use the conda packages from
the latest release and use Python 3, as Python 2 support ended January 1st,
2020 and many packages including Py-ART no longer support Python 2.
To do this you must download and install
`Anaconda <https://www.anaconda.com/download/#>`_ or
`Miniconda <https://conda.io/miniconda.html>`_.
With Anaconda or Miniconda install, it is recommended to create a new conda
environment when using Py-ART or even other packages. To create a new
environment based on the `environment.yml <https://github.com/MeteoSwiss/pyart/blob/master/environment.yml>`_::

    conda env create -f environment.yml

Or for a basic environment and downloading optional dependencies as needed::

    conda create -n pyart_env -c conda-forge python=3.8 pyart_mch

Basic command in a terminal or command prompt to install the latest version of
Py-ART::

    conda install -c conda-forge pyart_mch

To update an older version of Py-ART to the latest release use::

    conda update -c conda-forge pyart_mch

If you do not wish to use Anaconda or Miniconda as a Python environment or want
to use the latest, unreleased version of Py-ART clone the git repository or
download the repositories zip file and extract the file. Then run:

$ python setup.py install

Additional detail on installing Py-ART can be found in the installation section.

Dependencies
============

Py-ART is tested to work under Python 3.6, 3.7 and 3.8

The required dependencies to install Py-ART in addition to Python are:

* `NumPy <https://www.numpy.org/>`_
* `SciPy <https://www.scipy.org>`_
* `matplotlib <https://matplotlib.org/>`_
* `netCDF4 <https://github.com/Unidata/netcdf4-python>`_

A working C/C++ compiler is required for some optional modules. An easy method
to install these dependencies is by using a
`Scientific Python distributions <http://scipy.org/install.html>`_.
`Anaconda Compilers <https://www.anaconda.com/distribution/>`_ will install
all of the above packages by default on Windows, Linux and Mac computers and is
provided free of charge by Anaconda. Anaconda also has their own compilers,
which may be required for optional dependencies such as CyLP. These compilers
can be found here:
https://docs.conda.io/projects/conda-build/en/latest/resources/compiler-tools.html

Optional Dependences
====================

The above Python modules are require before installing Py-ART, additional
functionality is available of the following modules are installed.

* `TRMM Radar Software Library (RSL)
  <https://trmm-fc.gsfc.nasa.gov/trmm_gv/software/rsl/>`_.
  If installed Py-ART will be able t`o read in radar data in a number of
  additional formats (Lassen, McGill, Universal Format, and RADTEC) and
  perform automatic dealiasing of Doppler velocities.  RSL should be
  install prior to installing Py-ART. The environmental variable `RSL_PATH`
  should point to the location where RSL was installed if RSL was not
  installed in the default location (/usr/local/trmm), such as a anaconda path
  (/usr/anaconda3/envs/pyart_env/.

* In order to read files which are stored in HDF5 files the
  `h5py <https://www.h5py.org/>`_ package and related libraries must be
  installed.

* A linear programming solver and Python wrapper to use the LP phase
  processing method. `CyLP <https://github.com/mpy/CyLP>`_ is recommended as
  it gives the fastest results, but
  `PyGLPK <https://tfinley.net/software/pyglpk/>`_ and
  `CVXOPT <https://cvxopt.org/>`_ are also supported. The underlying LP
  solvers `CBC <https://projects.coin-or.org/Cbc>`_ or
  `GLPK <https://www.gnu.org/software/glpk/>`_ will also be required depending
  on which wrapper is used. When using `CyLP <https://github.com/mpy/CyLP>`_
  a path to coincbc is needed by setting the `COIN_INSTALL_DIR` path, such as
  (/usr/anaconda3/envs/pyart_env/).

* `Cartopy <https://scitools.org.uk/cartopy/docs/latest/>`_. If installed,
  the ability to plot grids on geographic maps is available.

* `xarray <https://xarray.pydata.org/en/stable/>`_. If installed, gives the
  ability to work with the grid dataset used in grid plotting.

* `Basemap <https://matplotlib.org/basemap/>`_. If installed, also gives the
  ability to plot grids on geographic maps, but Cartopy is recommended over
  Basemap.

* `wradlib <https://docs.wradlib.org/en/latest/>`_. Needed to calculate the texture
  of a differential phase field.

* `pytest <https://docs.pytest.org/en/latest/>`_.
  Required to run the Py-ART unit tests.

* `gdal <https://pypi.python.org/pypi/GDAL/>`_.
  Required to output GeoTIFFs from `Grid` objects.

Getting help
============
To get help please either open an  `issue on github <https://github.com/MeteoSwiss/pyart/issues>`_ or use the  `pyrad github discussions <https://github.com/MeteoSwiss/pyrad/discussions>`_

Contributing
============
Py-ART is an open source software package distributed under the `New BSD License <https://opensource.org/licenses/BSD-3-Clause>`_
Source code for the package is available on `GitHub <https://github.com/MeteoSwiss/pyart>`_. Feature requests and bug reports
can be submitted to the `Issue tracker <https://github.com/MeteoSwiss/pyart/issues>`_.

Contributions of source code, documentation or additional examples are always
appreciated from both developers and users. 
