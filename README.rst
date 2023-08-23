==================================================================================================
This branch is intended for MeteoSwiss Py-ART developers. Pull requests should target this branch!
==================================================================================================

The MeteoSwiss version of the Python ARM Radar Toolkit (Py-ART)
===============================================================

This is the MeteoSwiss version of `the Python ARM Radar Toolkit, Py-ART <http://arm-doe.github.io/pyart/>`_. This version contains features developed at MeteoSwiss that have not yet been pulled into the ARM-DOE Py-ART. MeteoSwiss contributes to the ARM-DOE Py-ART on a regular basis.

Users of `Pyrad <https://github.com/meteoswiss-mdr/pyrad>`_ that want to exploit its full functionality should preferably use this version instead of the ARM-DOE one. The MeteoSwiss Py-ART is a submodule of the Pyrad superproject.

Installation
============
To install the MeteoSwiss Py-ART as part of the Pyrad superproject have a look at the `Pyrad user manual(pdf) <https://github.com/meteoswiss-mdr/pyrad/blob/master/doc/pyrad_user_manual.pdf>`_
=======
|GithubCI|  |GithubDoc| |CodeCovStatus|

|DocsUsers| |DocsGuides| 

|AnacondaCloud| |CondaDownloads|

|CondaPlatforms| |License| 

.. |GithubCI| image:: https://github.com/MeteoSwiss/pyart/actions/workflows/ci.yml/badge.svg
    :target: https://github.com/MeteoSwiss/pyart/actions?query=workflow%3ACI

.. |GithubDoc| image:: https://github.com/MeteoSwiss/pyart/actions/workflows/build_docs.yml/badge.svg
    :target: https://github.com/MeteoSwiss/pyart/actions/workflows/build_docs.yml

.. |CodeCovStatus| image:: https://img.shields.io/codecov/c/github/MeteoSwiss/pyart.svg?logo=codecov
    :target: https://codecov.io/gh/MeteoSwiss/pyart

.. |DocsUsers| image:: https://img.shields.io/badge/docs-users-4088b8.svg
    :target: http://meteoswiss.github.io/pyart/API/index.html

.. |DocsGuides| image:: https://img.shields.io/badge/docs-guides-4088b8.svg
    :target: https://github.com/meteoswiss/pyart/tree/master/guides/
    
.. |AnacondaCloud| image:: https://anaconda.org/conda-forge/pyart_mch/badges/version.svg
    :target: https://anaconda.org/conda-forge/pyart_mch

.. |CondaLastUpdated| image:: https://img.shields.io/badge/Last%20updated-01%20Mar%202021-blue.svg?style=flat-square
    :target: https://anaconda.org/conda-forge/pyart_mch

.. |CondaDownloads| image:: https://anaconda.org/conda-forge/pyart_mch/badges/downloads.svg
    :target: https://anaconda.org/conda-forge/pyart_mch

.. |CondaPlatforms| image:: https://anaconda.org/conda-forge/pyart_mch/badges/platforms.svg
    :target: https://anaconda.org/conda-forge/pyart_mch

.. |License| image:: https://anaconda.org/conda-forge/pyart_mch/badges/license.svg
    :target: https://anaconda.org/conda-forge/pyart_mch

The MeteoSwiss version of the Python ARM Radar Toolkit (Py-ART)
===============================================================

This is the MeteoSwiss version of `the Python ARM Radar Toolkit, Py-ART <http://arm-doe.github.io/pyart/>`_. This version contains features developed at MeteoSwiss that have not yet been pulled into the ARM-DOE Py-ART. MeteoSwiss contributes to the ARM-DOE Py-ART on a regular basis.

Users of `Pyrad <https://github.com/meteoswiss-mdr/pyrad>`_ that want to exploit its full functionality should preferably use this version instead of the ARM-DOE one. The MeteoSwiss Py-ART is a submodule of the Pyrad superproject.

Installation
============
To install the MeteoSwiss Py-ART as part of the Pyrad superproject have a look at the `Pyrad user manual(pdf) <https://github.com/MeteoSwiss/pyrad/blob/master/additional_doc/pyrad_user_manual.pdf>`_


Use
===
For details on the implemented function check the `MeteoSwiss Py-ART library reference for users <https://pyart-mch.readthedocs.io/en/stable//>`_. Downloadable copies can be found in the Pyart readthedocs repository:

`For users(pdf) <https://media.readthedocs.org/pdf/pyart-mch/stable/pyart-mch.pdf>`_

Development
===========
Suggestions of developments and bug reports should use the `Issues page of the Pyrad github repository <https://github.com/meteoswiss-mdr/pyrad/issues>`_.

We welcome contributions. The process to contribute by partners external to MeteoSwiss is described in the `Pyrad user manual <https://github.com/meteoswiss-mdr/pyrad/blob/master/doc/pyrad_user_manual.pdf>`_. However consider contributing directly to the ARM-DOE Py-ART to serve a broader community.

Citation
========
Py-ART was originally developed in the context of the `ARM Research Facility <https://www.arm.gov/>`_. If you use the MeteoSwiss version of Py-ART for your work, please cite BOTH these papers:

Helmus J.J., S.M. Collis, (2016). The Python ARM Radar Toolkit (Py-ART), a Library for Working with Weather Radar Data in the Python Programming Language. Journal of Open Research Software. 4(1), p.e25. DOI: http://doi.org/10.5334/jors.119

Figueras i Ventura J., M. Lainer, Z. Schauwecker, J. Grazioli, U. Germann, (2020). Pyrad: A Real-Time Weather Radar Data Processing Framework Based on Py-ART. Journal of Open Research Software, 8(1), p.28. DOI: http://doi.org/10.5334/jors.330

Disclaimer
==========
The software is still in a development stage. Please let us know if you would like to test it.

MeteoSwiss cannot be held responsible for errors in the code or problems that could arise from its use.
