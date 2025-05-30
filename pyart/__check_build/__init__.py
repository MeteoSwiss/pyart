""" Module to give helpful messages to the user that did not
compile Py-ART properly.
"""

import os

INPLACE_MSG = """
It appears that you are importing a local pyart source tree. For this, you
need to have an inplace install. Maybe you are in the source directory and
you need to try from another location."""

STANDARD_MSG = """
If you have used an installer, please check that it is suited for your
Python version, your operating system and your platform."""


def raise_build_error(e):
    # Raise a comprehensible error and list the contents of the
    # directory to help debugging on the mailing list.
    local_dir = os.path.split(__file__)[0]
    msg = STANDARD_MSG
    if local_dir == "pyart/__check_build":
        # Picking up the local install: this will work only if the
        # install is an 'inplace build'
        msg = INPLACE_MSG
    dir_content = list()
    for i, filename in enumerate(os.listdir(local_dir)):
        if (i + 1) % 3:
            dir_content.append(filename.ljust(26))
        else:
            dir_content.append(filename + "\n")
    raise ImportError(
        f"""{e}
___________________________________________________________________________
Contents of {local_dir}:
{''.join(dir_content).strip()}
___________________________________________________________________________
It seems that Py-ART has not been built correctly.

If you have installed Py-ART from source, please do not forget
to build the package before using it: run `python setup.py install` in the
source directory.
{msg}"""
    )


try:
    from ._check_build import check_build  # noqa
except ImportError as e:
    raise_build_error(e)
