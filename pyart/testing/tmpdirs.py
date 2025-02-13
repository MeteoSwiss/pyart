"""
pyart.testing.tmpdirs
=====================

Classes for creating and cleaning temporary directories in unit tests.

.. autosummary::
    :toctree: generated/

    TemporaryDirectory
    InTemporaryDirectory
    InGivenDirectory


This module is taken from the nibable project.  The following license applies:

::

  The MIT License

  Copyright (c) 2009-2014 Matthew Brett <matthew.brett@gmail.com>
  Copyright (c) 2010-2013 Stephan Gerhard <git@unidesign.ch>
  Copyright (c) 2006-2014 Michael Hanke <michael.hanke@gmail.com>
  Copyright (c) 2011 Christian Haselgrove <christian.haselgrove@umassmed.edu>
  Copyright (c) 2010-2011 Jarrod Millman <jarrod.millman@gmail.com>
  Copyright (c) 2011-2014 Yaroslav Halchenko <debian@onerussian.com>

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
  THE SOFTWARE.

"""

import os
import shutil
from tempfile import mkdtemp, template

# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
""" Contexts for *with* statement providing temporary directories. """


class TemporaryDirectory:
    """Create and return a temporary directory. This has the same
    behavior as mkdtemp but can be used as a context manager.

    Upon exiting the context, the directory and everything contained
    in it are removed.

    Examples
    --------
    >>> import os
    >>> with TemporaryDirectory() as tmpdir:
    ...     fname = os.path.join(tmpdir, 'example_file.txt')
    ...     with open(fname, 'wt') as fobj:
    ...         _ = fobj.write('a string\\n')
    >>> os.path.exists(tmpdir)
    False
    """

    def __init__(self, suffix="", prefix=template, dir=None):
        self.name = mkdtemp(suffix, prefix, dir)
        self._closed = False

    def __enter__(self):
        return self.name

    def cleanup(self):
        if not self._closed:
            shutil.rmtree(self.name)
            self._closed = True

    def __exit__(self, exc, value, tb):
        self.cleanup()
        return False


class InTemporaryDirectory(TemporaryDirectory):
    """Create, return, and change directory to a temporary directory.

    Examples
    --------
    >>> import os
    >>> my_cwd = os.getcwd()
    >>> with InTemporaryDirectory() as tmpdir:
    ...     _ = open('test.txt', 'wt').write('some text')
    ...     assert os.path.isfile('test.txt')
    ...     assert os.path.isfile(os.path.join(tmpdir, 'test.txt'))
    >>> os.path.exists(tmpdir)
    False
    >>> os.getcwd() == my_cwd
    True
    """

    def __enter__(self):
        self._pwd = os.getcwd()
        os.chdir(self.name)
        return super().__enter__()

    def __exit__(self, exc, value, tb):
        os.chdir(self._pwd)
        return super().__exit__(exc, value, tb)


class InGivenDirectory:
    """Change directory to given directory for duration of ``with`` block.
    Useful when you want to use `InTemporaryDirectory` for the final test, but
    you are still debugging. For example, you may want to do this in the end:

    >>> with InTemporaryDirectory() as tmpdir:
    ...     # do something complicated which might break
    ...     pass

    But indeed the complicated thing does break, and meanwhile the
    ``InTemporaryDirectory`` context manager wiped out the directory with the
    temporary files that you wanted for debugging. So, while debugging, you
    replace with something like:

    >>> with InGivenDirectory() as tmpdir: # Use working directory by default
    ...     # do something complicated which might break
    ...     pass

    You can then look at the temporary file outputs to debug what is happening,
    fix, and finally replace ``InGivenDirectory`` with ``InTemporaryDirectory``
    again.

    """

    def __init__(self, path=None):
        """Initialize directory context manager

        Parameters
        ----------
        path : None or str, optional
            path to change directory to, for duration of ``with`` block.
            Defaults to ``os.getcwd()`` if None.

        """
        if path is None:
            path = os.getcwd()
        self.path = os.path.abspath(path)

    def __enter__(self):
        self._pwd = os.path.abspath(os.getcwd())
        if not os.path.isdir(self.path):
            os.mkdir(self.path)
        os.chdir(self.path)
        return self.path

    def __exit__(self, exc, value, tb):
        os.chdir(self._pwd)
