"""
pyart.retrieve.gecsx_functions
========================

Cython routines for the gecsx visibility processing

.. autosummary::
    :toctree: generated/

    grid_displacement_pc
    grid_shift

"""
# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# currently part of the Cython distribution).
cimport numpy as np
import numpy as np

# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
DTYPE = np.double

# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.double_t DTYPE_t

def argmin_abs(np.ndarray[DTYPE_t, ndim=1] vec, double elem):
    cdef double min = np.inf
    cdef int minidx = 0
    cdef int i
    for i in range(len(vec)):
        if abs(vec[i] - elem) < min:
            min = vec[i]
            minidx = i
    return minidx

def vis_weighting(np.ndarray[DTYPE_t, ndim=1] daz_vec, 
                  np.ndarray[DTYPE_t, ndim=1] del_vec,
                  np.ndarray[DTYPE_t, ndim=1] daz_area,
                  np.ndarray[DTYPE_t, ndim=2] ant_weight,
                  np.ndarray[DTYPE_t, ndim=1] minvisvals,
                  double el):
    
    cdef int ndaz = len(daz_vec)
    cdef int ndel = len(del_vec)
    cdef int idxmin = 0
    cdef double vis = 0
    cdef int idaz, idel
    for idaz in range(ndaz):
        mina = argmin_abs(daz_area, daz_vec[idaz])
        for idel in range(ndel):
            if minvisvals[mina] <= el + del_vec[idel]:
                vis += ant_weight[idaz, idel]
    return vis
