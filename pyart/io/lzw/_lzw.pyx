# cython: language_level=3
import numpy as np
cimport numpy as np

cdef extern from "lzw.h":
    int Expand(unsigned char *inputaddr, int insize,
               unsigned char *outputaddr, int outsize)

def expand(np.ndarray[np.uint8_t, ndim=1] inp,
           int insize,
           np.ndarray[np.uint8_t, ndim=1] out,
           int outsize):
    cdef int ret
    ret = Expand(&inp[0], insize, &out[0], outsize)
    return ret
