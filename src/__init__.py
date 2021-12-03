from __future__ import absolute_import

import numpy as np
from . import _pymemalign


def empty_aligned(shape, dtype=np.float32, order='C', alignment=None):
    """
    """
    return _pymemalign.allocate(dtype, shape)


def empty_like_aligned(array, alignment=None):
    order = 'F' if numpy.isfortran(array) else 'C'
    return empty_aligned(shape=array.shape, dtype=array.dtype,
                         order=order, alignment=alignment)
