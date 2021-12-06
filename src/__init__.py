from __future__ import absolute_import

import numpy as np
from . import _pymemalign

__all__ = ['empty_aligned', 'empty_like_aligned']


PYMEMALIGN_TYPEMAP = {
    np.byte: np.int8,
    np.short: np.int16,
    int: np.int64,
    np.int: np.int64,
    np.long: np.int64,
    np.longlong: np.int64,
    np.ubyte: np.uint8,
    np.ushort: np.uint16,
    np.uint: np.uint32,
    np.uint64: np.uint32,
    np.ulonglong: np.uint32,
    np.half: np.float16,
    float: np.float32,
    np.float: np.float32,
    np.double: np.float64,
    np.complex: np.complex64,
    np.cdouble: np.complex128,
}


def _get_mapped_type(dtype):
    return PYMEMALIGN_TYPEMAP.get(dtype, dtype)


def _to_tuple(v):
    if isinstance(v, tuple):
        return v
    elif hasattr(v, '__iter__'):
        return tuple(v)
    else:
        return (v,)


def empty_aligned(shape, dtype=np.float32, order='C', alignment=None):
    """
    """
    shape_tuple = _to_tuple(shape)
    align = alignment if alignment is not None else 32
    return _pymemalign.allocate(_get_mapped_type(dtype), shape_tuple, align)


def empty_like_aligned(array, alignment=None):
    order = 'F' if np.isfortran(array) else 'C'
    return empty_aligned(shape=array.shape, dtype=array.dtype,
                         order=order, alignment=alignment)
