import gc
import time

import memory_profiler
import numpy as np
import pytest

import pymemalign


def _allocate_large_mem(length, dtype):
    time.sleep(1)
    arr = pymemalign.empty_aligned(shape=length, dtype=dtype)
    arr[:] = 0
    time.sleep(1)
    del arr
    time.sleep(1)
    gc.collect()
    time.sleep(1)


@pytest.mark.parametrize("input_type, expected_type", [
    (np.int8, np.int8),
    (np.byte, np.int8),
    (np.int16, np.int16),
    (np.short, np.int16),
    (np.int32, np.int32),
    (np.int64, np.int64),
    (int, np.int64),
    (np.int, np.int64),
    (np.long, np.int64),
    (np.longlong, np.int64),
    (np.uint8, np.uint8),
    (np.ubyte, np.uint8),
    (np.uint16, np.uint16),
    (np.ushort, np.uint16),
    (np.uint32, np.uint32),
    (np.uint, np.uint32),
    (np.uint64, np.uint32),
    (np.ulonglong, np.uint32),
    (np.float16, np.float16),
    (np.half, np.float16),
    (np.float32, np.float32),
    (float, np.float32),
    (np.float, np.float32),
    (np.float64, np.float64),
    (np.double, np.float64),
    (np.complex64, np.complex64),
    (np.complex, np.complex64),
    (complex, np.complex64),
    (np.complex128, np.complex128),
])
def test_allocate(input_type, expected_type):
    shape = (10, 10,)
    alignments = [32, 64, 128]
    for a in alignments:
        arr = pymemalign.empty_aligned(shape=shape, dtype=input_type)
        print(f'expected shape {shape}, actual shape {arr.shape}')
        print(f'input type {input_type}, output type {arr.dtype}')
        assert shape == arr.shape
        assert expected_type == arr.dtype


def test_memory_leak():
    """Test memory leak"""
    gc.collect()
    mem_profile = memory_profiler.memory_usage((_allocate_large_mem, (10 ** 7, np.float64,),))
    mem_max = max(mem_profile)
    mem_start = mem_profile[0]
    mem_end = mem_profile[-1]
    print(f'mem_start={mem_start}, mem_end={mem_end}, mem_max={mem_max}')
    incr = mem_max - mem_start
    diff = abs((mem_end - mem_start) / mem_start)
    assert incr > 50
    assert diff < 0.1
