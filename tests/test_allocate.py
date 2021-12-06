import unittest

import numpy as np

import pymemalign


class TestAllocation(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def __test_allocate(self, shape, dtypes, expected_type):
        alignments = [32, 64, 128]
        for a in alignments:
            for t in dtypes:
                arr = pymemalign.empty_aligned(shape=shape, dtype=t)
                print(f'expected shape {shape}, actual shape {arr.shape}')
                print(f'input type {t}, output type {arr.dtype}')
                self.assertEqual(shape, arr.shape)
                self.assertEqual(expected_type, arr.dtype)

    def test_allocate_int8(self):
        """Test memory allocation (int8)."""
        shape = (10, 10)
        expected = np.int8
        types_tested = [np.int8, np.byte]
        self.__test_allocate(shape, types_tested, expected)

    def test_allocate_int16(self):
        """Test memory allocation (int16)."""
        shape = (10, 10)
        expected = np.int16
        types_tested = [np.int16, np.short]
        self.__test_allocate(shape, types_tested, expected)

    def test_allocate_int32(self):
        """Test memory allocation (int32)."""
        shape = (10, 10)
        expected = np.int32
        types_tested = [np.int32]
        self.__test_allocate(shape, types_tested, expected)

    def test_allocate_int64(self):
        """Test memory allocation (int64)."""
        shape = (10, 10)
        expected = np.int64
        types_tested = [int, np.int, np.int64, np.long, np.longlong]
        self.__test_allocate(shape, types_tested, expected)

    def test_allocate_uint8(self):
        """Test memory allocation (uint8)."""
        shape = (10, 10)
        expected = np.uint8
        types_tested = [np.uint8, np.ubyte]
        self.__test_allocate(shape, types_tested, expected)

    def test_allocate_uint16(self):
        """Test memory allocation (uint16)."""
        shape = (10, 10)
        expected = np.uint16
        types_tested = [np.uint16, np.ushort]
        self.__test_allocate(shape, types_tested, expected)

    def test_allocate_uint32(self):
        """Test memory allocation (uint32)."""
        shape = (10, 10)
        expected = np.uint32
        types_tested = [np.uint32, np.uint]
        self.__test_allocate(shape, types_tested, expected)

    def test_allocate_uint64(self):
        """Test memory allocation (uint64)."""
        shape = (10, 10)
        expected = np.uint32
        types_tested = [np.uint64, np.ulonglong]
        self.__test_allocate(shape, types_tested, expected)

    def test_allocate_float16(self):
        """Test memory allocation (float16)."""
        shape = (10, 10)
        expected = np.float16
        types_tested = [np.half, np.float16]
        self.__test_allocate(shape, types_tested, expected)

    def test_allocate_float32(self):
        """Test memory allocation (float32)."""
        shape = (10, 10)
        expected = np.float32
        types_tested = [float, np.float, np.float32]
        self.__test_allocate(shape, types_tested, expected)

    def test_allocate_float64(self):
        """Test memory allocation (float64)."""
        shape = (10, 10)
        expected = np.float64
        types_tested = [np.double, np.float64]
        self.__test_allocate(shape, types_tested, expected)

    def test_allocate_complex64(self):
        """Test memory allocation (complex64)."""
        shape = (10, 10)
        expected = np.complex64
        types_tested = [complex, np.complex, np.complex64]
        self.__test_allocate(shape, types_tested, expected)

    def test_allocate_complex128(self):
        """Test memory allocation (complex128)."""
        shape = (10, 10)
        expected = np.complex128
        types_tested = [np.complex128]
        self.__test_allocate(shape, types_tested, expected)
