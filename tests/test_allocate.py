import unittest

import pymemalign


class TestAllocation(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_allocate(self):
        """Test memory allocation."""
        shape = (10, 10)
        arr = pymemalign.empty_aligned(shape=shape)
        print(f'expected shape {shape}, actual shape {arr.shape}')
        self.assertEqual(shape, arr.shape)
