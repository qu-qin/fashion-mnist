import unittest
import numpy as np

from vanilla import conv_util


class TestCNNUtil(unittest.TestCase):

    def setUp(self):
        np.random.seed(1)

    def test_zero_padding(self):
        arr = np.random.randn(4, 3, 3, 2)
        padded_arr = conv_util.zero_padding(arr, 2)
        self.assertEqual(padded_arr.shape, (4, 7, 7, 2))

    def test_conv_slice(self):
        a_slice = np.random.randn(3, 3, 3)
        W = np.random.randn(3, 3, 3)
        b = np.random.randn(1, 1, 1)
        result = conv_util.conv_slice(a_slice, W, b)
        self.assertIsInstance(result, float)

    def test_conv_forward(self):
        A_prev = np.random.randn(10, 4, 4, 3)
        W = np.random.randn(2, 2, 3, 8)
        b = np.random.randn(1, 1, 1, 8)
        output, _ = conv_util.conv_forward(A_prev, W, b, 1, 2)
        self.assertEqual(output.shape, (10, 7, 7, 8))

    def test_conv_backward(self):
        A_prev = np.random.randn(10, 4, 4, 3)
        W = np.random.randn(2, 2, 3, 8)
        b = np.random.randn(1, 1, 1, 8)
        output, cache = conv_util.conv_forward(A_prev, W, b, 1, 1)
        dA_prev, dW, db = conv_util.conv_backward(output, cache)
        self.assertEqual(dA_prev.shape, A_prev.shape)
        self.assertEqual(dW.shape, W.shape)
        self.assertEqual(db.shape, b.shape)

    def test_max_pool_forward(self):
        A_prev = np.random.randn(2, 4, 4, 3)
        output, _ = conv_util.max_pool_forward(A_prev, stride=1, f=3)
        self.assertEqual(output.shape, (2, 2, 2, 3))

    def test_max_pool_backward(self):
        A_prev = np.random.randn(2, 4, 4, 3)
        output, cache = conv_util.max_pool_forward(A_prev, stride=1, f=3)
        dA = np.random.randn(2, 2, 2, 3)
        dA_prev = conv_util.max_pool_backward(dA, cache)
        self.assertEqual(dA_prev.shape, A_prev.shape)

    def tearDown(self):
        print u'\U0001f604'

if __name__ == '__main__':
    unittest.main()
