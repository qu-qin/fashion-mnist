import unittest
import numpy as np

from vanilla import nn_util


class TestNNUtil(unittest.TestCase):

    def setUp(self):
        np.random.seed(1)

    def test_rand_mini_batches(self):

        batch_size = 64
        X = np.random.randn(148, 56)
        Y = np.random.randn(148, 1)

        # pick random x, y to verify the mini batches
        rand_index = np.random.randint(0, 148)
        test_x, test_y = X[rand_index, :], Y[rand_index, :]

        mini_batches = nn_util.rand_mini_batches(X, Y, batch_size)
        correct_sizes = [64, 64, 20]

        for idx, (batch_X, batch_Y) in enumerate(mini_batches):
            self.assertEqual(batch_X.shape, (correct_sizes[idx], 56)) # X
            self.assertEqual(batch_Y.shape, (correct_sizes[idx], 1)) # Y
            for batch_idx, _ in enumerate(batch_X):
                if np.array_equal(batch_X[batch_idx], test_x):
                    self.assertTrue(np.array_equal(batch_Y[batch_idx], test_y))

    def tearDown(self):
        print u'\U0001f604'

if __name__ == '__main__':
    unittest.main()
