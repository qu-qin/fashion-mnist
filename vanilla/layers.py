import numpy as np

from vanilla import conv_util, nn_util


class Conv2d(object):

    def __init__(self, stride, pad):
        self.stride = stride
        self.pad = pad
        self.W = None
        self.b = None
        self.running_mean = None
        self.running_var = None
        self.conv_cache = None
        self.relu_cache = None
        self.batchnorm_cache = None

    def forward(self, A_prev, W, b, gamma, beta, train):

        self.W, self.b = W, b

        if self.running_mean is None or self.running_var is None:
            self.running_mean = np.zeros(gamma.shape)
            self.running_var = np.zeros(gamma.shape)

        conv_output, self.conv_cache = conv_util.conv_forward_fast(A_prev, W, b, self.stride, self.pad)
        activation, self.relu_cache = nn_util.relu(conv_output)

        output, self.batchnorm_cache, self.running_mean, self.running_var = conv_util.conv_batchnorm_forward(
            activation, gamma, beta, self.running_mean, self.running_var, train=train
        )

        return output

    def backward(self, dA_prev):
        dA_prev, dgamma, dbeta = conv_util.conv_batchnorm_backward(dA_prev, self.batchnorm_cache)
        dA_prev = nn_util.relu_backward(dA_prev, self.relu_cache)
        dA_prev, dW, db = conv_util.conv_backward_fast(dA_prev, self.conv_cache)
        assert self.W.shape == dW.shape
        assert self.b.shape == db.shape
        return dA_prev, dW, db, dgamma, dbeta


class Maxpool(object):

    def __init__(self, stride, pool_size):
        self.stride = stride
        self.pool_size = pool_size
        self.pool_cache = None
        self.cache_test = None

    def forward(self, A_prev):
        output, self.pool_cache = conv_util.max_pool_forward_fast(A_prev, self.stride, self.pool_size)
        return output

    def backward(self, dA_prev):
        return conv_util.max_pool_backward_fast(dA_prev, self.pool_cache)


class FullyConnected(object):

    def __init__(self, dropout_rate):
        self.dropout_rate = dropout_rate
        self.W = None
        self.b = None
        self.running_mean = None
        self.running_var = None
        self.fc_cache = None
        self.relu_cache = None
        self.batchnorm_cache = None
        self.dropout_cache = None

    def forward(self, A_prev, W, b, gamma, beta, train):

        self.W, self.b = W, b

        if self.running_mean is None or self.running_var is None:
            self.running_mean = np.zeros(gamma.shape)
            self.running_var = np.zeros(gamma.shape)

        fc_output, self.fc_cache = nn_util.fc_forward(A_prev, W, b)
        activation, self.relu_cache = nn_util.relu(fc_output)

        output, self.batchnorm_cache, self.running_mean, self.running_var = nn_util.batchnorm_forward(
            activation, gamma, beta, self.running_mean, self.running_var, train=train
        )

        assert fc_output.shape == output.shape, \
               'fc shape: %s, batchnorm shape: %s' % (fc_output.shape, output.shape)

        output, self.dropout_cache = nn_util.dropout_forward(output, self.dropout_rate)
        return output

    def backward(self, dA_prev):
        dA_prev = nn_util.dropout_backward(dA_prev, self.dropout_cache)
        dA_prev, dgamma, dbeta = nn_util.batchnorm_backward(dA_prev, self.batchnorm_cache)
        dA_prev = nn_util.relu_backward(dA_prev, self.relu_cache)
        dA, dW, db = nn_util.fc_backward(dA_prev, self.fc_cache)
        assert self.W.shape == dW.shape
        assert self.b.shape == db.shape
        return dA, dW, db, dgamma, dbeta


class Softmax(object):

    def __init__(self):
        self.W = None
        self.b = None
        self.fc_cache = None
        self.forward_output = None

    def forward(self, A_prev, W, b):
        self.W, self.b = W, b
        fc_output, fc_cache = nn_util.fc_forward(A_prev, W, b)
        self.forward_output = nn_util.softmax(fc_output)
        self.fc_cache = fc_cache
        return self.forward_output

    def backward(self, Y):
        dout = nn_util.softmax_cross_entropy(self.forward_output, Y)
        dA, dW, db = nn_util.fc_backward(dout, self.fc_cache)
        assert self.W.shape == dW.shape
        assert self.b.shape == db.shape
        return dA, dW, db
