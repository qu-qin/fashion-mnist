import numpy as np

from vanilla import nn_util
from utils import im2col


def zero_padding(arr, padding):
    return np.pad(array=arr,
                  pad_width=((0, 0), (padding, padding), (padding, padding), (0, 0)),
                  mode='constant')


def conv_slice(a_slice, W, b):
    """
    Apply filter on a slice of the matrix.
    @param a_slice: slice of the matrix, shape (f, f, n_C)
    @param W: shape (f, f, n_C)
    @param b: shape (1, 1, 1)
    @return: result of convolving the sliding window on a slice of the matrix
    """
    s = np.multiply(a_slice, W) + b
    return np.sum(s)


def conv_forward(A_prev, W, b, stride, pad):
    """
    @param A_prev: output of activation from the previous layer, shape (m, n_H_prev, n_W_prev, n_C_prev)
    @param W: shape (f, f, n_C_prev, n_C)
    @param b: shape (1, 1, 1, n_C)
    @param stride
    @param pad
    @return: conv output, shape (m, n_H, n_W, n_C)
             cache for the back propagation
    """
    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    f, f, n_C_prev, n_C = W.shape

    # compute the dimensions of the conv output
    n_H = int((n_H_prev - f + 2 * pad) / stride ) + 1
    n_W = int((n_W_prev - f + 2 * pad) / stride ) + 1

    # initialize the output
    output = np.zeros((m, n_H, n_W, n_C))

    A_prev_pad = zero_padding(A_prev, pad)

    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):

                    # locate the current slice
                    y1, y2, x1, x2 = _locate_slice(h, w, f, stride)
                    a_slice_prev = a_prev_pad[y1:y2, x1:x2, :]
                    output[i, h, w, c] = conv_slice(a_slice_prev, W[:, :, :, c], b[:, :, :, c])

    cache = (A_prev, W, b, stride, pad)
    return output, cache


def conv_forward_fast(A_prev, W, b, stride, pad):
    """
    im2col implementation of the conv forward, to speed up the training
    """
    A_prev = A_prev.transpose(0, 3, 1, 2)
    W = W.transpose(3, 2, 0, 1)
    b = b.transpose(3, 2, 0, 1)

    m, n_C, n_H, n_W = A_prev.shape
    num_filters, _, filter_height, filter_width = W.shape

    assert (n_W + 2 * pad - filter_width) % stride == 0, 'width does not work'
    assert (n_H + 2 * pad - filter_height) % stride == 0, 'height does not work'

    out_height = (n_H + 2 * pad - filter_height) / stride + 1
    out_width = (n_W + 2 * pad - filter_width) / stride + 1
    out = np.zeros((m, num_filters, out_height, out_width), dtype=A_prev.dtype)

    x_cols = im2col.im2col_indices(A_prev, W.shape[2], W.shape[3], pad, stride)
    res = W.reshape((W.shape[0], -1)).dot(x_cols) + b.reshape(-1, 1)

    out = res.reshape(W.shape[0], out.shape[2], out.shape[3], A_prev.shape[0])
    out = out.transpose(3, 1, 2, 0)

    cache = (A_prev, W, b, stride, pad, x_cols)
    return out, cache


def conv_backward(dZ, cache):
    """
    @param dZ: gradient of the cost to the output of the conv layer, shape (m, n_H, n_W, n_C)
    @param cache: cached from the forward propagation
    @return: dA_prev gradient of the cost to the input of the conv layer, shape (m, n_H_prev, n_W_prev, n_C_prev)
             dW weight gradient of the conv layer, shape (f, f, n_C_prev, n_C)
             db biases gradient of the conv layer, shape (1, 1, 1, n_C)
    """
    A_prev, W, b, stride, pad = cache

    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    f, f, n_C_prev, n_C = W.shape
    m, n_H, n_W, n_C = dZ.shape

    # initialize dA_prev, dW, db
    dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))
    dW = np.zeros((f, f, n_C_prev, n_C))
    db = np.zeros((1, 1, 1, n_C))

    # add padding
    A_prev_pad = zero_padding(A_prev, pad)
    dA_prev_pad = zero_padding(dA_prev, pad)

    for i in range(m):

        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]

        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):

                    # locate the current slice
                    y1, y2, x1, x2 = _locate_slice(h, w, f, stride)
                    a_slice = a_prev_pad[y1:y2, x1:x2, :]

                    # update gradients for the slice and the filter
                    da_prev_pad[y1:y2, x1:x2, :] += W[:, :, :, c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]
                    db[:, :, :, c] += dZ[i, h, w, c]

        if pad == 0:
            dA_prev[i, :, :, :] = da_prev_pad
        else:
            dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]

    return dA_prev, dW, db


def conv_backward_fast(dout, cache):
    """
    im2col implementation of the conv backward, to speed up the training
    """
    dout = dout.transpose(0, 3, 1, 2)
    A_prev, W, b, stride, pad, x_cols = cache

    db = np.sum(dout, axis=(0, 2, 3))

    num_filters, _, filter_height, filter_width = W.shape
    dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(num_filters, -1)
    dw = dout_reshaped.dot(x_cols.T).reshape(W.shape)

    dx_cols = W.reshape(num_filters, -1).T.dot(dout_reshaped)
    dx = im2col.col2im_indices(dx_cols, A_prev.shape, filter_height, filter_width, pad, stride)

    dx = dx.transpose(0, 2, 3, 1)
    dw = dw.transpose(2, 3, 1, 0)
    db = db.reshape((1, 1, 1, db.shape[0]))

    return dx, dw, db


def max_pool_forward(A_prev, stride, f):
    """
    @param A_prev: output of activation from the previous layer, shape (m, n_H_prev, n_W_prev, n_C_prev)
    @param stride
    @param f
    @return: max pool output, shape (m, n_H, n_W, n_C)
             cache for the back propagation
    """
    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape

    # Compute the dimensions of the max pool output
    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    n_C = n_C_prev

    # Initialize the output
    output = np.zeros((m, n_H, n_W, n_C))

    for i in range(m):
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):

                    # locate the current slice
                    y1, y2, x1, x2 = _locate_slice(h, w, f, stride)
                    a_slice_prev = A_prev[i][y1:y2, x1:x2, c]
                    output[i, h, w, c] = np.max(a_slice_prev)

    cache = (A_prev, stride, f)
    return output, cache


def max_pool_forward_fast(A_prev, stride, f):

    A_prev = A_prev.transpose(0, 3, 1, 2)

    m, n_C, n_H, n_W = A_prev.shape
    h_out = (n_H - f) / stride + 1
    w_out = (n_W - f) / stride + 1

    A_prev_reshaped = A_prev.reshape(m * n_C, 1, n_H, n_W)
    x_col = im2col.im2col_indices(A_prev_reshaped, f, f, padding=0, stride=stride)

    max_idx = np.argmax(x_col, axis=0)
    out = x_col[max_idx, range(max_idx.size)]

    out = out.reshape(h_out, w_out, m, n_C)
    out = out.transpose(2, 0, 1, 3)

    pool_cache = max_idx
    cache = (A_prev, f, stride, x_col, pool_cache)

    return out, cache


def max_pool_backward(dA, cache):
    """
    @param dA: gradient of cost to the output of the max pool layer, shape (m, n_H_prev, n_W_prev, n_C_prev)
    @param cache: cached from the forward propagation
    @return: dA_prev gradient of the cost to the input of the max pool layer, shape (m, n_H_prev, n_W_prev, n_C_prev)
    """
    A_prev, stride, f = cache

    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    m, n_H, n_W, n_C = dA.shape

    # initialize dA_prev
    dA_prev = np.zeros(A_prev.shape)

    for i in range(m):
        a_prev = A_prev[i]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):

                    # locate the current slice
                    y1, y2, x1, x2 = _locate_slice(h, w, f, stride)
                    a_prev_slice = a_prev[y1:y2, x1:x2, c]

                    # create a mask from the slice, to identify the max entry
                    mask = a_prev_slice == np.max(a_prev_slice)
                    # update gradients for the slice
                    dA_prev[i, y1: y2, x1: x2, c] += np.multiply(mask, dA[i, h, w, c])

    return dA_prev


def max_pool_backward_fast(dout, cache):

    dout = dout.transpose(0, 3, 1, 2)

    A_prev, f, stride, x_col, pool_cache = cache
    m, n_C, n_H, n_W = A_prev.shape

    dx_col = np.zeros(x_col.shape)
    dout_col = dout.transpose(2, 3, 0, 1).ravel()

    dx_col[pool_cache, range(pool_cache.size)] = dout_col
    dx = im2col.col2im_indices(dx_col, (m * n_C, 1, n_H, n_W), f, f, padding=0, stride=stride)
    dx = dx.reshape(A_prev.shape)
    dx = dx.transpose(0, 2, 3, 1)

    return dx


def conv_batchnorm_forward(X, gamma, beta, running_mean, running_var, momentum=0.9, train=True):
    m, H, W, C = X.shape
    X_reshaped = X.reshape(-1, C)
    output, cache, running_mean, running_var = nn_util.batchnorm_forward(X_reshaped, gamma, beta, running_mean, running_var, momentum, train)
    output = output.reshape(m, H, W, C)
    return output, cache, running_mean, running_var


def conv_batchnorm_backward(dx, cache):
    m, H, W, C = dx.shape
    dx_reshaped = dx.reshape(-1, C)
    dx, dgamma, dbeta = nn_util.batchnorm_backward(dx_reshaped, cache)
    dx = dx.reshape(m, H, W, C)
    return dx, dgamma, dbeta


def _locate_slice(h, w, f, stride):
    """
    @return: y1, y2, x1, x2
    """
    return h * stride, h * stride + f, w * stride, w * stride + f
