import numpy as np


def relu(X):
    output = np.maximum(0, X)
    cache = X
    return output, cache


def relu_backward(dout, cache):
    dX = dout.copy()
    dX[cache <= 0] = 0
    return dX


def fc_forward(X, W, b):
    output = np.dot(X, W) + b
    cache = X, W
    return output, cache


def fc_backward(dout, cache):
    X, W = cache
    dW = np.dot(X.T, dout)
    db = np.sum(dout, axis=0, keepdims=True)
    dX = np.dot(dout, W.T)
    return dX, dW, db


def dropout_forward(X, dropout_rate):
    keep_rate = 1 - dropout_rate
    dropout_units = np.random.binomial(1, keep_rate, size=X.shape) / keep_rate
    output = X * dropout_units
    cache = dropout_units
    return output, cache


def dropout_backward(dout, cache):
    dX = dout * cache
    return dX


def one_hot(Y, C):
    return np.eye(C)[Y.reshape(-1)]


def init_params(layer_dims):
    """
    he normal initializer
    @param layer_dims: list of dimensions of each layer
    @return: dictionary of initialised params, W1...Wl, b1...bl
    """
    params = {}

    for idx, shapes in enumerate(layer_dims):

        w_shape, b_shape = shapes.get('W'), shapes.get('b')
        gamma_shape, beta_shape = shapes.get('gamma'), shapes.get('beta')

        w_key = 'W%s' % (idx + 1)
        b_key = 'b%s' % (idx + 1)
        gamma_key = 'gamma%s' % (idx + 1)
        beta_key = 'beta%s' % (idx + 1)

        if w_shape:
            fan_in = w_shape[0] if len(w_shape) == 2 else w_shape[-2] * np.prod(w_shape[:-2])
            scale = 2.0 / max(1.0, fan_in)
            stddev = np.sqrt(scale)
            rnd_normal = np.random.normal(loc=0.0, scale=scale, size=w_shape)
            params[w_key] = np.clip(rnd_normal, -(2 * stddev), 2 * stddev)
        else:
            params[w_key] = np.empty(0)

        if b_shape:
            params[b_key] = np.zeros(b_shape)
        else:
            params[b_key] = np.empty(0)

        if gamma_shape:
            params[gamma_key] = np.ones(gamma_shape)
        else:
            params[gamma_key] = np.empty(0)

        if beta_shape:
            params[beta_key] = np.zeros(beta_shape)
        else:
            params[beta_key] = np.empty(0)

    return params


def init_adam(layers_dims):
    """
    Initialise the exponentially moving average of the gradient and the squared gradient.
    @param layers_dims: list of dimensions of each layer
    @return: the exponentially moving average of the gradient and the squared gradient.
    """

    g = {}
    sg = {}

    for l in range(len(layers_dims)):

        dw_key = 'dW%s' % (l + 1)
        db_key = 'db%s' % (l + 1)
        dgamma_key = 'dgamma%s' % (l + 1)
        dbeta_key = 'dbeta%s' % (l + 1)

        layer_dims = layers_dims[l]

        g[dw_key] = np.zeros((layer_dims.get('W')))
        g[db_key] = np.zeros((layer_dims.get('b')))
        g[dgamma_key] = np.zeros((layer_dims.get('gamma')))
        g[dbeta_key] = np.zeros((layer_dims.get('beta')))

        sg[dw_key] = np.zeros((layer_dims.get('W')))
        sg[db_key] = np.zeros((layer_dims.get('b')))
        sg[dgamma_key] = np.zeros((layer_dims.get('gamma')))
        sg[dbeta_key] = np.zeros((layer_dims.get('beta')))

    return g, sg


def init_rmsprop(layers_dims):

    rmsprop_cache = {}

    for l in range(len(layers_dims)):
        dw_key = 'dW%s' % (l + 1)
        db_key = 'db%s' % (l + 1)
        dgamma_key = 'dgamma%s' % (l + 1)
        dbeta_key = 'dbeta%s' % (l + 1)

        layer_dims = layers_dims[l]

        rmsprop_cache[dw_key] = np.zeros((layer_dims.get('W')))
        rmsprop_cache[db_key] = np.zeros((layer_dims.get('b')))
        rmsprop_cache[dgamma_key] = np.zeros((layer_dims.get('gamma')))
        rmsprop_cache[dbeta_key] = np.zeros((layer_dims.get('beta')))

    return rmsprop_cache


def update_adam(layers_dims, params, gradients, g, sg, counter, learning_rate, beta1, beta2):
    """
    Update params with adam
    @param layers_dims: list of dimensions of each layer
    @param params: dictionary of initialised params, W1...Wl, b1...bl
    @param grads: gradients for params
    @param g: exponential moving average of the gradient
    @param sg: exponential moving average of the squared gradient
    @param counter: adam counter
    @param learning_rate
    @param beta1: exponential decay for the first moment estimates
    @param beta2: exponential decay for the second moment estimates
    @return: params, updated g and sg
    """

    for l in range(len(layers_dims)):

        dw_key = 'dW%s' % (l + 1)
        db_key = 'db%s' % (l + 1)
        dgamma_key = 'dgamma%s' % (l + 1)
        dbeta_key = 'dbeta%s' % (l + 1)

        w_key = 'W%s' % (l + 1)
        b_key = 'b%s' % (l + 1)
        gamma_key = 'gamma%s' % (l + 1)
        beta_key = 'beta%s' % (l + 1)

        if dw_key in gradients:
            params[w_key] -= _adam_param_offset(dw_key, gradients, g, sg, counter, learning_rate, beta1, beta2)

        if db_key in gradients:
            params[b_key] -= _adam_param_offset(db_key, gradients, g, sg, counter, learning_rate, beta1, beta2)

        if dgamma_key in gradients:
            params[gamma_key] -= _adam_param_offset(dgamma_key, gradients, g, sg, counter, learning_rate, beta1, beta2)

        if dbeta_key in gradients:
            params[beta_key] -= _adam_param_offset(dbeta_key, gradients, g, sg, counter, learning_rate, beta1, beta2)

    return params, g, sg


def update_rmsprop(layers_dims, params, gradients, cache, learning_rate, decay_rate):

    for l in range(len(layers_dims)):

        dw_key = 'dW%s' % (l + 1)
        db_key = 'db%s' % (l + 1)
        dgamma_key = 'dgamma%s' % (l + 1)
        dbeta_key = 'dbeta%s' % (l + 1)

        w_key = 'W%s' % (l + 1)
        b_key = 'b%s' % (l + 1)
        gamma_key = 'gamma%s' % (l + 1)
        beta_key = 'beta%s' % (l + 1)

        if dw_key in gradients:
            params[w_key] -= _rmsprop_param_offset(dw_key, gradients, cache, learning_rate, decay_rate)

        if db_key in gradients:
            params[b_key] -= _rmsprop_param_offset(db_key, gradients, cache, learning_rate, decay_rate)

        if dgamma_key in gradients:
            params[gamma_key] -= _rmsprop_param_offset(dgamma_key, gradients, cache, learning_rate, decay_rate)

        if dbeta_key in gradients:
            params[beta_key] -= _rmsprop_param_offset(dbeta_key, gradients, cache, learning_rate, decay_rate)

    return params, cache


def batchnorm_forward(X, gamma, beta, running_mean, running_var, momentum=0.9, train=True):

    out, cache = None, None

    if train:

        mu = np.mean(X, axis=0)
        var = np.var(X, axis=0)

        X_norm = (X - mu) / np.sqrt(var + 1e-8)
        out = gamma * X_norm + beta

        cache = (X, X_norm, mu, var, gamma, beta)

        running_mean = momentum * running_mean + (1.0 - momentum) * mu
        running_var = momentum * running_var + (1.0 - momentum) * var

    else:

        X_norm = (X - running_mean) / np.sqrt(running_var + 1e-8)
        out = gamma * X_norm + beta

    return out, cache, running_mean, running_var


def batchnorm_backward(dx, cache):

    X, X_norm, mu, var, gamma, _ = cache

    D, N = X.shape

    X_mu = X - mu
    std_inv = 1.0 / np.sqrt(var + 1e-8)

    dX_norm = dx * gamma
    dvar = np.sum(dX_norm * X_mu, axis=0, keepdims=True) * -0.5 * std_inv ** 3
    dmu = np.sum(dX_norm * -std_inv, axis=0, keepdims=True) + dvar * np.mean(-2.0 * X_mu, axis=0)

    dX = (dX_norm * std_inv) + (dvar * 2 * X_mu / N) + (dmu / N)
    dgamma = np.sum(dx * X_norm, axis=0, keepdims=True)
    dbeta = np.sum(dx, axis=0, keepdims=True)

    return dX, dgamma, dbeta


def rand_mini_batches(X, Y, batch_size=64):
    """
    Split (X, Y) to mini batches, randomly
    @param X: input data, shape (m, input size)
    @param Y: labelled results (m, 1)
    @param batch_size: size of each batch
    @return: list of mini batches, format [(batch_X, batch_Y)]
    """
    m = X.shape[0]
    mini_batches = []

    # shuffle
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :]
    shuffled_Y = Y[permutation, :]

    # split into chunks
    for i in xrange(0, m, batch_size):
        mini_batch_X = shuffled_X[i: i + batch_size, :]
        mini_batch_Y = shuffled_Y[i: i + batch_size, :]
        yield (mini_batch_X, mini_batch_Y)


def flatten(X):
    return X.reshape(X.shape[0], -1)


def softmax(X):
    # ensure numeric stability to avoid nan
    shifted_X = X - np.amax(X, axis=1, keepdims=True)
    exp_scores = np.exp(shifted_X)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def softmax_cross_entropy(probs, Y):
    m = Y.shape[0]
    m_range = np.arange(m)
    m_range = m_range.reshape((m_range.shape[0], 1))
    dX = probs
    dX[m_range, Y.astype(int)] -= 1
    dX /= m
    return dX


def compute_cost(probs, Y):
    m = Y.shape[0]
    m_range = np.arange(m)
    m_range = m_range.reshape((m_range.shape[0], 1))
    correct_logprobs = -np.log(probs[m_range, Y.astype(int)])
    loss = np.sum(correct_logprobs) / m
    return loss


def _adam_param_offset(dx_key, grads, g, sg, counter, learning_rate, beta1, beta2):

    g[dx_key] = beta1 * g[dx_key] + (1 - beta1) * grads[dx_key]
    sg[dx_key] = beta2 * sg[dx_key] + (1 - beta2) * np.power(grads[dx_key], 2)

    g_param = g[dx_key] / (1 - np.power(beta1, counter))
    sg_param = sg[dx_key] / (1 - np.power(beta2, counter))

    return learning_rate * g_param / (np.sqrt(sg_param) + 1e-8)


def _rmsprop_param_offset(dx_key, grads, cache, learning_rate, decay_rate):
    cache[dx_key] = decay_rate * cache[dx_key] + (1 - decay_rate) * grads[dx_key] ** 2
    return learning_rate * grads[dx_key] / (np.sqrt(cache[dx_key]) + 1e-8)
