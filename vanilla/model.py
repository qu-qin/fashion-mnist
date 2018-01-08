import time
import numpy as np

from utils import mnist_reader
from vanilla import nn_util
from vanilla.layers import Conv2d, Maxpool, FullyConnected, Softmax


_IMG_WIDTH, _IMG_HEIGHT, _IMG_CHANNELS = 28, 28, 1
_EPOCHS = 200
_BATCH_SIZE = 64
_DROP_OUT_RATE = 0.35
_LEARNING_RATE = 0.001
_ADAM_BETA1 = 0.9
_ADAM_BETA2 = 0.999
_RMSPROP_DECAY_RATE = 0.99


def forward_propagation(X, layers, params, train):

    # conv2d_1, maxpool_2, conv2d_3, maxpool_4, fc_5, softmax_6 = layers
    # a1 = conv2d_1.forward(X, params['W1'], params['b1'], params['gamma1'], params['beta1'], train)
    # a2 = maxpool_2.forward(a1)
    # a3 = conv2d_3.forward(a2, params['W3'], params['b3'], params['gamma3'], params['beta3'], train)
    # a4 = maxpool_4.forward(a3)
    # a4 = nn_util.flatten(a4)
    # a5 = fc_5.forward(a4, params['W5'], params['b5'], params['gamma5'], params['beta5'], train)
    # return softmax_6.forward(a5, params['W6'], params['b6'])

    conv2d_1, maxpool_2, conv2d_3, maxpool_4, fc_5, fc_6, softmax_7 = layers
    a1 = conv2d_1.forward(X, params['W1'], params['b1'], params['gamma1'], params['beta1'], train)
    a2 = maxpool_2.forward(a1)
    a3 = conv2d_3.forward(a2, params['W3'], params['b3'], params['gamma3'], params['beta3'], train)
    a4 = maxpool_4.forward(a3)
    a4 = nn_util.flatten(a4)
    a5 = fc_5.forward(a4, params['W5'], params['b5'], params['gamma5'], params['beta5'], train)
    a6 = fc_6.forward(a5, params['W6'], params['b6'], params['gamma6'], params['beta6'], train)
    return softmax_7.forward(a6, params['W7'], params['b7'])


def backward_propagation(layers, X, Y):

    m = X.shape[0]
    grads = {}

    # conv2d_1, maxpool_2, conv2d_3, maxpool_4, fc_5, softmax_6 = layers

    # dA5, grads['dW6'], grads['db6'] = softmax_6.backward(Y)
    # dA4, grads['dW5'], grads['db5'], grads['dgamma5'], grads['dbeta5'] = fc_5.backward(dA5)
    # dA4 = dA4.reshape((m, 4, 4, 64))
    # dA3 = maxpool_4.backward(dA4)
    # dA2, grads['dW3'], grads['db3'], grads['dgamma3'], grads['dbeta3'] = conv2d_3.backward(dA3)
    # dA1 = maxpool_2.backward(dA2)
    # _, grads['dW1'], grads['db1'], grads['dgamma1'], grads['dbeta1'] = conv2d_1.backward(dA1)

    conv2d_1, maxpool_2, conv2d_3, maxpool_4, fc_5, fc_6, softmax_7 = layers

    dA6, grads['dW7'], grads['db7'] = softmax_7.backward(Y)
    dA5, grads['dW6'], grads['db6'], grads['dgamma6'], grads['dbeta6'] = fc_6.backward(dA6)
    dA4, grads['dW5'], grads['db5'], grads['dgamma5'], grads['dbeta5'] = fc_5.backward(dA5)
    dA4 = dA4.reshape((m, 7, 7, 64))
    dA3 = maxpool_4.backward(dA4)
    dA2, grads['dW3'], grads['db3'], grads['dgamma3'], grads['dbeta3'] = conv2d_3.backward(dA3)
    dA1 = maxpool_2.backward(dA2)
    _, grads['dW1'], grads['db1'], grads['dgamma1'], grads['dbeta1'] = conv2d_1.backward(dA1)

    return grads


def predict(X, Y, layers, params):

    batch_accuracy = []
    # split into mini batches to avoid memory error
    mini_batches = nn_util.rand_mini_batches(X, Y, _BATCH_SIZE)

    for mini_batch_X, mini_batch_Y in mini_batches:
        scores = forward_propagation(mini_batch_X, layers, params, train=False)
        batch_accuracy.append(calc_accuracy(scores, mini_batch_Y))

    accuracy = np.average(batch_accuracy)

    print 'training accuracy: %.2f' % accuracy
    with open('accuracy.csv', 'a') as result:
        result.write('%.2f\n' % accuracy)


def calc_accuracy(scores, Y):
    m = scores.shape[0]
    predicted_class = np.argmax(scores, axis=1).reshape(m, 1)
    return np.mean(predicted_class == Y)


def model(X_train, Y_train, X_test, Y_test, layers):

    # layer dimensions
    layers_dims = [

        # {'W': (5, 5, 1, 32), 'b': (1, 1, 1, 32), 'gamma': (1, 32), 'beta': (1, 32)},
        # {'W': None, 'b': None},
        # {'W': (5, 5, 32, 64), 'b': (1, 1, 1, 64), 'gamma': (1, 64), 'beta': (1, 64)},
        # {'W': None, 'b': None},
        # {'W': (1024, 64), 'b': (1, 64), 'gamma': (1, 64), 'beta': (1, 64)},
        # {'W': (64, 10), 'b': (1, 10)}

        {'W': (3, 3, 1, 64), 'b': (1, 1, 1, 64), 'gamma': (1, 64), 'beta': (1, 64)},
        {'W': None, 'b': None},
        {'W': (3, 3, 64, 64), 'b': (1, 1, 1, 64), 'gamma': (1, 64), 'beta': (1, 64)},
        {'W': None, 'b': None},
        {'W': (3136, 128), 'b': (1, 128), 'gamma': (1, 128), 'beta': (1, 128)},
        {'W': (128, 64), 'b': (1, 64), 'gamma': (1, 64), 'beta': (1, 64)},
        {'W': (64, 10), 'b': (1, 10)}

    ]

    params = nn_util.init_params(layers_dims)
    g, sg = nn_util.init_adam(layers_dims)
    adam_count = 0

    rmsprop_cache = nn_util.init_rmsprop(layers_dims)

    for i in xrange(_EPOCHS):

        mini_batches = nn_util.rand_mini_batches(X_train, Y_train, _BATCH_SIZE)

        for mini_batch_X, mini_batch_Y in mini_batches:

            start = time.time()

            activation = forward_propagation(mini_batch_X, layers, params, train=True)

            mini_batch_accuracy = calc_accuracy(activation, mini_batch_Y)

            cost = nn_util.compute_cost(activation, mini_batch_Y)
            grads = backward_propagation(layers, mini_batch_X, mini_batch_Y)

            adam_count += 1
            params, g, sg = nn_util.update_adam(layers_dims, params, grads, g, sg, adam_count, _LEARNING_RATE, _ADAM_BETA1, _ADAM_BETA2)
            # params, rmsprop_cache = nn_util.update_rmsprop(layers_dims, params, grads, rmsprop_cache, _LEARNING_RATE, _RMSPROP_DECAY_RATE)

            end = time.time()
            with open('minibatch.csv', 'a') as result:
                result.write('%f\n' % cost)

            print 'Minibatch cost %f, finished in %.2f seconds, accuracy %s' % (cost, end - start, '{0:.2f}%'.format(mini_batch_accuracy * 100))

        print 'Cost after epoch %i: %f' % (i, cost)
        with open('output.csv', 'a') as result:
            result.write('%i,%f\n' % (i, cost))

        yield params


def setup_layers():
    return (
        # Conv2d(1, 0),
        # Maxpool(2, 2),
        # Conv2d(1, 0),
        # Maxpool(2, 2),
        # FullyConnected(_DROP_OUT_RATE),
        # Softmax()

        Conv2d(1, 1),
        Maxpool(2, 2),
        Conv2d(1, 1),
        Maxpool(2, 2),
        FullyConnected(_DROP_OUT_RATE),
        FullyConnected(_DROP_OUT_RATE),
        Softmax()
    )


if __name__ == '__main__':

    # load data
    X_train, Y_train = mnist_reader.load_mnist('data/fashion', kind='train')
    X_test, Y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

    # reshape X to (m, width, height, channels)
    X_train = X_train.reshape(X_train.shape[0], _IMG_WIDTH, _IMG_HEIGHT, _IMG_CHANNELS)
    X_test = X_test.reshape(X_test.shape[0], _IMG_WIDTH, _IMG_HEIGHT, _IMG_CHANNELS)

    # reshape Y to (m, 1)
    Y_train = Y_train.reshape(Y_train.shape[0], 1)
    Y_test = Y_test.reshape(Y_test.shape[0], 1)

    # one hot
    # Y_train = nn_util.one_hot(Y_train, 10)

    # normalization
    X_train = X_train / 255
    X_test = X_test / 255

    # DEBUG MODE
    # X_train, Y_train = X_train[:10, ], Y_train[:10, ]
    # X_test, Y_test = X_test[:100, ], Y_test[:100, ]

    nn_layers = setup_layers()

    for trained_params in model(X_train, Y_train, X_test, Y_test, nn_layers):
        predict(X_test, Y_test, nn_layers, trained_params)
