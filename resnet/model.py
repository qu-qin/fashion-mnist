from keras.models import Model
from keras import backend as K
from keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, AveragePooling2D, Flatten, Dense
from utils import mnist_reader
from vanilla import nn_util
from resnet import res_util


_IMG_WIDTH, _IMG_HEIGHT, _IMG_CHANNELS = 28, 28, 1
_EPOCHS = 180
_BATCH_SIZE = 64


def res18(input_shape, classes):

    X_input = Input(input_shape)

    # First stage
    X = Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same')(X_input)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(X)

    layer_blocks = [2, 2, 2, 2]
    filters = 64

    # Basic blocks
    for i, blocks in enumerate(layer_blocks):
        is_first_layer = i == 0
        for j in range(blocks):
            is_first_block = j == 0
            X = res_util.basic_block(X, filters, is_first_layer, is_first_block)
        filters *= 2

    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    # Last stage
    X_shape = K.int_shape(X)
    X = AveragePooling2D(pool_size=(X_shape[1], X_shape[2]), strides=(1, 1))(X)
    X = Flatten()(X)
    X = Dense(classes, activation='softmax')(X)

    # Create model
    return Model(inputs=X_input, outputs=X)


if __name__ == '__main__':

    # load data
    X_train, Y_train = mnist_reader.load_mnist('data/fashion', kind='train')
    X_test, Y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

    # one hot
    Y_train = nn_util.one_hot(Y_train, 10)
    Y_test = nn_util.one_hot(Y_test, 10)

    # reshape X to (m, width, height, channels)
    X_train = X_train.reshape(X_train.shape[0], _IMG_WIDTH, _IMG_HEIGHT, _IMG_CHANNELS)
    X_test = X_test.reshape(X_test.shape[0], _IMG_WIDTH, _IMG_HEIGHT, _IMG_CHANNELS)

    # normalization
    X_train = X_train / 255
    X_test = X_test / 255

    model = res18(input_shape=(_IMG_WIDTH, _IMG_HEIGHT, _IMG_CHANNELS), classes=10)
    print model.summary()

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=_EPOCHS, batch_size=_BATCH_SIZE)

    preds = model.evaluate(X_test, Y_test)
    print 'Loss: %s' % preds[0]
    print 'Accuracy: %s' % preds[1]
