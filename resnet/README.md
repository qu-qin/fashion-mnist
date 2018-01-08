__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input_1 (InputLayer)            (None, 28, 28, 1)    0
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 14, 14, 64)   3200        input_1[0][0]
__________________________________________________________________________________________________
batch_normalization_1 (BatchNor (None, 14, 14, 64)   256         conv2d_1[0][0]
__________________________________________________________________________________________________
activation_1 (Activation)       (None, 14, 14, 64)   0           batch_normalization_1[0][0]
__________________________________________________________________________________________________
max_pooling2d_1 (MaxPooling2D)  (None, 7, 7, 64)     0           activation_1[0][0]
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 7, 7, 64)     36928       max_pooling2d_1[0][0]
__________________________________________________________________________________________________
batch_normalization_2 (BatchNor (None, 7, 7, 64)     256         conv2d_2[0][0]
__________________________________________________________________________________________________
activation_2 (Activation)       (None, 7, 7, 64)     0           batch_normalization_2[0][0]
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 7, 7, 64)     36928       activation_2[0][0]
__________________________________________________________________________________________________
add_1 (Add)                     (None, 7, 7, 64)     0           max_pooling2d_1[0][0]
                                                                 conv2d_3[0][0]
__________________________________________________________________________________________________
batch_normalization_3 (BatchNor (None, 7, 7, 64)     256         add_1[0][0]
__________________________________________________________________________________________________
activation_3 (Activation)       (None, 7, 7, 64)     0           batch_normalization_3[0][0]
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, 7, 7, 64)     36928       activation_3[0][0]
__________________________________________________________________________________________________
batch_normalization_4 (BatchNor (None, 7, 7, 64)     256         conv2d_4[0][0]
__________________________________________________________________________________________________
activation_4 (Activation)       (None, 7, 7, 64)     0           batch_normalization_4[0][0]
__________________________________________________________________________________________________
conv2d_5 (Conv2D)               (None, 7, 7, 64)     36928       activation_4[0][0]
__________________________________________________________________________________________________
add_2 (Add)                     (None, 7, 7, 64)     0           add_1[0][0]
                                                                 conv2d_5[0][0]
__________________________________________________________________________________________________
batch_normalization_5 (BatchNor (None, 7, 7, 64)     256         add_2[0][0]
__________________________________________________________________________________________________
activation_5 (Activation)       (None, 7, 7, 64)     0           batch_normalization_5[0][0]
__________________________________________________________________________________________________
conv2d_6 (Conv2D)               (None, 4, 4, 128)    73856       activation_5[0][0]
__________________________________________________________________________________________________
batch_normalization_6 (BatchNor (None, 4, 4, 128)    512         conv2d_6[0][0]
__________________________________________________________________________________________________
activation_6 (Activation)       (None, 4, 4, 128)    0           batch_normalization_6[0][0]
__________________________________________________________________________________________________
conv2d_8 (Conv2D)               (None, 4, 4, 128)    8320        add_2[0][0]
__________________________________________________________________________________________________
conv2d_7 (Conv2D)               (None, 4, 4, 128)    147584      activation_6[0][0]
__________________________________________________________________________________________________
add_3 (Add)                     (None, 4, 4, 128)    0           conv2d_8[0][0]
                                                                 conv2d_7[0][0]
__________________________________________________________________________________________________
batch_normalization_7 (BatchNor (None, 4, 4, 128)    512         add_3[0][0]
__________________________________________________________________________________________________
activation_7 (Activation)       (None, 4, 4, 128)    0           batch_normalization_7[0][0]
__________________________________________________________________________________________________
conv2d_9 (Conv2D)               (None, 4, 4, 128)    147584      activation_7[0][0]
__________________________________________________________________________________________________
batch_normalization_8 (BatchNor (None, 4, 4, 128)    512         conv2d_9[0][0]
__________________________________________________________________________________________________
activation_8 (Activation)       (None, 4, 4, 128)    0           batch_normalization_8[0][0]
__________________________________________________________________________________________________
conv2d_10 (Conv2D)              (None, 4, 4, 128)    147584      activation_8[0][0]
__________________________________________________________________________________________________
add_4 (Add)                     (None, 4, 4, 128)    0           add_3[0][0]
                                                                 conv2d_10[0][0]
__________________________________________________________________________________________________
batch_normalization_9 (BatchNor (None, 4, 4, 128)    512         add_4[0][0]
__________________________________________________________________________________________________
activation_9 (Activation)       (None, 4, 4, 128)    0           batch_normalization_9[0][0]
__________________________________________________________________________________________________
conv2d_11 (Conv2D)              (None, 2, 2, 256)    295168      activation_9[0][0]
__________________________________________________________________________________________________
batch_normalization_10 (BatchNo (None, 2, 2, 256)    1024        conv2d_11[0][0]
__________________________________________________________________________________________________
activation_10 (Activation)      (None, 2, 2, 256)    0           batch_normalization_10[0][0]
__________________________________________________________________________________________________
conv2d_13 (Conv2D)              (None, 2, 2, 256)    33024       add_4[0][0]
__________________________________________________________________________________________________
conv2d_12 (Conv2D)              (None, 2, 2, 256)    590080      activation_10[0][0]
__________________________________________________________________________________________________
add_5 (Add)                     (None, 2, 2, 256)    0           conv2d_13[0][0]
                                                                 conv2d_12[0][0]
__________________________________________________________________________________________________
batch_normalization_11 (BatchNo (None, 2, 2, 256)    1024        add_5[0][0]
__________________________________________________________________________________________________
activation_11 (Activation)      (None, 2, 2, 256)    0           batch_normalization_11[0][0]
__________________________________________________________________________________________________
conv2d_14 (Conv2D)              (None, 2, 2, 256)    590080      activation_11[0][0]
__________________________________________________________________________________________________
batch_normalization_12 (BatchNo (None, 2, 2, 256)    1024        conv2d_14[0][0]
__________________________________________________________________________________________________
activation_12 (Activation)      (None, 2, 2, 256)    0           batch_normalization_12[0][0]
__________________________________________________________________________________________________
conv2d_15 (Conv2D)              (None, 2, 2, 256)    590080      activation_12[0][0]
__________________________________________________________________________________________________
add_6 (Add)                     (None, 2, 2, 256)    0           add_5[0][0]
                                                                 conv2d_15[0][0]
__________________________________________________________________________________________________
batch_normalization_13 (BatchNo (None, 2, 2, 256)    1024        add_6[0][0]
__________________________________________________________________________________________________
activation_13 (Activation)      (None, 2, 2, 256)    0           batch_normalization_13[0][0]
__________________________________________________________________________________________________
conv2d_16 (Conv2D)              (None, 1, 1, 512)    1180160     activation_13[0][0]
__________________________________________________________________________________________________
batch_normalization_14 (BatchNo (None, 1, 1, 512)    2048        conv2d_16[0][0]
__________________________________________________________________________________________________
activation_14 (Activation)      (None, 1, 1, 512)    0           batch_normalization_14[0][0]
__________________________________________________________________________________________________
conv2d_18 (Conv2D)              (None, 1, 1, 512)    131584      add_6[0][0]
__________________________________________________________________________________________________
conv2d_17 (Conv2D)              (None, 1, 1, 512)    2359808     activation_14[0][0]
__________________________________________________________________________________________________
add_7 (Add)                     (None, 1, 1, 512)    0           conv2d_18[0][0]
                                                                 conv2d_17[0][0]
__________________________________________________________________________________________________
batch_normalization_15 (BatchNo (None, 1, 1, 512)    2048        add_7[0][0]
__________________________________________________________________________________________________
activation_15 (Activation)      (None, 1, 1, 512)    0           batch_normalization_15[0][0]
__________________________________________________________________________________________________
conv2d_19 (Conv2D)              (None, 1, 1, 512)    2359808     activation_15[0][0]
__________________________________________________________________________________________________
batch_normalization_16 (BatchNo (None, 1, 1, 512)    2048        conv2d_19[0][0]
__________________________________________________________________________________________________
activation_16 (Activation)      (None, 1, 1, 512)    0           batch_normalization_16[0][0]
__________________________________________________________________________________________________
conv2d_20 (Conv2D)              (None, 1, 1, 512)    2359808     activation_16[0][0]
__________________________________________________________________________________________________
add_8 (Add)                     (None, 1, 1, 512)    0           add_7[0][0]
                                                                 conv2d_20[0][0]
__________________________________________________________________________________________________
batch_normalization_17 (BatchNo (None, 1, 1, 512)    2048        add_8[0][0]
__________________________________________________________________________________________________
activation_17 (Activation)      (None, 1, 1, 512)    0           batch_normalization_17[0][0]
__________________________________________________________________________________________________
average_pooling2d_1 (AveragePoo (None, 1, 1, 512)    0           activation_17[0][0]
__________________________________________________________________________________________________
flatten_1 (Flatten)             (None, 512)          0           average_pooling2d_1[0][0]
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 10)           5130        flatten_1[0][0]
==================================================================================================
Total params: 11,186,186
Trainable params: 11,178,378
Non-trainable params: 7,808
__________________________________________________________________________________________________