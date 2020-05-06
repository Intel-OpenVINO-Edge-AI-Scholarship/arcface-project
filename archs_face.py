import keras
from keras.models import Model
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import Input, MaxPooling2D, Dropout, Flatten, Embedding, AveragePooling2D
from keras import regularizers
import numpy as np
from metrics_face import *

weight_decay = 1e-4


def vgg_block(x, filters, layers):
    for _ in range(layers):
        x = Conv2D(filters, (3, 3), padding='same', kernel_initializer='glorot_uniform',
                    kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

    return x


def vgg8(args):
    input = Input(shape=(28, 28, 1))

    x = vgg_block(input, 16, 2)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = vgg_block(x, 32, 2)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = vgg_block(x, 64, 2)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = Dense(args.num_features, kernel_initializer='he_normal',
                kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    output = Dense(10, activation='softmax', kernel_regularizer=regularizers.l2(weight_decay))(x)

    return Model(input, output)


def vgg8_arcface(args, n_classes=10):
    input = Input(shape=(160, 160, 3))
    y = Input(shape=(n_classes,))

    x = vgg_block(input, 16, 2)
    x = MaxPooling2D(pool_size=(3, 3))(x)
    x = vgg_block(x, 32, 2)
    x = MaxPooling2D(pool_size=(3, 3))(x)
    x = vgg_block(x, 38, 4)
    x = MaxPooling2D(pool_size=(3, 3))(x)
    x = vgg_block(x, 40, 2)
    x = AveragePooling2D(pool_size=(3, 3))(x)

    x = Dropout(0.4)(x)
    x = Flatten()(x)
    x = Dense(args.num_features, kernel_initializer='glorot_uniform',
                kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    output = ArcFace(n_classes, regularizer=regularizers.l2(weight_decay))([x, y])

    return Model([input, y], output)


def vgg8_cosface(args):
    input = Input(shape=(28, 28, 1))
    y = Input(shape=(10,))

    x = vgg_block(input, 16, 2)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = vgg_block(x, 32, 2)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = vgg_block(x, 64, 2)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = Dense(args.num_features, kernel_initializer='he_normal',
                kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    output = CosFace(10, regularizer=regularizers.l2(weight_decay))([x, y])

    return Model([input, y], output)


def vgg8_sphereface(args):
    input = Input(shape=(28, 28, 1))
    y = Input(shape=(10,))

    x = vgg_block(input, 16, 2)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = vgg_block(x, 32, 2)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = vgg_block(x, 64, 2)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = Dense(args.num_features, kernel_initializer='he_normal',
                kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    output = SphereFace(10, regularizer=regularizers.l2(weight_decay))([x, y])

    return Model([input, y], output)
