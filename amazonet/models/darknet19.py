'''
Implementation of pjreddie's Darknet19.
'''

from keras.models import Model
from keras.layers import (Input, Conv2D, BatchNormalization,
                          Activation, MaxPooling2D, GlobalAveragePooling2D)
from keras.layers.advanced_activations import LeakyReLU

name = 'darknet19'

def create_model():
    _input = Input((256, 256, 4))

    # layer 1
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(_input)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    # layer 2
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    # layer 3
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    # layer 4
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    # layer 5
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    # layer 6
    x = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    # layer 7
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    # layer 8
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    # layer 9
    x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    # layer 10
    x = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    # layer 11
    x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    # layer 12
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    # layer 13
    x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    # layer 14
    x = Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    # layer 15
    x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    # layer 16
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    # layer 17
    x = Conv2D(filters=1024, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    # layer 18
    x = Conv2D(filters=512, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    # layer 19
    x = Conv2D(filters=1024, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    # out
    x = Conv2D(filters=17, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    x = GlobalAveragePooling2D()(x)
    out = Activation('sigmoid')(x)

    return Model(_input, out)


if __name__ == "__main__":
    create_model().summary()