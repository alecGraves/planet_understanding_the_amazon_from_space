'''
An implementation of a network using all elu's.

* insert dramatic poem about alleluna *

'''
from keras.layers.advanced_activations import ELU

from keras.models import Model
from keras.layers import (Input, Conv2D, Activation, Dropout,
                          GlobalAveragePooling2D,
                          BatchNormalization)

name = 'alleluna'

def create_model():
    '''
    Returns alleluna in the flesh.
    '''
    the_beginning = Input((256, 256, 4))
    alleluna = Conv2D(filters=128, kernel_size=(11, 11), strides=(4, 4))(the_beginning)
    alleluna = ELU()(alleluna)
    alleluna = BatchNormalization()(alleluna)

    alleluna = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1))(alleluna)
    alleluna = ELU()(alleluna)
    alleluna = BatchNormalization()(alleluna)


    alleluna = Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2))(alleluna)
    alleluna = ELU()(alleluna)
    alleluna = BatchNormalization()(alleluna)

    alleluna = Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1))(alleluna)
    alleluna = ELU()(alleluna)
    alleluna = BatchNormalization()(alleluna)

    alleluna = Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1))(alleluna)
    alleluna = ELU()(alleluna)
    alleluna = BatchNormalization()(alleluna)

    alleluna = Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2))(alleluna)
    alleluna = ELU()(alleluna)
    alleluna = BatchNormalization()(alleluna)

    alleluna = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1))(alleluna)
    alleluna = ELU()(alleluna)
    alleluna = BatchNormalization()(alleluna)

    alleluna = Conv2D(filters=512, kernel_size=(1, 1), strides=(1, 1))(alleluna)
    alleluna = ELU()(alleluna)
    alleluna = BatchNormalization()(alleluna)

    alleluna = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1))(alleluna)
    alleluna = ELU()(alleluna)
    alleluna = BatchNormalization()(alleluna)
    alleluna = Dropout(2/3)(alleluna)

    alleluna = Conv2D(filters=1280, kernel_size=(3, 3), strides=(1, 1))(alleluna)
    alleluna = ELU()(alleluna)
    alleluna = BatchNormalization()(alleluna)

    alleluna = Conv2D(filters=1280, kernel_size=(1, 1), strides=(1, 1))(alleluna)
    alleluna = ELU()(alleluna)
    alleluna = BatchNormalization()(alleluna)

    alleluna = Conv2D(filters=17, kernel_size=(1, 1), strides=(1, 1))(alleluna)
    alleluna = ELU()(alleluna)

    alleluna = GlobalAveragePooling2D()(alleluna)
    alleluna = Activation('sigmoid')(alleluna)

    alleluna = Model(the_beginning, alleluna)

    return alleluna


if __name__ == "__main__":
    create_model().summary()