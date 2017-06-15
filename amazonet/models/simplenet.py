'''

'''
from keras.models import Model

from keras.layers import (Input, Conv2D, Activation, Dense,
                          GlobalAveragePooling2D, MaxPool2D,
                          Concatenate, BatchNormalization)

from keras.layers.advanced_activations import ELU

name = 'simplenet'

def create_model():
    '''
    NULL
    '''
    # _input = Input((256, 256, 4))
    # x = make_conv_bn_elu(_input, 64, 7, 2)
    # x = MaxPool2D(pool_size=(3, 3), strides=2)(x)
    # x = make_conv_bn_elu(x, 192, 3)
    _input = Input((64, 64, 3))
    x = make_conv_bn_elu(_input, 8)
    x = make_conv_bn_elu(x, 8)
    x = make_conv_bn_elu(x, 8)

    x = make_block(x, 32)

    x = make_block(x, 32)

    outa = make_block(x, 64)

    outb = make_block(outa, 128)


    # add dropout to a and b here if overfitting
    outa = GlobalAveragePooling2D()(outa)
    outb = GlobalAveragePooling2D()(outb)

    out = Concatenate()([outa, outb])

    out = Dense(512)(out)
    out = BatchNormalization()(out)
    out = ELU()(out)
    out = Dense(512)(out)
    out = BatchNormalization()(out)
    out = ELU()(out)

    out = Dense(17)(out)
    out = Activation('sigmoid')(out)

    return Model(_input, out)


def make_block(x, filters):
    x = make_conv_bn_elu(x, filters, 3, 1, 1)
    x = make_conv_bn_elu(x, filters)
    x = make_conv_bn_elu(x, filters)
    x = make_conv_bn_elu(x, filters)
    x = make_conv_bn_elu(x, filters, 3, 1, 1)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
    return x

def make_conv_bn_elu(x, filters, kernel_size=1, stride=1, padding=0):
    if padding == 0:
        padding = 'valid'
    else:
        padding = 'same'
    x = Conv2D(filters=filters, kernel_size=(kernel_size, kernel_size), strides=(stride, stride), padding=padding)(x)
    x = BatchNormalization()(x)
    x = ELU()(x)
    return x

if __name__ == "__main__":
    create_model().summary()