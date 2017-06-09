'''

'''
from keras.models import Model

from keras.layers import (Input, Conv2D, Activation, Dense,
                          GlobalAveragePooling2D, MaxPool2D,
                          Concatenate, BatchNormalization, Dropout)

from keras.layers.advanced_activations import ELU

name = 'simplenet_v2_m'

def create_model():
    '''
    NULL
    '''
    _input = Input((256, 256, 4))

    x = make_conv_bn_elu(_input, 32, 3, 2)
    x = make_conv_bn_elu(x, 32, 3, 1, 1)
    x = make_conv_bn_elu(x, 64, 3, 1, 1)

    x1 = MaxPool2D(pool_size=(3, 3), strides=2)(x)
    x2 = make_conv_bn_elu(x, 96, 3, 2)
    x = Concatenate()([x1, x2])

    x = make_conv_bn_elu(x, 64)
    x = make_conv_bn_elu(x, 96)
    x = make_conv_bn_elu(x, 96)

    x = make_block(x, 64)

    outa = make_block(x, 64)

    outb = make_block(outa, 128)

    outc = make_block(outb, 256)


    # add dropout to a and b here if overfitting
    outa = GlobalAveragePooling2D()(outa)
    outb = GlobalAveragePooling2D()(outb)
    outc = GlobalAveragePooling2D()(outc)

    out = Concatenate()([outa, outb, outc])
    out = Dropout(1/3)(out)

    out = Dense(1280)(out)
    out = BatchNormalization()(out)
    out = ELU()(out)
    out = Dense(1280)(out)
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
    x1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
    x2 = make_conv_bn_elu(x, int(filters*3/2), 3, 2)
    return Concatenate()([x1, x2])

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