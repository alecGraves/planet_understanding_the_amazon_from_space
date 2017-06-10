'''
resnet 18
'''
from keras.models import Model
from keras.layers import (Input, Conv2D, Activation, Add, Concatenate,
                          BatchNormalization, GlobalAveragePooling2D,
                          MaxPooling2D)
from keras.layers.advanced_activations import PReLU as Act


name = 'resnet18_prelu'

def create_model():
    '''
    returns resnet18 model
    '''
    _input = Input((256, 256, 3))
    x = Conv2D(62, kernel_size=(7, 7), strides=(2, 2))(_input)
    x1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    x2 = Conv2D(96, kernel_size=(3, 3), strides=(2, 2))(x)
    x = Concatenate()([x1, x2])

    x = resize_block(x, 64)
    x = identity_block(x, 64)

    x = resize_block(x, 128)
    x = identity_block(x, 128)

    x = resize_block(x, 256)
    x = identity_block(x, 256)

    x = resize_block(x, 512)
    x = identity_block(x, 512)

    x = out(x)
    return Model(_input, x)

def resize_block(x0, filters):
    x = Conv2D(filters=filters, kernel_size=(3, 3), strides=(2, 2), padding='same')(x0)
    x = BatchNormalization()(x)
    x = Act()(x)
    x = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)

    x1 = Conv2D(filters=filters, kernel_size=(1, 1), strides=(2, 2))(x0)

    x = Add()([x, x1])
    x = Act()(x)
    return x

def identity_block(x0, filters):
    x = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(x0)
    x = BatchNormalization()(x)
    x = Act()(x)
    x = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)

    x = Add()([x0, x])
    x = Act()(x)
    return x

def out(x):
    x = Conv2D(filters=17, kernel_size=(1, 1), strides=(1, 1))(x)
    x = GlobalAveragePooling2D()(x)
    x = Activation('sigmoid')(x)
    return x

if __name__ == "__main__":
    create_model().summary()