'''
A simple implmentation of a network using inception modules.
'''
import keras.backend as K
from keras.models import Model
from keras.layers import (Input, Conv2D, BatchNormalization, 
                          Activation, Average, MaxPooling2D,
                          Concatenate, GlobalAveragePooling2D,
                          Reshape, Dropout, Flatten, Dense,
                          AveragePooling2D)
from keras.layers.advanced_activations import LeakyReLU

name = 'incepnet'

def create_model():
    "Generate a single inception_net model"
    _input = Input((256, 256, 4))
    incep1 = inception_net(_input)
    out = incep1
    model = Model(inputs=_input, outputs=[out])
    return model

def inception_net(_input):
    '''
    My version of the 'inception net',
    '''
    x = Conv2D(64, (7, 7),strides=(2, 2))(_input)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D(pool_size=(3,3), strides=2)(x)
    x = Conv2D(192, (3, 3),strides=(1, 1))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D(pool_size=(3,3), strides=2)(x)

    x = my_inception_module(x, 1)
    x = my_inception_module(x, 2)
    x = MaxPooling2D(pool_size=(3,3), strides=2)(x)
    x = my_inception_module(x, 2)
    x, pred1 = my_inception_module(x, 3, True)

    x = my_inception_module(x, 3)
    x = my_inception_module(x, 3)
    x, pred2 = my_inception_module(x, 4, True)

    x = MaxPooling2D((3, 3), strides=(2,2))(x)
    x = my_inception_module(x, 4)
    x = my_inception_module(x, 5)
    x = AveragePooling2D((5, 5), strides=(1, 1))(x)
    x = Dropout(0.4)(x)
    pred3 = Conv2D(17, kernel_size=(1, 1))(x)
    pred3 = GlobalAveragePooling2D()(pred3)
    out = Average()([pred1, pred2, pred3])
    out = Activation('sigmoid')(out)
    return out

def my_inception_module(x, scale=1, do_predict=False):
    '''
    x is input layer, scale is factor to scale kernel sizes by

    This is a version of the 'inception module', 
    with Batch Norm added
    '''
    scale *= 5
    x11 = Conv2D(int(16*scale), (1, 1), padding='valid')(x)
    x11 = BatchNormalization()(x11)
    x11 = LeakyReLU()(x11)

    x33 = Conv2D(int(24*scale), (1, 1))(x)
    x33 = BatchNormalization()(x33)
    x33 = LeakyReLU()(x33)
    x33 = Conv2D(int(32*scale), (3, 3), padding='same')(x33)
    x33 = BatchNormalization()(x33)
    x33 = LeakyReLU()(x33)

    x55 = Conv2D(int(4*scale), (1, 1))(x)
    x55 = BatchNormalization()(x55)
    x55 = LeakyReLU()(x55)
    x55 = Conv2D(int(8*scale), (5, 5), padding='same')(x55)
    x55 = BatchNormalization()(x55)
    x55 = LeakyReLU()(x55)

    x33p = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    x33p = Conv2D(int(8*scale), (1, 1))(x33p)
    x33p = BatchNormalization()(x33p)
    x33p = LeakyReLU()(x33p)

    out = Concatenate(axis=3)([x11, x33, x55, x33p])

    if do_predict:
        predict = AveragePooling2D((5, 5), strides=(1, 1))(x)
        predict = Conv2D(int(8*scale), (1, 1))(predict)
        predict = BatchNormalization()(predict)
        predict = LeakyReLU()(predict)
        predict = Conv2D(int(100*scale), (1, 1))(predict)
        predict = BatchNormalization()(predict)
        predict = LeakyReLU()(predict)
        predict = Dropout(0.25)(predict)
        predict = Conv2D(17, kernel_size=(1, 1))(predict)
        predict = GlobalAveragePooling2D()(predict)
        return out, predict

    return out

def test_model():
    "Test to ensure model compiles successfully"
    print("LOADING MODEL")
    model = create_model()
    print("COMPILING MODEL")
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    print("MODEL COMPILES SUCCESSFULLY")

if __name__ == "__main__":
    test_model()
