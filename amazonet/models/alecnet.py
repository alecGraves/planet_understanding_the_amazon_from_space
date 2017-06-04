'''
A simple implmentation of a network using inception modules.
'''
import keras.backend as K
from keras.models import Model
from keras.layers import (Input, Conv2D, BatchNormalization, 
                          Activation, Average, MaxPooling2D,
                          Concatenate, AveragePooling2D,
                          Reshape, Dropout, Flatten, Dense,
                          GlobalMaxPooling2D)
from keras.layers.advanced_activations import LeakyReLU

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
    x = Conv2D(64, (3, 3),strides=(2, 2))(_input)
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
    x, soft1 = my_inception_module(x, 3, True)

    x = my_inception_module(x, 3)
    x = my_inception_module(x, 3)
    x, soft2 = my_inception_module(x, 4, True)

    x = MaxPooling2D((3, 3), strides=(2,2))(x)
    x = my_inception_module(x, 4)
    x = my_inception_module(x, 5)
    x = AveragePooling2D((5, 5), strides=(1, 1))(x)
    x = Dropout(0.4)(x)
    spatial = get_spatial_dims(x)
    x = Conv2D(17, kernel_size=spatial)(x)
    soft3 = Activation('softmax')(x)
    out = Average()([soft1, soft2, soft3])
    out = GlobalMaxPooling2D()(out) # Flatten
    return out

def my_inception_module(x, scale=1, do_predict=False):
    '''
    x is input layer, scale is factor to scale kernel sizes by

    This is a version of the 'inception module', 
    with Batch Norm added
    '''
    scale *= 4
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
        predict = Dropout(0.25)(predict)
        spatial = get_spatial_dims(predict)
        predict = Conv2D(1024, kernel_size=spatial)(predict)
        predict = BatchNormalization()(predict)
        predict = LeakyReLU()(predict)
        predict = Conv2D(17, kernel_size=(1,1), activation='softmax')(predict)
        return out, predict

    return out

def get_spatial_dims(layer):
    if K.image_data_format() == 'channels_first':
        channel_shape = (int(layer.shape[2]), int(layer.shape[3]))
    else:
        channel_shape = (int(layer.shape[1]), int(layer.shape[2]))
    return channel_shape

def test_model():
    "Test to ensure model compiles successfully"
    print("LOADING MODEL")
    model = create_model()
    print("COMPILING MODEL")
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    print("MODEL COMPILES SUCCESSFULLY")

if __name__ == "__main__":
    test_model()
