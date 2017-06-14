'''
simple conductor net that finds the optimal weight for each category of each prediction network
'''
import keras.backend as K
from keras.layers import Input, Dense, Reshape, Lambda, Multiply, Add, Average, Activation
from keras.models import Model

def create_model(n=1):
    '''
    creates a model to learn weights for an ensemble with 'n' members.
    '''
    x = Input((n, 17))
    one = Lambda(lambda x : K.ones((1, 1)))(x)
    m = Dense(17*n, use_bias=False, kernel_initializer='ones')(one)
    m = Reshape((n, 17))(m)
    b = Dense(17*n, use_bias=False, kernel_initializer='glorot_uniform')(one)
    b = Reshape((n, 17))(b)
    y = Multiply()([m, x])
    y = Add()([y, b])
    y = Lambda(mean, output_shape=mean_output_shape)(y)
    return Model(x, y)

def mean(x):
    x = K.mean(x, axis=1)
    return x

def mean_output_shape(input_shape):
    return (input_shape[0], input_shape[2])

if __name__ == "__main__":
    create_model(2).summary()
