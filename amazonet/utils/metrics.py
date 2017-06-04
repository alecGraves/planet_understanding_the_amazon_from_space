'''
Definitions for various loss functions to be used dring training.
'''

import keras.backend as K

def FScore2(y_true, y_pred):
    '''
    The F score, beta=2
    '''
    beta2 = K.variable(4)
    pred = K.cast(K.greater(y_pred, 0.55), 'float32')
    tp = K.sum(y_true, 1) # true positive
    fp = K.sum(K.cast(K.not_equal(pred, K.clip(y_true, .5, 1.)), 'float32'), 1)
    fn = K.sum(K.cast(K.not_equal(pred, K.clip(y_true, 0, 0.5)), 'float32'), 1)

    p = tp / (tp + fp)
    r = tp / (tp + fn)

    f2 = K.variable(5) * p * r / (beta2 * p + r)

    return K.mean(f2)


