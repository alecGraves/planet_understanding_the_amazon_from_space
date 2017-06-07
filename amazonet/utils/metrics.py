'''
Definitions for various loss functions to be used dring training.
'''

import keras.backend as K

def FScore2(y_true, y_pred):
    '''
    The F score, beta=2
    '''
    beta2 = K.variable(4)
    pred = K.cast(K.round(y_pred), 'float32')
    tp = K.sum(K.cast(y_true, 'float32'), -1) # true positive
    fp = K.sum(K.cast(K.less(K.abs(pred - K.clip(y_true, .5, 1.)), 0.01), 'float32'), -1) 
    fn = K.sum(K.cast(K.less(K.abs(pred - K.clip(y_true, 0., .5)), 0.01), 'float32'), -1)

    p = tp / (tp + fp)
    r = tp / (tp + fn)

    f2 = K.variable(5) * p * r / (beta2 * p + r)

    return K.mean(f2)

def FScore2_python(y_true, y_pred):
    '''
    python implementation of F_B Score, B=2
    # Inputs
        y_true: list of lists of 'true' values
        y_pred: list of lists of predicted values
    # Outputs
        returns the average F score
    '''
    B = 2
    B2 = B ** 2
    OnePlusB2 = 1 + B2
    FScore = []

    for i, true in enumerate(y_true):
        true = [int(category) for category in true]
        pred = [int(round(category)) for category in y_pred[i]]

        true_positives = 0
        false_positives = 0
        false_negatives = 0
        for j, true_cat in enumerate(true):
            if true_cat == 1:
                if y_pred[i][j] == 1:
                    true_positives += 1
                else:
                    false_negatives += 1
            elif y_pred[i][j] == 1:
                false_positives += 1

        _fscore = OnePlusB2 * true_positives / (OnePlusB2 * true_positives + B2 * false_negatives + false_positives)
        FScore.append(_fscore)

    avg = 0
    n = len(FScore)
    for score in FScore:
        avg += score/n

    return avg

def test_FScore2():
    '''test for FScore2'''
    # Test 1:
    y_true = [[1, 0, 0, 1]]
    y_pred = [[1, 1, 1, 1]]

    score_python = FScore2_python(y_true, y_pred)

    y_true = K.constant(y_true)
    y_pred = K.constant(y_pred)
    score_keras = K.eval(FScore2(y_true, y_pred))

    assert(abs(score_keras-score_python) < 0.0001)
    print('Test 1 passed!')


def competition_loss(y_true, y_pred):
    '''
    Loss for the kaggle competition data.
    ### Note:
    will break if y_pred == 1 or y_pred == 0
    * Always use sigmoid
    '''
    # evaluation: F2 Score: Recall weighted twice as high as Precision
    #
    # Recall = true_positives / (true_positives + false_negatives)
    #        = maximized when model does not miss categories.
    #
    # Precision = true_positives / (true_positives + false_positives)
    #           = maximized when model does not get false positives.
    #
    # y_true.shape = y_test.shape = (batch_sze, 17)
    # sqrt(neg) = 0, this multiplier only affects false_positives.
    loss_multiplier = (K.variable(1.41421356237) * K.sqrt(y_true-y_pred) + K.variable(1))
    binary_crossentropy = y_true * K.log(y_pred) + (K.variable(1) - y_true) * K.log(K.variable(1) - y_pred)
    recall_preferred_logloss = K.mean(K.variable(-1) * K.mean(binary_crossentropy * loss_multiplier, axis=-1))
    return recall_preferred_logloss

def test_competition_loss():
    y_true = K.constant([[1, 0, 0, 1]])
    y_pred = K.constant([[0.1, .9, .9, .9]])
    loss = competition_loss(y_true, y_pred)
    print(K.eval(loss))


if __name__ == "__main__":
    #test_FScore2()
    test_competition_loss()

