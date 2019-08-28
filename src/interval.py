import pandas as pd
from scipy import stats
import math
import numpy as np
import config
from sklearn.metrics import mean_squared_error


# Estimating prediction intervals
def prediction_interval(x_train, y_train, y_mean, y_std, x_test, model, confidence=95.0):
    """
    n is Number of Sample data while training
    mse is Mean Square error of trained dataset
    xi is independent variable (x_test) to be useds while predicting model
    x is independent variable (x_train) which is used while training the model
    model is trained sklearn model
    confidence interval is how large your 
    """
    #Studnt, n=999, p<0.05, 2-tail
    # equivalent to Excel TINV(0.05,999)
    n = len(x_train)

    if len(x_train) != len(y_train):
        raise('Training data and variables shape is not equal')

    y_train_pred = model.predict(x_train)
    mse = mean_squared_error(y_train, y_train_pred)

    confidence = (100-confidence)/100/2

    t = stats.t.ppf(1-confidence, n)

    if x_train.shape[1] != x_test.shape[1]:
        raise('Parameters in x_train, x_test are not same. x has %d and xi has %d' % (
            x_train.shape[1], x_test.shape[1]))

    temp = math.sqrt(mse*(1 + 1/n + np.sum(np.square(x_test - x_train.mean(axis=0)).values
                                           ) / np.sum(np.square(x_train - x_train.mean(axis=0)).values)))
    lower = -t*temp
    upper = t*temp

    x_test['target'] = model.predict(x_test)
    x_test['lower'] = x_test.target + lower
    x_test['upper'] = x_test.target + upper

    x_test['target'] = np.round(x_test.target * y_std + y_mean)
    x_test['lower'] = np.round(x_test.lower * y_std + y_mean)
    x_test['upper'] = np.round(x_test.upper * y_std + y_mean)
    var = '%s percentile variation' % (config.confidence_prediction_interval)
    x_test[var] = np.round((x_test.target - x_test.upper)*100/x_test.target)
    x_test.reset_index(drop=True)
    return x_test


