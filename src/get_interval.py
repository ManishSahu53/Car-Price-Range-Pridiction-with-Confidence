import pandas as pd
from scipy import stats
import math
import numpy as np
import config
from sklearn.metrics import mean_squared_error


# Estimating prediction intervals
def prediction_interval(n, x_train_mean, y_train_mean, x_test, model, y_mse, x_mse, confidence=95.0):
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

    confidence = (100-confidence)/100/2

    t = stats.t.ppf(1-confidence, n)
    
    # If Datatypr if Dataframe or NDArray
    if type(x_test) is pd.core.frame.DataFrame:
        temp = math.sqrt(y_mse*(1 + 1/n + np.sum(np.square(x_test.values - x_train_mean)) / x_mse))
    elif type(x_test) is np.ndarray:
        temp = math.sqrt(y_mse*(1 + 1/n + np.sum(np.square(x_test - x_train_mean)) / x_mse))
    
    lower = -t*temp
    upper = t*temp

    x_test['target'] = np.round(model.predict(x_test),0)
    x_test['lower'] = np.round(x_test.target + lower, 0)
    x_test['upper'] = np.round(x_test.target + upper, 0)

    var = '%s_percentile_variation' % (config.confidence_prediction_interval)
    x_test[var] = np.round((x_test.target - x_test.upper)*100/x_test.target)
    x_test.reset_index(drop=True)
    return x_test


