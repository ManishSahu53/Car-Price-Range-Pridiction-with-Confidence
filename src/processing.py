import numpy as np
from sklearn.metrics import mean_squared_error

# Removeing dots
def removedot(s):
    s = s.replace('.', '')
    return s


# Calculating R2 value
def r2(y_pred, y_true):
    res = np.sum(np.square(y_pred - y_true))
    tot = np.sum(np.square(y_true - np.mean(y_true)))
    return 1 - res/tot


