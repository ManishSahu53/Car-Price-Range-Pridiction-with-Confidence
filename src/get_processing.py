import numpy as np
from sklearn.metrics import mean_squared_error

# Removeing dots
def removedot(s):
    s = s.replace('.', '')
    return s


def convert_price(s: str) -> float:
    if s !=s:
        return -1
    
    a = removedot(s)
    b = a.replace(' ', ',')
    c = b.split(',')
    d = c[1:][0]
    return d


def convert_distance(s: str) -> float:
    if s!=s:
        return -1.0

    a = removedot(s)
    b = a.replace(' ', ',')
    c = b.replace('>', ',')
    d = c.replace('-', ',')
    e = d.split(',')
    try:
        avg = (float(e[0]) + float(e[1]))/2
    except:
        avg = -1.0
    return avg

# Calculating R2 value
def r2(y_pred, y_true):
    res = np.sum(np.square(y_pred - y_true))
    tot = np.sum(np.square(y_true - np.mean(y_true)))
    return 1 - res/tot


