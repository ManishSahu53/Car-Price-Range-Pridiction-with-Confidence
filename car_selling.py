# import nltk
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np


from sklearn.metrics import mean_squared_error
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# from keras.models import Sequential
# from keras import layers
# from keras.layers.normalization import BatchNormalization
from scipy import stats
import math

random_state=10

def removedot(s):
    s = s.replace('.', '')
    return s


def r2(y_pred, y_true):
    res = np.sum(np.square(y_pred - y_true))
    tot = np.sum(np.square(y_true - np.mean(y_true)))
    return 1 - res/tot


def interval(x_train, y_train, y_mean, y_std, x_test, model, confidence=95.0):
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
    var = '%s percentile variation'%(percentile_confidence)
    x_test[var] = np.round((x_test.target - x_test.upper)*100/x_test.target)
    x_test.reset_index(drop=True)
    return x_test


# Reading dataset
path_json = 'oto_car_json.json'

df = pd.read_json(path_json, lines=True)
print('Total Dataset available before removing duplicates : %d' % (len(df)))


# Preprocessing dataset
df['km'] = df['km'].apply(lambda x: removedot(x))
df['km'] = df['km'].apply(lambda x: float(x))
df['price'] = df['price'].apply(lambda x: float(x))
df['fuel'] = df['fuel'].apply(lambda x: x.lower())
df['model'] = df['model'].apply(lambda x: x.lower())
df['brand'] = df['brand'].apply(lambda x: x.lower())
df['city'] = df['city'].apply(lambda x: x.lower())
df['type'] = df['type'].apply(lambda x: x.lower())

df = df.drop_duplicates()
print('Total Dataset available after removing duplicates : %d' % (len(df)))

df['count'] = 1

# 4. Creating delta year wrt 2019
# df['delta_year'] = 2019 - df['year']
base_year = 2019
df["delta_year"] = df["year"].apply(lambda x: base_year - x)
df["model"] = df["model"].apply(lambda x: x.lower())

# Collecting different models
model = df.groupby('model').agg({'count': 'sum'}).reset_index()
model = model.sort_values('count').reset_index(drop=True)
print('Number of model : %d' % (len(model)))

# Copying data
data = df.copy()

# ## Removing < 1%tile and >99% data
# Since a lot of pricing given is incorrect
lower = 0.01
upper = 0.99
percentile_confidence = 90.0

print('Length of data before outlier removal: %d' % (len(data)))

data = data[data.price > data.price.quantile(lower)]
data = data[data.price < data.price.quantile(upper)]

print('Length of data after outlier removal: %d' % (len(data)))

# List of car models availables with atleast 200 data points
models = list(model.model[model['count'] > 200])
# models = ['honda crv (2012-2017) 2.4 i-vtec at']

# Variables initialization
forest_accuracy = {}
nn_accuracy = {}
cars = []

for i in range(len(models)):
    print('Running %s model' % (models[i]))

    # Taking one particular model
    car_model = data[data.model == models[i]].reset_index(drop=True)

    # Extracting x and y values
    y = car_model.price
    y = y.values

    # Normalizing target datasets
    y_mean = y.mean()
    y_std = y.std()

    y = (y - y_mean)/y_std

    x = car_model[['km', 'delta_year']].copy()

    # Normalizing paramters datasets
    x_km_mean = x.km.mean()
    x_km_std = x.km.std()

    x['km'] = (x['km'] - x_km_mean) / (x_km_std)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=random_state)

    forest = []
    forest = RandomForestRegressor(
        max_depth=10, random_state=random_state, n_estimators=100)
    forest.fit(x_train, y_train)

    y_pred = forest.predict(x_test)
    score = math.sqrt(mean_squared_error(y_pred, y_test))

    # Training data MSE
    y_train_pred = forest.predict(x_train)
    mse = mean_squared_error(y_train_pred*y_std + y_mean,
                             y_train*y_std + y_mean)

    # Accuracy Metrics
    acc_RMSE = round(score, 4)
    acc_R2 = r2(y_pred, y_test)

    # Getting to original dataset
    y_abs_pred = y_pred*y_std + y_mean
    y_abs_test = y_test*y_std + y_mean

    err = abs(y_abs_test - y_abs_pred)

    # Percentage Error
    met = abs(y_abs_test - y_abs_pred)*100/y_abs_pred

    acc_mean = met.mean()
    acc_min = min(met)
    acc_max = max(met)

    met.sort()
    acc_95 = met[int(len(met)*percentile_confidence/100)]

    forest_accuracy[models[i]] = {'mean%': round(acc_mean, 2),
                                  'min %': round(acc_min, 2),
                                  'max %': round(acc_max, 2),
                                  '%s percentile'%(percentile_confidence): round(acc_95, 2)}

    predict = interval(x_train, y_train, y_mean, y_std, x_test, model=forest, confidence=percentile_confidence)
    predict['model'] = models[i]
    cars.append(predict)

    # # Neural Network training
    # # x_train = x_train.values

    # input_dim = x_train.shape[1]  # Number of features
    # model = []
    # model = Sequential()
    # model.add(layers.Dense(30
    #                     , input_dim=input_dim, activation='relu',kernel_initializer='uniform'))
    # model.add(layers.Dense(20,kernel_initializer='uniform', activation='relu'))
    # model.add(layers.Dense(10,kernel_initializer='uniform', activation='relu'))
    # model.add(layers.Dense(1))

    # model.compile(loss='mean_squared_error',
    #             optimizer='adam')
    # model.summary()

    # history = model.fit(x_train, y_train,
    #                 epochs=100,
    #                 verbose=True,
    #                 validation_data=(x_test, y_test),
    #                 batch_size=32)

    # y_pred = model.predict(x_test)
    # score = math.sqrt(mean_squared_error(y_pred, y_test))

    # # Accuracy Metrics
    # acc_RMSE = round(score,4)
    # acc_R2 = r2(y_pred, y_test)

    # # Getting to original dataset
    # y_abs_pred = y_pred*y_std + y_mean
    # y_abs_test = y_test*y_std + y_mean

    # err = abs(y_abs_test - y_abs_pred)

    # # Percentage Error
    # met = np.abs(y_abs_test- y_abs_pred[:,0])*100/y_abs_pred[:,0]

    # acc_mean = met.mean()
    # acc_min = min(met)
    # acc_max =  max(met)

    # met.sort()
    # acc_95 = met[int(len(met)*0.95)]

    # nn_accuracy[models[i]] = {'mean%': round(acc_mean, 2),
    #                  'min %': round(acc_min, 2),
    #                  'max %': round(acc_max, 2),
    #                  '95 %tile': round(acc_95, 2)}

# df1 = pd.DataFrame(nn_accuracy)
# print('Saving NN results to csv')
# df1.to_csv('nn.csv')
df = pd.DataFrame(forest_accuracy)

print('Saving random forest results to csv')
df.to_csv('forest.csv')

data = pd.concat(cars)
data = data.reset_index(drop=True)
data.to_csv('predict.csv')
