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

from keras.models import Sequential
from keras import layers
from keras.layers.normalization import BatchNormalization


def removedot(s):
    s = s.replace('.','')
    return s


def r2(y_pred, y_true):
    res = np.sum(np.square(y_pred - y_true))
    tot = np.sum(np.square(y_true - np.mean(y_true)))
    return 1 - res/tot


def rmse(y_pred, y_true):
    sq_pred = np.power(y_pred, 2)


path_json = 'oto_car_json.json'

df = pd.read_json(path_json,lines=True)
print('Total Dataset available before removing duplicates : %d' %(len(df)))


df['km'] = df['km'].apply(lambda x: removedot(x))
df['km'] = df['km'].apply(lambda x: float(x))
df['price'] = df['price'].apply(lambda x: float(x))
df['fuel'] = df['fuel'].apply(lambda x: x.lower())
df['model'] = df['model'].apply(lambda x: x.lower())
df['brand'] = df['brand'].apply(lambda x: x.lower())
df['city'] = df['city'].apply(lambda x: x.lower())
df['type'] = df['type'].apply(lambda x: x.lower())

df = df.drop_duplicates()
print('Total Dataset available after removing duplicates : %d' %(len(df)))

df['count'] = 1

# 4. Creating delta year wrt 2019
# df['delta_year'] = 2019 - df['year']
base_year = 2019
df["delta_year"] = df["year"].apply(lambda x: base_year - x)
df["model"] = df["model"].apply(lambda x: x.lower())

# Collecting different models
model = df.groupby('model').agg({'count': 'sum'}).reset_index()
model = model.sort_values('count').reset_index(drop=True)
print('Number of model : %d' %(len(model)))

# Copying data
data = df.copy()
# ## Removing < 1%tile and >99% data 
# Since a lot of pricing given is incorrect

lower = 0.01
upper = 0.99

print('Length of data before outlier removal: %d' %(len(data)))

data = data[data.price > data.price.quantile(lower)]
data = data[data.price < data.price.quantile(upper)]

print('Length of data after outlier removal: %d' %(len(data)))

# List of car models availables with atleast 50 data points
models = list(model.model[model['count'] > 50])

forest_accuracy = {}
nn_accuracy = {}

for i in range(len(models)):
    print('Running %s model' %(models[i]))

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
    x_km_range = x.km.max() - x.km.min()

    x['km'] = (x['km'] - x_km_mean) / (x_km_range)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    forest = []
    forest = RandomForestRegressor(max_depth=10, random_state=0, n_estimators=100)
    forest.fit(x_train, y_train)

    y_pred = forest.predict(x_test)
    score = math.sqrt(mean_squared_error(y_pred, y_test))

    # Accuracy Metrics
    acc_RMSE = round(score,4)
    acc_R2 = r2(y_pred, y_test)

    # Getting to original dataset
    y_abs_pred = y_pred*y_std + y_mean
    y_abs_test = y_test*y_std + y_mean


    err = abs(y_abs_test - y_abs_pred)

    # Percentage Error 
    met = abs(y_abs_test- y_abs_pred)*100/y_abs_pred

    acc_mean = met.mean()
    acc_min = min(met)
    acc_max =  max(met)

    met.sort()
    acc_95 = met[int(len(met)*0.95)]

    forest_accuracy[models[i]] = {'mean%': round(acc_mean, 2),
                     'min %': round(acc_min, 2),
                     'max %': round(acc_max, 2),
                     '95 %tile': round(acc_95, 2)}

    print('Completed %s model' %(models[i]))



    # Neural Network training
    x_train = x_train.values


    input_dim = x_train.shape[1]  # Number of features
    model = []
    model = Sequential()
    model.add(layers.Dense(30
                        , input_dim=input_dim, activation='relu',kernel_initializer='uniform'))
    model.add(layers.Dense(20,kernel_initializer='uniform', activation='relu'))
    model.add(layers.Dense(10,kernel_initializer='uniform', activation='relu'))
    model.add(layers.Dense(1))


    model.compile(loss='mean_squared_error', 
                optimizer='adam')
    model.summary()

    history = model.fit(x_train, y_train,
                    epochs=100,
                    verbose=True,
                    validation_data=(x_test, y_test),
                    batch_size=32)


    y_pred = model.predict(x_test)
    score = math.sqrt(mean_squared_error(y_pred, y_test))

    # Accuracy Metrics
    acc_RMSE = round(score,4)
    acc_R2 = r2(y_pred, y_test)

    # Getting to original dataset
    y_abs_pred = y_pred*y_std + y_mean
    y_abs_test = y_test*y_std + y_mean


    err = abs(y_abs_test - y_abs_pred)

    # Percentage Error 
    met = np.abs(y_abs_test- y_abs_pred[:,0])*100/y_abs_pred[:,0]

    acc_mean = met.mean()
    acc_min = min(met)
    acc_max =  max(met)

    met.sort()
    acc_95 = met[int(len(met)*0.95)]

    nn_accuracy[models[i]] = {'mean%': round(acc_mean, 2),
                     'min %': round(acc_min, 2),
                     'max %': round(acc_max, 2),
                     '95 %tile': round(acc_95, 2)}

df1 = pd.DataFrame(nn_accuracy)
print('Saving NN results to csv')
df1.to_csv('nn.csv')

df = pd.DataFrame(forest_accuracy)
print('Saving random forest results to csv')
df.to_csv('forest.csv')