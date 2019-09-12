# import nltk
import pandas as pd
import json
import numpy as np
import config
import os

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Creating Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# from keras.models import Sequential
# from keras import layers
# from keras.layers.normalization import BatchNormalization
from scipy import stats
from joblib import dump, load
import math


from src import get_processing
from src import get_interval
from src import io

# Reading dataset
path_json = config.path_data

try:
    df = pd.read_csv(path_json)
    print('Total Dataset available before removing duplicates : %d' % (len(df)))
except Exception as e:
    raise('Unable to load dataset. Error %s' % (e))

# Preprocessing dataset
try:
    df['km'] = df['Distance'].apply(lambda x: get_processing.convert_distance(x))
    df['model'] = df['Brand']+ ' ' + df['Model']
    df['model'] = df['model'].apply(lambda x: x.lower())
    df['year'] = df['Year'].apply(lambda x: float(x))
    df['price'] = df['Price'].apply(lambda x: get_processing.convert_price(x))
    df['price'] = df['price'].apply(lambda x: float(x))

except Exception as e:
    raise('Unable to preprocess dataset. Error %s' % (e))

# Droping duplicates
df = df.drop_duplicates()
print('Total Dataset available after removing duplicates : %d' % (len(df)))

# Creating a count column for EDA
df['count'] = 1

# 4. Creating delta year wrt 2019
# df['delta_year'] = 2019 - df['year']
df["delta_year"] = df["year"].apply(lambda x: config.base_year - x)
df["model"] = df["model"].apply(lambda x: x.lower())

# Collecting different car models
model = df.groupby('model').agg({'count': 'sum'}).reset_index()
model = model.sort_values('count').reset_index(drop=True)
print('Number of model : %d' % (len(model)))

# Copying data
data = df.copy()

# Removing < 1%tile and >99% data to remove outliers in selling prices

print('Length of data before outlier removal: %d' % (len(data)))

data = data[data.price > data.price.quantile(config.lower)]
data = data[data.price < data.price.quantile(config.upper)]

print('Length of data after outlier removal: %d' % (len(data)))

# List of car models availables with atleast 200 data points
models = list(model.model[model['count'] > config.minimum_datapoint])
# models = ['honda crv (2012-2017) 2.4 i-vtec at']

# Variables initialization
forest_accuracy = {}
nn_accuracy = {}
cars = []

# Iterating to all the models
for i in range(len(models)):
    print('Running %s model' % (models[i]))

    # Taking one particular model
    car_model = data[data.model == models[i]].reset_index(drop=True)
    
    # Creating a new KPI, Delta_year * KM
    car_model['delta_year_km'] = car_model['km'] * car_model['delta_year']

    # Extracting x and y values
    y = car_model.price
    y = y.values

    # Creating a copy of dataset to get ready for training
    x = car_model[['km', 'delta_year', 'delta_year_km']].copy()

    # Performing train test split
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=config.test_size, random_state=config.random_state)

    # Initializing Random Forest Pipeline model
    pkl = {}

    forest = []
    forest = make_pipeline(StandardScaler(), RandomForestRegressor(
        max_depth=config.max_depth, random_state=config.random_state, n_estimators=config.n_estimators))

    # Fitting model
    forest.fit(x_train, y_train)

    # Predicting on validation dataset
    y_pred = forest.predict(x_test)
    score = math.sqrt(mean_squared_error(y_pred, y_test))

    # Training data MSE
    y_train_pred = forest.predict(x_train)
    mse = mean_squared_error(y_train_pred, y_train)

    # Accuracy Metrics
    acc_RMSE = round(score, 4)
    acc_R2 = get_processing.r2(y_pred, y_test)

    # Percentage Error
    met = abs(y_test - y_pred)*100/y_pred

    acc_mean = met.mean()
    acc_min = min(met)
    acc_max = max(met)

    met.sort()
    acc_95 = met[int(len(met)*config.confidence_prediction_interval/100)]

    forest_accuracy[models[i]] = {'mean_%': round(acc_mean, 2),
                                  'min_%': round(acc_min, 2),
                                  'max_%': round(acc_max, 2),
                                  '%s_percentile' % (config.confidence_prediction_interval): round(acc_95, 2)
                                  }

    # Creating PKL file
    pkl['model'] = forest
    pkl['y_mean'] = y_train.mean()
    pkl['x_mean'] = x_train.mean(axis=0).values
    pkl['x_mse'] = np.sum(np.square(x_train - x_train.mean(axis=0)).values)
    pkl['y_mse'] = mse
    pkl['length'] = len(y_train)

    print(pkl)

    predict = get_interval.prediction_interval(n=len(x_train), x_train_mean=pkl['x_mean'], y_train_mean=pkl['y_mean'],
                                           x_test=x_test, model=forest, y_mse=pkl['y_mse'], x_mse=pkl['x_mse'],
                                           confidence=config.confidence_prediction_interval)

    predict['model'] = models[i]
    cars.append(predict)

    # Saving Trained models
    path_output = os.path.join(
        config.path_pretrained_model, models[i].replace('/', ''))

    # Check if exist or not
    if not os.path.exists(path_output):
        os.makedirs(path_output)

    path_output = os.path.join(path_output, 'forest_' +
                               str(config.version) + '.pkl')
    io.save_model(pkl, path_output)

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

# Creating dataframe for accuracy assessment
df = pd.DataFrame(forest_accuracy)

# Saving results to csv
print('Saving random forest results to csv')
df.to_csv(os.path.join(config.path_output,
                       'forest_' + str(config.version) + '.csv'))

# Saving prediction results for different models
data = pd.concat(cars)
data = data.reset_index(drop=True)
data.to_csv(os.path.join(config.path_output,
                         'predict_' + str(config.version) + '_.csv'))

print('Successfully Completed')
