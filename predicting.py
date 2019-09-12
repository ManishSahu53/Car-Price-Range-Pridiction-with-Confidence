import pandas as pd
from sklearn.externals import joblib
import config
import argparse
import numpy as np
import os

from src import get_processing
from src import io
from src import get_interval

# Arg Parser
parser = argparse.ArgumentParser(
    description='See description below to see all available options')

parser.add_argument('-i', '--input',
                    help='Input CSV file containing data points',
                    required=True)

parser.add_argument('-m', '--model',
                    help='Path directory of pretrained data car model wise',
                    default='results/pretrained_model',
                    required=True)

parser.add_argument('-o', '--output',
                    help='Output CSV file. [Default] = Data written in input file itself',
                    default=None,
                    required=False)

# Parsing arguments
args = parser.parse_args()
path_data = args.input
path_output = args.output
path_model = args.model

# path_data = 'data/old.csv'
# path_output = 'data/old_re.csv'
# path_model = 'results/pretrained_model'

# Loading dataset
data = pd.read_csv(path_data)

if path_output is None:
    path_output = path_data


# Preprocessing dataset
if 'km' in data.columns:
    print('km column found. Converting to Float datatype')
    # data['km'] = data['km'].apply(lambda x: str(x))
    # data['km'] = data['km'].apply(lambda x: processing.removedot(x))
    data['km'] = data['km'].apply(lambda x: float(x))
else:
    raise('Unable to get km column from CSV')

if 'year' in data.columns:
    print('year column found')
    data['year'] = data['year'].apply(lambda x: float(x))
else:
    raise('Unable to get year column from CSV')

if 'model' in data.columns:
    print('model column found')
    data['model'] = data['model'].apply(lambda x: x.lower())
    data['model'] = data['model'].apply(lambda x: x.replace('/', ''))

else:
    raise('Unable to get model column from CSV')


# Adding delta year column
data["delta_year"] = data["year"].apply(lambda x: config.base_year - x)

# Loading pretrained models
dir = []
for root, dirs, files in os.walk(path_model):
    if dirs:
        dir.append(dirs)

models = dir[0]
print('List of pretrained models : ', models)

prediction = []

for i in range(len(models)):
    print('Running %s model' % (models[i]))

    # Taking one particular model
    car_model = data[data.model == models[i]].reset_index(drop=True)

    if car_model.empty:
        print('Model data %s not found! Skipping this model' %(models[i]))
        continue
    else:
        print('Model data %s found' %(models[i]))
        # Loading pickle file
        path_pkl = os.path.join(
            path_model, models[i], 'forest_%s' % (config.version) + '.pkl')

        try:
            model = joblib.load(path_pkl)
        except Exception as e:
            raise('Unable to load model from %s. Error: %s' % (path_pkl, e))

        forest = model['model']
        x_predict = car_model[['km', 'delta_year']].copy()
        y_predict = forest.predict(x_predict)

        x_mean = model['x_mean']
        y_mean = model['y_mean']
        x_mse = model['x_mse']
        y_mse = model['y_mse']
        n = model['length']

        x_predict = get_interval.prediction_interval(n, x_mean, y_mean, x_predict, forest,
                                                 y_mse, x_mse, confidence=config.confidence_prediction_interval)
        prediction.append(x_predict)

if prediction:
    output = pd.concat(prediction)
    output['price'] = data.price

    output = output.reset_index(drop=True)
    output.to_csv(path_output)
else:
    print('No Suitable model found')