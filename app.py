from flask import Flask, request, jsonify, Response, send_file
import base64
from requests.utils import requote_uri
"""app.main: handle request for lambda-tiler"""

import re
import json
import os
import joblib

import numpy as np
from flask_compress import Compress
from flask_cors import CORS

import time
import gzip

import config
from src import get_exception
from src import get_processing
from src import get_interval
# from lambda_proxy.proxy import API


class TilerError(Exception):
    """Base exception class."""


app = Flask(__name__)
cors = CORS(app)
app.config['COMPRESS_LEVEL'] = 9
app.config['COMPRESS_MIN_SIZE'] = 0
Compress(app)


# Loading pretrained models
dir = []
for root, dirs, files in os.walk(config.path_pretrained_model):
    if dirs:
        dir.append(dirs)

models = dir[0]
print('List of pretrained models : ', models)

pretrained = {}

# Loading models
for i in range(len(models)):
    path_pkl = os.path.join(config.path_pretrained_model, models[i], 'forest_%s' % (config.version) + '.pkl')
    try:
        model = joblib.load(path_pkl)
        pretrained[models[i]] = model

    except Exception as e:
        raise('Unable to load model from %s. Error: %s' % (path_pkl, e))

print('Models Loaded!')

# Welcome page
@app.route('/')
def hello():
    return "Welcome to, Price Prediction API!"

@app.route('/api/v1/')
def hello_version():
    return "Welcome to, Price Prediction API v1!"

# Generates bounds of Raster data
@app.route('/api/v1/price', methods=['GET'])
def price():

    # Handling input parameters
    """Handle price requests."""
    company = request.args.get('company', default=None)
    model = request.args.get('model')
    km_driven = request.args.get('km_driven')
    year = request.args.get('year')
    confidence = request.args.get('confidence', default=90, type=float)

    if company is None or model is None or km_driven is None or year is None:
        msg = 'Parameter not specified. Make sure you have given all (company, model, km_driven. year, confidence=90) the parameters'
        return get_exception.general(msg)
    
    company = company.split(',')
    model = model.split(',')

    km_driven = np.array(km_driven.split(','), dtype=float)
    year = np.array(year.split(','), dtype=float)
 
    # Checking with length of parameters
    if not len(company) == len(model) == len(km_driven) == len(year):
        msg = 'Length of parameters not equal'
        return get_exception.general(msg)

    info = {
        'length': len(km_driven),
        'data': []
    }
    # Interating throughtout the model
    for i in range(len(km_driven)):
        # Preprocessing inputs

        # Removing leading and ending spaces
        company[i] = company[i].strip()
        model[i] = model[i].strip()

        car_model = company[i] + ' ' + model[i]
        
        # Preprocessing model
        car_model = car_model.lower()
        # Removing Leading and ending spaces
        car_model = car_model.strip()
        
        delta_year = 2019 - year[i]

        # Checking if car model exist in our list
        print('model: ', car_model)
        if car_model not in pretrained:
            return get_exception.general('Model given was not Found. %s from %s' %(car_model, models))

        # Loading pretraining model
        print('Loading Model')
        try:
            forest = pretrained[car_model]['model']
        except Exception as e:
            msg = 'Unable to load pretrained model'
            return get_exception.general(msg)
        
        x_predict = np.array([km_driven[i], delta_year, km_driven[i]*delta_year])
        
        x_predict = x_predict.reshape(1, -1)

        try:
            y_predict = forest.predict(x_predict)
        except Exception as e:
            msg = 'Error: While predicting from model. %s' %(e)
            return get_exception.general(msg)
        
        x_mean = pretrained[car_model]['x_mean']
        y_mean = pretrained[car_model]['y_mean']
        x_mse = pretrained[car_model]['x_mse']
        y_mse = pretrained[car_model]['y_mse']
        n = pretrained[car_model]['length']

        # try:
        #     x_predict = get_interval.prediction_interval(n, x_mean, y_mean, x_predict, forest,
        #                                                 y_mse, x_mse, confidence=confidence)
        # except Exception as e:
        #     msg = 'Error: Unable to get prediction interval. %s' %(e)
        #     return get_exception.general(msg)
        info['data'].append(round(y_predict[0]))
    
    return (jsonify(info))


@app.route('/api/v1/favicon.ico', methods=['GET'])
def favicon():
    """Favicon."""
    output = {}
    output['status'] = '205'  # Not OK
    output['type'] = 'text/plain'
    output['data'] = ''
    return (json.dumps(output))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000, debug=True)
