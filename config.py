import os

# Data source
path_data = 'data/oto_car_json.json'

# Minimum data points to the model to take
minimum_datapoint = 200

# Version
version = 0.2

# States
random_state = 10
test_size = 0.25
base_year = 2019

# Prediction interval
confidence_prediction_interval = 90

# Outlier removal
lower = 0.01
upper = 0.99


# Random Forest Parameters
max_depth = 10
n_estimators = 100

# Results path
path_output = 'results'
path_pretrained_model = os.path.join(path_output, 'pretrained_model')