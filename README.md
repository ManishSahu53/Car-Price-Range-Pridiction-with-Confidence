# car_prediction
This repo is used to predict second hand car prices. It is developed in Python3 env.


## Model Description
Car Prediction model is used to predict prices of old cars based on
parameters like : Car model and company, KM driven, Number of years old.

We estimate the price using random forest regressor algorithm. We have testing linear regression also but since random forest regressor gave us better result in validation dataset, we went ahead with it.

Also Random Forest regressor generalizes well and reduces overfitting possibilities because of it is ensemble of multiple decision trees with sampling from input dataset.

### **Input dataset**
There are two datasources from which data is scrapped.
1. Olx (Used for training model)
2. Oto (Used for testing model accuracy)

Since OLX contains larger repository of dataset across muliple car brands and models, we decided to use it for training purpose.

#### **Assumptions**
1. We have taken an assumption that price quote by seller in Olx is true price of the used car. It might be possible that the car is beloved to the owner and he has quoted much higher price for his car.
2. Due to above reason, we are removing top and bottom 1% of datapoints assuming them to be an outlier.
3. Olx dataset is good representation sample of the population we are trying to monitor 
4. Variant of particular car model (Example- Maruki Suzuki Ertiga L, Maruti Suzuki Ertiga M, Maruti Suzuki Ertiga AC etc. are assumed to have similar reselling prices and a single ML model. We are invariant to Car Model Variant.
5. There will be separate ML model for each car type. This ML model is responsible to predict price of a particular car model.



### **Model Parameters**
All the parameters used in this model are given in [config.py](config.py). 
* **Forest** - Number of trees in forest are taken as 100, and max_depth
* **Base** - Base year is assumed to be 2019. Number of years old is calculated by subtracting 2019 from the year.
* **Outlier** - Top and Bottom 1% of the training dataset is assumed to be outliers.
* **Validation** - Training dataset is divided into 75-25 as training and validating dataset.
* **Trained Weights** - All the weights trained are saved in [pretrained_weights](results/pretrained_model/)
* **Accuracy** - Accuracy is reported as if ML model is able to predict model in a given price range.


### **Results**
All the results are given in [docs](https://docs.google.com/document/d/1XG84zG2qVd0ufv2gSTpq3s0hvim7qRiubw-NI6mEmSM/edit) here


## How to Train
Tweek config.py to training dataset and output parameters. THen simply run.

```
python training.py
```

## How to Test
```
python predicting.py -i testing.csv -m results/pretrained
```

## How to use it as API
API is containerized using docker. To run docker simply do following
1. Build Image - docker build -t car_prediction .
2. RUN image - docker run -p 4000:4000 -d {image_id}
3. PostMan - Hit API in postman with 

>http://0.0.0.0:4000/api/v1/price?company=xyz&model=xyz&km_driven=123&year=2018


## References
1. https://gdcoder.com/random-forest-regressor-explained-in-depth/
2. https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
3. https://www.real-statistics.com/regression/confidence-and-prediction-intervals/