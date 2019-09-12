import pandas as pd

def percentage_accuracy(path_prediction, percentage=10):
    try:
        data = pd.read_csv(path_prediction)
    except Exception as e:
        raise('Error: Unable to load prediction csv. %s' %(e))
    
    data['per_lower'] = data.target*(1 - percentage/100)
    data['per_upper'] = data.target*(1 + percentage/100)
    
    data['status'] = (data.price > data.per_lower) & (data.price < data.per_upper)
    
    n = len(data)
    overall_accuracy = len(data[data.status==True]) *100 / n
    return overall_accuracy