import joblib

# Saving model
def save_model(model, path_output):
    joblib.dump(model, path_output)
    print('Model saved to %s' % (path_output))


# Loading model
def load_model(path_model):
    model = joblib.load(path_model)
    return model
