from joblib import dump, load

# Saving model
def save_model(model, path_output):
    dump(model, path_output)
    print('Model saved to %s' % (path_output))