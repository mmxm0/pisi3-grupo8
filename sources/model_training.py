from sklearn.linear_model import LogisticRegression
import pickle
import os

def train_model(x_train, y_train):
    model = LogisticRegression()
    model.fit(x_train, y_train)
    return model

def save_model(model, file_path):
    models_dir = 'models'
    file_path = os.path.join(models_dir, 'logistic_regression_model.pkl')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)