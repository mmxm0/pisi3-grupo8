from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pickle
import os

from sklearn.neural_network import MLPClassifier

def train_model_lr(x_train, y_train, solver='lbfgs'):
    model = LogisticRegression(solver=solver, max_iter=802, random_state=42)
    model.fit(x_train, y_train)
    return model

def save_model(model, filename):
    models_dir = 'models'
    file_path = os.path.join(models_dir, filename)
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)

def train_random_forest(x_train_scaled, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(x_train_scaled, y_train)
    
    return model

def train_mpl_classifier(x_train_scaled, y_train):
    model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    model.fit(x_train_scaled, y_train)

    return model