from sklearn.linear_model import LogisticRegression
import pickle
import os

def train_model(x_train, y_train, solver='lbfgs'):
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


        