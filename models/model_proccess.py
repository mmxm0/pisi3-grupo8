import sys
import os


# Adiciona o diretório raiz do projeto ao PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import read_df
from data import data_preparation
from sources.model_training import train_model, save_model
from sources.model_evaluation import evaluate_model

'''x_train, x_test, y_train, y_test = data_preparation.load_and_split_data()

model = train_model(x_train, y_train)

save_model(model, 'models/logistic_regression_model.pkl')

accuracy = evaluate_model(model, x_test, y_test)
print(f"Acurácia do modelo: {accuracy:.2f}")'''

data_preparation.calculate_description_statistics()