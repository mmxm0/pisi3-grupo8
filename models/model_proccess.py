import sys
import os
# Adiciona o diretório raiz do projeto ao PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import read_df
from data import data_preparation
from sources.model_training import train_model, save_model
from sources.model_evaluation import *

x_train, x_test, y_train, y_test = data_preparation.load_and_split_data()

model = train_model(x_train, y_train)

save_model(model, 'models/logistic_regression_model.pkl')

accuracy = evaluate_model(model, x_test, y_test)
report = evaluate_classification_report(model, x_test, y_test)
# print(f"Acurácia do modelo: {accuracy:.2f}")
# print(f"Classificação de métricas precisão, recall e F1-score: \n{report}")

solvers = ['liblinear', 'newton-cg', 'lbfgs']

for s in solvers:
    models = train_model(x_train, y_train, s)
    save_model(model, f'logistic_regression_model_{s}.pkl')
    y_pred = models.predict(x_test)
    print(evaluate_multiple_solvers(y_test, y_pred, s))
    print('\n')