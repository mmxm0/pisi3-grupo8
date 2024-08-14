import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import read_df
from data import data_preparation
from sources.model_training import *
from sources.model_evaluation import *

x_train, x_test, y_train, y_test = data_preparation.load_and_split_data()

model = train_model_lr(x_train, y_train)

#save_model(model, 'models/logistic_regression_model.pkl')

accuracy = evaluate_model(model, x_test, y_test)
report = evaluate_classification_report(model, x_test, y_test)
print(f"Acurácia do modelo: {accuracy:.2f}")
print(f"Classificação de métricas precisão, recall e F1-score: \n{report}")


x_train_tree, x_test_tree, y_train_tree, y_test_tree = data_preparation.load_and_split_data_bi()

tree_model= train_random_forest(x_train_tree, y_train_tree)

y_pred_tree = tree_model.predict(y_test_tree)
print("Acurácia da Árvore de Decisão:", accuracy_score(y_test_tree, y_pred_tree))
print(classification_report(y_test_tree, y_pred_tree))

'''
x_train_neural, x_test_neural, y_train_tree, y_test_tree = data_preparation.load_and_split_data()

tree_model= train_random_forest(x_train_tree, y_train_tree)

y_pred_tree = tree_model.predict(y_test_tree)
print("Acurácia da Árvore de Decisão:", accuracy_score(y_test_tree, y_pred_tree))
print(classification_report(y_test_tree, y_pred_tree))
'''
