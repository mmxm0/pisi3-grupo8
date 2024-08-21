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

x_train_svm, x_test_svm, y_train_svm, y_test_svm = data_preparation.load_and_split_data()
svm_model = train_svm(x_train=x_train_svm, y_train=y_train_svm)

y_pred = svm_model.predict(x_test_svm)

y_prob = svm_model.predict_proba(x_test_svm)


# Relatório de classificação
print("Relatório de Classificação:")
print(classification_report(y_test, y_pred))
