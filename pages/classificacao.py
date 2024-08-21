import plotly.express as px
import streamlit as st
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

from data import data_preparation
from pages.util.pages_util import build_dataframe_section
from sources.model_training import train_svm
from imblearn.over_sampling import SMOTE

# Título da página
st.write("Classificação do modelo SVM", unsafe_allow_html=True)

# Carregamento e preparação dos dados
x_train_svm, x_test_svm, y_train_svm, y_test_svm = data_preparation.load_and_split_data()

# Treinamento do modelo SVM
svm_model = train_svm(x_train=x_train_svm, y_train=y_train_svm)

# Previsões
y_pred = svm_model.predict(x_test_svm)

# Geração da matriz de confusão
cm = confusion_matrix(y_test_svm, y_pred)
cm_df = pd.DataFrame(cm, index=["Classe Verdadeira: 0", "Classe Verdadeira: 1"],
                     columns=["Classe Predita: 0", "Classe Predita: 1"])

# Exibição da matriz de confusão na interface
st.title('Matriz de Confusão - SVM')
fig = px.imshow(cm_df, text_auto=True, color_continuous_scale='Blues')
fig.update_layout(title='Matriz de Confusão', xaxis_title='Classe Predita', yaxis_title='Classe Verdadeira')
st.plotly_chart(fig)

# Nome exato da classe a ser prevista no relatório
class_name = "outcome_type_Adoption       "  # Incluindo os espaços no final

# Geração do classification_report
report = classification_report(y_test_svm, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

# Exibição do classification_report
st.subheader("Relatório de Classificação SVM")
st.dataframe(report_df)

# Verificação e insights sobre o modelo SVM
st.subheader("Insights sobre o modelo SVM")

# Verifique se a classe 'outcome_type_Adoption' está presente no relatório
if class_name in report_df.index:
    st.markdown(f"**Precision (Precisão):** A precisão para a classe '{class_name.strip()}' (Adoção) é {report_df.loc[class_name, 'precision']:.2f}, "
                f"o que significa que {report_df.loc[class_name, 'precision'] * 100:.2f}% das previsões para a adoção estão corretas.")
    st.markdown(f"**Recall (Sensibilidade):** O recall para a classe '{class_name.strip()}' (Adoção) é {report_df.loc[class_name, 'recall']:.2f}, "
                f"indicando que {report_df.loc[class_name, 'recall'] * 100:.2f}% das instâncias de adoção foram corretamente identificadas.")
    st.markdown(f"**F1-Score:** O F1-Score para a classe '{class_name.strip()}' (Adoção) é {report_df.loc[class_name, 'f1-score']:.2f}. "
                f"Este valor reflete o equilíbrio entre precisão e recall para essa classe.")
else:
    st.markdown(f"**Atenção:** A classe '{class_name.strip()}' (Adoção) não foi prevista pelo modelo ou não está presente no conjunto de teste.")

# Sempre mostre a acurácia geral
st.markdown(f"**Acurácia:** A acurácia geral do modelo é {report_df.loc['accuracy', 'f1-score']:.2f}, "
            "o que representa a proporção total de previsões corretas.")
