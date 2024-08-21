import plotly.express as px
import streamlit as st
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from data import data_preparation
from pages.util.pages_util import build_dataframe_section
from imblearn.over_sampling import SMOTE

# Título da página
st.write("Classificação do modelo Ensemble", unsafe_allow_html=True)

x_train, x_test, y_train, y_test = data_preparation.load_and_split_data()

# Modelos base
model_svm = SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced')
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')

# Ensemble: Voting Classifier
ensemble_model = VotingClassifier(estimators=[
    ('svm', model_svm),
    ('rf', model_rf),
    ('lr', model_lr)
], voting='soft') 

# Treinamento do modelo Ensemble
ensemble_model.fit(x_train, y_train)

# Previsões
y_pred = ensemble_model.predict(x_test)

# Geração da matriz de confusão
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=["Classe Verdadeira: 0", "Classe Verdadeira: 1"],
                     columns=["Classe Predita: 0", "Classe Predita: 1"])

st.title('Matriz de Confusão - Ensemble')
fig = px.imshow(cm_df, text_auto=True, color_continuous_scale='Blues')
fig.update_layout(title='Matriz de Confusão', xaxis_title='Classe Predita', yaxis_title='Classe Verdadeira')
st.plotly_chart(fig)

# Geração do classification_report
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

# Exibição do classification_report
st.subheader("Relatório de Classificação Ensemble")
st.dataframe(report_df)

# Verificação e insights sobre o modelo Ensemble
st.subheader("Insights sobre o modelo Ensemble")

# Nome exato da classe a ser prevista no relatório
class_name = "outcome_type_Adoption       "  # Incluindo os espaços no final

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
