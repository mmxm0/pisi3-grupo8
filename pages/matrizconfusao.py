import streamlit as st
import plotly.express as px
import pandas as pd
from sklearn.metrics import confusion_matrix
from data import data_preparation
from sources.model_training import train_model_lr

def build_page():
    build_header()
    build_body()

def build_header():
    st.set_page_config(page_title="Matriz de Confusão", layout="wide")

def build_body():
    x_train, x_test, y_train, y_test = data_preparation.load_and_split_data()

    model = train_model_lr(x_train, y_train)

    y_pred = model.predict(x_test)

    cm = confusion_matrix(y_test, y_pred)

    cm_df = pd.DataFrame(cm, index=["Classe Verdadeira: 0", "Classe Verdadeira: 1"],
                         columns=["Classe Predita: 0", "Classe Predita: 1"])

    st.title('Matriz de Confusão')

    fig = px.imshow(cm_df, text_auto=True, color_continuous_scale='Blues')
    fig.update_layout(title='Matriz de Confusão', xaxis_title='Classe Predita', yaxis_title='Classe Verdadeira')

    st.plotly_chart(fig)

def main():
    build_page()

if __name__ == "__main__":
    main()
