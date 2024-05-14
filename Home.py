import streamlit as st
import pandas as pd
import plotly.express as px
from pages.util.pages_util import read_ACC_df, build_dataframe_section, dicionario_acc

st.set_page_config(
    page_title="PISI3 - Grupo 8",
    layout="wide"
)

def build_page():
    build_header()
    build_body()

def build_header():
    text ='<h1>Projetos 3 - Grupo 8</h1>' + \
       '<p>Este projeto visa realizar análises utilizando os dados disponibilizados pelo Centro Animal de Austin. '+ \
       'Os dados foram obtidos a partir do conjunto de dados disponível no ' + \
       '<a href="https://www.kaggle.com/datasets/aaronschlegel/austin-animal-center-shelter-intakes-and-outcomes?select=aac_intakes_outcomes.csv" target="_blank">Kaggle</a>.</p>'
    st.write(text, unsafe_allow_html=True)

def build_body():
    dicionario_acc()
    df = read_ACC_df()
    build_dataframe_section(df)

build_page()