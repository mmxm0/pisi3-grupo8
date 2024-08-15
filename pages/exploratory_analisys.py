import streamlit as st
import pandas as pd
import plotly.express as px
from pages.util.pages_util import read_ACC_df, build_dataframe_section, dicionario_acc


def build_page():
    build_header()
    build_body()

def build_header():
    text ='<h1>Análises Exploratórias</h1>'+\
    '<p>Esta página apresenta alguns gráficos a partir da base de dados fornecida pelo Centro Animal de Austin. '+\
    '(https://www.kaggle.com/datasets/aaronschlegel/austin-animal-center-shelter-intakes-and-outcomes?select=aac_intakes_outcomes.csv).</p>'
    st.write(text, unsafe_allow_html=True)

def build_body():
    df = read_ACC_df()
    st.markdown('<h2>Gráficos relativos à idade do animal no momento de saída do abrigo</h2>', unsafe_allow_html=True)
    grafico_barra(df)
    grafico_boxplot(df)
    grafico_barras(df)
    grafico_histograma(df)

def grafico_barra(df):
    age_counts = df['age_upon_outcome'].value_counts().reset_index()
    age_counts.columns = ['Idade', 'Número de Animais']
    bar_fig = px.bar(age_counts, x='Idade', y='Número de Animais', labels={'Idade': 'Idade no Momento da Saída', 'Número de Animais': 'Número de Animais'})
    st.plotly_chart(bar_fig, use_container_width=True)

def grafico_boxplot(df):
    st.write('**Boxplot de Idade (em anos)**')
    selected_category = st.selectbox('Selecione a categória para análise', options=['outcome_type', 'animal_type'], key='category_boxplot')
    box_fig = px.box(df, x=selected_category, y='age_upon_outcome_(years)', points="all", labels={selected_category: selected_category, 'age_upon_outcome_(years)': 'Idade (anos)'})
    st.plotly_chart(box_fig, use_container_width=True)

def grafico_barras(df):
    st.write('**Gráfico de Barras: Comparação dos Tipos de Saída por Tipo de Animal**')
    barras_data = df.groupby(['animal_type', 'outcome_type']).size().reset_index(name='count')
    bar_fig = px.bar(barras_data, 
                     x='animal_type', 
                     y='count', 
                     color='outcome_type', 
                     barmode='group',
                     labels={'animal_type': 'Tipo de Animal', 
                             'count': 'Quantidade', 
                             'outcome_type': 'Tipo de Saída'})
    st.plotly_chart(bar_fig, use_container_width=True)

def grafico_histograma(df):
    st.write('**Gráfico Histograma: Análise do Tipo de Saída por Condição de Entrada**')
    
    bar_fig = px.histogram(df, 
                           x='intake_condition', 
                           color='outcome_type',
                           labels={'intake_condition': 'Condição na Entrada', 
                                   'outcome_type': 'Tipo de Saída',
                                   'count': 'Quantidade'},
                           barmode='group')  # Agrupa as barras
    
    st.plotly_chart(bar_fig, use_container_width=True)

build_page()