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
    grafico_barra(df)
    grafico_boxplot(df)
    grafico_barras(df)
    grafico_histograma(df)

def grafico_barra(df):
    st.write('**Gráfico relativos à idade do animal no momento de saída do abrigo**')
    age_counts = df['age_upon_outcome'].value_counts().reset_index()
    age_counts.columns = ['Idade', 'Número de Animais']

    limite_relevancia = 200 
    age_counts_filtrado = age_counts[age_counts['Número de Animais'] > limite_relevancia]

    ordem_idades = ['1 month', '2 months', '3 months', '4 months', '5 months', '6 months', '7 months', '8 months', '9 months', '10 months', '11 months', '12 months']
    ordem_idades = [idade for idade in ordem_idades if idade in age_counts_filtrado['Idade'].values]

    idades_selecionadas = st.multiselect(
        'Selecione as Idades que deseja analisar:',
        options=ordem_idades, default=ordem_idades)
    
    age_counts_filtrado = age_counts_filtrado[age_counts_filtrado['Idade'].isin(idades_selecionadas)]
    age_counts_filtrado['Idade'] = pd.Categorical(age_counts_filtrado['Idade'], categories=ordem_idades, ordered=True)
    age_counts_filtrado = age_counts_filtrado.sort_values('Idade')

    bar_fig = px.bar(
        age_counts_filtrado, 
        x='Número de Animais', 
        y='Idade', 
        orientation='h', 
        labels={'Idade': 'Idade no Momento da Saída', 'Número de Animais': 'Número de Animais'})

    st.plotly_chart(bar_fig, use_container_width=True)

def grafico_boxplot(df):
    st.write('**Boxplot de Idade (em anos)**')
    selected_category = st.selectbox('Selecione a categória para análise', options=['outcome_type', 'animal_type'], key='category_boxplot')
    box_fig = px.box(df, x=selected_category, y='age_upon_outcome_(years)', points="all", labels={selected_category: selected_category, 'age_upon_outcome_(years)': 'Idade (anos)'})
    st.plotly_chart(box_fig, use_container_width=True)

def grafico_barras(df):
    st.write('**Gráfico de Barras: Comparação dos Tipos de Saída por Tipo de Animal**')
    animal_tipo_selecionado = st.selectbox(
        'Selecione o Tipo de Animal para análise:',
        options=df['animal_type'].unique())
    df_filtrado = df[df['animal_type'] == animal_tipo_selecionado]
    barras_data = df_filtrado.groupby(['animal_type', 'outcome_type']).size().reset_index(name='count')
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

    tipo_saida_selecionado = st.selectbox(
        'Selecione o Tipo de Saída para análise:',
        options=df['outcome_type'].unique())
    
    df_filtrado = df[df['outcome_type'] == tipo_saida_selecionado]
    bar_fig = px.histogram(df_filtrado, 
                           x='intake_condition', 
                           color='outcome_type',
                           labels={'intake_condition': 'Condição na Entrada', 
                                   'outcome_type': 'Tipo de Saída',
                                   'count': 'Quantidade'},
                           barmode='group')
    
    st.plotly_chart(bar_fig, use_container_width=True)

build_page()