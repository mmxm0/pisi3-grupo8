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
    grafico_scatter(df)
    grafico_parallel_categories(df)

def grafico_barra(df):
    age_counts = df['age_upon_outcome'].value_counts().reset_index()
    age_counts.columns = ['Idade', 'Número de Animais']
    bar_fig = px.bar(age_counts, x='Idade', y='Número de Animais', labels={'Idade': 'Idade no Momento da Saída', 'Número de Animais': 'Número de Animais'})
    st.plotly_chart(bar_fig, use_container_width=True)

def grafico_boxplot(df):
    st.write('**Boxplot de Idade na Saída por Tipo de Saída (em anos)**')
    selected_category = st.selectbox('Selecione a variável categórica para análise', options=['outcome_type', 'animal_type'], key='category_boxplot')
    box_fig = px.box(df, x=selected_category, y='age_upon_outcome_(years)', points="all", labels={selected_category: selected_category, 'age_upon_outcome_(years)': 'Idade (anos)'})
    st.plotly_chart(box_fig, use_container_width=True)

def grafico_scatter(df):
    st.write('**Gráfico Scatter relativo à idade na saída por tempo total no abrigo**')
    fig = px.scatter(df, x = 'age_upon_outcome_(years)', y = 'time_in_shelter_days', 
                     color = 'animal_type', hover_data = ['breed'], 
                     labels = {'age_upon_outcome_(years)': 'Idade na Saída (anos)', 
                             'time_in_shelter_days': 'Tempo no Abrigo (dias)',
                             'animal_type': 'Animal',
                             'breed': 'Raça'})
    st.plotly_chart(fig, use_container_width=True)

def grafico_parallel_categories(df):
    st.write('**Gráfico de Categorias Paralelas relativo à idade na saída por tempo total no abrigo**')
    fig = px.parallel_categories(df, color='intake_month',
                                 dimensions=["animal_type","outcome_type", "sex_upon_outcome", "intake_condition", "intake_number"], 
                                 labels={'animal_type': 'Animal', 'outcome_type':'Tipo de Entrada', 'sex_upon_outcome':'Gênero', 'intake_condition':'Condição de Entrada', 'intake_number':'Nº de Entradas'},
                                 color_continuous_scale=px.colors.sequential.Inferno)
    st.plotly_chart(fig, use_container_width=True)

def grafico_bubble(df):
    st.write('**Gráfico Bubble relativo ao tempo total no abrigo por raça**')
    df_aux = df[['breed', 'time_in_shelter_days']].copy()
    df_aux = df_aux.groupby('breed').agg(
        qtd=('time_in_shelter_days', 'size'),
        avg_time=('time_in_shelter_days', 'mean')
    ).reset_index()

    
    df_aux = df_aux[df_aux['avg_time'] > df_aux['avg_time'].quantile(0.75)]  

    fig = px.scatter(df_aux, 
                     x='breed', 
                     y='avg_time', 
                     size='qtd', 
                     color='breed', 
                     size_max=60,
                     labels={'breed': 'Raça', 
                             'avg_time': 'Tempo Médio no Abrigo (dias)', 
                             'qtd': 'Quantidade de Animais'})

    fig.update_layout(
        xaxis_title="Raça",
        yaxis_title="Tempo Médio no Abrigo (dias)",
        xaxis={'categoryorder':'total descending', 'visible': False},
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)


build_page()
