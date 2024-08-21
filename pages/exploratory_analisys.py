import streamlit as st
import pandas as pd
import plotly.express as px
from pages.util.pages_util import read_ACC_df, build_dataframe_section, dicionario_acc
import numpy as np

def build_page():
    build_header()
    build_body()

def build_header():
    text = '<h1>Análises Exploratórias</h1>' + \
           '<p>Esta página apresenta alguns gráficos a partir da base de dados fornecida pelo Centro Animal de Austin. ' + \
           '(https://www.kaggle.com/datasets/aaronschlegel/austin-animal-center-shelter-intakes-and-outcomes?select=aac_intakes_outcomes.csv).</p>'
    st.write(text, unsafe_allow_html=True)

def build_body():
    df = read_ACC_df()
    grafico_saida_animal(df)
    st.write("")
    grafico_saida_entrada(df)
    st.write("")
    grafico_pizza(df)
    st.write("")
    grafico_barra_por_idade(df)
    
    st.write("")
    grafico_racas_adocao(df) 

import streamlit as st
import plotly.express as px

def grafico_racas_adocao(df):
    st.write('**Análise do tempo total no abrigo por raça**')

    # Adicionar filtro para selecionar o tipo de animal
    animal_filter = st.selectbox('Selecione o Tipo de Animal:', ('Cachorro', 'Gato'))

    # Filtrar o DataFrame com base no tipo de animal selecionado
    if animal_filter == 'Cachorro':
        df = df[df['animal_type'].str.lower() == 'dog']
    elif animal_filter == 'Gato':
        df = df[df['animal_type'].str.lower() == 'cat']

    # Adicionar filtro para selecionar entre as raças mais e menos adotadas
    opcao = st.selectbox('Selecione uma opção:', ('5 Raças Mais Adotadas', '5 Raças Menos Adotadas'))

    # Adicionar opção para selecionar o tipo de gráfico
    tipo_grafico = st.selectbox('Selecione o Tipo de Gráfico:', ('Barra', 'Pizza'))

    df_aux = df[['breed', 'time_in_shelter_days']].copy()
    df_aux = df_aux.groupby('breed').agg(
        qtd=('time_in_shelter_days', 'size'),
        avg_time=('time_in_shelter_days', 'mean')
    ).reset_index()

    # Converter o tempo médio de dias para anos
    df_aux['avg_time_years'] = df_aux['avg_time'] / 365

    if opcao == '5 Raças Mais Adotadas':
        df_filtered = df_aux.nlargest(5, 'qtd')
    else:
        df_filtered = df_aux.nsmallest(5, 'qtd')

    if tipo_grafico == 'Barra':
        fig = px.histogram(df_filtered, 
                           x='breed', 
                           y='avg_time_years', 
                           nbins=5,
                           labels={'breed': 'Raça', 
                                   'avg_time_years': 'Tempo Médio no Abrigo (anos)'})
        fig.update_layout(
            xaxis_title="Raça",
            yaxis_title="Tempo Médio no Abrigo (anos)",
            xaxis={'categoryorder':'total descending'},
            showlegend=False
        )

    else:  # Tipo de gráfico Pizza
        fig = px.pie(df_filtered, 
                     names='breed', 
                     values='avg_time_years',
                     labels={'breed': 'Raça', 
                             'avg_time_years': 'Tempo Médio no Abrigo (anos)'},
                     title='Distribuição do Tempo Médio no Abrigo por Raça')

    st.plotly_chart(fig, use_container_width=True)

def grafico_barra_por_idade(df):
    st.write('**Distribuição de Animais por Idade no Abrigo**')

    # Criar uma coluna para categorizar os tipos de animais
    df['categoria_animal'] = df['animal_type'].apply(lambda x: 'Cachorro' if x.lower() == 'dog' 
                                                    else 'Gato' if x.lower() == 'cat' 
                                                    else 'Pássaro' if x.lower() == 'bird' 
                                                    else 'Outros Animais')

    # Adicionar um filtro para selecionar a categoria de animal
    categorias_animais = ['Cachorro', 'Gato', 'Pássaro', 'Outros Animais']
    categoria_selecionada = st.selectbox('Selecione o Tipo de Animal para análise:', categorias_animais, key='tipo_animal_barra')

    # Filtrar o DataFrame pela categoria de animal selecionada
    df_filtrado = df[df['categoria_animal'] == categoria_selecionada]

    # Filtrar os dados para idade até 10 anos
    df_filtrado = df_filtrado[df_filtrado['age_upon_outcome_(years)'] <= 10]

    # Definir os bins para 0-1, 1-2, ..., 9-10
    bins = np.arange(0, 11, 1)
    
    # Criar os grupos de idade e adicionar uma coluna para as labels
    df_filtrado['age_group'] = pd.cut(df_filtrado['age_upon_outcome_(years)'], bins=bins, include_lowest=True, right=False)
    age_groups = df_filtrado.groupby('age_group').size().reset_index(name='Quantidade')
    age_groups['Idade'] = age_groups['age_group'].apply(lambda x: f'{int(x.left)}-{int(x.right)}')

    # Criar o gráfico de barras
    bar_fig = px.bar(
        age_groups,
        x='Idade',
        y='Quantidade',
        labels={'Idade': 'Idade (anos)', 'Quantidade': 'Quantidade'},
        title=f"Distribuição de Animais por Idade no Abrigo ({categoria_selecionada})"
    )

    # Configurar o eixo X para exibir as categorias na ordem correta
    bar_fig.update_xaxes(type='category', categoryorder='category ascending')
    bar_fig.update_yaxes(title_text="Quantidade")

    st.plotly_chart(bar_fig, use_container_width=True)


def grafico_saida_animal(df):
    st.write('**Comparação dos Tipos de Saída por Tipo de Animal**')
    
    tipos_animais_traduzidos = {
        'Dog': 'Cachorro',
        'Cat': 'Gato',
        'Bir': 'Pássaro',
        'Oth': 'Outro'
    }
    
    opcoes_traduzidas = [tipos_animais_traduzidos.get(tipo, tipo) for tipo in df['animal_type'].unique()]
    
    animal_tipo_selecionado = st.selectbox(
        'Selecione o Tipo de Animal para análise:',
        options=opcoes_traduzidas)

    tipo_original = {v: k for k, v in tipos_animais_traduzidos.items()}.get(animal_tipo_selecionado, animal_tipo_selecionado)

    df_filtrado = df[df['animal_type'] == tipo_original]

    hist_fig = px.histogram(df_filtrado,
                            x='outcome_type',
                            color_discrete_sequence=['#636EFA'],
                            labels={'outcome_type': 'Tipo de Saída',
                                    'count': 'Quantidade'},
                            barmode='group')

    hist_fig.update_xaxes(title_text="Tipo de Saída")
    hist_fig.update_yaxes(title_text="Quantidade")
    hist_fig.update_layout(showlegend=False)

    st.plotly_chart(hist_fig, use_container_width=True)

    st.write("**Legenda das Traduções:**")
    st.write("""
    - Transfer: Transferência  
    - Adoption: Adoção  
    - Euthanasia: Eutanásia  
    - Died: Morte  
    - Missing: Desaparecido  
    - Return to Owner: Devolução ao Dono
    """)

def grafico_saida_entrada(df):
    st.write('**Análise do Tipo de Saída por Condição de Entrada**')

    tipo_saida_selecionado = st.selectbox(
        'Selecione o Tipo de Saída para análise:',
        options=df['outcome_type'].unique())

    df_filtrado = df[df['outcome_type'] == tipo_saida_selecionado]

    bar_fig = px.histogram(df_filtrado,
                           x='intake_condition',
                           color_discrete_sequence=['#556b2f'],
                           labels={'intake_condition': 'Condição na Entrada',
                                   'outcome_type': 'Tipo de Saída',
                                   'count': 'Quantidade'},
                           barmode='group')

    bar_fig.update_yaxes(title_text="Quantidade")
    bar_fig.update_layout(showlegend=False)

    st.plotly_chart(bar_fig, use_container_width=True)

    st.write("**Legenda das Traduções:**")
    st.write("""
    - Normal: Normal  
    - Sick: Doente  
    - Injured: Ferido  
    - Aged: Idoso  
    - Feral: Selvagem  
    - Other: Outro  
    - Nursing: Amamentando  
    - Pregnant: Grávida
    """)
    
def grafico_pizza(df):
    st.write('**Distribuição dos Tipos de Saída dos Animais**')
    def determinar_genero(sexo_intake, sexo_outcome):
        if 'Male' in sexo_intake or 'Male' in sexo_outcome:
            return 'Masculino'
        elif 'Female' in sexo_intake or 'Female' in sexo_outcome:
            return 'Feminino'
        else:
            return 'Desconhecido'

    df['gender'] = df.apply(lambda row: determinar_genero(row['sex_upon_intake'], row['sex_upon_outcome']), axis=1)

    sexo_selecionado = st.selectbox(
        'Selecione o Sexo do Animal para análise:',
        options=['Todos', 'Masculino', 'Feminino'],
        key='sexo_animal_pizza'
    )

    if sexo_selecionado != 'Todos':
        df_filtrado = df[df['gender'] == sexo_selecionado]
    else:
        df_filtrado = df[df['gender'] != 'Desconhecido']
    df_filtrado = df_filtrado.dropna(subset=['outcome_type'])

    pie_fig = px.pie(
        df_filtrado,
        names='outcome_type',
        hole=0.3,
        labels={'outcome_type': 'Tipo de Saída'},
        title=f"Distribuição dos Tipos de Saída dos Animais ({sexo_selecionado})"
    )
    
    st.plotly_chart(pie_fig, use_container_width=True)

    st.write("**Legenda das Traduções:**")
    st.write("""
    - Transfer: Transferência  
    - Adoption: Adoção  
    - Euthanasia: Eutanásia  
    - Died: Morte  
    - Missing: Desaparecido  
    - Return to Owner: Devolução ao Dono
    """)

def grafico_linha(df):
    st.write('**Tendências de Entrada e Saída de Animais ao Longo do Tempo**')
    
    filtro = st.selectbox('Selecione o que deseja analisar:', ['Entradas', 'Saídas', 'Ambos'])

    df['intake_monthyear'] = pd.to_datetime(df['intake_monthyear'])
    df['outcome_monthyear'] = pd.to_datetime(df['outcome_monthyear'])

    df_grouped = df.groupby(['intake_monthyear']).size().reset_index(name='Entradas')
    df_grouped_outcome = df.groupby(['outcome_monthyear']).size().reset_index(name='Saídas')

    line_fig = None

    if filtro == 'Entradas':
        line_fig = px.line(
            df_grouped, x='intake_monthyear', y='Entradas', labels={'intake_monthyear': 'Mês/Ano', 'Entradas': 'Número de Entradas'},
            title="Número de Entradas de Animais ao Longo do Tempo"
        )
    elif filtro == 'Saídas':
        line_fig = px.line(
            df_grouped_outcome, x='outcome_monthyear', y='Saídas', labels={'outcome_monthyear': 'Mês/Ano', 'Saídas': 'Número de Saídas'},
            title="Número de Saídas de Animais ao Longo do Tempo"
        )
    elif filtro == 'Ambos':
        line_fig = px.line(
            df_grouped, x='intake_monthyear', y='Entradas', labels={'intake_monthyear': 'Mês/Ano', 'Entradas': 'Número de Entradas'},
            title="Número de Entradas e Saídas de Animais ao Longo do Tempo"
        )
        line_fig.add_trace(
            px.line(df_grouped_outcome, x='outcome_monthyear', y='Saídas', labels={'outcome_monthyear': 'Mês/Ano', 'Saídas': 'Número de Saídas'}).data[0]
        )
    
    line_fig.update_xaxes(title_text="Mês/Ano")
    line_fig.update_yaxes(title_text="Quantidade")
    line_fig.update_layout(showlegend=(filtro == 'Ambos'))

    st.plotly_chart(line_fig, use_container_width=True)



build_page()