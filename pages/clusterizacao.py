import streamlit as st
import pandas as pd
from utils import read_df
from sklearn.cluster import KMeans
import plotly.express as px

st.set_page_config(page_title="Clusterização", layout="wide")

# Definindo a escala de cores personalizada
color_scale = ['#00ccff', '#cc00ff', '#ffcc00', '#0066bb', '#6600bb', 
               '#bb0066', '#bb6600', '#ff0066', '#66ff66', '#ee0503']

def main():
    st.title("Clusterização")
    
    # Carregar os dados normalizados e padronizados
    normalized_data_name = 'normalized_ACC_INTAKES_OUTCOMES'
    scaled_data_name = 'scaled_ACC_INTAKES_OUTCOMES'
    data_normalized = read_df(normalized_data_name, extension='parquet')
    data_scaled = read_df(scaled_data_name, extension='parquet')

    # Colunas a serem utilizadas para a clusterização
    cluster_features = [
        'age_upon_intake_(days)', 'age_upon_outcome_age_group_encoded',
        'animal_type_Bir', 'animal_type_Cat', 'animal_type_Dog', 'animal_type_Oth',
        'is_mix_breed', 'intake_condition_Aged   ', 'intake_condition_Feral  ',
        'intake_condition_Injured', 'intake_condition_Normal', 'intake_condition_Nursing',
        'intake_condition_Other  ', 'intake_condition_Pregnant', 'intake_condition_Sick   ',
        'intake_type_Euthanasia Request', 'intake_type_Owner Surrender   ',
        'intake_type_Public Assist     ', 'intake_type_Stray             ', 'intake_type_Wildlife          ',
        'sex_upon_intake_Intact Female', 'sex_upon_intake_Intact Male  ', 
        'sex_upon_intake_Neutered Male', 'sex_upon_intake_Spayed Female',
        'sex_upon_intake_Intact Unknown','time_in_shelter_days', 'age_upon_intake_age_group_encoded'
    ]

    # Filtrar as colunas disponíveis no dataset
    available_features_norm = [feature for feature in cluster_features if feature in data_normalized.columns]
    available_features_scaled = [feature for feature in cluster_features if feature in data_scaled.columns]

    # Função para calcular o Elbow
    def calculate_elbow(data, features):
        sse = []
        k_values = range(1, 11)  # Testar de 1 a 10 clusters
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(data[features])
            sse.append(kmeans.inertia_)
        return k_values, sse
    
    st.markdown("""
    
    Um dos objetivos específicos deste trabalho é utilizar técnicas de clusterização para identificar padrões nos perfis de animais que passam mais tempo nos abrigos, buscando compreender as categorias predominantes.
    """)

    # Calcular o Elbow para os dados padronizados e normalizados
    st.markdown("<h2>Método Elbow</h2>", unsafe_allow_html=True)

    k_values_norm, sse_norm = calculate_elbow(data_normalized, available_features_norm)
    k_values_scaled, sse_scaled = calculate_elbow(data_scaled, available_features_scaled)

    # Criar os gráficos de Elbow
    fig_norm = px.line(x=k_values_norm, y=sse_norm, markers=True, title='Método Elbow para dataset normalizado',
                       labels={'x': 'Número de Clusters', 'y': 'Soma dos quadrados das distâncias'})
    
    fig_scaled = px.line(x=k_values_scaled, y=sse_scaled, markers=True, title='Método Elbow para dataset padronizado',
                         labels={'x': 'Número de Clusters', 'y': 'Soma dos quadrados das distâncias'})

    # Dropdown para selecionar qual gráfico exibir
    elbow_option = st.selectbox(
        'Selecione o gráfico de Elbow para exibir:',
        ('Dataset normalizado', 'Dataset padronizado')
    )

    # Mostrar o gráfico correspondente
    if elbow_option == 'Dataset normalizado':
        st.plotly_chart(fig_norm)
    elif elbow_option == 'Dataset padronizado':
        st.plotly_chart(fig_scaled)

    # Seção de Clusterização
    st.markdown("<h2>Clusterização - Análise de Padrões</h2>", unsafe_allow_html=True)

    # Função para clusterização
    def clusterize_data(data, features, num_clusters=3):
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        clusters = kmeans.fit_predict(data[features])
        data['cluster'] = clusters
        return data

    # Função para exibir gráficos de clusters com cores definidas
    def plot_clusters(data, x_axis, y_axis, title):
        data['cluster'] = data['cluster'].astype(str)  # Converter para string para forçar as cores
        fig = px.scatter(data, x=x_axis, y=y_axis, color='cluster', 
                         color_discrete_sequence=color_scale[:len(data['cluster'].unique())],
                         title=title)
        st.plotly_chart(fig)

    # Selecionar o dataset e o número de clusters
    dataset_option = st.selectbox('Selecione o dataset para clusterização:', 
                                  ['Dataset Padronizado', 'Dataset Normalizado'])
    num_clusters = st.slider('Número de Clusters:', 2, 10, 3)
    
    # Selecionar o eixo X e Y
    x_axis = st.selectbox('Selecione a variável para o eixo X:', cluster_features)
    y_axis = st.selectbox('Selecione a variável para o eixo Y:', cluster_features)

    # Executar a clusterização
    if dataset_option == 'Dataset Padronizado':
        selected_data = data_scaled
        available_features = available_features_scaled
    else:
        selected_data = data_normalized
        available_features = available_features_norm

    # Aplicar a clusterização
    data_clustered = clusterize_data(selected_data, available_features, num_clusters)

    # Exibir gráficos de clusters
    plot_clusters(data_clustered, x_axis, y_axis, f'Clusters - {dataset_option}')

if __name__ == "__main__":
    main()