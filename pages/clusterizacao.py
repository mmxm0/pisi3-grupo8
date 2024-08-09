import streamlit as st
import pandas as pd
from utils import read_df
from sklearn.cluster import KMeans
import plotly.express as px

st.set_page_config(page_title="Clusterização", layout="wide")

def main():
    st.title("Análise de Clusterização - Elbow Method")

    # Carregar os dados normalizados, padronizados e limpos
    normalized_data_name = 'normalized_ACC_INTAKES_OUTCOMES'
    scaled_data_name = 'scaled_ACC_INTAKES_OUTCOMES'
    cleaned_data_name = 'data_cleaned'
    data_normalized = read_df(normalized_data_name, extension='parquet')
    data_scaled = read_df(scaled_data_name, extension='parquet')
    data_cleaned = read_df(cleaned_data_name, extension='parquet')

    # Selecionar características relevantes para a clusterização
    features = ['age_upon_intake_(days)', 'time_in_shelter_days', 'dob_year', 'intake_hour', 'outcome_year']

    # Função para calcular o Elbow
    def calculate_elbow(data, features):
        sse = []
        k_values = range(1, 11)  # Testar de 1 a 10 clusters
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(data[features])
            sse.append(kmeans.inertia_)
        return k_values, sse

    # Elbow para os dados limpos
    k_values_cleaned, sse_cleaned = calculate_elbow(data_cleaned, features)
    fig_cleaned = px.line(x=k_values_cleaned, y=sse_cleaned, markers=True, title='Método Elbow para Dados Limpos',
                          labels={'x': 'Número de Clusters', 'y': 'Soma dos Quadrados das Distâncias'})
    st.plotly_chart(fig_cleaned)

    # Elbow para os dados normalizados
    k_values_norm, sse_norm = calculate_elbow(data_normalized, features)
    fig_norm = px.line(x=k_values_norm, y=sse_norm, markers=True, title='Método Elbow para Dados Normalizados',
                       labels={'x': 'Número de Clusters', 'y': 'Soma dos Quadrados das Distâncias'})
    st.plotly_chart(fig_norm)

    # Elbow para os dados padronizados
    k_values_scaled, sse_scaled = calculate_elbow(data_scaled, features)
    fig_scaled = px.line(x=k_values_scaled, y=sse_scaled, markers=True, title='Método Elbow para Dados Padronizados',
                         labels={'x': 'Número de Clusters', 'y': 'Soma dos Quadrados das Distâncias'})
    st.plotly_chart(fig_scaled)

if __name__ == "__main__":
    main()