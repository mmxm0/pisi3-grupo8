import streamlit as st
import pandas as pd
from utils import read_df
from sklearn.cluster import KMeans
import plotly.express as px

st.set_page_config(page_title="Clusterização", layout="wide")

# Definindo a escala de cores personalizada
color_scale = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', 
               '#f1c40f', '#1abc9c', '#e67e22', '#34495e', 
               '#ff6b6b', '#48dbfb']

def main():
    st.write('<h1>Clusterização (<i>clustering</i>)</h1>', unsafe_allow_html=True)
    st.write('''Para a geração de grupos, foi usado o método K-means, que separa o <i>dataset</i> em <i>k</i> grupos distintos.
         Como a quantidade de <i>clusters</i> é subjetiva, para que pudesse ter uma base da quantidade adequada de
         <i>clusters</i>, foi aplicado o Método do Cotovelo, em que consiste em executar o algoritmo em <i>k</i> vezes, e
         calcular a inércia (soma das distâncias quadráticas dos pontos para o centro do <i>cluster</i> mais próximo), a partir do
         ponto que a inércia começa a diminuir de forma mais lenta, é chamado de "cotovelo", o número ideal de <i>clusters</i>.''', unsafe_allow_html=True)

    st.write('----')
    
    # Carregar os dados normalizados e padronizados
    normalized_data_name = 'normalized_ACC_INTAKES_OUTCOMES'
    scaled_data_name = 'scaled_ACC_INTAKES_OUTCOMES'
    data_normalized = read_df(normalized_data_name, extension='parquet')
    data_scaled = read_df(scaled_data_name, extension='parquet')

    # Colunas a serem utilizadas para a clusterização (ajustadas para incluir as colunas normalizadas/padronizadas)
    cluster_features_norm = [
        'age_upon_outcome_age_group_encoded',
        'animal_type_Bir', 'animal_type_Cat', 'animal_type_Dog', 'animal_type_Oth',
        'is_mix_breed', 'intake_condition_Aged   ', 'intake_condition_Feral  ',
        'intake_condition_Injured', 'intake_condition_Normal ', 'intake_condition_Nursing',
        'intake_condition_Other  ', 'intake_condition_Pregnan', 'intake_condition_Sick   ',
        'intake_type_Euthanasia Request', 'intake_type_Owner Surrender   ',
        'intake_type_Public Assist     ', 'intake_type_Stray             ', 'intake_type_Wildlife          ',
        'sex_upon_intake_Intact Female', 'sex_upon_intake_Intact Male  ', 
        'sex_upon_intake_Neutered Male', 'sex_upon_intake_Spayed Female',
        'sex_upon_intake_Unknown      ','outcome_type_Died           ','outcome_type_Euthanasia     ', 
        'outcome_type_Missing        ','outcome_type_Return to Owner', 'outcome_type_Transfer       ',
        'outcome_type_Adoption       ',
        'sex_upon_outcome_Intact Female', 'sex_upon_outcome_Intact Male  ', 
        'sex_upon_outcome_Neutered Male', 'sex_upon_outcome_Spayed Female', 
        'sex_upon_outcome_Unknown      ', 'age_upon_intake_age_group_encoded',
        'time_in_shelter_days_normalized', 'age_upon_outcome_(years)_normalized',
        'age_upon_intake_(years)_normalized',
    ]

    cluster_features_scaled = [
        'age_upon_outcome_age_group_encoded',
        'animal_type_Bir', 'animal_type_Cat', 'animal_type_Dog', 'animal_type_Oth',
        'is_mix_breed', 'intake_condition_Aged   ', 'intake_condition_Feral  ',
        'intake_condition_Injured', 'intake_condition_Normal ', 'intake_condition_Nursing',
        'intake_condition_Other  ', 'intake_condition_Pregnan', 'intake_condition_Sick   ',
        'intake_type_Euthanasia Request', 'intake_type_Owner Surrender   ',
        'intake_type_Public Assist     ', 'intake_type_Stray             ', 'intake_type_Wildlife          ',
        'sex_upon_intake_Intact Female', 'sex_upon_intake_Intact Male  ', 
        'sex_upon_intake_Neutered Male', 'sex_upon_intake_Spayed Female',
        'sex_upon_intake_Unknown      ','outcome_type_Died           ','outcome_type_Euthanasia     ', 
        'outcome_type_Missing        ','outcome_type_Return to Owner', 'outcome_type_Transfer       ',
        'outcome_type_Adoption       ',
        'sex_upon_outcome_Intact Female', 'sex_upon_outcome_Intact Male  ', 
        'sex_upon_outcome_Neutered Male', 'sex_upon_outcome_Spayed Female', 
        'sex_upon_outcome_Unknown      ', 'age_upon_intake_age_group_encoded',
        'time_in_shelter_days_scaled', 'age_upon_outcome_(years)_scaled',
        'age_upon_intake_(years)_scaled',
    ]

    # Função para calcular o Elbow
    def calculate_elbow(data, features):
        sse = []
        k_values = range(1, 11)  # Testar de 1 a 10 clusters
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(data[features])
            sse.append(kmeans.inertia_)
        return k_values, sse

    # Selecionar o dataset e o número de clusters
    dataset_option = st.selectbox('Selecione o dataset para clusterização:', 
                                  ['Dataset Padronizado', 'Dataset Normalizado'])

    # Ajustar as colunas de acordo com a seleção do dataset
    if dataset_option == 'Dataset Padronizado':
        selected_data = data_scaled
        cluster_features = cluster_features_scaled
    else:
        selected_data = data_normalized
        cluster_features = cluster_features_norm

    # Calcular o Elbow para os dados padronizados e normalizados
    st.markdown("<h2>Método Elbow</h2>", unsafe_allow_html=True)

    k_values, sse = calculate_elbow(selected_data, cluster_features)

    # Criar o gráfico de Elbow
    fig_elbow = px.line(x=k_values, y=sse, markers=True, title=f'Método Elbow para {dataset_option}',
                        labels={'x': 'Número de Clusters', 'y': 'Soma dos quadrados das distâncias'})

    st.plotly_chart(fig_elbow)

    st.write('----')

    # Função para clusterização
    def clusterize_data(data, features, num_clusters=3):
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        clusters = kmeans.fit_predict(data[features])
        data['cluster'] = clusters
        return data

    # Aplicar a clusterização
    num_clusters = st.slider('Número de Clusters:', 2, 10, 3)
    data_clustered = clusterize_data(selected_data, cluster_features, num_clusters)

    # Adicionar gráfico de boxplot para tempo no abrigo por cluster
    st.markdown("<h2>Distribuição do Tempo no Abrigo por Cluster</h2>", unsafe_allow_html=True)
    
    # Para garantir que os valores originais de 'time_in_shelter_days' sejam usados no boxplot:
    original_time_in_shelter = read_df(scaled_data_name if dataset_option == 'Dataset Padronizado' else normalized_data_name, extension='parquet')['time_in_shelter_days']

    data_clustered['time_in_shelter_days_original'] = original_time_in_shelter

    # Gráfico de boxplot
    fig_boxplot = px.box(data_clustered, x='cluster', y='time_in_shelter_days_original', 
                         color='cluster', color_discrete_sequence=color_scale[:num_clusters],
                         labels={'cluster': 'Cluster', 'time_in_shelter_days_original': 'Tempo no Abrigo (dias)'})
    
    st.plotly_chart(fig_boxplot)

    st.write('----')

    # Adicionar gráfico de Violin Plot para idade por cluster
    st.markdown("<h2>Distribuição da idade no momento de saída do Abrigo por Cluster</h2>", unsafe_allow_html=True)
    
    # Para garantir que os valores originais de 'age_upon_outcome_(years)' sejam usados no Violin Plot:
    original_age_upon_outcome = read_df(scaled_data_name if dataset_option == 'Dataset Padronizado' else normalized_data_name, extension='parquet')['age_upon_outcome_(years)']

    data_clustered['age_upon_outcome_(years)_original'] = original_age_upon_outcome

     # Criar o Violin Plot
    fig_violin = px.violin(data_clustered, x='cluster', y='age_upon_outcome_(years)_original', 
                           color='cluster', color_discrete_sequence=color_scale[:num_clusters],
                           box=True, points='all', labels={'cluster': 'Cluster', 'age_upon_outcome_(years)_original': 'Idade (anos)'})
    
    st.plotly_chart(fig_violin)

    st.write('----')

    # Adicionar gráfico de Stacked Bar Chart para tipos de saída por cluster
    st.markdown("<h2>Distribuição dos Tipos de Saída por Cluster</h2>", unsafe_allow_html=True)

    # Filtrar as colunas relevantes para os tipos de saída
    outcome_columns = [
        'outcome_type_Died           ', 'outcome_type_Euthanasia     ', 
        'outcome_type_Missing        ', 'outcome_type_Return to Owner', 
        'outcome_type_Transfer       ', 'outcome_type_Adoption       '
    ]

    # Criar um dicionário para mapear os nomes das colunas para as legendas desejadas
    outcome_labels = {
        'outcome_type_Died           ': 'Morreu',
        'outcome_type_Euthanasia     ': 'Eutanásia',
        'outcome_type_Missing        ': 'Perdido',
        'outcome_type_Return to Owner': 'Retornou ao dono',
        'outcome_type_Transfer       ': 'Transferido para abrigo parceiro',
        'outcome_type_Adoption       ': 'Adotado'
    }

    # Criar o dataframe para o gráfico de barras empilhadas
    outcome_data = data_clustered.groupby('cluster')[outcome_columns].sum().reset_index()

    # Renomear as colunas de acordo com as legendas desejadas
    outcome_data = outcome_data.rename(columns=outcome_labels)

    # Converter o dataframe para um formato longo para facilitar o uso com Plotly
    outcome_data_melted = outcome_data.melt(id_vars='cluster', 
                                            value_vars=list(outcome_labels.values()), 
                                            var_name='Tipo de Saída', 
                                            value_name='Quantidade')

    # Criar o gráfico de barras empilhadas
    fig_stacked_bar = px.bar(outcome_data_melted, x='cluster', y='Quantidade', 
                             color='Tipo de Saída', 
                             color_discrete_sequence=color_scale[:len(outcome_labels)],
                             labels={'cluster': 'Cluster', 'Quantidade': 'Quantidade'},
                             title='Distribuição dos Tipos de Saída por Cluster')

    st.plotly_chart(fig_stacked_bar)

    st.write('----')

    # Adicionar gráfico de Stacked Bar Chart para condições de saúde na entrada por cluster
    st.markdown("<h2>Distribuição das Condições de Saúde na Entrada por Cluster</h2>", unsafe_allow_html=True)

    # Filtrar as colunas relevantes para as condições de saúde na entrada
    intake_condition_columns = [
        'intake_condition_Aged   ', 'intake_condition_Feral  ', 
        'intake_condition_Injured', 'intake_condition_Normal ', 
        'intake_condition_Nursing', 'intake_condition_Other  ', 
        'intake_condition_Pregnan', 'intake_condition_Sick   '
    ]

    # Criar um dicionário para mapear os nomes das colunas para as legendas desejadas
    intake_condition_labels = {
        'intake_condition_Aged   ': 'Idoso',
        'intake_condition_Feral  ': 'Feroz',
        'intake_condition_Injured': 'Ferido',
        'intake_condition_Normal ': 'Normal',
        'intake_condition_Nursing': 'Amamentando',
        'intake_condition_Other  ': 'Outros',
        'intake_condition_Pregnan': 'Grávida',
        'intake_condition_Sick   ': 'Doente'
    }

    # Criar o dataframe para o gráfico de barras empilhadas
    intake_condition_data = data_clustered.groupby('cluster')[intake_condition_columns].sum().reset_index()

    # Renomear as colunas de acordo com as legendas desejadas
    intake_condition_data = intake_condition_data.rename(columns=intake_condition_labels)

    # Converter o dataframe para um formato longo para facilitar o uso com Plotly
    intake_condition_data_melted = intake_condition_data.melt(id_vars='cluster', 
                                                              value_vars=list(intake_condition_labels.values()), 
                                                              var_name='Condição de Saúde', 
                                                              value_name='Quantidade')

    # Criar o gráfico de barras empilhadas
    fig_intake_condition_stacked_bar = px.bar(intake_condition_data_melted, x='cluster', y='Quantidade', 
                                              color='Condição de Saúde', 
                                              color_discrete_sequence=color_scale[:len(intake_condition_labels)],
                                              labels={'cluster': 'Cluster', 'Quantidade': 'Quantidade'},
                                              title='Distribuição das Condições de Saúde na Entrada por Cluster')

    st.plotly_chart(fig_intake_condition_stacked_bar)

    st.write('----')

    # Adicionar gráfico de Stacked Bar Chart para tipos de animais por cluster
    st.markdown("<h2>Distribuição dos Tipos de Animais por Cluster</h2>", unsafe_allow_html=True)

    # Filtrar as colunas relevantes para os tipos de animais
    animal_columns = [
        'animal_type_Bir', 'animal_type_Cat', 'animal_type_Dog', 'animal_type_Oth'
    ]

    # Renomear as colunas para exibição mais clara
    column_rename_mapping = {
        'animal_type_Bir': 'Pássaro',
        'animal_type_Cat': 'Gato(a)',
        'animal_type_Dog': 'Cachorro(a)',
        'animal_type_Oth': 'Outros tipos de animais'
    }

    data_clustered = data_clustered.rename(columns=column_rename_mapping)

    # Criar o dataframe para o gráfico de barras empilhadas
    animal_data = data_clustered.groupby('cluster')[list(column_rename_mapping.values())].sum().reset_index()

    # Converter o dataframe para um formato longo para facilitar o uso com Plotly
    animal_data_melted = animal_data.melt(id_vars='cluster', 
                                          value_vars=list(column_rename_mapping.values()), 
                                          var_name='Tipo de Animal', 
                                          value_name='Quantidade')

    # Criar o gráfico de barras empilhadas
    fig_stacked_bar_animals = px.bar(animal_data_melted, x='cluster', y='Quantidade', 
                                     color='Tipo de Animal', 
                                     color_discrete_sequence=color_scale[:len(animal_columns)],
                                     labels={'cluster': 'Cluster', 'Quantidade': 'Quantidade'},
                                     title='Distribuição dos Tipos de Animais por Cluster')

    st.plotly_chart(fig_stacked_bar_animals)

    st.write('----')

    # Adicionar gráfico de Stacked Bar Chart para raça pura ou misturada por cluster
    st.markdown("<h2>Distribuição de Raça Pura ou Misturada por Cluster</h2>", unsafe_allow_html=True)

    # Criar o dataframe para o gráfico de barras empilhadas
    breed_data = data_clustered.groupby(['cluster', 'is_mix_breed']).size().unstack(fill_value=0)

    # Renomear as colunas para exibição mais clara
    breed_data = breed_data.rename(columns={0: 'Raça Pura', 1: 'Raça Misturada'})

    # Converter o dataframe para um formato longo para facilitar o uso com Plotly
    breed_data_melted = breed_data.reset_index().melt(id_vars='cluster', 
                                                      value_vars=['Raça Pura', 'Raça Misturada'], 
                                                      var_name='Tipo de Raça', 
                                                      value_name='Quantidade')

    # Criar o gráfico de barras empilhadas
    fig_stacked_bar_breeds = px.bar(breed_data_melted, x='cluster', y='Quantidade', 
                                    color='Tipo de Raça', 
                                    color_discrete_sequence=['#3498db', '#e74c3c'],  # Cores específicas para cada tipo de raça
                                    labels={'cluster': 'Cluster', 'Quantidade': 'Quantidade'},
                                    title='Distribuição de Raça Pura ou Misturada por Cluster')

    st.plotly_chart(fig_stacked_bar_breeds)


if __name__ == "__main__":
    main()

