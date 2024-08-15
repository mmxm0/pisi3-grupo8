import streamlit as st
import pandas as pd
import plotly.express as px
from utils import read_df
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import io

st.set_page_config(page_title="Pré-processamento dos dados", layout="wide")

def display_dataframe_info(data):
    buffer = io.StringIO()
    data.info(buf=buffer)
    s = buffer.getvalue()
    return s

def analyze_missing_values(data):
    missing_values = data.isnull().sum()
    missing_values = missing_values[missing_values > 0].sort_values(ascending=False)
    return missing_values

def analyze_duplicates(data):
    duplicate_count = data.duplicated().sum()
    return duplicate_count

def remove_duplicates(data):
    data_cleaned = data.drop_duplicates()
    return data_cleaned

def convert_data_types(data):
    # Converter colunas de datas e horas para o tipo datetime com formato específico
    date_columns = [
        'date_of_birth', 'outcome_datetime', 'intake_datetime'
    ]
    
    for col in date_columns:
        data[col] = pd.to_datetime(data[col], format='%d/%m/%Y %H:%M', errors='coerce')
    
    # Converter colunas 'monthyear' para o formato mm/yyyy
    monthyear_columns = ['dob_monthyear', 'outcome_monthyear', 'intake_monthyear']
    for col in monthyear_columns:
        data[col] = pd.to_datetime(data[col], format='%Y-%m', errors='coerce').dt.strftime('%m/%Y')
    
    # Tratar colunas de mês como números
    month_columns = ['outcome_month', 'dob_month', 'intake_month']
    for col in month_columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    # Converter colunas numéricas para tipos numéricos
    numeric_columns = [
        'age_upon_outcome_(days)', 'age_upon_outcome_(years)', 'outcome_year', 'outcome_hour',
        'outcome_number', 'dob_year', 'age_upon_intake_(days)', 'age_upon_intake_(years)',
        'intake_year', 'intake_hour', 'intake_number', 'time_in_shelter_days'
    ]
    for col in numeric_columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    return data

def identify_outliers(data):
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    outlier_columns = []
    for col in numeric_columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        outlier_step = 1.5 * IQR
        if data[(data[col] < Q1 - outlier_step) | (data[col] > Q3 + outlier_step)].shape[0] > 0:
            outlier_columns.append(col)
    return outlier_columns

def remove_outliers(data, outlier_columns):
    for col in outlier_columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        outlier_step = 1.5 * IQR
        data = data[~((data[col] < (Q1 - outlier_step)) | (data[col] > (Q3 + outlier_step)))]
    return data

def plot_outliers(data, outlier_columns):
    st.markdown("<h2>Visualização de Outliers:</h2>", unsafe_allow_html=True)
    for col in outlier_columns:
        fig = px.box(data, x=col, title=f'Outliers na Coluna: {col}', orientation='h', color_discrete_sequence=['#A9A9A9'])  # Cinza
        st.plotly_chart(fig)
        # Remover outliers e exibir o novo box plot
        data_no_outliers = remove_outliers(data.copy(), [col])
        fig_no_outliers = px.box(data_no_outliers, x=col, title=f'Coluna: {col} (Sem Outliers)', orientation='h', color_discrete_sequence=['#87CEEB'])  # Azul claro
        st.plotly_chart(fig_no_outliers)

def categorize_breeds(data):
    data['is_mix_breed'] = data['breed'].str.contains('Mix|/', case=False, na=False).astype(int)
    return data

def scale_numeric_data(data):
    scaler = StandardScaler()
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    data_scaled = data.copy()
    data_scaled[numeric_cols] = scaler.fit_transform(data_scaled[numeric_cols])
    return data_scaled, numeric_cols

def normalize_numeric_data(data):
    scaler = MinMaxScaler()
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    data_normalized = data.copy()
    data_normalized[numeric_cols] = scaler.fit_transform(data_normalized[numeric_cols])
    return data_normalized, numeric_cols

def apply_label_encoding(data):
    le = LabelEncoder()
    columns_to_encode = [
        'age_upon_outcome_age_group', 'outcome_weekday', 
        'outcome_number', 'intake_weekday', 'intake_number', 'age_upon_intake_age_group', 'outcome_subtype', 
    ]
    
    st.markdown("<h2>Comparação de colunas com Label Encoding (antes e depois):</h2>", unsafe_allow_html=True)
    
    for col in columns_to_encode:
        data[f'{col}_encoded'] = le.fit_transform(data[col])
        
        st.markdown(f"{col}", unsafe_allow_html=True)
        comparison_df = pd.DataFrame({
            'Original': data[col].head(10),
            'Encoded': data[f'{col}_encoded'].head(10)
        })
        st.write(comparison_df)
    
    return data, columns_to_encode

def apply_one_hot_encoding(data):
    categorical_columns = [
        'animal_type', 'outcome_type', 'sex_upon_outcome',
        'sex_upon_intake', 'intake_condition', 'intake_type'
    ]
    data_encoded = pd.get_dummies(data, columns=categorical_columns)
    return data_encoded, categorical_columns

def apply_and_display_one_hot_encoding(data):
    categorical_columns = [
        'animal_type', 'outcome_type', 'sex_upon_outcome',
        'sex_upon_intake', 'intake_condition', 'intake_type'
    ]
    
    data_encoded = pd.get_dummies(data, columns=categorical_columns)
    
    st.markdown("<h2>Comparação de colunas com One-Hot Encoding (antes e depois):</h2>", unsafe_allow_html=True)
    for col in categorical_columns:
        before = data[col].head(10).to_frame()  
        after = data_encoded.filter(like=col).head(10)
        comparison_df = pd.concat([before, after], axis=1)
        st.write(f"Coluna: {col}")
        st.write(comparison_df)
    
    return data_encoded, categorical_columns

def main():
    st.title("Pré-processamento dos dados")
    
    df_name = 'ACC_INTAKES_OUTCOMES'  
    data = read_df(df_name, extension='parquet')
    
    st.markdown("<h2>Dataset original:</h2>", unsafe_allow_html=True)
    st.write(data.head(20))  
    
    st.markdown("<h2>Informações do dataset:</h2>", unsafe_allow_html=True)
    st.text(display_dataframe_info(data))
    
    data = convert_data_types(data)
    st.markdown("<h2>Informações do dataset após correção dos tipos:</h2>", unsafe_allow_html=True)
    st.text(display_dataframe_info(data))

    st.markdown("<h2>Análise de valores ausentes:</h2>", unsafe_allow_html=True)
    missing_values = analyze_missing_values(data)
    if missing_values.empty:
        st.write("Não há valores ausentes no DataFrame.")
    else:
        st.write(missing_values)
    
    st.markdown("<h2>Análise de Duplicatas:</h2>", unsafe_allow_html=True)
    duplicate_count = analyze_duplicates(data)
    if duplicate_count == 0:
        st.write("Não há registros duplicados no DataFrame.")
    else:
        st.write(f"Número de registros duplicados: {duplicate_count}")
        data = remove_duplicates(data)
        st.write(f"Registros duplicados removidos. Número de registros restantes: {len(data)}")

    st.markdown("<h2>Identificação de Outliers:</h2>", unsafe_allow_html=True)
    outlier_columns = identify_outliers(data)
    if outlier_columns:
        st.write(f"Foi identificado outliers nas seguintes colunas: {', '.join(outlier_columns)}")
        plot_outliers(data, outlier_columns)
        data = remove_outliers(data, outlier_columns)
        st.write(f"Outliers removidos. Número de registros restantes: {len(data)}")
    else:
        st.write("Não foram identificados outliers nas colunas numéricas.")

    data.to_parquet('data/data_cleaned.parquet')
    
    # Categorização de raças
    st.markdown("<h2>Categorização de raças:</h2>", unsafe_allow_html=True)
    data = categorize_breeds(data)
    st.write(data[['breed', 'is_mix_breed']].head(20))  # Exibir as primeiras 20 linhas da tabela com a nova coluna

    # Escalonamento de Dados Numéricos
    st.markdown("<h2>Escalonamento de dados:</h2>", unsafe_allow_html=True)
    st.write("""
        1. Padronização (Standard Scaling):
           - Transforma os dados para que tenham média zero e desvio padrão igual a um.
           - É útil para algoritmos que assumem que os dados são distribuídos em uma forma normal (gaussiana).
        
        2. Normalização (Min-Max Scaling):
           - Transforma os dados para que todos os valores estejam dentro do intervalo [0, 1].
           - É útil para algoritmos que não fazem suposições sobre a distribuição dos dados, como redes neurais e métodos de distância (por exemplo, k-NN).
        
    ### Abaixo estão as tabelas mostrando como as colunas foram alteradas antes e depois do escalonamento (padronização e normalização):
    """)

    # Padronizar e Normalizar os dados
    data_scaled, numeric_cols_scale = scale_numeric_data(data)
    data_normalized, numeric_cols_norm = normalize_numeric_data(data)
    
    for col in numeric_cols_scale:
        if not data_scaled[col].equals(data_normalized[col]):
            st.write(f"{col} (Padronizado)")
            comparison_df = pd.DataFrame({
                'Antes': data[col].head(8),
                'Depois (Padronizado)': data_scaled[col].head(8)
            })
            st.write(comparison_df)
    
    for col in numeric_cols_norm:
        if not data_scaled[col].equals(data_normalized[col]):
            st.write(f"{col} (Normalizado)")
            comparison_df = pd.DataFrame({
                'Antes': data[col].head(8),
                'Depois (Normalizado)': data_normalized[col].head(8)
            })
            st.write(comparison_df)
    
    # Explicação da Codificação
    st.markdown("<h2>Codificação dos dados:</h2>", unsafe_allow_html=True)
    st.write("""
        ### One-Hot Encoding
        Utilizamos One-Hot Encoding para colunas categóricas nominais onde não há uma ordem implícita entre os valores, como animal_type, outcome_type, sex_upon_outcome, sex_upon_intake, intake_condition e intake_type. Esta técnica cria uma nova coluna binária para cada valor único da categoria, permitindo que o modelo trate cada valor de maneira igual.

        ### Label Encoding
        Utilizamos Label Encoding para colunas categóricas ordinais onde há uma ordem implícita entre os valores, como age_upon_outcome_age_group, outcome_subtype, outcome_weekday, outcome_number, intake_weekday e intake_number. Esta técnica converte os valores categóricos em valores numéricos que refletem a ordem implícita entre eles.
    """)

    # Aplicar Label Encoding nas colunas específicas
    data_encoded, columns_to_encode = apply_label_encoding(data)

    # Aplicar One-Hot Encoding e Mostrar Comparação
    data_encoded, categorical_columns = apply_and_display_one_hot_encoding(data_encoded)
    
    # Criar os datasets finais
    data_final_scaled = data_encoded.drop(columns=numeric_cols_scale, errors='ignore').join(data_scaled[numeric_cols_scale])
    data_final_normalized = data_encoded.drop(columns=numeric_cols_norm, errors='ignore').join(data_normalized[numeric_cols_norm])

    # Exibir o dataset final padronizado
    st.markdown("<h2>Dataset final padronizado após pré-processamento:</h2>", unsafe_allow_html=True)
    st.write(data_final_scaled.head(20))  # Exibir as primeiras 20 linhas do dataset final padronizado
    
    # Exibir o dataset final normalizado
    st.markdown("<h2>Dataset final normalizado após pré-processamento:</h2>", unsafe_allow_html=True)
    st.write(data_final_normalized.head(20))  # Exibir as primeiras 20 linhas do dataset final normalizado
    
    # Verificação de duplicatas no dataset final
    st.markdown("<h2>Verificação de duplicatas no Dataset Final:</h2>", unsafe_allow_html=True)
    final_duplicate_count_scaled = analyze_duplicates(data_final_scaled)
    final_duplicate_count_normalized = analyze_duplicates(data_final_normalized)
    
    if final_duplicate_count_scaled == 0:
        st.write("Não há registros duplicados no Dataset Final Padronizado.")
    else:
        st.write(f"Número de registros duplicados no Dataset Final Padronizado: {final_duplicate_count_scaled}")

    if final_duplicate_count_normalized == 0:
        st.write("Não há registros duplicados no Dataset Final Normalizado.")
    else:
        st.write(f"Número de registros duplicados no Dataset Final Normalizado: {final_duplicate_count_normalized}")
    
    # Salvar os datasets pré-processados
    data_final_scaled.to_parquet('data/scaled_ACC_INTAKES_OUTCOMES.parquet')
    data_final_normalized.to_parquet('data/normalized_ACC_INTAKES_OUTCOMES.parquet')

if __name__ == "__main__":
    main()