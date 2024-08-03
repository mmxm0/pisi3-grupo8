import streamlit as st
import pandas as pd
import plotly.express as px
from utils import read_df
from sklearn.preprocessing import StandardScaler, LabelEncoder
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

def scale_numeric_data(data):
    scaler = StandardScaler()
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    data_scaled = data.copy()
    data_scaled[numeric_cols] = scaler.fit_transform(data_scaled[numeric_cols])
    return data, data_scaled, numeric_cols

def apply_label_encoding(data):
    # Instanciar o LabelEncoder
    le = LabelEncoder()
    # Colunas para aplicar Label Encoding
    columns_to_encode = [
        'age_upon_outcome_age_group', 'outcome_weekday', 
        'outcome_number', 'intake_weekday', 'intake_number'
    ]
    
    st.markdown("<h2>Comparação de colunas com Label Encoding (antes e depois):</h2>", unsafe_allow_html=True)
    
    for col in columns_to_encode:
        data[f'{col}_encoded'] = le.fit_transform(data[col])
        
        # Exibir comparação antes e depois
        st.markdown(f"**{col}**", unsafe_allow_html=True)
        comparison_df = pd.DataFrame({
            'Original': data[col].head(10),
            'Encoded': data[f'{col}_encoded'].head(10)
        })
        st.write(comparison_df)
    
    return data

def apply_one_hot_encoding(data):
    # Variáveis para One-Hot Encoding
    categorical_columns = [
        'animal_type', 'outcome_type', 'sex_upon_outcome',
        'sex_upon_intake', 'intake_condition', 'intake_type'
    ]
    data_encoded = pd.get_dummies(data, columns=categorical_columns)
    return data_encoded

def apply_and_display_one_hot_encoding(data):
    # Variáveis para One-Hot Encoding
    categorical_columns = [
        'animal_type', 'outcome_type', 'sex_upon_outcome',
        'sex_upon_intake', 'intake_condition', 'intake_type'
    ]
    
    # Aplicar One-Hot Encoding
    data_encoded = pd.get_dummies(data, columns=categorical_columns)
    
    # Exibir comparação antes e depois
    st.markdown("<h2>Comparação de colunas com One-Hot Encoding (antes e depois):</h2>", unsafe_allow_html=True)
    for col in categorical_columns:
        before = data[col].head(10).to_frame()  # Exibir apenas as 10 primeiras linhas para comparação
        after = data_encoded.filter(like=col).head(10)
        comparison_df = pd.concat([before, after], axis=1)
        st.write(f"**Coluna: {col}**")
        st.write(comparison_df)
    
    return data_encoded

def main():
    st.title("Pré-processamento dos dados")
    
    # Carregar os dados
    df_name = 'ACC_INTAKES_OUTCOMES'  # Nome do arquivo Parquet (sem extensão)
    data = read_df(df_name, extension='parquet')
    
    st.markdown("<h2>Dataset original:</h2>", unsafe_allow_html=True)
    st.write(data.head(20))  # Exibir as primeiras 20 linhas da tabela
    
    st.markdown("<h2>Informações do dataset:</h2>", unsafe_allow_html=True)
    st.text(display_dataframe_info(data))
    
    # Correção de Tipos de Variáveis
    data = convert_data_types(data)
    st.markdown("<h2>Informações do dataset após correção dos tipos:</h2>", unsafe_allow_html=True)
    st.text(display_dataframe_info(data))

    # Análise de Valores Ausentes
    st.markdown("<h2>Análise de valores ausentes:</h2>", unsafe_allow_html=True)
    missing_values = analyze_missing_values(data)
    if missing_values.empty:
        st.write("Não há valores ausentes no DataFrame.")
    else:
        st.write(missing_values)
    
    # Análise de Duplicatas
    st.markdown("<h2>Análise de Duplicatas:</h2>", unsafe_allow_html=True)
    duplicate_count = analyze_duplicates(data)
    if duplicate_count == 0:
        st.write("Não há registros duplicados no DataFrame.")
    else:
        st.write(f"Número de registros duplicados: {duplicate_count}")
        # Remover duplicatas
        data = remove_duplicates(data)
        st.write(f"Registros duplicados removidos. Número de registros restantes: {len(data)}")
    
    # Identificação de Outliers
    st.markdown("<h2>Identificação de Outliers:</h2>", unsafe_allow_html=True)
    outlier_columns = identify_outliers(data)
    if outlier_columns:
        st.write(f"Foi identificado outliers nas seguintes colunas: {', '.join(outlier_columns)}")
        plot_outliers(data, outlier_columns)
    else:
        st.write("Não foram identificados outliers nas colunas numéricas.")
    
    # Explicação da Técnica de Padronização
    st.markdown("<h2>Escalonamento de dados:</h2>", unsafe_allow_html=True)
    st.write("""
        Para este dataset, escolhemos a **padronização** usando a técnica de Standard Scaling.
        A padronização transforma os dados para que tenham média zero e desvio padrão igual a um.
        
        ### Por que escolhemos a padronização para este dataset?
        Nosso dataset contém variáveis numéricas com diferentes escalas, como idade, tempo de permanência no abrigo e condições de saúde. Padronizar os dados ajuda a garantir que todas as variáveis tenham a mesma importância no modelo de machine learning.
        
        - **Algoritmos Sensíveis à Escala**: Muitos algoritmos de machine learning, como regressão linear, SVMs e redes neurais, são sensíveis à escala das variáveis. A padronização garante que todas as variáveis contribuam igualmente para o modelo.
        
        - **Variabilidade das Variáveis**: As variáveis do nosso dataset possuem diferentes unidades e amplitudes de valores. Padronizar os dados ajuda a nivelar essas diferenças, facilitando a análise e a construção do modelo.
        
        - **Melhor Desempenho**: Escalonar os dados pode melhorar o desempenho e a estabilidade numérica dos algoritmos, resultando em modelos mais precisos e robustos.
        
        - **Convergência Mais Rápida**: Para algoritmos de otimização, como o gradiente descendente, a padronização pode levar a uma convergência mais rápida e eficiente.
        
        ### Colunas não Padronizadas
        As colunas "dob_month" e "outcome_month" não foram padronizadas porque representam meses do ano como valores inteiros de 1 a 12, que são valores categóricos e não contínuos. 

        As colunas "dob_monthyear", "outcome_monthyear" e "intake_monthyear" não foram padronizadas porque representam um mês e ano combinado no formato mm/yyyy, que é uma representação categórica de um período de tempo.

        A coluna "age_upon_outcome_age_group" não foi padronizada pois representa grupos de idade em intervalos, que são categóricos e não contínuos.

        Essas colunas são categóricas ou representações de tempo, e padronizá-las não seria apropriado, pois a padronização é usada para ajustar variáveis numéricas contínuas para uma média zero e desvio padrão igual a um.
    """)

    # Escalonamento de Dados Numéricos
    st.markdown("<h2>Colunas Padronizadas (antes e depois):</h2>", unsafe_allow_html=True)
    data_before, data_after, numeric_cols = scale_numeric_data(data)
    for col in numeric_cols:
        if not data_before[col].equals(data_after[col]):
            st.write(f"**{col}**")
            comparison_df = pd.DataFrame({
                'Antes': data_before[col].head(8),
                'Depois': data_after[col].head(8)
            })
            st.write(comparison_df)
    
    # Explicação da Codificação
    st.markdown("<h2>Codificação dos dados:</h2>", unsafe_allow_html=True)
    st.write("""
        ### One-Hot Encoding
        Utilizamos One-Hot Encoding para colunas categóricas nominais onde não há uma ordem implícita entre os valores, como `animal_type`, `outcome_type`, `sex_upon_outcome`, `sex_upon_intake`, `intake_condition` e `intake_type`. Esta técnica cria uma nova coluna binária para cada valor único da categoria, permitindo que o modelo trate cada valor de maneira igual.

        ### Label Encoding
        Utilizamos Label Encoding para colunas categóricas ordinais onde há uma ordem implícita entre os valores, como `age_upon_outcome_age_group`, `outcome_weekday`, `outcome_number`, `intake_weekday` e `intake_number`. Esta técnica converte os valores categóricos em valores numéricos que refletem a ordem implícita entre eles.
    """)

    # Aplicar Label Encoding nas colunas específicas
    data = apply_label_encoding(data)

    # Aplicar One-Hot Encoding e Mostrar Comparação
    data_encoded = apply_and_display_one_hot_encoding(data)
    
    # Exibir o dataset final
    st.markdown("<h2>Dataset final após pré-processamento:</h2>", unsafe_allow_html=True)
    st.write(data_encoded.head(20))  # Exibir as primeiras 20 linhas do dataset final
    
    # Verificação de duplicatas no dataset final
    st.markdown("<h2>Verificação de duplicatas no Dataset Final:</h2>", unsafe_allow_html=True)
    final_duplicate_count = analyze_duplicates(data_encoded)
    if final_duplicate_count == 0:
        st.write("Não há registros duplicados no Dataset Final.")
    else:
        st.write(f"Número de registros duplicados no Dataset Final: {final_duplicate_count}")

if __name__ == "__main__":
    main()

