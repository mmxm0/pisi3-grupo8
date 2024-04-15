import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def main():
    st.title('Projetos III - Grupo 8')

    file_path = 'C:/Users/julia/Documents/UFRPE/PROJETOS3/projetos3-julia/pisi3-grupo8/data/ACC_INTAKES_OUTCOMES.parquet'

    df = pd.read_parquet(file_path, engine='pyarrow')

   
    selected_columns = st.multiselect('Selecione as colunas para visualizar', options=df.columns.tolist(), default=df.columns.tolist(), key='multiselect')
     #Filtro
    filtered_df = df[selected_columns]

    st.write('**Visualização do Dataset:**')
    st.write(filtered_df)

        #Gráfico coluna "age_upon_outcome"
    if 'age_upon_outcome' in filtered_df.columns:

        st.write('**Gráfico de Barras da Coluna "age_upon_outcome":**')
        bar_fig = plt.figure(figsize=(10, 6))
        filtered_df['age_upon_outcome'].value_counts().plot(kind='bar')
        plt.xlabel('Idade no Momento da Saída')
        plt.ylabel('Número de animais')
        st.pyplot(bar_fig)

        # Gráfico coluna "animal_type"
    if 'animal_type' in filtered_df.columns:
        animal_type_counts = filtered_df['animal_type'].value_counts()

        st.write('**Tipos de animais:**')
        pie_fig = plt.figure(figsize=(8, 8))
        plt.pie(animal_type_counts, labels=animal_type_counts.index, autopct='%1.1f%%', startangle=140)
        plt.axis('equal') 
        st.pyplot(pie_fig)

if __name__ == '__main__':
    main()
