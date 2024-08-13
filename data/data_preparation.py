import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Carregando o dataset escalonado
parquet_file_path = 'c:/Users/marta.maria.x.melo/OneDrive - Accenture/Desktop/PISI3/pisi3-grupo8/data/scaled_ACC_INTAKES_OUTCOMES.parquet'

parquet_file_path_intact = 'c:/Users/marta.maria.x.melo/OneDrive - Accenture/Desktop/PISI3/pisi3-grupo8/data/ACC_INTAKES_OUTCOMES.parquet'

parquet_file_path_normalized = 'c:/Users/marta.maria.x.melo/OneDrive - Accenture/Desktop/PISI3/pisi3-grupo8/data/normalized_ACC_INTAKES_OUTCOMES.parquet'

features = [
        'age_upon_intake_(days)', 'age_upon_outcome_age_group_encoded',
        'animal_type_Bir', 'animal_type_Cat', 'animal_type_Dog', 'animal_type_Oth',
        'is_mix_breed', 'intake_condition_Aged   ', 'intake_condition_Feral  ',
        'intake_condition_Injured', 'intake_condition_Normal ', 'intake_condition_Nursing',
        'intake_condition_Other  ', 'intake_condition_Pregnan', 'intake_condition_Sick   ',
        'intake_type_Euthanasia Request', 'intake_type_Owner Surrender   ',
        'intake_type_Public Assist     ', 'intake_type_Stray             ', 'intake_type_Wildlife          ',
        'sex_upon_intake_Intact Female', 'sex_upon_intake_Intact Male  ', 
        'sex_upon_intake_Neutered Male', 'sex_upon_intake_Spayed Female',
        'sex_upon_intake_Unknown      ','time_in_shelter_days', 'age_upon_intake_age_group_encoded'
    ]



def load_and_split_data():
    
    df = pd.read_parquet(parquet_file_path)

    # Dividindo os dados em características (x) e rótulos (y)
    x = df[features]
    # Target outcome subtype
    y = df['outcome_type_Adoption       ']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    return x_train, x_test, y_train, y_test

def calculate_description_statistics():
    df = pd.read_parquet(parquet_file_path_intact)
    print("Estatísticas dos Dados Originais:")
    print(df.describe())

    df_processed = pd.read_parquet(parquet_file_path_normalized)
    print("Estatísticas dos Dados Normalizados:")
    print(df_processed.describe())