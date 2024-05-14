import pandas as pd


csv_file_path = 'C:/Users/julia/Documents/UFRPE/PROJETOS3/projetos3-julia/pisi3-grupo8/data/ACC_INTAKES_OUTCOMES_1.csv'

parquet_file_path = 'C:/Users/julia/Documents/UFRPE/PROJETOS3/projetos3-julia/pisi3-grupo8/data/ACC_INTAKES_OUTCOMES.parquet'

df = pd.read_csv(csv_file_path, encoding='utf-8', on_bad_lines='skip')


df.to_parquet(parquet_file_path, engine='pyarrow')


