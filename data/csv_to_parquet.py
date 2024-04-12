import csv
import pandas as pd


    #Abre o CSV e l^e linhas ignorando os erros
with open('C:/Users/julia/Documents/UFRPE/PROJETOS3/projetos3-julia/pisi3-grupo8/data/ACC_INTAKES_OUTCOMES.csv', 'r', errors='ignore') as f:
    reader = csv.reader(f)
    lines = []
    for line in reader:
            lines.append(line)
    
df = pd.DataFrame(lines)
    
df.to_parquet('C:/Users/julia/Documents/UFRPE/PROJETOS3/projetos3-julia/pisi3-grupo8/data/ACC_INTAKES_OUTCOMES.parquet')


