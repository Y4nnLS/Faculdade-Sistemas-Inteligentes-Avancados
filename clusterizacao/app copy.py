import pandas as pd
from pickle import load

# Carregar os nomes das colunas categóricas
colunas_categoricas = load(open("clusterizacao\\colunas_Categoricas.pkl", "rb"))
# print(colunas_categoricas)
# Lista de teste
teste_instancia = ['Female', 21, 1.62, 64, 'yes', 'no', 2, 3, 'Sometimes', 'no', 2, 'no', 0, 1, 'no', 'Public_Transportation', 'Normal_Weight']
pd.set_option('display.max_columns', None)

# Criar DataFrame a partir da lista de teste
df_teste = pd.DataFrame([teste_instancia], columns=['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight', 'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE', 'CALC', 'MTRANS', 'NObeyesdad'])

# Converter colunas categóricas em dummy variables
dados_categoricos_normalizados = pd.get_dummies(data=df_teste[['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS', 'NObeyesdad']], dtype=int)

# Exibir o DataFrame resultante
# print(dados_categoricos_normalizados)

dados_categoricos = pd.DataFrame(columns=colunas_categoricas)

# Concatenar os DataFrames
dados_completos = dados_categoricos.combine_first(dados_categoricos_normalizados)
# dados_completos = pd.merge(dados_categoricos,dados_categoricos_normalizados ,how='outer')
print(dados_completos)
# Substituir os valores NaN por pd.NA
dados_completos = dados_completos.where(pd.notna(dados_completos), other=pd.NA)

# Exibir o DataFrame resultante
# print(dados_completos)