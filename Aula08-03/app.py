import pandas as pd

# Normalizar os dados

dados = pd.read_csv('Aula08-03\dados\dados_normalizar.csv', sep = ';')
# print(dados.head(5))

# Segmentar os dados entre escalares e categóricos
dados_numericos = dados.drop(columns = ['sexo'])
# print(dados_numericos.head(5))

dados_categoricos = dados['sexo'] # NÃO USAR O MAP .map({'F':0, 'M':1})
                                  # Ele muda o formato dos dados. Ele meio que atribui um peso diferente ao dado, tornando ele em um valor numerico e não escalar
# print(dados_categoricos.head(5))

# todo meta estimador trata todas as suas colunas como valores escalares


dados_categoricos_normalizados = pd.get_dummies(data = dados_categoricos, prefix = 'sexo', prefix_sep = '_')
print(dados_categoricos_normalizados)

# Treinar o modelo normalizador para os dado numéricos
from sklearn import preprocessing
normalizador = preprocessing.MinMaxScaler()
modelo_normalizador = normalizador.fit(dados_numericos)

# b-sample(organico) -> normalizar -> b-normalizada -> treinar ML -> MODELO(modelo gravado em disco) 
# Nova Instancia -> normalizar -> MODELO - > inferencia

# salvar o modelo normalizador para quando uma nova instancia chegar, vc submete ela ao modelo normalizador

from pickle import dump
dump(modelo_normalizador, open('Aula08-03\\dados\\normalizador1.pkl', 'wb'))

# Normalizar a base de dados de entrada
dados_numericos_normalizados = modelo_normalizador.fit_transform(dados_numericos)
print(dados_numericos)
print(dados_numericos_normalizados)