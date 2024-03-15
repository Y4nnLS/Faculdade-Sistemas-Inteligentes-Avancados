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
# print(dados_categoricos_normalizados)

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
# print(dados_numericos)
# print(dados_numericos_normalizados)

# Converter os dados numericos normalizados em DataFrame
dados_numericos_normalizados = pd.DataFrame(data = dados_numericos_normalizados, columns = ['idade', 'altura', 'peso'])
# print(dados_numericos_normalizados)

# Juntar com os dados categoricos normalizados
dados_normalizados_final = dados_numericos_normalizados.join(dados_categoricos_normalizados, how = 'left')
print(dados_normalizados_final.head(10))



# Aplicar inverse_transform aos dados numéricos
dados_numericos_legiveis = pd.DataFrame(modelo_normalizador.inverse_transform(dados_numericos_normalizados))

# Juntar com os dados categóricos normalizados
dados_legiveis_final = dados_numericos_legiveis.join(dados_categoricos_normalizados, how='left')

print(dados_legiveis_final.head(10))



### Está célula contem exemplos de abertura do modelo normalizador e de normalização de novas instancias
### Foi elaborada como mero exemplo

# Abrir o modelo normalizador
from pickle import load
modelo_normalizador = load(open('Aula08-03\\dados\\normalizador1.pkl', 'rb'))

nova_instancia = [[13, 1, 98]]

# Normalizar a nova instancia com o modelo salvo inicialmente
nova_instancia_normalizada = modelo_normalizador.transform(nova_instancia)
# print(nova_instancia_normalizada)

# Recompor os dados normalizados para uma representação legivel
dados_legiveis = modelo_normalizador.inverse_transform(nova_instancia_normalizada)
# print(dados_legiveis)



