# Aplicação que utiliza o modelo de cluster treinado
# Abrir o modelo
from pickle import load
import pandas as pd
from sklearn import preprocessing

obesity_clusters_kmeans = load(open("clusterizacao\obesity_cluster.pkl", "rb"))
# print(obesity_clusters_kmeans.cluster_centers_)

# Obter uma nova instancia de dados
nova_instancia = ['Female',21,1.62,64,'yes','no',2,3,'Sometimes','no',2,'no',0,1,'no','Public_Transportation','Normal_Weight']
df_teste_instancia = pd.DataFrame([nova_instancia], columns=['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight', 'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE', 'CALC', 'MTRANS', 'NObeyesdad'])
# print(df_teste_instancia.head(5))

dados_numericos = df_teste_instancia.drop(columns=['Gender','family_history_with_overweight', 'FAVC', 'CAEC','SMOKE','SCC','CALC','MTRANS','NObeyesdad' ])
dados_categoricos = df_teste_instancia[['Gender','family_history_with_overweight', 'FAVC', 'CAEC','SMOKE','SCC','CALC','MTRANS','NObeyesdad' ]]
dados_categoricos_normalizados = pd.get_dummies(data=dados_categoricos, dtype=int)
# print(dados_categoricos_normalizados)
normalizador = preprocessing.MinMaxScaler()
modelo_normalizador = normalizador.fit(dados_numericos)

dados_numericos_normalizados = modelo_normalizador.fit_transform(dados_numericos)

dados_numericos_normalizados = pd.DataFrame(data = dados_numericos_normalizados, columns=['Age','Height','Weight','FCVC','NCP','CH2O','FAF','TUE'])

dados_normalizados_final = dados_numericos_normalizados.join(dados_categoricos_normalizados, how='left')

dados_normalizados_final_legiveis = modelo_normalizador.inverse_transform(dados_numericos_normalizados)

dados_normalizados_final_legiveis = pd.DataFrame(data= dados_normalizados_final_legiveis, columns=['Age','Height','Weight','FCVC','NCP','CH2O','FAF','TUE']).join(dados_categoricos_normalizados)
pd.set_option('display.max_columns', None)
print(dados_normalizados_final_legiveis)

print(f"índice do grupo da nova instancia{obesity_clusters_kmeans.predict(dados_normalizados_final_legiveis.values)}")
print(f"Centroide da nova instancia: {obesity_clusters_kmeans.cluster_centers_[obesity_clusters_kmeans.predict([dados_normalizados_final_legiveis])]}")