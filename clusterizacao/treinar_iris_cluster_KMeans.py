# CLUSTER
# Determinação do número ótimo de grupos

import pandas as pd

obesity = pd.read_csv('clusterizacao\ObesityDataSet_raw_and_data_sinthetic.csv', sep = ',')


# Excluir o atribudo class
obesity = obesity.drop(columns=['Gender','family_history_with_overweight', 'FAVC', 'CAEC','SMOKE','SCC','CALC','MTRANS','NObeyesdad' ])
print(obesity.head(5))

# Determinar o número ótimo de grupos pela distorcao
from sklearn.cluster import KMeans # Clusterização
import matplotlib.pyplot as plt
import math
from scipy.spatial.distance import cdist # Para calcular as distâncias e distorções
import numpy as np # Para procedimentos númericos

distortions = []
K = range(1, 101)

# treinar iterativamente conforme n_clusters = K[i]
for i in K:
    obesity_kmeans_model = KMeans(n_clusters = i).fit(obesity)
    distortions.append(sum(np.min(cdist(obesity, obesity_kmeans_model.cluster_centers_, 'euclidean'), axis = 1)/obesity.shape[0]))

print(distortions)

# Exibir o gráfico das distorções
fig, ax = plt.subplots()
ax.plot(K, distortions)
ax.set(xlabel = 'n Clusters', ylabel = 'Distorção', title = 'Elbow pela distorção')
ax.grid()
fig.savefig('clusterizacao\elbow_distorcao.png')
plt.show()

# Calcular o número ótimo de clusters
x0 = K[0]
y0 = distortions[0]
xn = K[len(K) - 1]
yn = distortions[len(distortions)-1]
# Iterar nos pontos gerados durante os treinamentos preliminares
distancias = []
for i in range(len(distortions)):
    x = K[i]
    y = distortions[i]
    numerador = abs((yn-y0)*x - (xn-x0)*y + xn*y0 - yn*x0)
    denominador = math.sqrt((yn-y0)**2 + (xn-x0)**2)
    distancias.append(numerador/denominador)

# Maior distância
n_clusters_otimo = K[distancias.index(np.max(distancias))]

obesity_kmeans_model = KMeans(n_clusters = n_clusters_otimo, random_state=42).fit(obesity)

from pickle import dump
dump(obesity_kmeans_model, open("C:\\Users\\yann_\\OneDrive\\Documentos\\GitHub\\Faculdade-Sistemas-Inteligentes-Avancados\\clusterizacao\\obesity_cluster.pkl", "wb"))