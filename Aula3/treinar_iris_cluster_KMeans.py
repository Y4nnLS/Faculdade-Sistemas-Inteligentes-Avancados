# CLUSTER
# Determinação do número ótimo de grupos

import pandas as pd

iris = pd.read_csv('Aula15-03\\iris.csv', sep = ';')


# Excluir o atribudo class
iris = iris.drop(columns='class')
print(iris.head(5))

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
    iris_kmeans_model = KMeans(n_clusters = i).fit(iris)
    distortions.append(sum(np.min(cdist(iris, iris_kmeans_model.cluster_centers_, 'euclidean'), axis = 1)/iris.shape[0]))

print(distortions)

# Exibir o gráfico das distorções
fig, ax = plt.subplots()
ax.plot(K, distortions)
ax.set(xlabel = 'n Clusters', ylabel = 'Distorção', title = 'Elbow pela distorção')
ax.grid()
fig.savefig('elbow_distorcao.png')
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

iris_kmeans_model = KMeans(n_clusters = n_clusters_otimo, random_state=42).fit(iris)

from pickle import dump
dump(iris_kmeans_model, open("C:\\Users\\yann_\\OneDrive\\Documentos\\GitHub\\Faculdade-Sistemas-Inteligentes-Avancados\\Aula15-03\\iris_cluster.pkl", "wb"))