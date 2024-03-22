# Aplicação que utiliza o modelo de cluster treinado
# Abrir o modelo
from pickle import load
iris_clusters_kmeans = load(open("Aula15-03\\iris_cluster.pkl", "rb"))
print(iris_clusters_kmeans.cluster_centers_)