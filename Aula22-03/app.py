# Aplicação que utiliza o modelo de cluster treinado
# Abrir o modelo
from pickle import load
iris_clusters_kmeans = load(open("Aula15-03\\iris_cluster.pkl", "rb"))
print(iris_clusters_kmeans.cluster_centers_)

# Obter uma nova instancia de dados
nova_instancia = [6.5, 3.0, 5.2, 2.0]
print(f"índice do grupo da nova flor{iris_clusters_kmeans.predict([nova_instancia])}")
print(f"Centroide da nova flor: {iris_clusters_kmeans.cluster_centers_[iris_clusters_kmeans.predict([nova_instancia])]}")