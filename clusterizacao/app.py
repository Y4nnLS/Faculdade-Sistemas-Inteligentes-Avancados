# Aplicação que utiliza o modelo de cluster treinado
# Abrir o modelo
from pickle import load
obesity_clusters_kmeans = load(open("clusterizacao\obesity_cluster.pkl", "rb"))
print(obesity_clusters_kmeans.cluster_centers_)

# Obter uma nova instancia de dados
nova_instancia = [20, 1.59, 60, 2.0, 3.0, 2.0, 2.0, 0.6, 0,0,0,1,1,0,0,1,0,0,1,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,0,0,0,0]
print(f"índice do grupo da nova instancia{obesity_clusters_kmeans.predict([nova_instancia])}")
print(f"Centroide da nova instancia: {obesity_clusters_kmeans.cluster_centers_[obesity_clusters_kmeans.predict([nova_instancia])]}")