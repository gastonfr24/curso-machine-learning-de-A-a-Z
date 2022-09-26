                                        # K-Means
# importar librerias a usar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# importar dataset
df = pd.read_csv('Mall_Customers.csv')

X = df.iloc[:, [3,4]].values


# Numero optimo de clusters
# metodo del codo:
from sklearn.cluster import KMeans
wcss = []

for clusters in range(1,11):
    kmeans = KMeans(n_clusters=clusters, init="k-means++", n_init=10, max_iter=300, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    

# grafico de los resultados
plt.plot(range(1,11), wcss)
plt.title('Método del codo')
plt.xlabel('clusters', color='blue')
plt.ylabel('wcss(k)', color='blue')

# Número optimo de kmeans = 5
kmeans = KMeans(n_clusters=5, init="k-means++", n_init=10, max_iter=300, random_state=0)
y_pred = kmeans.fit_predict(X)


colors = ['g', 'm', 'y', 'c', 'b']
# Visualización de los clusters o agrupaciones
for n_cluster,color in enumerate(colors):
    plt.scatter(X[y_pred == n_cluster, 0], X[y_pred == n_cluster, 1], c=color, label=f'Cluster {n_cluster+1}')

plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=150, label='Baricentros')
plt.title('Clusters de Clientes')
plt.xlabel('Ingresos Anuales')
plt.ylabel('Puntuación de Gastos')
plt.legend()
plt.slow()

plt.scatter(X[:,0], X[:,1])
plt.show()