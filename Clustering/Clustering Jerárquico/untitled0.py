#Created on Tue Sep 27 13:28:02 2022
#@author: Gaston Franco

                                # Clutering Jerárquico
# importamos librerias a usar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# importamos el dataset
df = pd.read_csv('Mall_Customers.csv')

X = df.iloc[:,[3,4]].values


# Dendrograma para encontrar número óptimo de clusters
import scipy.cluster.hierarchy as sch

# grafico del Dendrograma
dendrograma = sch.dendrogram(sch.linkage(X, method='ward'))

plt.title('Dendrograma')
plt.xlabel('Clientes')
plt.ylabel('Distancia euclidea')
plt.show()


# Ajustar el clustering Jerárquico a nuestro conjunto de datos
from sklearn.cluster import AgglomerativeClustering

cj = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
# metodo ward trata de minimizar la varianza entre clusters***

y_pred = cj.fit_predict(X)


# Grafico de las segmentaciones
colors = ['g', 'm', 'y', 'c', 'b']
# Visualización de los clusters o agrupaciones
for n_cluster,color in enumerate(colors):
    plt.scatter(X[y_pred == n_cluster, 0], X[y_pred == n_cluster, 1], c=color, label=f'Cluster {n_cluster+1}')
plt.title('Clusters de Clientes')
plt.xlabel('Ingresos Anuales')
plt.ylabel('Puntuación de Gastos')
plt.legend()
plt.slow()

plt.scatter(X[:,0], X[:,1])
plt.show()