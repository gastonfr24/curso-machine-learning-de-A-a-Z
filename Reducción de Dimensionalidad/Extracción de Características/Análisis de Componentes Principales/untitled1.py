"""
Created on Sat Oct  1 15:59:27 2022
@author: Gaston Franco
"""
                            # Análisis de Componentes Principales
# importar las librerias a usar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# importar el dataset
df = pd.read_csv('Wine.csv')


# variable dependiente e independientes
X = df.iloc[:,0:-1].values
y = df.iloc[:,-1].values


# división entre train y test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# escalado de variables
# las variables tienen que estar centradas cuando tratamos de reducir las dimensiones
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# Reducir dimension del dataset con ACP
from sklearn.decomposition import PCA

#pca = PCA(n_components= None)
pca = PCA(n_components= 2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

explained_variance = pca.explained_variance_ratio_

# Elegimos solo 2 componentes para hacer el gráfico
# Estos dos representan el 55% de la varianza de nuestro dataset


# creación del modelo de Regresión Logística
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state=0).fit(X_train, y_train)

# Testeo deo modelo
y_pred = classifier.predict(X_test)

score = classifier.score(X_test,y_test) # 0.97

# matriz de confusión
from sklearn.metrics import confusion_matrix

cf = confusion_matrix(y_test, y_pred)






