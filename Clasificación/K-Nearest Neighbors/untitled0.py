                                # K-Nearest Neighbors (KNN)
# importar librerias a usar
import numpy as np
import pandas as pd
import matplotlib.pyplot as pls

# importar el dataset
df = pd.read_csv('Social_Network_Ads.csv')

# Variable dependiente e independiente/s
X = df.iloc[:,2:4]
y = df.iloc[:,-1]


# división entre train y test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/4, random_state=0)

# escalado de variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# creacion del modelo K-NN
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(5, p=2, metric='minkowski')
classifier.fit(X_train, y_train)


# Testeo deo modelo
y_pred = classifier.predict(X_test)

score = classifier.score(X_test,y_test) #0.93

# matriz de confusión
from sklearn.metrics import confusion_matrix

cf = confusion_matrix(y_test, y_pred)


