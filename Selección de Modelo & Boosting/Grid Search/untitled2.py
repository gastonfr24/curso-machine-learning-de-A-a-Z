"""
Created on Mon Oct  3 01:35:38 2022
@author: Gaston Franco
"""
                                        # Grid Search
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


# creacion del modelo SVM Classifier
from sklearn.svm import SVC

classifier = SVC(
    kernel='rbf',
    random_state=0
    )

classifier.fit(X_train, y_train)


# Testeo deo modelo
y_pred = classifier.predict(X_test)

score = classifier.score(X_test,y_test) #0.93

# matriz de confusión
from sklearn.metrics import confusion_matrix

cf = confusion_matrix(y_test, y_pred)


# Aplicar K-Fold Cross Validation
from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(classifier, X_train, y_train, cv=10) 
# sesgo
accuracies.mean()
# varianza
accuracies.std()


# Aplicar Grid Search para encontrar mejores hiperparámetros
from sklearn.model_selection import GridSearchCV

parametros = [{
                'C':[1, 10, 100, 1000],
                'kernel': ['linear']
               },
    
    
              {
                'C':[1, 10, 100, 1000],
                'kernel': ['rbf'],
                'gamma': [
                    #0.5, 0.01, 0.001, 0.0001
                    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
                    ]
              }
             ]

grid_search = GridSearchCV(classifier, parametros, scoring='accuracy', cv=10, n_jobs=-1)

grid_search = grid_search.fit(X_train, y_train)

# Mejor score
best_score = grid_search.best_score_

# Mejores hiperparámetros
best_params = grid_search.best_params_












