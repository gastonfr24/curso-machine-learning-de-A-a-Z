                                    # Random Forest
# importamos las librerias a usar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# importamos el dataset
df = pd.read_csv('Social_Network_Ads.csv')


# variable dependiente e independientes
X = df.iloc[:,2:4]
y = df.iloc[:,-1]


# división entre train y test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/4, random_state=0)

##escalado de variables
# from sklearn.preprocessing import StandardScaler
# sc_X = StandardScaler()

# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)


# creación del modelo Random Forest
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=10,criterion="entropy", random_state=1)

# gini mide la dispersion de los datos
# entropia mide el caos (se tiene que minimizar)

classifier.fit(X_train, y_train)

# Testeo deo modelo
y_pred = classifier.predict(X_test)

score = classifier.score(X_test,y_test) # 0.93

# matriz de confusión
from sklearn.metrics import confusion_matrix

cf = confusion_matrix(y_test, y_pred)
#64	04
#03	29