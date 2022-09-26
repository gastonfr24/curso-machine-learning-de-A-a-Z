                                    # Support Vector Regression

# Importar librerias a usar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Importamos el Dataset
df = pd.read_csv('Position_Salaries.csv')

X = df.iloc[:,1:2].values

y =df.iloc[:,2:3].values


# Escalado Estándar
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()

X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)


# Creación del modelo
# Support Vector Machine
from sklearn.svm import SVR

svreg = SVR(
    kernel= "rbf",
    
    )

svreg.fit(X, y)

y_pred = svreg.predict(sc_X.transform([[6.5]]))
y_pred = sc_y.inverse_transform([y_pred])

# Visualización
plt.scatter(X, y, color='orange')
plt.plot(X, svreg.predict(X), color='blue')
plt.show()

