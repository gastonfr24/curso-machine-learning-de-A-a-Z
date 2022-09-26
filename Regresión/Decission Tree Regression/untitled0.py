                            # Regresión con Árboles de Decisión

# Importar librerias a usar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Importar el Dataset
df = pd.read_csv('Position_Salaries.csv')

X = df.iloc[:,1:2].values

y =df.iloc[:,2:3].values


# Escalado Estándar
#from sklearn.preprocessing import StandardScaler
#sc_X = StandardScaler()
#sc_y = StandardScaler()

#X = sc_X.fit_transform(X)
#y = sc_y.fit_transform(y)


# Creación del modelo
# Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor

reg= DecisionTreeRegressor(random_state=0)

reg.fit(X,y)

reg.predict([[6.5]])

# Visualización
plt.scatter(X, y, color='orange')
plt.plot(X, reg.predict(X), color='blue')
plt.show()
