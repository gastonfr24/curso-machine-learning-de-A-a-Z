                            # Regresión con Bosques Aleatorios

# Importar librerias a usar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Importar el Dataset
df = pd.read_csv('Position_Salaries.csv')

X = df.iloc[:,1:2].values

y =df.iloc[:,2].values


# Escalado Estándar
#from sklearn.preprocessing import StandardScaler
#sc_X = StandardScaler()
#sc_y = StandardScaler()

#X = sc_X.fit_transform(X)
#y = sc_y.fit_transform(y)


# Creación del modelo
# Random Forest Regression
from sklearn.ensemble import RandomForestRegressor 

reg= RandomForestRegressor(
    # numero de árboles
    n_estimators=300,
    random_state=0
    )

reg.fit(X,y)

reg.predict([[6.5]])

# Visualización
X_grid = np.arange(min(X), max(X),0.01)
X_grid = X_grid.reshape(-1,1)

plt.scatter(X, y, color='orange')
plt.plot(X_grid, reg.predict(X_grid), color='blue')
plt.show()
