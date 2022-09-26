                                   #Reegresión Lineal Polinómica

# Importar librerias a usar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Importamos em Dataset
df = pd.read_csv('Position_Salaries.csv')

X = df.iloc[:,1:2].values

y =df.iloc[:,2].values




# Creación del modelo

# Regresión Lineal Simple
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression().fit(X,y) 

# Regresión Lineal Polinómica
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(2)
X_poly = poly_reg.fit_transform(X)

lin_reg_2 = LinearRegression().fit(X_poly, y)


plt.scatter(X, y, color='orange')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.show()

plt.scatter(X, y, color='orange')
plt.plot(X, lin_reg_2.predict(X_poly), color='blue')
plt.show()

