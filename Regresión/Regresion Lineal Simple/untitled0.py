# Regresión Lineal Simple

#Importar las librerías
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Cargamos el Dataset
df = pd.read_csv('Salary_Data.csv')


# Dividimos en variable dependiente e independiente
X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values


# Division de set de entrenamiento y prueba
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 1/3, random_state=0)

                # el modelo de regresion lineal simple no lleva escalado #
                
# Creación del modelo de Regresión Lineal Simple
from sklearn.linear_model import LinearRegression

reg = LinearRegression().fit(X_train, y_train)


# Predecir el conjunto de Test
y_pred = reg.predict(X_test)


# Visualizar los resultados del conjunto de train
plt.scatter(X_train, y_train, color='orange')
plt.plot(X_train, reg.predict(X_train), color='blue')
plt.title('Salarios x Años de Exp.(Train Data)' )
plt.xlabel('Años de Experiencia')
plt.ylabel('Salario(USD)')
plt.show()



# Visualizar los resultados del conjunto de test
plt.scatter(X_test, y_test, color='orange')
plt.plot(X_train, reg.predict(X_train), color='blue')
plt.title('Salarios x Años de Exp.(Train Data)' )
plt.xlabel('Años de Experiencia')
plt.ylabel('Salario(USD)')
plt.show()

reg.score(X_test, y_test)
























