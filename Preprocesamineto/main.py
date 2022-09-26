                                # Machine Learning: Preprocesamiento - 1

# axis = 0  //columnas
# axis = 1 //filas

                                        # Importar Librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


                                        # Importación del dataset
df = pd.read_csv('Data.csv')
 
#variable independiente
X = df.iloc[:,:-1].values

#variable dependiente
y = df.iloc[:,-1].values

                                        # Datos faltantes o null
from sklearn.impute import SimpleImputer

#entrenamos el inpute
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer.fit(X[:,1:3])

# aplicamos el inpute
X[:,1:3] = imputer.transform(X[:,1:3])

                                       # Datos Categóricos

# Label Encoder sirve para tallas de ropa( S, M, L, XL)
# Sirve para datos ORDINALES que llevan un orden y NO para datos Categóricos
# Sirve para datos de 2 categorias o Booleanas (Verdadero o Falso, Si o No) ya que los convierte a 1 y 0
from sklearn.preprocessing import LabelEncoder

"""le = LabelEncoder()
X[:,0] = le.fit_transform(X[:,0])"""

# Variables Dummy o OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
 
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [0])],   
    remainder='passthrough'                        
)
X = np.array(ct.fit_transform(X), dtype=np.float)

# Label Encoder
le = LabelEncoder()
y = le.fit_transform(y)


                                    # Dividir el dataset en Train y Test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)



                                    #Escalado de variables datos numéricos
# Hay 2 tipos de escalados: Estandarización y Normalización

# Estándar
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Y no se escala por que es un modelo de clasificación binaria













