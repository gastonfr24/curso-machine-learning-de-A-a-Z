#Created on Tue Sep 27 17:18:27 2022
#@author: Gaston Franco

                                          # Apriori
# Importar librerias a usar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Importar dataset
df = pd.read_csv('Market_Basket_Optimisation.csv', header=None)

# Preprocesamiento
transacciones = []

for compra in range(0,len(df)):
    transacciones.append([str(df.values[compra,producto]) for producto in range(0,20)])
    

# Entrenar algoritmo Apriori
from apyori import apriori

# reglas de asocianción
rules = apriori(transacciones,
                # soporte minimo
                min_support = 0.003,
                # minimo nivel de confiaza
                min_confidence = 0.2,
                # minimo lift para considerar asociación
                min_lift = 3,
                # minimo de asociaciones 
                min_length = 2,
                # maximo de asociaciones
                #max_length = 
                )

# para elegir el soporte minimo tenemos que elegir los productos que mas se compren
# por ejemplo uno que se compre 3 veces al dia minimo, 3 X 7 dias a la semana = 21 
# (7 si es que el dataframe es el reporte de una semana, si no 30 si es un mes )
# Ahora lo dividimos por la cantidad de transacciones o compras 21/7500 = 0.0028

results = list(rules)

for i in range(len(results)):
    print(results[i][0])




