#Created on Wed Sep 28 16:04:26 2022
#@author: Gaston Franco
                                # Upper Confidence Bound
# importar librerias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# importar dataset
df = pd.read_csv('Ads_CTR_Optimisation.csv')


#Algoritmo UCB

#Paso 1

N = len(df)
d = df.shape[1]

numero_de_selecciones = [0]* d
suma_de_recompensas = [0] * d

anuncios_seleccionados = []
recompensa_total = 0
for n in range(0,N):
    limite_superior_maximo = 0
    ad = 0
    for i in range(0,d):
        if numero_de_selecciones[i]>0:
            recompensa_media = suma_de_recompensas[i]/numero_de_selecciones[i]
            delta_i = math.sqrt(3/2* math.log(n+1)/numero_de_selecciones[i])
            intevalo_superior = recompensa_media + delta_i
        else:
            intevalo_superior = 1e400
            
            
        if intevalo_superior > limite_superior_maximo:
            limite_superior_maximo = intevalo_superior
            ad = i
    
    anuncios_seleccionados.append(ad)
    numero_de_selecciones[ad] += 1
    recompensa = df.values[n, ad]
    suma_de_recompensas[ad] += recompensa 
    recompensa_total += recompensa
    
    
    
    
# Histograma de resultados
plt.hist(anuncios_seleccionados, color='orange')
plt.title('Histograma de Anuncios', fontsize=12)
plt.xlabel('ID del Anuncio')
plt.ylabel('Frecuencia de visualización del anuncio')
plt.show()
    
    
    
    