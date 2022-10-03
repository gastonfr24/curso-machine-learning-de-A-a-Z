                    # Eliminación hacia Atrás


#Construir el modelo óptimo de RLM posible usando la eliminación hacia atrás
import statsmodels.formula.api as sm

X = np.append(arr=np.ones((50,1)).astype(int), values= X , axis=1)

# en esta X se quedarán las variables independientes con valores estadisticamente significativos
# para ser capaces de predecir la variable dependinte


# PASO 1 : SELECCION DEL NIVEL DE SIGNIFICACIÓN
# Nivel de significación
SL = 0.05

#PASO 2 : SE CALCULA EL MODELO CON TODAS LAS POSIBLES VARIABLES PREDICTORIAS

# en esta X se quedarán las variables independientes con valores estadisticamente significativos
# para ser capaces de predecir la variable dependinte

# Partimos con todas las columnas y eliminará la menos significativa 
X_opt = X[:,[0, 1, 2, 3, 4, 5, 6, ...]].tolist()

reg_ols = sm.OLS(y, X_opt).fit()

#PASO 3 : CONSIDERAR LA VARIABLE PRODICTORA CON EL P-VALUE MAS ALTO. 
# SI P-V > SL ENTONCES VAMOS AL PASO 4 SI NO VAMOS AL FIN

reg_ols.summary()


#PASO 4: ELIMINAR LA DE P-VALUE MAS ALTO

#4
X_opt = X[:,[0, 1, 2, 3, """4,""" 5, 6, ...]].tolist()

# Cuando el P-Value sea mas bajo que el sl terminamos con el modelo


#PASO 5: FINAL



#Eliminación hacia atrás utilizando solamente p-valores:

import statsmodels.api as sm
def backwardElimination(x, sl):    
    numVars = len(x[0])    
    for i in range(0, numVars):        
        regressor_OLS = sm.OLS(y, x.tolist()).fit()        
        maxVar = max(regressor_OLS.pvalues).astype(float)        
        if maxVar > sl:            
            for j in range(0, numVars - i):                
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):                    
                    x = np.delete(x, j, 1)    
    regressor_OLS.summary()    
    return x 
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)


#Eliminación hacia atrás utilizando  p-valores y el valor de  R Cuadrado Ajustado:


import statsmodels.formula.api as sm
def backwardElimination(x, SL):    
    numVars = len(x[0])    
    temp = np.zeros((50,6)).astype(int)    
    for i in range(0, numVars):        
        regressor_OLS = sm.OLS(y, x.tolist()).fit()        
        maxVar = max(regressor_OLS.pvalues).astype(float)        
        adjR_before = regressor_OLS.rsquared_adj.astype(float)        
        if maxVar > SL:            
            for j in range(0, numVars - i):                
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):                    
                    temp[:,j] = x[:, j]                    
                    x = np.delete(x, j, 1)                    
                    tmp_regressor = sm.OLS(y, x.tolist()).fit()                    
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)                    
                    if (adjR_before >= adjR_after):                        
                        x_rollback = np.hstack((x, temp[:,[0,j]]))                        
                        x_rollback = np.delete(x_rollback, j, 1)     
                        print (regressor_OLS.summary())                        
                        return x_rollback                    
                    else:                        
                        continue    
    regressor_OLS.summary()    
    return x 
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)

