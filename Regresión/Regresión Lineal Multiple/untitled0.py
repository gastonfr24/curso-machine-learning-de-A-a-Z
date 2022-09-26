                                   # Regresión Lineal Mútiple
# Importar librerias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Importamos el DataFrame
df = pd.read_csv('50_Startups.csv')


# Variables Independientes

X = df.iloc[:,:-1]


# Variable Dependiente 
y = df.iloc[:, -1].values


# Variables Dummy o OneHotEncoder
df['State'].value_counts()

X = pd.get_dummies(X, columns=['State']).values

X = X[:,:-1] 


# División del set de datos (train, test)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Ajustar el modelo de Regresión Lineal con el conjunto de entrenamiento
# Creación del modelo de Regresión Lineal Múltiple
from sklearn.linear_model import LinearRegression

reg = LinearRegression().fit(X_train, y_train)


# Exportar modelo
#from joblib import dump
#dump(reg, 'regression.joblib') 

# Importar modelo
#from joblib import load
#regression_model = load('regression.joblib')

# Predicción de los resultados en el contunto de test
y_pred = reg.predict(X_test)


#Construir el modelo óptimo de RLM posible usando la eliminación hacia atrás

# Agregamos una columna de puros "1" para simular la ordenada al origen
import statsmodels.api as sm

X = np.append(arr=np.ones((50,1)).astype(int), values= X , axis=1)


# PASO 1 : SELECCION DEL NIVEL DE SIGNIFICACIÓN
# Nivel de significación
SL = 0.05


#PASO 2 : SE CALCULA EL MODELO CON TODAS LAS POSIBLES VARIABLES PREDICTORIAS

# en esta X se quedarán las variables independientes con valores estadisticamente significativos
# para ser capaces de predecir la variable dependinte

# Partimos con todas las columnas y eliminaremos la menos significativa 
X_opt = X[:, [0,1,2,3,4,5]].tolist()

reg_ols = sm.OLS(y, X_opt).fit()


#PASO 3 : CONSIDERAR LA VARIABLE PRODICTORA CON EL P-VALUE MAS ALTO. 
# SI P-V > SL ENTONCES VAMOS AL PASO 4 SI NO VAMOS AL FIN

reg_ols.summary()
#                 coef    std err          t      P>|t|      [0.025      0.975]
#const       5.008e+04   6952.587      7.204      0.000    3.61e+04    6.41e+04 |Coeficiente u ordenada al origen
#x1             0.8060      0.046     17.369      0.000       0.712       0.900 |R&D Spend
#x2            -0.0270      0.052     -0.517      0.608      -0.132       0.078 |Administracion
#x3             0.0270      0.017      1.574      0.123      -0.008       0.062 |Marketing
#x4            41.8870   3256.039      0.013      0.990   -6520.229    6604.003 |State(Califronia)
#x5           240.6758   3338.857      0.072      0.943   -6488.349    6969.701 |State(Florida)


#Dep. Variable:                      y   R-squared:                       0.951
#Model:                            OLS   Adj. R-squared:                  0.945
#Method:                 Least Squares   F-statistic:                     169.9
#Date:                Sun, 18 Sep 2022   Prob (F-statistic):           1.34e-27
#Time:                        19:52:04   Log-Likelihood:                -525.38
#No. Observations:                  50   AIC:                             1063.
#Df Residuals:                      44   BIC:                             1074.
#Df Model:                           5                                         
#Covariance Type:            nonrobust                                         

#PASO 4: ELIMINAR LA DE P-VALUE MAS ALTO

# En este caso es este:
#x5           240.6758   3338.857      0.072      0.943   -6488.349    6969.701 |State(Florida)

#Volvemos a ejecutar el paso 2, 3 y 4 hasta que el P-Value de las que queden sean menos que el SL

X_opt = X[:, [0,1,2,3,4]].tolist()
reg_ols = sm.OLS(y, X_opt).fit()
reg_ols.summary()

#                 coef    std err          t      P>|t|      [0.025      0.975]
#------------------------------------------------------------------------------
#const       5.016e+04   6798.992      7.377      0.000    3.65e+04    6.39e+04
#x1             0.8057      0.046     17.646      0.000       0.714       0.898
#x2            -0.0268      0.052     -0.520      0.606      -0.131       0.077
#x3             0.0272      0.017      1.627      0.111      -0.006       0.061
#x4           -70.2265   2828.752     -0.025      0.980   -5767.625    5627.172


X_opt = X[:, [0,1,2,3]].tolist()
reg_ols = sm.OLS(y, X_opt).fit()
reg_ols.summary()


#[0.025      0.975]
#------------------------------------------------------------------------------
#const       5.012e+04   6572.353      7.626      0.000    3.69e+04    6.34e+04
#x1             0.8057      0.045     17.846      0.000       0.715       0.897
#x2            -0.0268      0.051     -0.526      0.602      -0.130       0.076
#x3             0.0272      0.016      1.655      0.105      -0.006       0.060



X_opt = X[:, [0,1,2]].tolist()
reg_ols = sm.OLS(y, X_opt).fit()
reg_ols.summary()


#                 coef    std err          t      P>|t|      [0.025      0.975]
#------------------------------------------------------------------------------
#const       5.489e+04   6016.718      9.122      0.000    4.28e+04     6.7e+04
#x1             0.8621      0.030     28.589      0.000       0.801       0.923
#x2            -0.0530      0.049     -1.073      0.289      -0.152       0.046

X_opt = X[:, [0,1]].tolist()
reg_ols = sm.OLS(y, X_opt).fit()
reg_ols.summary()

#                 coef    std err          t      P>|t|      [0.025      0.975]
#------------------------------------------------------------------------------
#const       4.903e+04   2537.897     19.320      0.000    4.39e+04    5.41e+04
#x1             0.8543      0.029     29.151      0.000       0.795       0.913


X_opt = X[:,1:2]

X_train_opt, X_test_opt, y_train, y_test = train_test_split(X_opt, y, test_size=0.2, random_state=0)

reg_opt = LinearRegression().fit(X_train_opt, y_train)

y_pred_opt = reg_opt.predict(X_test_opt)

#np.savetxt('x.csv', X, delimiter=",")

