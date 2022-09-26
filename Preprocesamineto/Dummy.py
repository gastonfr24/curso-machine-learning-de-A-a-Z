                    # Dummy o OneHotEncoder

import pandas as pd

df = pd.read_csv('data.csv')


# columna con los datos dummies
columnas_dummies = pd.get_dummies(datos['paises'])

# Dataframe con los datos dummies
new_df = pd.get_dummies(datos, comuns=['paises'])


# Quitando una columna dummy
df_final = new_df[:,:-1] 