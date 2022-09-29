#Created on Thu Sep 29 14:36:30 2022
#@author: Gaston Franco

                        # Natural Processing Lenguage
# Importar librerias 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importar dataset
df = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)

# Limpieza de texto
import re
import nltk

# Descargar palabras irrelevantes
#nltk.download('stopwords')
from nltk.corpus import stopwords
noise = stopwords.words('english')

# Eliminar conjugaciones de palabras
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

"""
review = re.sub('[^a-zA-z]',' ', df.Review[0])
review = review.lower()
review = review.split()
review = [ps.stem(word) for word in review if not word in set(noise)]
review = ' '.join(review)
"""
corpus = []

for sentence in range(0, len(df)):
    review = re.sub('[^a-zA-z]',' ', df.Review[sentence])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(noise)]
    review = ' '.join(review)
    corpus.append(review)
    

# Crear Bolsa de Palabras (Bag of Words)
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(max_features=1500)

# Variables independientes y dependiete
X = vectorizer.fit_transform(corpus).toarray()
y = df.iloc[:,1].values


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)


# escalado de variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# creaciÃ³n del modelo NaÃ¯ves Bayes
from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit(X_train, y_train)

# Testeo deo modelo
y_pred = classifier.predict(X_test)

score = classifier.score(X_test,y_test) # 0.705

# matriz de confusiÃ³n
from sklearn.metrics import confusion_matrix

cf = confusion_matrix(y_test, y_pred)
















