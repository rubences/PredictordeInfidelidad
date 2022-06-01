#Importamos los módulos y librerías que vamos a necesitar
#!/usr/bin/env python -W ignore::DeprecationWarning
#!/usr/bin/env python -W ignore::FutureWarning

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import statsmodels.api as sm
import seaborn as sns

from patsy import dmatrices
from scipy import stats
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#Cargamos los datos
dta = sm.datasets.fair.load_pandas().data
dta.head(10)

#Información sobre el dataset: descripción general, origen, 
#definición de variables,tipo de variables

print(sm.datasets.fair.NOTE)
print(sm.datasets.fair.SOURCE)
print(sm.datasets.fair.DESCRLONG)

dta.info()

#Comprobamos que no falten datos (Resultado booleano: true=falta dato, false=dato)
#También se puede visualizar si faltan datos con los mapas de calor de seaborn.
#En este caso, no hace falta.

dta.isnull().head(10)

# Veamos ahora la matriz de correlación. 
# Deberíamos eliminar las variables altamente correlacionadas >0,90
# Edad, años matrimonio-- lógica
# Correlación positiva--religious/rate marriage,age/yrs_marriage
# Correlación negativa: affairs/children, religious

print(dta.corr())

#Edad, años matrimonio-- lógicamente no son independientes, para eliminarlos habría que hacer:
#dta.drop(['age','yrs_married'],axis=1,inplace=True)
#dta.head()

#También podemos ver la matriz de correlación de forma gráfica
#Si los coeficientes son muy bajos, significa que la influencia 
#de esa variable es muy pequeña y,podríamos plantearnos una "reducción" 
#de estas para simplifacar el modelo, pero en este ejemplo no vamos a
#quitar ninguna

%matplotlib inline
sns.heatmap(dta.corr(), annot=True)

#En la fase de exploración, podemos visualizar las variables y 
#sus relaciones mediante histogramas

#Para que muestre los gráficos en el notebook añadimos:
%matplotlib inline

# histograma sobre influencia del nivel educativo
dta.educ.hist()
plt.title('Influencia del Nivel Educativo')
plt.xlabel('Nivel Académico')
plt.ylabel('Frecuencia infidelidad')

# Creamos una nueva variable binaria "infidelity" para tratarlo
#como un problema de clasificación 0=fiel, 1=infiel
# Mostramos los 10 primeros ... infieles

dta['infidelity'] = (dta.affairs > 0).astype(int)
print(dta.head(10))
dta.shape

#Patsy es una librería de Python que permite convertir los datos 
#en el formato de matriz necesario para aplicar el modelo
#También permite generar variables dummy mediante la función C()
#La sintaxis es:

#patsy.dmatrix(formula_like, data={}, eval_env=0, NA_action='drop', return_type='matrix')

from patsy import dmatrices

y, X = dmatrices('infidelity ~ rate_marriage + age +  yrs_married + children + religious+ educ + C(occupation) + C(occupation_husb) ', dta, return_type = 'dataframe')

#Comprobamos las dimensiones y los índices de las matrices resultado
print(X.shape)
print(y.shape)
print (X.columns)
print(y.columns)

#Para que scikit-learn entienda y como variable dependiente (target)
#debemos convertirla de vector columna en matriz 1D
y=np.ravel(y)
print(y)

# sklearn output
model = LogisticRegression(fit_intercept = False, C = 1e9)
mdl = model.fit(X, y)
model.coef_

# Veamos la precisión del modelo (sobre los datos de entrenamiento) 73%
# Más adelante volveremos a entrenar el modelo, pero separando antes los
# datos en train y test. Volveremos a calcular la precisión del modelo
# entrenado sobre los datos train al aplicarlo a los test

model.score(X,y)

#¿Qué porcentaje tiene aventuras?: 32%-- 
#Si predecimos siempre "no", acertaríamos el 68% de las veces,
#algo mejor que el error nulo pero no mucho

y.mean()

# Podemos examinar la matriz de coeficientes, para ver qué peso tiene
# cada uno de los coeficientes. List(zip) permite crear una matriz
# a partir de dos listas, el nombre de los índices, en la primera columna
# y en la segunda, los valores
#pd.DataFrame(list(zip(X.columns, np.transpose(model.coef_)))

pd.DataFrame(list(zip(X.columns, np.transpose(model.coef_))))

# Por ejemplo, los incrementos en rate_marrige y religiousnes disminuyen 
# la probabilidad de infidelidad (coefientes negativos)

# Para evaluar el modelo, dividimos el dataset en dos partes
# un 75% de los datos para entrenar el modelo
# y el 25% restante para evaluarlo

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Ahora, aplicamos el modelo sobre los datos reservados para entrenamiento "train"

model2 = LogisticRegression()
model2.fit(X_train, y_train)

# Una vez entrenado el modelo, lo aplicamos a los datos reservados para "test"

predicted = model2.predict(X_test)
print (predicted)

# Generamos las métricas de evaluación 
# Cuidado con los "false friends"
# "Accuracy" es la precisión, y "Precision" es la exactitud

print("Accuracy:",metrics.accuracy_score(y_test, predicted))
print("Precision:",metrics.precision_score(y_test, predicted))

# Generamos la matriz de confusión

cnf_matrix = metrics.confusion_matrix(y_test, predicted)
cnf_matrix

#Podemos visualizarla con un mapa de calor
#Diagonal values represent accurate predictions, while non-diagonal elements are inaccurate predictions.
#In the output, 119 and 36 are actual predictions, and 26 and 11 are incorrect predictions.


#The accuracy is 73%, which is the same as we experienced when training and predicting on the same data.
#We can also see the confusion matrix and a classification report with other metrics.

# Importamos los módulos necesarios

class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

# Creamos un mapa de calor

sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

# Vamos a usar el modelo para hacer una predicción

# Para que nos resulta más fácil "escribir" los datos de entrada, 
# vamos a sacar un ejemplo de uno de los registros de la matriz de 
# datos, por ejemplo, el 4º, y después lo usaremos de base para 
# introducir en el modelo los datos que querarmos.

print(X.iloc[4])
F=X.iloc[4]
F.shape

#Con reshape(1,-1) indicamos que lo convierta en una matriz de 1 fila y 
#el número de columnas que corresponda para que la nueva forma sea  
#compatible con la original

F.values.reshape(1,-1)
model.predict_proba(F.values.reshape(1, -1))

F.keys();
F['age']=35; F['children']=3; F['yrs_married']=10; F['religious']=1; F['religious']=1; F['C(occupation_husb)[T.3.0]']=1
print(F.values)

# Aplicamos el modelo a este nuevo conjunto de valores y obtenmos
# la probabilidad de infidelidad que, en este caso es de un 29%

F.values.reshape(1,-1)
model.predict_proba(F.values.reshape(1, -1))