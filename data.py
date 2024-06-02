import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Referencia
print('Heart Failure Clinical Records (2020) - https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records')

# Cargamos el dataset
data = pd.read_csv('heart_failure.csv')

# Resumen de los datos. No hay nulos. Todas las variables son numéricas. Y usaremos todas las columnas para predecir DEATH_EVENT.
print(data.info())

# Buscamos datos atipicos
# Calculamos percentiles 25 y 75
q25 = data.quantile(0.25)
q75 = data.quantile(0.75)

# Calculamos el rango intercuartílico
iqr = q75 - q25

# Calculamos los límites inferior y superior
lower_bound = q25 - 1.5 * iqr
upper_bound = q75 + 1.5 * iqr

# Filtramos los datos atípicos. No hay.
outliers = data[(data < lower_bound) | (data > upper_bound)].count()

# Descripción de los datos
print(data.describe())

# Correlación de las variables
print(data.corr())

# Usamos la librearía seaborn para visualizar la correlación
f, ax = plt.subplots(figsize = (12,12))
sns.heatmap(data.corr(), annot = True, linewidths=0.5, linecolor = "black", fmt = ".4f", ax = ax)
plt.show()

