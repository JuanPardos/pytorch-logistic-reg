import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Referencia
print('Heart Failure Clinical Records (2020) - https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records')

# Cargamos el dataset
data = pd.read_csv('heart_failure.csv')

# Grafico con la distribución de las variables
data.hist(figsize = (10,10))
plt.show()

# Resumen de los datos. No hay nulos. Todas las variables son numéricas. Y usaremos todas las columnas para predecir DEATH_EVENT.
print(data.info())

# Descripción de los datos
print(data.describe())

# Correlación de las variables
print(data.corr())

# Usamos la librearía seaborn para visualizar la correlación
f, ax = plt.subplots(figsize = (12,12))
sns.heatmap(data.corr(), annot = True, linewidths=0.5, linecolor = "black", fmt = ".4f", ax = ax)
plt.show()

# Guardamos el dataset con los cambios realizados
data.to_csv('heart_failure.csv', index=False)

