from sklearn.preprocessing import StandardScaler
from train import LogisticRegression
import pandas as pd
import numpy as np
import torch

#FIXME

# Cargamos datos para calcular las medias
data = pd.read_csv('heart_failure.csv')

# Calculamos las medias
edad_mean = data['age'].mean()
creatinina_mean = data['creatinine_phosphokinase'].mean()
ejection_mean = data['ejection_fraction'].mean()
plaquetas_mean = data['platelets'].mean()
serum_mean = data['serum_creatinine'].mean()
sodio_mean = data['serum_sodium'].mean()
tiempo_mean = data['time'].mean()

# 12 variables independientes
model = LogisticRegression(12)

# Cargamos el modelo entrenado. Primero intentamos cargar el modelo preentrenado, si no existe, intentamos cargar el modelo menos exacto.
try:
    model.load_state_dict(torch.load('pretrained_model.pth'))
except FileNotFoundError:
    try:
        model.load_state_dict(torch.load('model.pth'))
    except FileNotFoundError:
        print('No se ha encontrado el modelo. Por favor, ejecute train.py para entrenar el modelo antes de realizar una predicción.')
        exit()

# Se pone el modelo en modo de evaluación
model.eval()

# Dispositivo donde se ejecutará el modelo. Para predecir con CPU nos vale.
device = 'cpu'

print(
    """A continuación puede insertar los valores de las variables independientes.
        Se muestra la media y unidad de medida de cada variable númerica entre paréntesis. 
        Se usará la media o 0 en las binarias si no se especifica un valor"""
)

# Pide al usuario que ingrese los valores de las variables independientes
edad_i = input(f'Edad ({edad_mean} años): ')
anemia_i = input('Anemia (0 o 1): ')
creatinina_i = input(f'Creatinina fosfoquinasa ({creatinina_mean} mcg/L): ')
diabetes_i = input('Diabetes (0 o 1): ')
ejection_i = input(f'Fracción de eyección ({ejection_mean} %): ')
hipertension_i = input('Hipertensión (0 o 1): ')
plaquetas_i = input(f'Plaquetas ({plaquetas_mean} kilopl/mL): ')
serum_i = input(f'Serum creatinina ({serum_mean} mg/dL): ')
sodio_i = input(f'Sodio ({sodio_mean}): ')
sexo_i = input('Sexo (0 mujer o 1 hombre): ')
smoking_i = input('Fumador (0 o 1): ')
tiempo_i = input(f'Tiempo ({tiempo_mean} dias): ')

# Asigna los valores ingresados por el usuario o la media si no se especifica un valor
edad_v = edad_mean if edad_i == '' else float(edad_i)
anemia_v = 0 if anemia_i == '' else int(anemia_i)
creatinina_v = creatinina_mean if creatinina_i == '' else float(creatinina_i)
diabetes_v = 0 if diabetes_i == '' else int(diabetes_i)
ejection_v = ejection_mean if ejection_i == '' else float(ejection_i)
hipertension_v = 0 if hipertension_i == '' else int(hipertension_i)
plaquetas_v = plaquetas_mean if plaquetas_i == '' else float(plaquetas_i)
serum_v = serum_mean if serum_i == '' else float(serum_i)
sodio_v = sodio_mean if sodio_i == '' else float(sodio_i)
sexo_v = 0 if sexo_i == '' else int(sexo_i)
smoking_v = 0 if smoking_i == '' else int(smoking_i)
tiempo_v = tiempo_mean if tiempo_i == '' else float(tiempo_i)

# Prepara los datos de entrada
X_new = np.array([edad_v, anemia_v, creatinina_v, diabetes_v, ejection_v, hipertension_v, plaquetas_v, serum_v, sodio_v, sexo_v, smoking_v, tiempo_v])

# Normaliza los datos
sc = StandardScaler()
X_new = sc.fit_transform(X_new.reshape(1, -1))

# Convierte los datos de entrada a un tensor de PyTorch
X_new = torch.from_numpy(X_new.astype(np.float32))

# Realiza la predicción
with torch.no_grad():
    y_pred = model(X_new)
    print('Probabilidad de fallecimiento: {:.2%}'.format(y_pred.item()))




