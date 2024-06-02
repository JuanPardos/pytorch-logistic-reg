from train import LogisticRegression
import pandas as pd
import torch

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

# Asegúrate de que la arquitectura del modelo sea la misma que la del modelo guardado
model = LogisticRegression(12)

# Carga los pesos del modelo
model.load_state_dict(torch.load('model.pth'))

# Asegúrate de que el modelo esté en modo de evaluación
model.eval()

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

#X_new = torch.tensor([edad_v, anemia_v, creatinina_v, diabetes_v, ejection_v, hipertension_v, plaquetas_v, serum_v, sodio_v, sexo_v, smoking_v, tiempo_v], dtype=torch.float32)


X_new = torch.tensor([95,1,582,0,30,0,461000,2,132,1,0,50], dtype=torch.float32)

# Realiza la predicción
with torch.no_grad():
    y_pred = model(X_new)
    #y_pred = y_pred.round()
    print(f'Predicción: {y_pred.item()}')




