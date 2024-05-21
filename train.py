from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import pandas as pd
import numpy as np
import torch

# Cargamos los datos
data = pd.read_csv('heart_failure.csv')

# Dividimos los datos en variables dependiente(y) e independientes(X). Nuestro objetivo es predecir la variable DEATH_EVENT.
X, y = data.drop('DEATH_EVENT', axis=1).values, data['DEATH_EVENT'].values

#Dividir los datos en entrenamiento y test. 66% para entrenamiento y 33% para test. Semilla para reproducibilidad.
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.66, random_state=420)

# Definimos el dispositivo donde se ejecutará el modelo. GPU si está disponible, de lo contrario, CPU.
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

# Normalizar los datos. 
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Convertir los datos a tensores.
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

# Mover los datos a la GPU
X_train = X_train.to(device)
X_test = X_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)

# Redimensionar y_train y y_test.
y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)


# Definir el modelo. Heredamos de la clase nn.Module.
class LogisticRegression(nn.Module):
    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1) # Regresión lineal. y = wx + b.

    def forward(self, x):
        y_predicted = torch.sigmoid(self.linear(x)) # Función sigmoide para predecir una variable logística.
        return y_predicted

n_samples, n_features = X.shape
model = LogisticRegression(n_features).to(device)

# Tasa de aprendizaje
learning_rate = 0.01

# Funciones de pérdida y optimizador
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Epochs (iteraciones)
num_epochs = 5000

# Entrenamiento del modelo
for epoch in range(num_epochs):
    # Forward pass. Se calculan las predicciones "iniciales"
    y_predicted = model(X_train)

    # Calculamos la pérdida.
    loss = criterion(y_predicted, y_train)

    # Backward pass. Se calcula el gradiente de la función de pérdida con respecto a los parámetros del modelo. Una especie de retropropagación.
    loss.backward()
    
    # Update. Actualizamos los parámetros del modelo con la nueva información.
    optimizer.step()
    
    # Limpiamos los gradientes.
    optimizer.zero_grad()
    
    # Imprimimos la pérdida cada 10 iteraciones.
    if (epoch+1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

# Evaluamos la precisión del modelo. 
with torch.no_grad():
    y_predicted = model(X_test)  # Predicciones
    y_predicted_cls = y_predicted.round() # Redondeamos las predicciones. Tenemos que redondear porque estamos trabajando con una función sigmoide que predice una variable logística
    acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0]) # Calculamos la precisión. Dividimos el número de predicciones correctas entre el número total de predicciones.
    print(f'accuracy = {acc:.4f}')

# Podemos guardar el modelo para usarlo más tarde.
#torch.save(model.state_dict(), 'model.ckpt')       

print('\n Modelo entrenado. Ahora puedes hacer predicciones.')
a = input('\n Introduzca la edad: ')
b = input('\n Introduzca si tiene anemia (1/0): ')
c = input('\n Introduzca si tiene diabetes (1/0): ')
d = input('\n Introduzca si tiene hipertensión (1/0): ')
e = input('\n Introduzca el sexo (1/0 Hombre/Mujer): ')
f = input('\n Introduzca si fuma (1/0): ')



# Normalizamos los datos
new_data = np.array([[60, 0, 100, 0, 60, 0, 200000, 1.1, 140, 1, 0, 100]])
new_data = sc.transform(new_data)

# Convertimos los datos a tensores
new_data = torch.from_numpy(new_data.astype(np.float32))

# Movemos los datos a la GPU
new_data = new_data.to(device)

# Hacemos la predicción
with torch.no_grad():
    new_data = model(new_data)
    #new_data = new_data.round()
    print(new_data) # Si la predicción es 1, el paciente morirá. Si es 0, el paciente no morirá.

