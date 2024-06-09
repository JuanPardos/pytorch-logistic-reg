from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from data import get_data
import torch.nn as nn
import pandas as pd
import numpy as np
import torch
import time
import os

# Cargamos los datos
if os.path.isfile('heart_failure.csv'):
    data = pd.read_csv('heart_failure.csv')
else:
    data = get_data()
    input_save = input('Datos cargados correctamente. Desea guardarlos en un fichero ? (S/N): ')
    if input_save.lower() == 's':
        data.to_csv('heart_failure.csv', index=False)
        print('Guardados correctamente.')

# Copiamos los datos originales para la predicción final
original_data = data.copy()

# Definimos el dispositivo donde se ejecutará el modelo. GPU si está disponible, de lo contrario, CPU.
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

# Variables que NO vamos a usar para predecir.
var_ignore = ['DEATH_EVENT']  # 'time' 

# Variables independientes(X) y dependendientes(y). Nuestro objetivo es predecir la variable dependiente DEATH_EVENT. Values devuelve un array de numpy.
X, y = data.drop(var_ignore, axis=1).values, data['DEATH_EVENT'].values

#Dividir los datos en entrenamiento y test. 66% para entrenamiento y 33% para test. Semilla para garantizar reproducibilidad.
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.66, random_state=33)

# Normalizar los datos. Todos los datos deben tener la misma escala.
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Convertir los datos a tensores con los que podamos trabajar
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

# Mover los datos a la GPU
X_train = X_train.to(device)
X_test = X_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)

# Redimensionar y_train y y_test. Pasar de (n,) a (n, 1)
y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)

# Definir el modelo. Heredamos de la clase nn.Module.
class LogisticRegression(nn.Module):
    def __init__(self, n_input_features, n_hidden_layers):
        super(LogisticRegression, self).__init__()
        self.hidden = nn.Linear(n_input_features, n_hidden_layers) 
        self.output = nn.Linear(n_hidden_layers, 1)

    def forward(self, x):
        x = torch.sigmoid(self.hidden(x))
        y_predicted = torch.sigmoid(self.output(x)) #Función de activación sigmoide para obtener valores entre 0 y 1
        return y_predicted

# Número de características. n_samples = número de registros, n_features = número de columnas.
n_samples, n_features = X.shape

# Semilla de torch. Para garantizar reproducibilidad en las capas ocultas.
torch.manual_seed(13)  #13

# Número de neuronas en la capa oculta
hidden_layers = 20  #20

# Tasa de aprendizaje
learning_rate = 0.05  #0.05

# Epochs (iteraciones)
num_epochs = 15000  #15000

# Instanciar el modelo
model = LogisticRegression(n_features, hidden_layers).to(device)

# Funciones de pérdida y optimizador
criterion = nn.MSELoss()     #MSELoss
optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate, weight_decay=5e-6)   #Adagrad

# Almacenar la pérdida. Se usará en la representación gráfica.
losses = []

# Entrenamiento
if __name__ == '__main__':
    t0 = time.time()
    for epoch in range(num_epochs):
        # Forward pass (predicción)
        y_predicted = model(X_train)

        # Calculamos la pérdida y la almacenamos para su representación gráfica
        loss = criterion(y_predicted, y_train)
        losses.append(loss.item())

        # Backward pass. Se calcula el gradiente de la función de pérdida con respecto a los parámetros del modelo.
        loss.backward()
        
        # Update. Actualizamos los pesos y otros parámetros del modelo con la nueva información
        optimizer.step()
        
        # Limpiamos los gradientes
        optimizer.zero_grad()
        
        # Imprimimos la pérdida cada 10 iteraciones
        if (epoch+1) % 10 == 0:
            print(f'Epoch: {epoch+1}, pérdida = {loss.item():.4f}')

    print(f'\nDispositivo usado en el entrenamiento: ', device)
    print(f'Tiempo de entrenamiento: {time.time()-t0:.2f} segundos')
    print('Epoch/s: {:.2f}'.format(num_epochs/(time.time()-t0)))
    
    # Representar pérdida a lo largo del tiempo. Da problemas en windows.
    """ plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show() """

    # Evaluamos la precisión del modelo con el conjunto de test
    with torch.no_grad():
        y_predicted = model(X_test)

        # Estadísticas
        acc = torch.sum(y_predicted.round() == y_test)/len(y_test)
        misses = torch.sum(y_predicted.round() != y_test)
        doubtful = torch.sum((y_predicted >= 0.4) & (y_predicted <= 0.6)) # Consideramos dudosos aquellos valores entre 0.4 y 0.6

        TP = torch.sum((y_predicted.round() == 1) & (y_test == 1))
        TN = torch.sum((y_predicted.round() == 0) & (y_test == 0))
        FP = torch.sum((y_predicted.round() == 1) & (y_test == 0))
        FN = torch.sum((y_predicted.round() == 0) & (y_test == 1))

        TPR = TP/(TP+FN)
        TNR = TN/(TN+FP)
        FPR = FP/(FP+TN)
        FNR = FN/(FN+TP)

        y_test = y_test.cpu().numpy()
        y_predicted = y_predicted.cpu().numpy()
        roc_auc = roc_auc_score(y_test, y_predicted)

        print('\n=== Validación ===')
        print(f'Nº registros: {len(y_test)}')
        print(f'Precisión: {acc.item()*100:.2f}%')
        print(f'Errores: {misses.item()}')
        print(f'Indecisos: {doubtful.item()} ({doubtful.item()/len(y_test)*100:.2f}%)')  

        print('\n=== Estadísticas ===')
        print(f'TPR: {TPR.item():.2f}')
        print(f'TNR: {TNR.item():.2f}')
        print(f'FPR: {FPR.item():.2f}')
        print(f'FNR: {FNR.item():.2f}')
        print(f'ROC AUC: {roc_auc:.2f}')

        print('\n=== Matriz de confusión ===')
        print(f'{"":<10} {"Predicción"}')
        print(f'{"Real":<10} {"0":<10} {"1":<10}')
        print(f'{"0":<10} {TN.item():<10} {FP.item():<10}')
        print(f'{"1":<10} {FN.item():<10} {TP.item():<10}')
        
    # Predecimos el conjunto original de datos. Añadimos columnas de predicción y error.
    with torch.no_grad():
        predicciones = model(torch.from_numpy(sc.transform(original_data.drop(var_ignore, axis=1).values).astype(np.float32)).to(device)).cpu().numpy()
        original_data['PREDICT'] = predicciones.round().astype(int) # Redondeamos a 0 o 1
        original_data['P1'] = predicciones.round(4) # Probabilidad con 4 decimales
        original_data['ERROR'] = np.abs(original_data['P1'] - original_data['DEATH_EVENT']).round(4)

    # Guardamos modelo y dataset con predicciones
    user_input = input('\n¿Desea guardar el modelo y dataset con las predicciones? (S/N): ')
    if user_input.lower() == 's':
        torch.save(model.state_dict(), 'model.pth')
        original_data.to_csv('heart_failure_predict.csv', index=False)
        print('Guardado correctamente.')