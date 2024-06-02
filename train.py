from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import torch.nn as nn
import pandas as pd
import numpy as np
import torch

# Cargamos los datos
data = pd.read_csv('heart_failure.csv')
original_data = data.copy()

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
        self.linear = nn.Linear(n_input_features, 1) # Regresión lineal. y = wx + b. 12 variables de entrada y 1 de salida.

    def forward(self, x):
        y_predicted = torch.sigmoid(self.linear(x)) # Función sigmoide para predecir una variable logística.
        return y_predicted

n_samples, n_features = X.shape
model = LogisticRegression(n_features).to(device)

# Tasa de aprendizaje
learning_rate = 0.1

# Funciones de pérdida y optimizador
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Epochs (iteraciones)
num_epochs = 15000

# Almacenar la pérdida. Se usará en la representación gráfica.
losses = []

# Entrenamiento del modelo
if __name__ == '__main__':
    for epoch in range(num_epochs):
        # Forward pass. Se calculan las predicciones "iniciales"
        y_predicted = model(X_train)

        # Calculamos la pérdida y la almacenamos para su representación gráfica.
        loss = criterion(y_predicted, y_train)
        losses.append(loss.item())

        # Backward pass. Se calcula el gradiente de la función de pérdida con respecto a los parámetros del modelo. Una especie de retropropagación.
        loss.backward()
        
        # Update. Actualizamos los parámetros del modelo con la nueva información.
        optimizer.step()
        
        # Limpiamos los gradientes.
        optimizer.zero_grad()
        
        # Imprimimos la pérdida cada 10 iteraciones.
        if (epoch+1) % 10 == 0:
            print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

    
    # Representamos gráficamente la pérdida.
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    # Evaluamos la precisión del modelo. 
    with torch.no_grad():
        y_predicted = model(X_test)
        acc = torch.sum(y_predicted.round() == y_test)/len(y_test)
        print(f'accuracy: {acc.item():.4f}')
    
    # Guardamos el modelo.
    torch.save(model.state_dict(), 'model.pth')

    with torch.no_grad():
        original_data['PREDICT'] = model(torch.from_numpy(sc.transform(original_data.drop('DEATH_EVENT', axis=1).values).astype(np.float32)).to(device)).cpu().numpy().round()
    
    fallos = original_data[original_data['DEATH_EVENT'] != original_data['PREDICT']]
    print(len(fallos))

    # Guardamos el dataframe con la predicción
    original_data.to_csv('heart_failure_predict.csv', index=False)
