import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim import Adam
from sklearn.model_selection import train_test_split

# Pre-procesamiento de datos
def preprocess_labels(Dataframe):
    encoder = LabelEncoder()
    Dataframe['Label'] = encoder.fit_transform(Dataframe['Label'])
    return Dataframe, encoder

class SignLanguageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        path = row['Path']
        label = row['Label']
        video_data = np.load(path)  # Cargar el archivo de video
        video_data = torch.tensor(video_data, dtype=torch.float32)  # Convertir a tensor
        if self.transform:
            video_data = self.transform(video_data)
        return video_data, label

class CNN3D(nn.Module):
    def __init__(self):
        super(CNN3D, self).__init__()

        # Primera capa convolucional
        self.conv1 = nn.Conv3d(
            in_channels=8,  # Videos en blanco y negro (1 canal)
            out_channels=32,  # 32 mapas de características
            kernel_size=(3, 3, 3),  # Tamaño del filtro
            stride=1,  # Paso
            padding=1  # Padding para mantener dimensiones
        )

        # Segunda capa convolucional
        self.conv2 = nn.Conv3d(
            in_channels=32,  # 32 mapas de características de la capa anterior
            out_channels=64,  # 64 mapas de características
            kernel_size=(3, 3, 3),  # Tamaño del filtro
            stride=1,  # Paso
            padding=1  # Padding para mantener dimensiones
        )

        # Capa de pooling (opcional para reducir las dimensiones espaciales)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

        # Capa totalmente conectada
        # Usamos un valor dinámico para el tamaño de la entrada de la capa fc1
        self.fc1 = None  # Lo definiremos después de calcular el tamaño dinámicamente
        self.fc2 = nn.Linear(512, 27)  # 27 clases de salida

    def forward(self, x):
        x = self.conv1(x)  # Aplicar la primera convolución 3D
        x = self.conv2(x)  # Aplicar la segunda convolución 3D
        x = self.pool(x)  # Aplicar el pooling 3D

        # Aplanar la salida para la capa totalmente conectada
        x = x.view(x.size(0), -1)  # Flattening

        # Definir la capa totalmente conectada fc1 usando el tamaño calculado dinámicamente
        if self.fc1 is None:
            self.fc1 = nn.Linear(x.size(1), 512)  # Crear la capa fc1 con el tamaño adecuado
        x = F.relu(self.fc1(x))  # Activación ReLU
        x = self.fc2(x)  # Salida (27 clases)

        return x


# Lectura del dataframe
DataFrame = pd.read_csv('Metadata.csv')

# Crear el dataset y procesar las etiquetas
DataFrame, encoder = preprocess_labels(DataFrame)
sign_language_dataset = SignLanguageDataset(DataFrame)

# Verificar si hay una GPU disponible, si no, usar la CPU
device = torch.device("cpu")
print(f"Usando dispositivo: {device}")

# Dividir el dataframe en entrenamiento y prueba
train_df, val_df = train_test_split(DataFrame, test_size=0.2, random_state=42)

# Crear datasets
train_dataset = SignLanguageDataset(train_df)
val_dataset = SignLanguageDataset(val_df)

# Crear DataLoaders para cargar los datos en lotes
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Crear el modelo y moverlo a la CPU
model = CNN3D().to(device)

# Configurar el optimizador y la función de pérdida
optimizer = Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Función de entrenamiento
def train(model, train_loader, optimizer, criterion, device):
    model.train()  # Poner el modelo en modo de entrenamiento
    running_loss = 0.0
    correct = 0
    total = 0

    for data, labels in train_loader:
        data, labels = data.to(device), labels.to(device)  # Mover los datos a la CPU

        optimizer.zero_grad()  # Limpiar los gradientes

        # Pasar los datos a través del modelo
        outputs = model(data)
        loss = criterion(outputs, labels)  # Calcular la pérdida

        loss.backward()  # Retropropagación
        optimizer.step()  # Actualizar los pesos

        running_loss += loss.item()

        # Calcular el número de aciertos
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

# Función de validación
def validate(model, val_loader, criterion, device):
    model.eval()  # Poner el modelo en modo de evaluación
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Desactivar el cálculo de gradientes durante la validación
        for data, labels in val_loader:
            data, labels = data.to(device), labels.to(device)  # Mover los datos a la CPU

            # Pasar los datos a través del modelo
            outputs = model(data)
            loss = criterion(outputs, labels)  # Calcular la pérdida

            running_loss += loss.item()

            # Calcular el número de aciertos
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(val_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

# Entrenamiento del modelo
num_epochs = 10
for epoch in range(num_epochs):
    model.to(device)
    train_loss, train_accuracy = train(model, train_loader, optimizer, criterion, device)
    val_loss, val_accuracy = validate(model, val_loader, criterion, device)

    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"Entrenamiento - Pérdida: {train_loss:.4f}, Precisión: {train_accuracy:.2f}%")
    print(f"Validación - Pérdida: {val_loss:.4f}, Precisión: {val_accuracy:.2f}%")
