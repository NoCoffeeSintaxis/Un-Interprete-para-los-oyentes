import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

#Pre-procesamiento de datos

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
        video_data = np.load(path)  # Cargar los datos del video

        # Cambiar la organización del shape: agregar una dimensión de canal
        video_data = np.expand_dims(video_data, axis=0)  # Forma: [1, frames, altura, anchura]
        video_data = torch.tensor(video_data, dtype=torch.float32)  # Convertir a tensor

        if self.transform:
            video_data = self.transform(video_data)
            
        return video_data, label


class CNN3D(nn.Module):
    def __init__(self):
        super(CNN3D, self).__init__()

        # Primera capa convolucional
        self.conv1 = nn.Conv3d(
            in_channels=1,
            out_channels=32,
            kernel_size=(3, 3, 3),
            stride=1,
            padding=1
        )

        # Segunda capa convolucional
        self.conv2 = nn.Conv3d(
            in_channels=32,
            out_channels=64,
            kernel_size=(3, 3, 3),
            stride=1,
            padding=1
        )

        # Capa de pooling
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

        # Capa fully connected
        self.fc1 = nn.Linear(64 * 2 * 76 * 43, 512)
        self.fc2 = nn.Linear(512, 29)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1) 
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
# Función de validación
def validate(model, val_loader, criterion, device):
    model.eval()  
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad(): 
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Realizar la predicción
            outputs = model(inputs)

            # Calcular la pérdida
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Obtener las predicciones
            _, predicted = torch.max(outputs, 1)

            # Contar las predicciones correctas
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)
    
    # Calcular la media de la pérdida y la precisión
    avg_loss = total_loss / len(val_loader)
    accuracy = correct_predictions / total_samples * 100
    
    print(f'Validación - Pérdida: {avg_loss:.4f}, Precisión: {accuracy:.2f}%')

    return avg_loss, accuracy

# Verificación de la GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"GPU disponible: {torch.cuda.get_device_name(0)}")
else:
    print("GPU no disponible, usando CPU.")

# Cargar el dataset
DataFrame = pd.read_csv('Metadata.csv')
sign_language_dataset = SignLanguageDataset(dataframe=DataFrame)
print('Dataframe cargado')

# Dividir el conjunto de datos en entrenamiento y validación
train_size = int(0.8 * len(sign_language_dataset))
val_size = len(sign_language_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(sign_language_dataset, [train_size, val_size])
print('Conjunto dividido')

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=0)

if __name__ == '__main__':
    # Instanciar el modelo y moverlo al dispositivo
    model = CNN3D().to(device)

    # Configurar la función de pérdida y optimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Función de entrenamiento
    def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=10):
        print('Librerías creadas!')
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        
        for epoch in range(epochs):
            model.train()
            running_train_loss = 0.0
            correct_train = 0
            total_train = 0

            # Entrenamiento
            print('En entrenamiento.')
            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

            # Calcular métricas de entrenamiento
            print('Calculando métricas de entrenamiento.')
            avg_train_loss = running_train_loss / len(train_loader)
            train_accuracy = 100 * correct_train / total_train
            train_losses.append(avg_train_loss)
            train_accuracies.append(train_accuracy)

            # Validación
            print('Validando época.')
            model.eval()
            running_val_loss = 0.0
            correct_val = 0
            total_val = 0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    running_val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()

            # Calcular métricas de validación
            print('Calculando métricas de validación.')
            avg_val_loss = running_val_loss / len(val_loader)
            val_accuracy = 100 * correct_val / total_val
            val_losses.append(avg_val_loss)
            val_accuracies.append(val_accuracy)

            # Imprimir estadísticas
            print(f"Epoch [{epoch+1}/{epochs}], "
                  f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
                  f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

        return train_losses, val_losses, train_accuracies, val_accuracies

    # Ejecutar el entrenamiento
    print('Iniciando entrenamiento!')
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model, train_loader, val_loader, criterion, optimizer, device, epochs=10
    )
    print('Entrenamiento finalizado.')

    # Graficar las métricas
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Pérdida Entrenamiento")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Pérdida Validación")
    plt.title("Pérdida vs Épocas")
    plt.xlabel("Épocas")
    plt.ylabel("Pérdida")
    plt.legend()

    plt.subplot(1, 2, 2)
    residuals = np.array(train_losses) - np.array(val_losses)
    plt.plot(range(1, len(residuals) + 1), residuals, label="Residuos")
    plt.title("Residuos entre Entrenamiento y Validación")
    plt.xlabel("Épocas")
    plt.ylabel("Residuos")
    plt.legend()

    plt.tight_layout()
    plt.show()