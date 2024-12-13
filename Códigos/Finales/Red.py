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
        video_data = np.load(path) 
        video_data = torch.tensor(video_data, dtype=torch.float32) 
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
        self.fc2 = nn.Linear(512, 27)

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

# Verificación de la GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando el dispositivo: {device}")

# Cargar el dataset
DataFrame = pd.read_csv('Metadata.csv')
sign_language_dataset = SignLanguageDataset(dataframe=DataFrame)

# Preparar un batch y moverlo al dispositivo
batch_sample = torch.stack([sign_language_dataset[i][0].unsqueeze(0) for i in range(16)]).to(device)
print(f"Dimensiones del batch: {batch_sample.shape}")

# Instanciar el modelo y moverlo al dispositivo
model = CNN3D().to(device)

# Pasar el batch por el modelo
output = model(batch_sample)
print(f"Dimensiones de salida después de la capa totalmente conectada: {output.shape}")