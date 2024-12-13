import numpy as np
import pandas as pd

# Leer el archivo CSV
df = pd.read_csv('Metadata.csv')

# Lista para guardar las formas únicas
shapes = []

for i in range(len(df)):  # Cambiado el rango para evitar índice fuera de rango
    try:
        path = df['Path'].iloc[i]
        # Cargar el archivo .npy
        array = np.load(path)
        shape = array.shape
        # Verificar si la forma ya está en la lista
        if shape not in shapes:
            shapes.append(shape)
    except Exception as e:
        print(f"Error al procesar el archivo en índice {i} ({path}): {e}")

# Imprimir las formas únicas
print("Formas únicas encontradas:", shapes)
