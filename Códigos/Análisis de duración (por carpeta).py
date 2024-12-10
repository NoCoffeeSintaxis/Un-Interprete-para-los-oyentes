import cv2
import os

## Este código obtiene el número de frames de cada video en una sola carpeta (de uan sola letra)
    # y calcula el promedio de frames por video en dicha carpeta.

# Ruta de la carpeta de videos
folder_path = 'Videos/h'

# Variables para almacenamiento
video_count = 0
num_frames = []

# Recorrer todos los archivos de la carpeta
for filename in os.listdir(folder_path):
    # Verificar el tipo de archivo
    if filename.endswith('.mp4'):
        video_path = os.path.join(folder_path, filename)
        cap = cv2.VideoCapture(video_path)
        name = os.path.splitext(filename)[0]  # Eliminar la extensión del archivo
        
        # Obtener las dimensiones de cada video
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        num_frames.append(frames)

        # Acumular las dimensiones
        video_count += 1

        #Printear las dimensiones
        print(f"El número de frames en {name} es: {frames}")

        cap.release()
    
# Calcular el promedio de las dimensiones
if video_count > 0:
    avg_frames = sum(num_frames) / video_count
    print(f"Promedio de frames por video es de ", avg_frames)
else:
    print("No se encontraron videos en la carpeta.")