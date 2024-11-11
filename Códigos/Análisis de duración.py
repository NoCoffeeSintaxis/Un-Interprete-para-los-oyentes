import cv2
import os

# Ruta de la carpeta
folder_path = 'C:/Users/juand/Desktop/Un-Interprete-para-los-oyentes/Videos/a'

# Variables para almacenamiento
total_frames = 0
video_count = 0

# Recorrer todos los archivos de la carpeta
for filename in os.listdir(folder_path):
    # Verificar el tipo de archivo
    if filename.endswith('.mp4'):
        video_path = os.path.join(folder_path, filename)
        cap = cv2.VideoCapture(video_path)
        
        # Obtener las dimensiones de cada video
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        # Acumular las dimensiones
        total_frames += frames
        video_count += 1

        cap.release()

# Calcular el promedio de las dimensiones
if video_count > 0:
    avg_frames = total_frames / video_count
    print(f"Promedio de frames por video es de ", avg_frames)
else:
    print("No se encontraron videos en la carpeta.")


# Muestra cantidad de frames de ejemplo
video_path = 'C:/Users/juand/Desktop/Un-Interprete-para-los-oyentes/Videos/k/K1.mp4'
cap = cv2.VideoCapture(video_path)
frames_ex = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Promedio de frames en el video de ejemplo es de ", frames_ex)