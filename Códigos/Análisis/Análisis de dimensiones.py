import cv2
import os

## Este código calcula el promedio de las dimensiones de los videos en una carpeta
### NOTA: Se usó solo en un inicio para conocer las dimensiones de los videos. 
    # No es relevante para el funcionamiento del proyecto final.

# Ruta de la carpeta con los videos
folder_path = 'Videos/a'

# Variables para almacenar las dimensiones totales
video_count = 0
width = []
height = []

# Recorrer todos los archivos de la carpeta
for filename in os.listdir(folder_path):
    # Verificar si el archivo es un video 
    if filename.endswith('.mp4'):
        video_path = os.path.join(folder_path, filename)
        cap = cv2.VideoCapture(video_path)
        name = os.path.splitext(filename)[0]  # Eliminar la extensión del archivo

        # Obtener las dimensiones del video
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        video_count += 1

        print(f"La resolución del video {name} es: {width}x{height}")

        # Liberar el video
        cap.release()

# Calcular el promedio de las dimensiones
if video_count > 0:
    avg_width = sum(width) / video_count
    avg_height = sum(height) / video_count
    print("Promedio de ancho:", avg_width)
    print("Promedio de alto:", avg_height)
else:
    print("No se encontraron videos en la carpeta.")

# Muestra de dimensiones de ejemplo
video_path = 'Videos/a/A8.mp4'
cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f'El ancho del video de muestra es de ', width, ', y su largo es de ', height)