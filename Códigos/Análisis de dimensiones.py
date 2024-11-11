import cv2
import os

# Ruta de la carpeta con los videos
folder_path = 'C:/Users/juand/Desktop/Un-Interprete-para-los-oyentes/Videos/a'

# Variables para almacenar las dimensiones totales
total_width = 0
total_height = 0
video_count = 0

# Recorrer todos los archivos de la carpeta
for filename in os.listdir(folder_path):
    # Verificar si el archivo es un video 
    if filename.endswith('.mp4'):
        video_path = os.path.join(folder_path, filename)
        cap = cv2.VideoCapture(video_path)
        
        # Obtener las dimensiones del video
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Acumular las dimensiones
        total_width += width
        total_height += height
        video_count += 1

        # Liberar el video
        cap.release()

# Calcular el promedio de las dimensiones
if video_count > 0:
    avg_width = total_width / video_count
    avg_height = total_height / video_count
    print("Promedio de ancho:", avg_width)
    print("Promedio de alto:", avg_height)
else:
    print("No se encontraron videos en la carpeta.")


# Muestra de dimensiones de ejemplo
video_path = 'C:/Users/juand/Desktop/Un-Interprete-para-los-oyentes/Videos/a/A8.mp4'
cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f'El ancho del video de muestra es de ', width, ', y su largo es de ', height)