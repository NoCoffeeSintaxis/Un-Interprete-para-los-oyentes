import numpy as np
import cv2
import os

folder_path = 'C:/Users/juand/Desktop/Un-Interprete-para-los-oyentes/Videos/a'
frame_lapse = 5

# Crear el directorio de destino si no existe
output_dir = 'C:/Users/juand/Desktop/Un-Interprete-para-los-oyentes/Datos'
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(folder_path):
    if filename.endswith('.mp4'):  
        # Inicializar una lista para almacenar los frames
        frames = []
        
        # Cargar el video
        cap = cv2.VideoCapture(os.path.join(folder_path, filename))
        name = os.path.splitext(filename)[0]  # Eliminar la extensión del archivo
        dirc = os.path.join(output_dir, name + '.npy')  # Crear la ruta completa con nombre de archivo

        for i in range(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), frame_lapse):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)  # Buscar el frame
            ret, frame = cap.read()  # Leer el frame
            
            if not ret:
                break
            
            frame = cv2.resize(frame, (474, 850))  # Redimensionar el frame
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convertir a escala de grises
            
            frames.append(gray_frame)  # Agregar el frame a la lista

        # Convertir la lista a una matriz tridimensional
        A = np.stack(frames, axis=2) if frames else None
        
        if A is not None:
            np.save(dirc, A)  # Guardar la matriz en un archivo .npy
        cap.release()
