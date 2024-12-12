import numpy as np
import cv2
import os
import string  

# Ruta base de los videos
videos_base_path = 'C:/Users/juand/Desktop/Un-Interprete-para-los-oyentes/Videos'
frame_lapse = 5

# Ruta base de guardado
datos_base_path = 'C:/Users/juand/Desktop/Un-Interprete-para-los-oyentes/Datos'

# Iterar sobre las letras del abecedario
for letter in string.ascii_lowercase:  # Letras en Minúscula
    folder_path = os.path.join(videos_base_path, letter)
    
    # Creación de la carpeta destino
    output_dir = os.path.join(datos_base_path, letter.upper()) 
    os.makedirs(output_dir, exist_ok=True)

    # Procesamiento
    for filename in os.listdir(folder_path):
        if filename.endswith('.mp4'):  
            # Inicializar una lista para almacenar los frames
            frames = []
            
            # Cargar el video
            video_path = os.path.join(folder_path, filename)
            cap = cv2.VideoCapture(video_path)
            name = os.path.splitext(filename)[0] 
            output_file_path = os.path.join(output_dir, name + '.npy') 

            # Leer frames según el intervalo definido
            for i in range(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), frame_lapse):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i) 
                ret, frame = cap.read() 
                
                if not ret:
                    break
                
                # Redimensionar y convertir el frame a escala de grises
                frame = cv2.resize(frame, (474, 850)) #Estos valores se proponen del análisis dado por "Análisis de dimensiones"
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                frames.append(gray_frame) #Agregar frame ya procesado a la lista

            # Convertir la lista de frames a una matriz tridimensional
            A = np.stack(frames, axis=2) if frames else None
            
            # Guardar la matriz en un archivo .npy
            if A is not None:
                np.save(output_file_path, A)
                print(f"Matriz guardada: {output_file_path}")
            else:
                print(f"No se generó matriz para el video {video_path}.")
            
            cap.release()
