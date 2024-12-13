import numpy as np
import cv2
import os
import string  

## Este código pasa por cada video de cada carpeta de la carpeta Videos, redimensiona los videos, 
    # escoge solo 5 frames por video y los guarda en una matriz numpy en la carpeta "Datos" con el 
    # mismo nombre del video.
### NOTA: Solo compilar para generar las matrices, si están ya están generadas, no compilar.
    # Si se desea verificar la funcionalidad del código, puede comentar el guardado de la matriz y printearla,
    # o bien, compilar el código "Tratamiento de video.py" que hace el mismo proceso, pero para un solo video
    # y las printea.

#Lista del abecedario (incluyendo ll, ñ y rr)
abc = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'll', 'm', 'n', 'ñ', 'o', 'p', 'q', 'r', 'rr', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'] 

#Inicializar contador de errores
errores = 0

#Pasar por cada letra del abecedario
for letra in abc:
    # Ruta de la carpeta
    folder_path = 'Videos/' + letra

    # Crear el directorio de destino si no existe
    output_dir = 'Datos/' + letra
    os.makedirs(output_dir, exist_ok=True)

    # Crear una lista con los nombres de los videos 1, 2 y 3
    list_name_videos = os.listdir(folder_path)
    videos_1_3 = [list_name_videos[0], list_name_videos[11], list_name_videos[20]]

    # Iterar sobre los archivos en la carpeta
    for filename in os.listdir(folder_path):
        if filename.endswith('.mp4'):  
            # Inicializar una lista para almacenar los frames
            frames = []
            
            # Cargar el video
            cap = cv2.VideoCapture(os.path.join(folder_path, filename))
            name = os.path.splitext(filename)[0]  # Eliminar la extensión del archivo
            dirc = os.path.join(output_dir, name + '.npy')  # Crear la ruta completa con nombre de archivo

            frame_want = 5
            frame_inicial = 19
            frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_lapse = int((frame_total-frame_inicial) / frame_want)

            count = 0
            for i in range(frame_inicial, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), frame_lapse):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i) # Buscar el frame
                ret, frame = cap.read()  # Leer el frame

                if count > 4:
                    break

                if not ret: 
                    break
                
                #Ancho y alto del frame
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                #Porcentaje de escala y definición de las nuevas dimensiones
                if width == 480 and height == 860:
                    scale_percent = 18
                    ancho = int(frame.shape[1] * scale_percent / 100)
                    videos_26_27 = True
                elif width == 474 and height == 850:
                    scale_percent = 18
                    ancho = int(frame.shape[1] * scale_percent / 100 + 1)
                else:
                    scale_percent = 8
                    ancho = int(frame.shape[1] * scale_percent / 100)
                alto = int(frame.shape[0] * scale_percent / 100)
                new_dim = (ancho, alto)

                frame = cv2.resize(frame, new_dim)

                #Convertir a escala de grises, suelta una matriz numpy
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convertir a escala de grises, suelta una matriz numpy
                
                frames.append(gray_frame)  # Agregar el frame a la lista
                count += 1

            # Convertir la lista a una matriz tridimensional
            A = np.stack(frames, axis=2) if frames else None
            if videos_26_27:
                A = A[1:-1, :, :] # Eliminar los bordes de la imagen (arriba y abajo)

            # Verificar que la matriz tenga 5 frames
            if A.shape[2] != 5 or A.shape[0] != 153 or A.shape[1] != 86:
                print('Error en la matriz', name)
                errores += 1

            # Guardar la matriz en un archivo .npy
            if A is not None:
                np.save(dirc, A)
            cap.release()

            # Printear mensaje de éxito
            print(f'Archivo {dirc} guardado exitosamente.')

#Printear número de errores
print(f'Número de errores: {errores}')