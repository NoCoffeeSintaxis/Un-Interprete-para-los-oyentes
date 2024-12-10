import cv2 
import numpy as np
import matplotlib.pyplot as plt

## Este código redimensiona un video, escoge solo 5 frames, los guarda en una matriz numpy y los printea.
### NOTA: Solo compilar para generar las matrices, si están ya están generadas, no compilar.
    # Si se desea verificar la funcionalidad del código, puede comentar el guardado de la matriz y printearla,
    # o bien, compilar el código "Tratamiento de video.py" que hace el mismo proceso, pero para un solo video
    # y las printea.

video_path = 'Videos/s/S27.mp4'
###Si el video es del 1 al 3, cambiar a TRUE
videos_1_3 = False

#Inicializar una lista para almacenar los frames
frames = []
cap = cv2.VideoCapture(video_path)

##### NOTA: ESTA SECCIÓN DEL CÓDIGO SOLO ES VÁLIDA PARA LOS VIDEOS 26 Y 27
#Lapso de frames para obtener el número de frames deseados
frame_want = 5
frame_inicial = 19
frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_lapse = int((frame_total-frame_inicial) / frame_want)

print(f'Número de frames del video original: {frame_total}')

count = 0
for i in range(frame_inicial, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), frame_lapse):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i) # Buscar el frame
    ret, frame = cap.read()  # Leer el frame

    if count > 4:
        break

    if not ret: 
        break
    
    #Porcentaje de escala y definición de las nuevas dimensiones
    if video_path.endswith('26.mp4') or video_path.endswith('27.mp4'):
        scale_percent = 18
        ancho = int(frame.shape[1] * scale_percent / 100)
    elif videos_1_3:
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
if video_path.endswith('26.mp4') or video_path.endswith('27.mp4'):
    A = A[1:-1, :, :] # Eliminar los bordes de la imagen (arriba y abajo)

# Descomentar solo si se desea guardar la matriz en un archivo .npy
'''dirc = 'Datos/s/S25.npy'  # Crear la ruta completa con nombre de archivo
if A is not None:
    np.save(dirc, A)  #Guardar la matriz en un archivo .npy'''

cap.release()

print(f'Tamaño de la matriz de datos A: {A.shape}')

#Mostrar todos los frames de la matriz A
num_ord = ['Primer', 'Segundo', 'Tercer', 'Cuarto', 'Quinto'] 
for i in range(A.shape[2]):
    i_frame = A[:, :, i]
    plt.imshow(i_frame, cmap='gray')
    plt.title(f"{num_ord[i]} frame de la matriz A")
    plt.axis('off')
    plt.show()

#Mostrar el último frame de la matriz A
'''last_frame = A[:, :, i]
plt.imshow(last_frame, cmap='gray')
plt.title(f"Último frame de la matriz A")
plt.axis('off')
plt.show()'''

# Crear un video a partir de la matriz A
'''if A is not None:
    height, width, num_frames = A.shape
    out = cv2.VideoWriter('output_video.avi', cv2.VideoWriter_fourcc(*'XVID'), 20, (width, height), isColor=False)
    
    for i in range(num_frames):
        out.write(A[:, :, i])
    
    out.release()'''