import cv2 
import numpy as np
import matplotlib.pyplot as plt

video_path = 'C:/Users/juand/Desktop/Un-Interprete-para-los-oyentes/Videos/a/A11.mp4'
frame_lapse = 5

# Inicializar una lista para almacenar los frames
frames = []
cap = cv2.VideoCapture(video_path)

for i in range(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), frame_lapse):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i) # Buscar el frame
    ret, frame = cap.read()  # Leer el frame
    
    if not ret: 
        break
    
    frame = cv2.resize(frame, (474, 850))  # Redimensionar el frame 474, 850
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convertir a escala de grises, suelta una matriz numpy
    
    frames.append(gray_frame)  # Agregar el frame a la lista


# Convertir la lista a una matriz tridimensional
A = np.stack(frames, axis=2) if frames else None

cap.release()

# Seleccionar el primer frame
first_frame = A[:, :, 9]

# Mostrar el frame usando matplotlib
plt.imshow(first_frame, cmap='gray')
plt.title("Primer frame de la matriz A")
plt.axis('off')
plt.show()
