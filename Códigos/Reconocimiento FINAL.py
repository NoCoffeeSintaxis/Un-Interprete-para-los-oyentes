import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
import time

## Este código le da instrucciones al usuario, se graba el video y se guarda en una matriz numpy.

def start_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "No se puede abrir la cámara")
        return

    out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (270, 480))

    # Cuenta regresiva
    start_time = time.time()
    countdown = 3
    while countdown > 0:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (640, 480))  # Redimensionar el frame
        # Recortar el frame a 480x270
        start_x = (frame.shape[1] - 270) // 2
        start_y = (frame.shape[0] - 480) // 2
        frame = frame[start_y:start_y + 480, start_x:start_x + 270]
        # Calcular el tamaño de la cuenta regresiva
        elapsed_time = time.time() - start_time
        if elapsed_time >= 1:
            countdown -= 1
            start_time = time.time()
        (text_width, text_height), baseline = cv2.getTextSize(str(countdown), cv2.FONT_HERSHEY_SIMPLEX, 10, 20)
        cv2.putText(frame, str(countdown), ((frame.shape[1] - text_width) // 2, (frame.shape[0] + text_height) // 2), cv2.FONT_HERSHEY_SIMPLEX, 10, (150, 150, 150), 20, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break
        
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (640, 480))  # Redimensionar el frame
        # Recortar el frame a 480x270
        start_x = (frame.shape[1] - 270) // 2
        start_y = (frame.shape[0] - 480) // 2
        frame = frame[start_y:start_y + 480, start_x:start_x + 270]
        out.write(frame)
        cv2.putText(frame, "Presiona 's' para salir", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    root.destroy()

# Crear la ventana principal
root = tk.Tk()
root.title("Un intérprete para los oyentes")

# Crear una etiqueta con instrucciones
label = tk.Label(root, text="Presione el botón \"Activar Cámara\" para activar la cámara. \
                 \nEspere a que la cuenta regresiva termine para hacer la seña deseada. \
                \nTan pronto termine de hacerla, presione 's' para salir. \n \
                 \nNOTA: Asegúrese de que el teclado se encuentre en minúsculas.")
label.pack(pady=10)

# Crear un botón para activar la cámara
button = tk.Button(root, text="Activar Cámara", command=start_camera)
button.pack(pady=10)

# Ejecutar la aplicación
root.mainloop()

# Cargar el video
cap = cv2.VideoCapture('output.avi')
frame_want = 5
frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_lapse = int((frame_total) / frame_want)
print(f'Número de frames del video original: {frame_total}')

count = 0
frames = []
for i in range(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), frame_lapse):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i) # Buscar el frame
    ret, frame = cap.read()  # Leer el frame

    if count > 4 or not ret:
        break
    
    #Redimensionar
    frame = cv2.resize(frame, (86, 153))

    #Convertir a escala de grises, suelta una matriz numpy
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convertir a escala de grises, suelta una matriz numpy
    frames.append(gray_frame)  # Agregar el frame a la lista
    count += 1

# Convertir la lista a una matriz tridimensional
A = np.stack(frames, axis=2) if frames else None

cap.release()

print(f'Tamaño de la matriz de datos A: {A.shape}')

if A is not None:
    #Mostrar todos los frames de la matriz A
    num_ord = ['Primer', 'Segundo', 'Tercer', 'Cuarto', 'Quinto'] 
    for i in range(A.shape[2]):
        i_frame = A[:, :, i]
        plt.imshow(i_frame, cmap='gray')
        plt.title(f"{num_ord[i]} frame de la matriz A")
        plt.axis('off')
        plt.show()
else:
    print("La matriz A no se ha creado.")