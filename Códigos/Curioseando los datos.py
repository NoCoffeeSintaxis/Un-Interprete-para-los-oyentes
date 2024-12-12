import numpy as np
import matplotlib.pyplot as plt

## Este código muestra frame a frame de la matriz de datos deseada o solo el último frame (si así lo quiere).
### Nota: Este código hace lo mismo que el de "Tratamiento de video", pero sin hacer todo el proceso de matrices.

# Cargar la matriz de datos deseada
<<<<<<< HEAD
A = np.load('Datos/s/S24.npy')
=======
A = np.load('Datos/a/A15.npy')
>>>>>>> 39c57481b9c2fc473f0aba277806c695db5bd543

# Seleccionar el último frame
last_frame = A[:, :, 4]

# Printear el tamaño del array
print(A.shape)

#Mostrar todos los frames de la matriz A
num_ord = ['Primer', 'Segundo', 'Tercer', 'Cuarto', 'Quinto'] 
for i in range(A.shape[2]):
    i_frame = A[:, :, i]
    plt.imshow(i_frame, cmap='gray')
    #plt.title(f"{num_ord[i]} frame de la matriz A")
    plt.axis('off')
    plt.show()