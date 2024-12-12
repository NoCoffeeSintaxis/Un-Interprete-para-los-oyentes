import numpy as np
import matplotlib.pyplot as plt

## Este código muestra frame a frame de la matriz de datos deseada o solo el último frame (si así lo quiere).
### Nota: Este código hace lo mismo que el de "Tratamiento de video", pero sin hacer todo el proceso de matrices.

# Cargar la matriz de datos deseada
A = np.load('Datos/a/A15.npy')

# Seleccionar el último frame
last_frame = A[:, :, 4]

# Printear el tamaño del array
print(A.shape)

#Mostrar todos los frames de la matriz A
num_ord = ['Primer', 'Segundo', 'Tercer', 'Cuarto', 'Quinto'] 
for i in range(A.shape[2]):
    i_frame = A[:, :, i]
    plt.imshow(i_frame, cmap='gray')
    plt.title(f"{num_ord[i]} frame de la matriz A")
    plt.axis('off')
    plt.show()