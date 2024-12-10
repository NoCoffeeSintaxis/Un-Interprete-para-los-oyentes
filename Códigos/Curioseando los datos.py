import numpy as np
import matplotlib.pyplot as plt

<<<<<<< HEAD
## Este código muestra frame a frame de la matriz de datos deseada o solo el último frame (si así lo quiere).
### Nota: Este código hace lo mismo que el de "Tratamiento de video", pero sin hacer todo el proceso de matrices.
=======
A = np.load('C:/Users/juand/Desktop/Un-Interprete-para-los-oyentes/Datos/R/R21.npy')
>>>>>>> 4ca6982dfa6e4f013714e0a1fb8a188df9737ee9

# Cargar la matriz de datos deseada
A = np.load('Datos/s/S25.npy')

# Seleccionar el último frame
last_frame = A[:, :, 4]

#Mostrar todos los frames de la matriz A
num_ord = ['Primer', 'Segundo', 'Tercer', 'Cuarto', 'Quinto'] 
for i in range(A.shape[2]):
    i_frame = A[:, :, i]
    plt.imshow(i_frame, cmap='gray')
    plt.title(f"{num_ord[i]} frame de la matriz A")
    plt.axis('off')
    plt.show()

'''# Mostrar el último frame
plt.imshow(last_frame, cmap='gray')
plt.title("Último frame de la matriz A")
plt.axis('off')
plt.show()'''