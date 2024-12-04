import numpy as np
import matplotlib.pyplot as plt

A = np.load('C:/Users/juand/Desktop/Un-Interprete-para-los-oyentes/Datos/R/R21.npy')

# Seleccionar el primer frame
first_frame = A[:, :, 1]

# Mostrar el frame usando matplotlib
plt.imshow(first_frame, cmap='gray')
plt.title("Primer frame de la matriz A")
plt.axis('off')
plt.show()