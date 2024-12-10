import numpy as np
import matplotlib.pyplot as plt
import os

## Este código verifica las dimensiones de las matrices de datos y muestra el último frame de cada una de ellas

#Lista del abecedario (incluyendo ll, ñ y rr)
abc = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'll', 'm', 'n', 'ñ', 'o', 'p', 'q', 'r', 'rr', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'] 

#Inicializar contador de errores
errores = 0

for letra in abc:
    folder_path = 'Datos/' + letra
    for filename in os.listdir(folder_path):
        matriz = np.load(folder_path + '/' + filename)
        #Printear la dimensión de cada matriz
        print(f'Tamaño de la matriz {os.path.splitext(filename)[0]}: {matriz.shape}')

        #Sumar la cantidad de errores respecto a los frames
        if matriz.shape[2] != 5:
            print(f'Error en la matriz {os.path.splitext(filename)[0]}')
            errores += 1

        #Mostrar el último frame de la matriz
        ###Nota: Esto se hizo para verificar que se mostraba la seña. No es necesario descomentar.
        '''plt.imshow(matriz[:, :, 4], cmap='gray')
        plt.title(f"Último frame de la matriz {filename}")
        plt.axis('off')
        plt.show()'''

print(f'Número de errores: {errores}')