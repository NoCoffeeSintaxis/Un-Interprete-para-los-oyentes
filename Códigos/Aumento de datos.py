import numpy as np
import cv2
import matplotlib.pyplot as plt


# La siguiente sección del presente código muestra los diferentes filtros que serán empleados para aumentar la cantidad de datos en el dataset utilizado
# para entrenar la red neuronal del presentre proyecto, para visualizar los diferentes filtros sige las siguientes instrucciones:

## INSTRUCCIONES DE USO
# 1. Borra las comillas simples (''') arriba y compila este código para visualizar el filtro aplicado.
# 2. Compila el código: esto resultará en una ventana emergente que mostrará la imagen con el filtro aplicado.
# 3. Una vez haya terminado de observar la imagen ponga nuevamente las comillas simples para evitar la tener que graficar más de una imagen.

path = 'Datos/A/A2.npy' # El path para visualizar sigue la siguiente forma: Datos/Letra en minuscula/Letra en mayúscula + número (1-27)
video = np.load(path)

'''
# Visualización de la imagen de referencia

plt.imshow(video[:, :, 4], cmap='gray')
plt.title(f'Frame de referencia')
plt.axis('off')
plt.show()
'''

# 0. Imagen real
# Esta es la imagen real

# 1. Rotación positiva
#Este flitro rota la imagen (deg) grados en dirección a las manecillas del reloj.
def rotacion_positiva(A, deg):
    (h, w, _) = A.shape # Conseguir el tamaño de la matriz
    centro = (w // 2, h // 2) # Encontrar el centro
    matriz_rotada = cv2.getRotationMatrix2D(centro, deg, 1.0) # Rotar la matriz (deg) grados 
    return np.stack([cv2.warpAffine(A[:, :, i], matriz_rotada, (w, h)) for i in range(A.shape[2])], axis=2)

'''
# Visualización del filtro

deg = 15
RP = rotacion_positiva(video, 15)

plt.imshow(RP[:, :, 4], cmap='gray')
plt.title(f'Frame rotado (-) {deg} grados')
plt.axis('off')
plt.show()
'''

# 2. Rotación negativa
#Este flitro rota la imagen (deg) grados en dirección contraria a las manecillas del reloj.
def rotacion_negativa(A, deg):
    (h, w, _) = A.shape # Conseguir el tamaño de la matriz
    centro = (w // 2, h // 2) # Encontrar el centro
    matriz_rotada = cv2.getRotationMatrix2D(centro, -deg, 1.0) # Rotar la matriz (-deg) grados
    return np.stack([cv2.warpAffine(A[:, :, i], matriz_rotada, (w, h)) for i in range(A.shape[2])], axis=2)

'''
# Visualización del filtro

deg = 15
RN = rotacion_negativa(video, 15)

plt.imshow(RN[:, :, 4], cmap='gray')
plt.title(f'Frame rotado (-) {deg} grados')
plt.show()
'''

# 3. Translación
# Este filtro mueve la imagen en sentido abajo-derecha, dejando un borde negro en la parte desplazada.
def translacion_frames(A, tx, ty):
    (h, w, _) = A.shape
    matriz_trasladada = np.float32([[1, 0, tx], [0, 1, ty]])
    return np.stack([cv2.warpAffine(A[:, :, i], matriz_trasladada, (w, h)) for i in range(A.shape[2])], axis=2)

'''
# Visualización del filtro

T = translacion_frames(video, 50, 50)

plt.imshow(T[:, :, 4], cmap='gray')
plt.title('Frame trasladado')
plt.axis('off')
plt.show()
'''

# 3. Zoom
# Este filtro realiza un zoom de (zoom_factor) a la imagen.
def zoom_frames(A, zoom_factor):
    (h, w, num_frames) = A.shape
    zoomed = []
    for i in range(num_frames):
        frame = A[:, :, i]
        h_scaled, w_scaled = int(h * zoom_factor), int(w * zoom_factor)
        resized = cv2.resize(frame, (w_scaled, h_scaled))
        if zoom_factor > 1:
            cropped = resized[(h_scaled - h) // 2: (h_scaled + h) // 2, (w_scaled - w) // 2: (w_scaled + w) // 2]
        else:
            pad_h, pad_w = (h - h_scaled) // 2, (w - w_scaled) // 2
            cropped = cv2.copyMakeBorder(resized, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=0)
        zoomed.append(cropped)
    return np.stack(zoomed, axis=2)

'''
# Visualización del filtro

zoom_factor = 1.3 #¡¡No excederse con el zoom_factor, ya que puede perderse la seña!! (Máximo: 1.3)
ZOOM = zoom_frames(video, zoom_factor) 

plt.imshow(ZOOM[:, :, 4], cmap='gray')
plt.title(f'Frame con zoom de  {zoom_factor}')
plt.axis('off')
plt.show()
'''

# 4. Normalización
# Esta función ajusta los valores de los píxeles para que estén entre 0 y 255.
def normalize_frames(A):
    A_min = A.min(axis=(0, 1), keepdims=True)
    A_max = A.max(axis=(0, 1), keepdims=True)
    return ((A - A_min) / (A_max - A_min) * 255).astype(np.uint8)

'''
# Visualización de la normalización

normalized_frames = normalize_frames(video)

plt.imshow(normalized_frames[:, :, 4], cmap='gray')
plt.title("frame normalizado")
plt.axis('off')
plt.show()
'''

# 5. Filtro Gaussiano
# Este filtro suaviza los frames mediante un desenfoque gaussiano para reducir el ruido.
def desenfoque_gaussiano(A, kernel_size):
    return np.stack([cv2.GaussianBlur(A[:, :, i], (kernel_size, kernel_size), 0) for i in range(A.shape[2])], axis=2)

'''
# Visualización del filtro

kernel_size = 9
gaussian_blur = desenfoque_gaussiano(video, kernel_size) # Kernel máx size: 9

plt.imshow(gaussian_blur[:, :, 4], cmap='gray')
plt.title("Frame con desenfoque Gaussiano")
plt.axis('off')
plt.show()
'''

# 6. Filtro Mediana
# Este filtro reemplaza el valor de cada píxel con la mediana de los píxeles vecinos para reducir el ruido.
def filtro_mediana(A, kernel_size):
    return np.stack([cv2.medianBlur(A[:, :, i], kernel_size) for i in range(A.shape[2])], axis=2)

'''
# Visualización del filtro

kernel_size = 5 #Kernel mín size = 5
FM = filtro_mediana(video, kernel_size)

plt.imshow(FM[:, :, 4], cmap='gray')
plt.title("Frame con filtro de mediana")
plt.axis('off')
plt.show()
'''

# 7. Umbralización Binaria
# Este filtro convierte cada frame en blanco y negro dependiendo de un valor de umbral.
def umbralizacion_binaria(A, umbral):
    return np.stack([cv2.threshold(A[:, :, i], umbral, 255, cv2.THRESH_BINARY)[1] for i in range(A.shape[2])], axis=2)

'''
# Visualización del filtro

UB = umbralizacion_binaria(video, 127)

plt.imshow(UB[:, :, 4], cmap='gray')
plt.title("Frame con umbralización binaria")
plt.axis('off')
plt.show()
'''

# 8. Umbralización Adaptativa
# Este filtro ajusta dinámicamente el umbral para cada píxel según su vecindario.
def umbralizacion_adaptativa(A, block_size, C):
    return np.stack([cv2.adaptiveThreshold(A[:, :, i], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, block_size, C) for i in range(A.shape[2])], axis=2)

'''
# Visualización del filtro

UA = umbralizacion_adaptativa(video, 11, 2)

plt.imshow(UA[:, :, 4], cmap='gray')
plt.title("Frame con umbralización adaptativa")
plt.axis('off')
plt.show()
'''

# 9. Detección de Bordes
# Este filtro identifica los bordes en cada frame utilizando el algoritmo de Canny.
def apply_canny_edge_detection(A, umbral_min, umbral_max):
    return np.stack([cv2.Canny(A[:, :, i], umbral_min, umbral_max) for i in range(A.shape[2])], axis=2)

'''
# Visualización del filtro

edges = apply_canny_edge_detection(video, 50, 150)

plt.imshow(edges[:, :, 4], cmap='gray')
plt.title("Frame con detección de bordes")
plt.axis('off')
plt.show()
'''

# 10. Filtro Laplaciano
# Este filtro detecta cambios rápidos de intensidad (bordes) mediante la segunda derivada.
def filtro_laplaciano(A):
    return np.stack([cv2.Laplacian(A[:, :, i], cv2.CV_64F) for i in range(A.shape[2])], axis=2).astype(np.uint8)

'''
# Visualización del filtro

FL = filtro_laplaciano(video)

plt.imshow(FL[:, :, 4], cmap='gray')
plt.title("Frame con filtro Laplaciano")
plt.axis('off')
plt.show()
'''

# 11. Filtro de Borde
# Este filtro detecta bordes en las imágenes utilizando el operador de Sobel.
def filtro_sobel(A):
    return np.stack([cv2.Sobel(A[:, :, i], cv2.CV_64F, 1, 1, ksize=3) for i in range(A.shape[2])], axis=2).astype(np.uint8)

'''
# Visualización del filtro

FS = filtro_sobel(video)

plt.imshow(FS[:, :, 4], cmap='gray')
plt.title("Primer frame con filtro Sobel")
plt.axis('off')
plt.show()
'''

# 12. Ecualización de Histograma
# Este filtro mejora el contraste de cada frame al redistribuir las intensidades del histograma.
def ecualizacion_histograma(A):
    return np.stack([cv2.equalizeHist(A[:, :, i]) for i in range(A.shape[2])], axis=2)

'''
# Visualización del filtro

EH = ecualizacion_histograma(video)

plt.imshow(EH[:, :, 4], cmap='gray')
plt.title("Frame con ecualización de histograma")
plt.axis('off')
plt.show()
'''

# 13. Inversión de Intensidades
# Este filtro invierte los valores de intensidad de cada frame.
# Las intensidades oscuras se vuelven claras y viceversa.
def inversion(A):
    return 255 - A  # Calcula el complemento de 255 para invertir las intensidades.

'''
# Visualización del filtro

INV = inversion(video)

plt.imshow(INV[:, :, 4], cmap='gray')
plt.title("Frame con inversión de intensidades")
plt.axis('off')
plt.show()
'''

# 14. Brillo Aleatorio
# Este filtro ajusta el brillo de cada frame añadiendo un valor constante (factor) a las intensidades.
def brillo_aleatorio(A, factor):
    return np.clip(A + factor, 0, 255).astype(np.uint8)

'''
# Visualización del filtro

factor_brillo = 55  # Ajuste del brillo: valor positivo para aumentar, negativo para reducir.
BRILLO = brillo_aleatorio(video, factor_brillo)

plt.imshow(BRILLO[:, :, 4], cmap='gray')
plt.title(f"Frame con brillo ajustado ({factor_brillo})")
plt.axis('off')
plt.show()
'''

# 15. Contraste Aleatorio
# Este filtro ajusta el contraste de cada frame mediante la fórmula: 128 + factor * (A - 128).
# Aquí, 128 representa la intensidad promedio, usada como punto de referencia para el ajuste.
def contraste_aleatorio(A, factor):
    return np.clip(128 + factor * (A - 128), 0, 255).astype(np.uint8)

'''
# Visualización del filtro

factor_contraste = 1.5  # Ajuste del contraste: valor mayor a 1 para aumentar, menor a 1 para reducir.
CONTRASTE = contraste_aleatorio(video, factor_contraste)

plt.imshow(CONTRASTE[:, :, 4], cmap='gray')
plt.title(f"Frame con contraste ajustado ({factor_contraste})")
plt.axis('off')
plt.show()
'''

## Augmenting data automatization ##
# ¡Usar esta parte de este código una única vez! 
# Este código está diseñado para aplicar los filtros anteriormente mostrados para añadirlos luego al set de datos, por lo tanto debe 
# usarse una única vez, ya que usarlo más ocasiones resultará en aplicar filtros sobre imagenes ya filtradas.

