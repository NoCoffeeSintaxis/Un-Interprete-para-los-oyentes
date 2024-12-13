import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

## Este código obtiene el número de frames de un solo video

# Ruta del video
video_path = 'Videos/y/Y26.mp4'
cap = cv2.VideoCapture(video_path)

num_fram = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
# Obtener el número de frames
print(f'Número de frames: {num_fram}')

cap.release()

