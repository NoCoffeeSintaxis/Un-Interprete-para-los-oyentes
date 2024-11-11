import cv2 

video_path = 'C:/Users/juand/Desktop/Un-Interprete-para-los-oyentes/Videos/a/A.mp4'
frame_lapse = 5

cap = cv2.VideoCapture(video_path)

for i in range(0,int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 5):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i) #Buscar el frame
    ret, frame = cap.read() #Leer el frame
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BRG2GRAY) #Convertir a una matriz de escala de grises
    
    