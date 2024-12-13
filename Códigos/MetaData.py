import pandas as pd
import os
import string
from sklearn.preprocessing import LabelEncoder

#Listas con las letras
mayus = list(string.ascii_uppercase) + ['LL', 'RR', 'Ñ']
minus = list(string.ascii_lowercase) + ['ll', 'rr', 'ñ']

# Listas con los path de las matrices
paths = []
labels = []
for letra, LETRA in zip(minus, mayus):
    for i in range(1,460):
        path = f'DATOS/{letra}/{LETRA}{i}.npy'
        label = f'{LETRA}'
        paths.append(path)
        labels.append(label)

encoder = LabelEncoder()            
labels = encoder.fit_transform(labels)            
            
df = pd.DataFrame({
    "Path" : paths,
    "Label" : labels
})

print(df.head())

df.to_csv("Metadata.csv", index=False)
            

        