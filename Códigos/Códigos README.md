# <p align='center'><b>Códigos</b></p>

Aquí podrá comprender la utilidad de cada uno de los códigos presentes en esta carpeta

## Análisis

**Análisis de dimensiones:** Una vez tomados los videos, se dispuso de este código para analizar las dimensiones de cada uno de ellos.

**Análisis de duración (por video):** De igual forma, este código realiza la medición de la cantidad de frames de un video en particular.

**Análisis de duración (por carpeta):** Esta es la automatización del <u> (Análisis de duración (por video)) </u> para aplicar sobre todos los videos en una carpeta.

**Revisión de dimensiones (Matrices):** Al finalizar se vió la necesidad de cerciorarse de las dimensiones de todas las matrices, pues, estas deben ser las mismas para todas las matrices.

## Tratamiento de datos

**Tratamiento de videos:** Este es la versión beta de <u> (Tratamiento automático de videos) </u>, convierte un video en una matriz 
tridimensional en formato numpy (.npy)

**Tratamiento automático de videos:** Lee cada uno de los videos, los convierte en matrices tridimensionales y los almacena en la carpeta "Datos"

**Metadata:** Almacena las etiquetas y direcciones de cada uno de los videos en "Datos" en un dataframe de pandas.

**Verificar dimensiones de las matrices:** Cuenta la cantidad de matrices que tienen diferentes dimensiones.
