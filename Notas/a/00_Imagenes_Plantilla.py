# Instalación de librerías
!pip install matplotlib mahotas opencv-python scikit-image

# Importaciones
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpi
from PIL import Image

# Configuración de matplotlib
%matplotlib inline

# Funciones de utilidad
def mostrar_imagen(titulo, imagen, cmap='gray'):
    plt.figure(figsize=(3,3))
    plt.imshow(imagen, cmap=cmap)
    plt.title(titulo)
    plt.axis('off')

def leer_mostrar_opencv(ruta, color=True):
    imagen = cv2.imread(ruta, cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE)
    cv2.imshow('Imagen', imagen)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Lectura y visualización de imágenes
ruta = './imagenes/barbara.png'
imagen_gris = cv2.imread(ruta, 0)
mostrar_imagen('Imagen en escala de grises', imagen_gris)

# Creación de imágenes
img = np.zeros((80, 80), dtype=np.uint8)
img[20:61, 20:61] = 255 - np.arange(20, 61)[:, None] - np.arange(20, 61)
img[30:51, 30:51] = 0
mostrar_imagen('Imagen creada', img)

# Grabar imagen
cv2.imwrite('MiImagen.png', img)

# Lectura de video
def leer_video(ruta):
    vreader = cv2.VideoCapture(ruta)
    while True:
        ret, frame = vreader.read()
        if not ret: break
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('x'): break
    vreader.release()
    cv2.destroyAllWindows()

# Estructura de una imagen
def info_imagen(ruta):
    imagen = cv2.imread(ruta)
    print(f"Tipo de dato: {imagen.dtype}")
    print(f"Dimensiones: {imagen.shape[0]} x {imagen.shape[1]}")
    print(f"Resolución: {imagen.shape[0] * imagen.shape[1]}")

info_imagen('./imagenes/_lena.png')
