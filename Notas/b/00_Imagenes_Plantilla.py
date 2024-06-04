# %% [markdown]
# # Manipulación de Imagenes

# %% [markdown]
# ## 1. Librerias
# %%
!pip install matplotlib mahotas opencv-python scikit-image

# %%
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpi
from PIL import Image
import mahotas
import mahotas.demos
%matplotlib inline

# %% [markdown]
# ## 2. Lectura

# %% [markdown]
# ### 2.1. Lectura de imagen en escala de grises con opencv
# %%
def mostrar_imagen(location, title, use_opencv=False):
    imagen = cv2.imread(location)
    if use_opencv:
        cv2.imshow(title, imagen)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        plt.figure(figsize=(3,3))
        plt.imshow(imagen)
        plt.title(title)
        plt.axis("off")

location = "./imagenes/barbara.png"
mostrar_imagen(location, "Mostrar con matplotlib")
mostrar_imagen(location, "Mostrar con opencv", use_opencv=True)

# %% [markdown]
# ### 2.2. Lectura de imagen a color con opencv
# %%
x = "./imagenes/_lena.png"
mostrar_imagen(x, "Mostrar con matplotlib")

# %% [markdown]
# ### 2.3. Lectura de imagenes con matplotlib
# %%
imagen = mpi.imread('./imagenes/_lena.png')
plt.figure(figsize=(3,3))
plt.imshow(imagen)
plt.title("Leer y mostrar con matplotlib")
plt.axis("off")

# %% [markdown]
# ### 2.4. Lectura de imagenes con PIL
# %%
im = Image.open('./imagenes/_lena.png')
im.show()

# %% [markdown]
# ### 2.5. Lectura de archivo de video
# %%
# Función y código para lectura de video omitidos por brevedad

# %% [markdown]
# ### 2.6. Lectura de video en tiempo real
# %%
# Código para captura de video en tiempo real omitido por brevedad

# %% [markdown]
# ## 3. Crear y grabar

# %% [markdown]
# ### 3.1. Imagen en escala de grises
# %%
# Código para crear imagen en escala de grises omitido por brevedad

# %% [markdown]
# ### 3.2. Imagen a color
# %%
# Código para crear imagen a color omitido por brevedad

# %% [markdown]
# ### 3.3. Grabar imagen
# %%
# Código para grabar imagen omitido por brevedad

# %% [markdown]
# ### 3.4. Grabar video
# %%
# Código para grabar video omitido por brevedad

# %% [markdown]
# ## 4.Estructura de una imagen

# %% [markdown]
# ### 4.1. Tamaño de la imagen.
# %%
def mostrar_info_imagen(location):
    imagen = cv2.imread(location)
    plt.figure(figsize=(3,3))
    plt.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    print(imagen.dtype)
    print('Dimension:', imagen.shape[0], 'x', imagen.shape[1])
    print('Resolución:', imagen.shape[0] * imagen.shape[1])

location = "./imagenes/_lena.png"
mostrar_info_imagen(location)
