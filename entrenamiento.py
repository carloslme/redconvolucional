import cv2 # Libreria OpenCV para manejo de imagenes
import tensorflow # Tensor Flow para algoritmos de machine learning
import keras # Algoritmos de redes neuronales
import numpy as np # Libreria para manejo de arreglos

#Keras api
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer, Input
from tensorflow.python.keras.layers import Reshape, MaxPooling2D
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten
from tensorflow.python.keras.optimizers import Adam

def cargar_datos(fase, num_categorias, limite):
    imagenes_cargadas = []
    tags = []
    expected_value = []
    for category in range (0, num_categorias):
        for id_img in range (0, limite):
            try:
            # dataset/0/0_0_0.jpg
                route = fase + str(category) + "/" + str(category) + "_0_"+ str(id_img) + ".jpg"
                print(route)
                imagen = cv2.imread(route, 0) #Se indica que la imágen estará a blanco y negro (0).
                # cv2.imshow("prueba", imagen)
                cv2.waitKey(50)
                imagen = imagen.flatten() #Convertir la imágen de matriz a vector. Aplanamiento.
                imagen = imagen/255 #Normalización de los valores del vector de la imágen, valores entre 0 y 1.
                imagenes_cargadas.append(imagen) 
                tags.append(category) #Cada imágen del ciclo interno pertenece a la categoría que referencia el ciclo externo.
                probabilidades = np.zeros(num_categorias) 
                probabilidades[category] = 1
                expected_value.append(probabilidades)
            except Exception as e:
                print("Error en la imagen: " + route + ". El problema es " + str(e))
    imagenes_entrenamiento = np.array(imagenes_cargadas)
    tags_entrenamiento = np.array(tags)
    valores_esperados = np.array(expected_value)

    # print (imagenes_cargadas)
    return imagenes_entrenamiento, tags_entrenamiento, valores_esperados

img_tamanio = 128 #Pixeles de las imágenes del data set.
img_tamanio_flat = img_tamanio * img_tamanio #Las imágenes son cuadradas.
num_canales = 1 #Escala de grises, RGB sería 3 canales.

img_shape = (img_tamanio, img_tamanio, num_canales) 
num_clases = 17 #Los números del 0 al 9 serán las posibles clasificaciones.
limite_tren_imagenes = 60 #Se entrenará cada clase con 60 imágenes.

imagenes,tags,probabilidades = cargar_datos("dataset/", num_clases, limite_tren_imagenes)

# print('imagenes: ' + str(imagenes))
model = Sequential()

model.add(InputLayer(input_shape=(img_tamanio_flat,))) #Capa de entrada. La coma se debe poner para ser reconocido como tupla.
model.add(Reshape(img_shape)) #Reformar la imágen, la convierte a matriz nuevamente.

#Primera capa convolucional
model.add(Conv2D(kernel_size=5, strides=1, filters=16, padding='same', activation='relu', name='layer_conv1'))
model.add(MaxPooling2D(pool_size=2, strides=2))

#Segunda capa convolucional
model.add(Conv2D(kernel_size=5, strides=1, filters=36, padding='same', activation='relu', name='layer_conv2'))
model.add(MaxPooling2D(pool_size=2, strides=2))

model.add(Flatten()) #Aplanamiento de la imágen.
model.add(Dense(128, activation='relu')) #Capa densa, se abstraen las características más relevantes.
model.add(Dense(num_clases, activation='softmax')) #Capa de salida.

#Compilación del modelo
optimizer = Adam(lr=1e-3)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

#Entrenamiento del modelo
model.fit(x=imagenes, y=probabilidades, epochs=25, batch_size=300) #Epochs: Número de veces que entrena. Batch_size: tamaño de lotes para entrenamiento.

limit_test_imagenes = 40
test_imagenes, test_tags, test_probabilidades = cargar_datos("test/", num_clases, limit_test_imagenes)
results = model.evaluate(x=test_imagenes, y=test_probabilidades)
print("{0}: {1:.2%}" .format(model.metrics_names[1], results[1]))

name_file = 'modelo.keras' #Debe ser .keras.
model.save(name_file) #Se guarda el archivo en la carpeta raíz.
model.summary() #Se imprime la estructura del archivo, reporte.
