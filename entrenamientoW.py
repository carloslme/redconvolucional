import cv2
import tensorflow
import keras
import numpy as np
import json
from flask import jsonify

#Keras api
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer, Input
from tensorflow.python.keras.layers import Reshape, MaxPooling2D
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten
from tensorflow.python.keras.optimizers import Adam

img_size = 128 #Pixeles de las imágenes del data set.
img_size_flat = img_size * img_size #Las imágenes son cuadradas.
num_chanels = 1 #Escala de grises, RGB sería 3 canales.

img_shape = (img_size, img_size, num_chanels) 
num_classes = 17 #Los números del 0 al 9 serán las posibles clasificaciones.
limit_train_images = 60 #Se entrenará cada clase con 60 imágenes.


def load_data(phase, num_categories, limit):
    # location = "dataset/" #Ubicación de los sets de imágenes.
    loaded_images = []
    tags = []
    expected_value = []
    for category in range (0, num_categories):
        for id_img in range (0, limit):
            try:
            # dataset/0/0_0_0.jpg
                route = phase + str(category) + "/" + str(category) + "_0_"+ str(id_img) + ".jpg"
                print(route)
                image = cv2.imread(route, 0) #Se indica que la imágen estará a blanco y negro (0).
                # cv2.imshow("prueba", image)
                cv2.waitKey(50)
                image = image.flatten() #Convertir la imágen de matriz a vector. Aplanamiento.
                image = image/255 #Normalización de los valores del vector de la imágen, valores entre 0 y 1.
                loaded_images.append(image) 
                tags.append(category) #Cada imágen del ciclo interno pertenece a la categoría que referencia el ciclo externo.
                probabilities = np.zeros(num_categories) 
                probabilities[category] = 1
                expected_value.append(probabilities)
            except Exception as e:
                print("ERROR on image::::: " + route + ". Problem is: " + str(e))
    images_training = np.array(loaded_images)
    tags_training = np.array(tags)
    expected_values = np.array(expected_value)

    # print (loaded_images)
    return images_training, tags_training, expected_values

images,tags,probabilities = load_data("dataset/", num_classes, limit_train_images)

# print('images: ' + str(images))
model = Sequential()

model.add(InputLayer(input_shape=(img_size_flat,))) #Capa de entrada. La coma se debe poner para ser reconocido como tupla.
model.add(Reshape(img_shape)) #Reformar la imágen, la convierte a matriz nuevamente.

#Primera capa convolucional
model.add(Conv2D(kernel_size=5, strides=1, filters=16, padding='same', activation='relu', name='layer_conv1'))
model.add(MaxPooling2D(pool_size=2, strides=2))

#Segunda capa convolucional
model.add(Conv2D(kernel_size=5, strides=1, filters=36, padding='same', activation='relu', name='layer_conv2'))
model.add(MaxPooling2D(pool_size=2, strides=2))

model.add(Flatten()) #Aplanamiento de la imágen.
model.add(Dense(128, activation='relu')) #Capa densa, se abstraen las características más relevantes.
model.add(Dense(num_classes, activation='softmax')) #Capa de salida.

#Compilación del modelo
optimizer = Adam(lr=1e-3)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

#Entrenamiento del modelo
model.fit(x=images, y=probabilities, epochs=25, batch_size=300) #Epochs: Número de veces que entrena. Batch_size: tamaño de lotes para entrenamiento.

limit_test_images = 40
test_images, test_tags, test_probabilities = load_data("test/", num_classes, limit_test_images)
results = model.evaluate(x=test_images, y=test_probabilities)
print("{0}: {1:.2%}" .format(model.metrics_names[1], results[1]))

name_file = 'modeloWeights.h5' #Debe ser .keras.
model.save_weights(name_file) #Se guarda el archivo en la carpeta raíz.
model.summary() #Se imprime la estructura del archivo, reporte.
