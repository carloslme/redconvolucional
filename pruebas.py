import cv2
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.models import Sequential
import numpy as np


def predecir(imagen):
    """
        Toma la imagen de entrada y realiza el proceso de predicción
    """
    model = load_model("modelo.keras") # Se carga el modelo
    imagen = cv2.resize(imagen,(128,128)) # Comienza la normalización de la imagen
    imagen = imagen.flatten() # Se convierte la imagen de matriz a vector
    imagen = np.array(imagen) # Se guardan en un arreglo los valores
    imagenNormalizada = imagen/255 # Se normalizan los valores del vector de la imágen, valores entre 0 y 1.
    pruebas = []
    pruebas.append(imagenNormalizada)
    imagenesAPredecir = np.array(pruebas)
    predicciones = model.predict(x = imagenesAPredecir) # rSe realiza la predicción
    claseMayorValor = np.argmax(predicciones,axis=1) # Se guarda el valor probabilístico más alto obtenido 
    return claseMayorValor[0] # Se retorna el valor más alto


categorias = ["Huevos","Arepas","Mantequilla","Chocolate","Pan","Cereales","Cafe","Leche","Tocino","Changua","Tamal","Papas","Calentado","Yuca frita","Jugo naranaja","Yogurth","Pollo"]
imagenPrueba = cv2.imread("test/0/0_0_0.jpg",0)
cv2.waitKey(0)
indiceCategoria = predecir(imagenPrueba) # Se ingresa la imagen en formato normal para su predicción 
print("La imagen cargada pertenece a la categoría: ",categorias[indiceCategoria]) # Se indica a qué categoría pertenece
while True:
   cv2.imshow("imagen",imagenPrueba)
   k=cv2.waitKey(30) & 0xff
   if k==27:
       break
cv2.destroyAllWindows()



