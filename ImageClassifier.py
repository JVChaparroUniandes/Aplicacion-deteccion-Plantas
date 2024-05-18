import numpy as np
import pandas as pd
import os
import re
import tensorflow as tf
import tensorflow_hub as hub

model_diseases = tf.keras.models.load_model(
       ('modelo_diseases_completo.h5'),
       custom_objects={'KerasLayer':hub.KerasLayer}
)

def predecir_nueva_imagen(ruta_imagen):
    imagen = tf.keras.preprocessing.image.load_img(ruta_imagen, target_size=(224, 224))
    imagen = tf.keras.preprocessing.image.img_to_array(imagen)
    imagen = imagen / 255.0
    imagen = tf.expand_dims(imagen, axis=0)
    predicciones = model_diseases.predict(imagen)
    clases = ['Corn_Healthy', 'Corn_Sick', 'Potato_Healthy', 'Potato_Sick', 'Rice_Healthy', 'Rice_Sick', 'Sugarcane_Healthy', 'Sugarcane_Sick']
    clase_diagnosticada = clases[np.argmax(predicciones)]
    return clase_diagnosticada

directory='/content/Test/CanaEnferma'
def listar_rutas_directorio(directorio):
    rutas = []
    # Iterar sobre los archivos y directorios dentro del directorio
    for root, dirs, files in os.walk(directorio):
        # Agregar la ruta de cada archivo encontrado a la lista de rutas
        for file in files:
            ruta_archivo = os.path.join(root, file)
            rutas.append(ruta_archivo)
    return rutas

rutas_archivos = listar_rutas_directorio(directory)

#Ejemplo de uso
predecir_nueva_imagen('/content/Test/0b3e5032-8ae8-49ac-8157-a1cac3df01dd___RS_HL 1817.JPG')