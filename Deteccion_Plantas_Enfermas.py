#Importamos Librerias
import streamlit as st
import numpy as np
import pandas as pd
import os
import re
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
from io import BytesIO
import tempfile

#Funcion para cargar el modelo
def load_model(uploaded_file):
    try:
        # Crear un archivo temporal para guardar el modelo
        with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp:
            # Leer el contenido del archivo cargado y escribirlo en el archivo temporal
            contents = uploaded_file.getvalue()
            tmp.write(contents)
            tmp_path = tmp.name  # Guardar la ruta del archivo temporal

        # Cargar el modelo desde la ruta del archivo temporal
        model = tf.keras.models.load_model(
            tmp_path,
            custom_objects={'KerasLayer': hub.KerasLayer}
        )
        
        # Opcional: Eliminar el archivo temporal si ya no es necesario
        os.remove(tmp_path)
        
        return model
    except Exception as e:
        st.error(f'Error al cargar el modelo: {e}')
        return None

#Funcion para procesar imagen
def procesar_imagen(imagen_cargada):
    # Suponiendo que 'imagen_cargada' es un objeto PIL.Image
    # Convertir la imagen a un array de numpy
    imagen = np.array(imagen_cargada)

    # Normalizar la imagen dividiendo cada pixel por 255
    imagen = imagen.astype('float32') / 255.0

    # Redimensionar la imagen usando TensorFlow
    imagen = tf.image.resize(imagen, [224, 224])

    # Añadir una dimensión extra al principio para crear un batch de una sola imagen
    imagen = np.expand_dims(imagen, axis=0)

    # Ahora 'imagen' está lista para ser usada con el modelo
    return imagen

#Funcion para predecir imagen
def predecir_nueva_imagen(imagen):
    predicciones = model_diseases.predict(imagen)
    clases = ['Corn_Healthy', 'Corn_Sick', 'Potato_Healthy', 'Potato_Sick', 'Rice_Healthy', 'Rice_Sick', 'Sugarcane_Healthy', 'Sugarcane_Sick']
    clase_diagnosticada = clases[np.argmax(predicciones)]
    return clase_diagnosticada

#Funcion para separar strings
def separar_string(s):
    # Elimina los dos puntos finales si están presentes
    s = s.rstrip(':')
    
    # Divide el string por raya al piso
    partes = s.split('_')
    
    # Si hay al menos dos partes y la segunda parte está vacía (doble raya al piso)
    if len(partes) > 1 and partes[1] == '':
        # La primera palabra es la primera parte
        primera_palabra = partes[0]
        # Lo que sigue después de la segunda raya al piso es el resto unido
        resto = '_'.join(partes[2:])  # Empieza desde el tercer elemento
    else:
        # Si no hay doble raya al piso, maneja el caso de una sola raya al piso
        primera_palabra = partes[0]
        resto = '_'.join(partes[1:])  # Todo después de la primera raya al piso

    return primera_palabra, resto

def traducir_palabras(planta,condicion):
    dic_planta={"Corn":"Maíz","Potato":"Papa","Rice":"Arroz","Sugarcane":"Caña de azucar"}
    dic_condicion={"Sick":"Enferma","Healthy":"Sana"}

    if planta in dic_planta:
        planta_espanol=dic_planta[planta]
    
    if condicion in dic_condicion:
        condicion_espanol=dic_condicion[condicion]
    return planta_espanol,condicion_espanol

#Titulo inicial
st.title("Detección de enfermedades en plantas")

#Descripción inicial
st.write("Esta aplicación fue diseñada para que puedas detectar si tu planta está enferma, por favor ten en cuenta que esta primera versión solo acepta plantas de \
         maíz, papa, caña de azucar y arroz ¡Empecemos!")

#Cargar Imagen
imagen_cargada=st.file_uploader("**Por favor carga tu imagen a continuación:**",type=['jpg','JPG'])

#Mostrar imagen cargada

if imagen_cargada is not None:
    st.write('**Esta es la imagen que procesaremos:**')
    imagen_planta=Image.open(imagen_cargada)
    st.image(imagen_planta)

   
#Cargamos el modelo
modelo=st.file_uploader("**Por favor carga el archivo de tu modelo aquí:**",type=['h5'])

if modelo is not None:
    model_diseases = load_model(modelo)
    if model_diseases is not None:
        print('El modelo fue cargado exitosamente...')

# Botón para iniciar la predicción
    if st.button('Iniciar análisis de la imagen'):
        imagen_procesada=procesar_imagen(imagen_planta)
        print('Imagen procesada existosamente...')
        resultado = predecir_nueva_imagen(imagen_procesada)
        planta,condicion=separar_string(resultado)
        planta,condicion=traducir_palabras(planta,condicion)
        st.subheader("Resultado para tu planta:")
        st.write(f'**Tipo de planta:** {planta}')
        st.write(f'**Condición:** {condicion}')
    else:
        st.write('Haz clic en el botón para analizar la imagen.')




