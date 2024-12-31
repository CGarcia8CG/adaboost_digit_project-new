
import streamlit as st
import numpy as np
import requests
from PIL import Image
import io

# Función para enviar datos a la API
def query_api(image):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    url = "http://localhost:9000/predict-image/"
    files = {"file": ("image.png", img_byte_arr, "image/png")}
    response = requests.post(url, files=files)
    return response.json()

# --- SIDEBAR INTERACTIVO ---
with st.sidebar:
    # Foto e Información Personal
    st.image("input_data/cropped_carlos.png", width=150)  # Cambia por tu foto
    st.markdown("## About Me:")
    st.write("""
    **Name:** Carlos David García Hernández  
    **Rol:** Data Scientist @ Tec de Monterrey
             Teacher of regional economic analysis @ Universidad Nacional Autónoma de México
             
    **Contact:** [carlos.garcia.economist@gmail.com](mailto:carlos.garcia.economist@gmail.com)  
    """)

    # Enlace a LinkedIn
    st.markdown("### Connect with me:")
    st.markdown("[LinkedIn](https://www.linkedin.com/in/cgarcia8cg/)")  # Cambia tu URL
    st.markdown("[GitHub](https://cgarcia8cg.github.io/)")  # Cambia tu URL

    st.write("---")

    # Información sobre el Proyecto
    st.markdown("## About the proyect")
    st.write("""
    This project analyzes the CDMX metro network using centrality metrics to identify key stations.
    It is based on libraries such as **GeoPandas**, **NetworkX** and **Folium** for interactive geospatial visualizations.
    """)

    st.write("**Objectives:**")
    st.markdown("- Identify critical stations by various centrality metrics.")
    st.markdown("- Visualize mobility patterns.")
    st.markdown("- Reflect on public policies to improve the mobility system.")

    st.write("---")
    st.write("Explore the results on the interactive map and analyze selected metrics.")


# Configurar título de la App
st.title("Clasificación de Dígitos con AdaBoost")

# Subida de archivo de imagen
st.markdown("### Sube una imagen en cualquier formato o tamaño:")
uploaded_file = st.file_uploader("Elige una imagen", type=["png", "jpg", "jpeg"])

# Botón para procesar la imagen
if st.button("Predecir"):
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagen cargada", use_container_width =True)
        st.write("")

        result = query_api(image)
        st.write(result)
        if 'prediction' in result:
            st.write(f"Predicción: {result['prediction']}")
        else:
            st.error(f"Error en la respuesta: {result}")
    else:
        st.error("Por favor, sube una imagen primero.")
