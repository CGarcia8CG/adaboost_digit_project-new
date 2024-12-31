
from fastapi import FastAPI, UploadFile, File
import joblib
import numpy as np
from PIL import Image
import io

import sys
import os

# Añadir manualmente la carpeta 'src' al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Importar configuración desde src/config.py
from config import Config

# Cargar configuración
config = Config()
#model_path = config.model_path1
model_path = os.path.join(os.getcwd(), "models", "logistic_adaboost.pkl")

# Crear aplicación
app = FastAPI()

# Cargar modelo
model = joblib.load(model_path)

# Preprocesar imagen para convertirla en el formato esperado (28x28 escala de grises)
def preprocess_image(image: Image.Image):
    image = image.convert("L")
    image = image.resize((8, 8))
    img_array = np.array(image) / 16.0
    return img_array.flatten().tolist()

@app.get("/")
async def root():
    return {"message": "API funcionando"}

# Endpoint para predicción desde imagen
@app.post("/predict-image/")
async def predict_image(file: UploadFile = File(...)):
    image = Image.open(file.file)
    features = preprocess_image(image)
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)
    return {"prediction": int(prediction[0])}

# Endpoint para estado de salud
@app.get("/health")
async def health():
    return {"status": "ok"}
