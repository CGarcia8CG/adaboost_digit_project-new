# Imagen base
FROM python:3.9-slim

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar archivos al contenedor
COPY . .

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Exponer los puertos para API y Streamlit
EXPOSE 9000 8501

# Comandos para iniciar la API y la App
CMD sh -c "uvicorn api.model_api:app --host 0.0.0.0 --port 9000 & \
           streamlit run app/app.py --server.port 8501 --server.address 0.0.0.0"