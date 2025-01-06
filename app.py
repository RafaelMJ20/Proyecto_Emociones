from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
from flask_cors import CORS
from PIL import Image, ImageDraw
import io
import base64
import json
import os
from tensorflow.keras.models import model_from_json
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

app = Flask(__name__)
CORS(app)

# ID de la carpeta donde deseas subir la imagen
FOLDER_ID = '1Z5oK0YBGg8HFsbpmsUWFwKqtczKXZxPX'

# Diccionario de emociones
label_to_text = {0: 'Ira', 1: 'Odio', 2: 'Tristeza', 3: 'Felicidad', 4: 'Sorpresa'}

# Cargar el modelo y los pesos
def load_model():
    with open('FacialExpression-model.json', 'r') as json_file:
        loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    model.load_weights('FacialExpression_weights.hdf5')
    return model

model = load_model()

# Función para preprocesar la imagen
def preprocess_image(image_bytes):
    """Preprocesa la imagen para el modelo desde la memoria."""
    image = Image.open(io.BytesIO(image_bytes)).convert('L')  # Leer imagen en escala de grises
    image = image.resize((64, 64))
    image = np.array(image).reshape(1, 64, 64, 1).astype('float32') / 255.0
    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No se subió ninguna imagen'}), 400

    # Leer la imagen directamente desde la memoria
    image_file = request.files['image']
    image_bytes = image_file.read()

    # Preprocesar la imagen en memoria
    image_data = preprocess_image(image_bytes)

    # Hacer predicción
    predictions = model.predict(image_data)
    emotion_index = np.argmax(predictions)
    emotion = label_to_text[emotion_index]
    confidence = predictions[0][emotion_index]

    # Subir la imagen procesada a Google Drive
    img_data = process_and_save_image(image_bytes, image_file.filename)

    return jsonify({
        'emotion': emotion,
        'confidence': float(confidence),
        'image_base64': img_data['image_base64'],
        'drive_id': img_data['drive_id']
    })

def obtener_servicio_drive():
    SCOPES = ['https://www.googleapis.com/auth/drive.file']
    creds_info = json.loads(os.environ.get('GOOGLE_APPLICATION_CREDENTIALS_JSON'))
    creds = service_account.Credentials.from_service_account_info(creds_info, scopes=SCOPES)
    service = build('drive', 'v3', credentials=creds)
    return service

def process_and_save_image(image_bytes, filename):
    """Convierte la imagen y la sube a Google Drive."""
    # Convertir la imagen a formato PIL para posibles manipulaciones
    image = Image.open(io.BytesIO(image_bytes))
    # Aquí puedes hacer más transformaciones a la imagen si lo deseas

    # Guardar la imagen procesada como base64
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_data = buffered.getvalue()

    # Subir a Google Drive
    service = obtener_servicio_drive()
    archivo_drive = MediaIoBaseUpload(io.BytesIO(img_data), mimetype='image/png')
    archivo_metadata = {
        'name': f'{filename}',
        'mimeType': 'image/png',
        'parents': [FOLDER_ID]
    }
    archivo_subido = service.files().create(body=archivo_metadata, media_body=archivo_drive).execute()

    return {
        'image_base64': base64.b64encode(img_data).decode('utf-8'),
        'drive_id': archivo_subido.get('id')
    }

if __name__ == '__main__':
    app.run(debug=True)
