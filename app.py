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
import mediapipe as mp

app = Flask(__name__)
CORS(app)

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

# Inicializar MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# Función para preprocesar la imagen
def preprocess_image(image_bytes):
    """Preprocesa la imagen para el modelo desde la memoria."""
    image = Image.open(io.BytesIO(image_bytes)).convert('L')  # Leer imagen en escala de grises
    image = image.resize((96, 96))
    image = np.array(image).reshape(1, 96, 96, 1).astype('float32') / 255.0
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

    # Detectar puntos faciales y obtener la imagen procesada
    img_data = process_image(image_bytes, image_file.filename)

    return jsonify({
        'emotion': emotion,
        'confidence': float(confidence),
        'image_base64': img_data['image_base64']
    })

def process_image(image_bytes, filename):
    """Convierte la imagen, detecta puntos faciales y la procesa."""
    # Convertir la imagen a formato PIL para posibles manipulaciones
    image = Image.open(io.BytesIO(image_bytes))
    img_np = np.array(image)

    # Procesar la imagen para detectar puntos faciales
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)

    # Si se detectan puntos faciales, dibujarlos
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            draw = ImageDraw.Draw(image)
            for idx, landmark in enumerate(face_landmarks.landmark):
                if idx in [70, 55, 285, 300, 33, 468, 133, 362, 473, 263, 4, 185, 0, 306, 17]:  # puntos clave
                    h, w, _ = img_rgb.shape
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    draw.line((x - 5, y - 5, x + 5, y + 5), fill='red', width=2)
                    draw.line((x - 5, y + 5, x + 5, y - 5), fill='red', width=2)

    # Guardar la imagen procesada como base64
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_data = buffered.getvalue()

    return {
        'image_base64': base64.b64encode(img_data).decode('utf-8')
    }

if __name__ == '__main__':
    app.run(debug=True)
