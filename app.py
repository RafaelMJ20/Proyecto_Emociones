import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from flask import Flask, request, render_template, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
from werkzeug.utils import secure_filename
import imghdr
import io

app = Flask(__name__)

# Diccionario de emociones (asegúrate de que coincida con tu modelo)
label_to_text = {0: 'Ira', 1: 'Odio', 2: 'Tristeza', 3: 'Felicidad', 4: 'Sorpresa'}

# Cargar el modelo y los pesos
def load_model():
    with open('FacialExpression-model.json', 'r') as json_file:
        loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    model.load_weights('FacialExpression_weights.hdf5')
    return model

model = load_model()

def preprocess_image_from_memory(file):
    """Preprocesa la imagen para el modelo desde la memoria."""
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("No se pudo procesar la imagen")
    image = cv2.resize(image, (96, 96))
    image = image.reshape(1, 96, 96, 1).astype('float32') / 255.0
    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No se subió ninguna imagen'}), 400

    # Leer la imagen desde la memoria
    image_file = request.files['image']
    image_file.filename = secure_filename(image_file.filename)  # Asegurarse de que el nombre del archivo sea seguro

    # Verificar si el archivo es una imagen válida
    file_type = imghdr.what(image_file)
    if file_type not in ['jpeg', 'png', 'jpg']:
        return jsonify({'error': 'Formato de imagen no soportado'}), 400

    try:
        # Preprocesar la imagen directamente desde la memoria
        image_data = preprocess_image_from_memory(image_file)

        # Hacer predicción
        predictions = model.predict(image_data)
        emotion_index = np.argmax(predictions)
        emotion = label_to_text[emotion_index]
        confidence = predictions[0][emotion_index]

        return jsonify({'emotion': emotion, 'confidence': float(confidence)})
    except Exception as e:
        return jsonify({'error': f'Error al procesar la imagen: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
