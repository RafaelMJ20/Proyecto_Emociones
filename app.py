import io
import os
import numpy as np
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import model_from_json
from PIL import Image

app = Flask(__name__)

# Diccionario de emociones
label_to_text = {0: 'Ira', 1: 'Odio', 2: 'Tristeza', 3: 'Felicidad', 4: 'Sorpresa'}

# Crear el directorio de uploads temporalmente si no existe
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Cargar el modelo y los pesos
def load_model():
    print("Cargando el modelo ...")
    with open('FacialExpression-model.json', 'r') as json_file:
        loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    model.load_weights('FacialExpression_weights.hdf5')
    print("Modelo cargado exitosamente.")
    return model

model = load_model()

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

    return jsonify({'emotion': emotion, 'confidence': float(confidence)})

if __name__ == '__main__':
    app.run(debug=True)
