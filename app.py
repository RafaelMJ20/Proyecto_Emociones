import os
import cv2
import numpy as np
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import model_from_json
from werkzeug.utils import secure_filename
import tempfile

app = Flask(__name__)

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

def preprocess_image(image_path):
    """Preprocesa la imagen para el modelo."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
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

    # Guardar la imagen temporalmente en un directorio temporal
    image_file = request.files['image']
    filename = secure_filename(image_file.filename)

    # Crear un archivo temporal
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(image_file.read())  # Escribir el contenido de la imagen en el archivo temporal
        temp_image_path = temp_file.name

    # Preprocesar la imagen
    image_data = preprocess_image(temp_image_path)

    # Hacer predicción
    predictions = model.predict(image_data)
    emotion_index = np.argmax(predictions)
    emotion = label_to_text[emotion_index]
    confidence = predictions[0][emotion_index]

    # Eliminar el archivo temporal
    os.remove(temp_image_path)

    return jsonify({'emotion': emotion, 'confidence': float(confidence)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
