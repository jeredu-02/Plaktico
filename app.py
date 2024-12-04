from flask import Flask, request, jsonify
from main import MultiLabelClassifier
from flask_cors import CORS
import os
import uuid

# Define las etiquetas
labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Inicializa el clasificador y carga el modelo
classifier = MultiLabelClassifier(labels)
classifier.load_model('multi_label_classifier.pkl')

# Inicializa la aplicación Flask
app = Flask(__name__)
CORS(app)

# Ruta para verificar si la API está activa (GET)
@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "La API está activa y funcionando."})

# Ruta para predecir una imagen (POST)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No se proporcionó una imagen'}), 400

        # Generar un nombre único para el archivo
        image = request.files['image']
        unique_filename = f"./temp/{uuid.uuid4().hex}.jpg"
        image.save(unique_filename)
        print(f"Imagen guardada en {unique_filename}")

        # Realizar la predicción
        predicted_label, confidence = classifier.predict(unique_filename)
        print(f"Predicción: {predicted_label}, Confianza: {confidence}")

        threshold = 0.90  # Ajusta según sea necesario
        if confidence < threshold:
            predicted_label = "None" 

        # Eliminar la imagen temporal
        os.remove(unique_filename)
        print("Imagen eliminada.")

        return jsonify({'predicted_label': predicted_label, 'confidence': confidence})

    except Exception as e:
        print(f"Error en el servidor: {str(e)}")
        return jsonify({'error': 'Error interno del servidor', 'details': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
