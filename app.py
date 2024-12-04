from flask import Flask, request, jsonify
from main import MultiLabelClassifier
from flask_cors import CORS
import os
import uuid
import tempfile

# Define las etiquetas
labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Inicializa el clasificador y carga el modelo
classifier = MultiLabelClassifier(labels)
classifier.load_model('multi_label_classifier.pkl')

# Inicializa la aplicación Flask
app = Flask(__name__)
temp_dir = tempfile.gettempdir()
CORS(app)

# Ruta para verificar si la API está activa (GET)
@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "La API está activa y funcionando."})

# Ruta para predecir una imagen (POST)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image part in the request"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Guarda la imagen en el directorio temporal
    file_path = os.path.join(temp_dir, file.filename)
    file.save(file_path)

    # Procesa la imagen
    try:
        predicted_label, confidence = classifier.predict(file_path)
        os.remove(file_path)  # Limpia el archivo temporal después de procesarlo
        return jsonify({"predicted_label": predicted_label, "confidence": confidence})
    except Exception as e:
        return jsonify({"error": "Error interno del servidor", "details": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
