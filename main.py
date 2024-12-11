import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from PIL import Image
import io

from utils import preprocess_image

app = Flask(__name__)
CORS(app)

cucumberModelUrl = os.getenv('CUCUMBER_MODEL_URL')
grapeModelUrl = os.getenv('GRAPE_MODEL_URL')
tomatoModelUrl = os.getenv('TOMATO_MODEL_URL')

try:
    cucumberModel = load_model(cucumberModelUrl)
    grapeModel = load_model(grapeModelUrl)
    tomatoModel = load_model(tomatoModelUrl)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    cucumberModel = None
    grapeModel = None
    tomatoModel = None

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'status': 'healthy',
        'cucumber_model': cucumberModel is not None,
        'grape_model': grapeModel is not None,
        'tomato_model': tomatoModel is not None
    })

@app.route('/predict/<int:plant_index>', methods=['POST'])
def predict(plant_index):
    print('plant_index: '+plant_index)
    
    if cucumberModel is None:
        return jsonify({'error': 'Cucumber Model not loaded'}), 500
    if grapeModel is None:
        return jsonify({'error': 'Grape Model not loaded'}), 500
    if tomatoModel is None:
        return jsonify({'error': 'Tomato Model not loaded'}), 500
    
    if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
    file = request.files['image']
        
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        image = Image.open(io.BytesIO(file.read()))
        
        processed_image = preprocess_image(image)
        
        prediction = None
        
        match plant_index:
            case 0:
                prediction = cucumberModel.predict(processed_image)
            case 1:
                prediction = grapeModel.predict(processed_image)
            case 2:
                prediction = tomatoModel.predict(processed_image)
        
        predictionList = None if prediction is None else prediction.tolist()
        
        return jsonify({
            'prediction': predictionList
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)