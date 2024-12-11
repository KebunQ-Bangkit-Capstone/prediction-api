import os
import uvicorn

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, Response, UploadFile

from utils import preprocess_image

app = FastAPI()

local_cucumber_model_path = 'models/cucumber_model.h5'
local_grape_model_path = 'models/grape_model.h5'
localCucumberModelPath = '/models/tomato_model.h5'

cucumber_model = None
grape_model = None
tomato_model = None

def prepare_model():
    global cucumber_model
    global grape_model
    global tomato_model
        
    cucumber_model = tf.keras.models.load_model(local_cucumber_model_path)
    grape_model = tf.keras.models.load_model(local_cucumber_model_path)
    tomato_model = tf.keras.models.load_model(local_cucumber_model_path)
        
    print('Model loaded successfully.')
    
prepare_model()

@app.get('/', status_code=200)
def index():
    return {
        'status': 'healthy',
        'cucumber_model': cucumber_model is not None,
        'grape_model': grape_model is not None,
        'tomato_model': tomato_model is not None
    }

@app.post('/predict/{plant_index}', status_code=200)
async def predict(plant_index: int, image: UploadFile, response: Response):
    if image.content_type not in ["image/jpeg", "image/png"]:
        response.status_code = 400
        return {'error': 'File is not an image'}

    if image.filename == '':
        response.status_code = 400
        return {'error': 'No file selected'}
    
    if cucumber_model is None:
        response.status_code = 500;
        return {'error': 'Cucumber Model not loaded'}
    if grape_model is None:
        response.status_code = 500;
        return {'error': 'Grape Model not loaded'}
    if tomato_model is None:
        response.status_code = 500;
        return {'error': 'Tomato Model not loaded'}
    
    try:
        image = await image.read()
        
        processed_image = preprocess_image(image)
        
        prediction = None
        
        if plant_index == 0:
            prediction = cucumber_model.predict(processed_image)
        elif plant_index == 1:
            prediction = grape_model.predict(processed_image)
        elif plant_index == 2:
            prediction = tomato_model.predict(processed_image)
        
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = float(np.max(prediction))
        
        return {
            'class': int(predicted_class),
            'confidence_score': confidence,
            'prediction': prediction.tolist()
        }
    
    except Exception as e:
        response.status_code = 400
        return {'error': str(e)}


port = os.environ.get("PORT", 5000)
print(f"Listening to http://0.0.0.0:{port}")
uvicorn.run(app, host='0.0.0.0',port=port)