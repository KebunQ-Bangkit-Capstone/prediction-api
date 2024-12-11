import os
import io
import uvicorn

import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
from fastapi import FastAPI, Response, UploadFile
import tempfile

from utils import download_model_from_gcs, preprocess_image

app = FastAPI()

cucumberModelUrl = os.getenv('CUCUMBER_MODEL_URL')
grapeModelUrl = os.getenv('GRAPE_MODEL_URL')
# tomatoModelUrl = os.getenv('TOMATO_MODEL_URL')

cucumber_model = None
grape_model = None
# tomato_model = None

def prepare_model():
    print("Downloading model from GCS...")
    
    global cucumber_model
    global grape_model
    # global tomato_model
    
    cucumber_model_bytes = download_model_from_gcs(cucumberModelUrl)
    grape_model_bytes = download_model_from_gcs(grapeModelUrl)
    
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as temp_file:
            temp_file.write(cucumber_model_bytes)
            cucumber_file_path = temp_file.name
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as temp_file:
            temp_file.write(grape_model_bytes)
            grape_file_path = temp_file
            
    cucumber_model = load_model(cucumber_file_path)
    grape_model = load_model(grape_file_path)
    
    print("Model loaded successfully!")
    
prepare_model()

@app.get('/', status_code=200)
def index():
    return {
        'status': 'healthy',
        'cucumber_model': cucumber_model is not None,
        'grape_model': grape_model is not None,
        # 'tomato_model': tomatoModel is not None
    }

@app.post('/predict/{plant_index}', status_code=200)
async def predict(image: UploadFile, plant_index: int, response: Response):
    if image.content_type not in ["image/jpeg", "image/png"]:
            response.status_code = 400
            return {'error': 'File is not an image'}

    if image.filename == '':
        response.status_code = 400
        return {'error': 'No file selected'}
        
    print('plant_index: '+plant_index)
    
    if cucumber_model is None:
        response.status_code = 500;
        return {'error': 'Cucumber Model not loaded'}
    if grape_model is None:
        response.status_code = 500;
        return {'error': 'Grape Model not loaded'}
    # if tomatoModel is None:
    #     response.status_code = 500;
    #     return {'error': 'Tomato Model not loaded'}
    
    try:
        image = Image.open(io.BytesIO(image.read()))
        
        processed_image = preprocess_image(image)
        
        prediction = None
        
        match plant_index:
            case 0:
                prediction = cucumber_model.predict(processed_image)
            case 1:
                prediction = grape_model.predict(processed_image)
            # case 2:
            #     prediction = tomatoModel.predict(processed_image)
        
        predictionList = None if prediction is None else prediction.tolist()
        
        return {
            'prediction': predictionList
        }
    
    except Exception as e:
        response.status_code = 400
        return {'error': str(e)}


port = os.environ.get("PORT", 5000)
print(f"Listening to http://0.0.0.0:{port}")
uvicorn.run(app, host='0.0.0.0',port=port)