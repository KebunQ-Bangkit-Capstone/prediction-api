import os
import io
import uvicorn

from tensorflow.keras.models import load_model
from PIL import Image
from fastapi import FastAPI, Response, UploadFile

from utils import preprocess_image

app = FastAPI()

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

@app.get('/', status_code=200)
def index():
    return {
        'status': 'healthy',
        'cucumber_model': cucumberModel is not None,
        'grape_model': grapeModel is not None,
        'tomato_model': tomatoModel is not None
    }

@app.post('/predict/{plant_index}', status_code=200)
def predict(image: UploadFile, plant_index: int, response: Response):
    if image.content_type not in ["image/jpeg", "image/png"]:
            response.status_code = 400
            return {'error': 'File is not an image'}

    if image.filename == '':
        response.status_code = 400
        return {'error': 'No file selected'}
        
    print('plant_index: '+plant_index)
    
    if cucumberModel is None:
        response.status_code = 500;
        return {'error': 'Cucumber Model not loaded'}
    if grapeModel is None:
        response.status_code = 500;
        return {'error': 'Grape Model not loaded'}
    if tomatoModel is None:
        response.status_code = 500;
        return {'error': 'Tomato Model not loaded'}
    
    try:
        image = Image.open(io.BytesIO(image.read()))
        
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
        
        return {
            'prediction': predictionList
        }
    
    except Exception as e:
        response.status_code = 400
        return {'error': str(e)}


port = os.environ.get("PORT", 5000)
print(f"Listening to http://0.0.0.0:{port}")
uvicorn.run(app, host='0.0.0.0',port=port)