import io
import numpy as np
from google.cloud import storage

def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

def download_model_from_gcs(model_path: str, local_model_path: str):
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket('kebunq-ml-model')
        blob = bucket.blob(model_path)
        blob.download_to_filename(local_model_path)
        print(f"Model downloaded successfully to {local_model_path}")
    except Exception as e:
        print(f"Error downloading model: {e}")