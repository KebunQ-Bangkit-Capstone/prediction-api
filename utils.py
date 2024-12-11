import io
import numpy as np
from google.cloud import storage

def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

def download_model_from_gcs(source_blob_name: str):
    storage_client = storage.Client()
    bucket = storage_client.bucket('kebunq-ml-model')
    blob = bucket.blob(source_blob_name)
    model_bytes = blob.download_as_bytes()
    return model_bytes