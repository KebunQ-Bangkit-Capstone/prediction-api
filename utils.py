import numpy as np
from google.cloud import storage

def preprocess_image(image):
    image = image.resize(224, 224)
    image_array = np.array(image)
    return np.expand_dims(image_array, axis=0)

def download_model_from_gcs(source_blob_name, destination_file_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket('kebunq-ml-model')
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Model downloaded to {destination_file_name}")