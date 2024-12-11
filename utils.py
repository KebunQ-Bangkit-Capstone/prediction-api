import numpy as np
from google.cloud import storage

def preprocess_image(image):
    image = image.resize(224, 224)
    return np.array(image)

def download_model_from_gcs(source_blob_name, destination_file_name):
    """
    Download a model from Google Cloud Storage to the local filesystem.
    Args:
        bucket_name (str): The name of the GCS bucket.
        source_blob_name (str): The path to the model in the bucket.
        destination_file_name (str): The local path to save the model.
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket('kebunq-ml-model')
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Model downloaded to {destination_file_name}")