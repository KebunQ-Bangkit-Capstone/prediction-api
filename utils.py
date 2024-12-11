import io
import numpy as np
from PIL import Image

def preprocess_image(file):
    image = Image.open(io.BytesIO(file))
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)