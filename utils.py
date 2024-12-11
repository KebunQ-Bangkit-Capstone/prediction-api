import numpy as np

def preprocess_image(image):
    image = image.resize(224, 224)
    return np.array(image)