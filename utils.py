import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import img_to_array

def preprocess_image(image, target_size=(256, 256)):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image
