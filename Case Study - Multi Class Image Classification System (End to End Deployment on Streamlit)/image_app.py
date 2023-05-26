import streamlit as st
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array


def load_classification_model():
    model = load_model('Multi_Class_Image_Classification_System.h5')
    return model

model = load_classification_model()

st.write("""
Image Classification System
"""
)

file = st.file_uploader("Upload an Image", type = ['jpeg', 'jpg', 'png'])

import cv2
from PIL import Image, ImageOps
import numpy as np

def import_and_predict(image_data, model):
    size = (180, 180)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = img_to_array(image)
    img = img.reshape(1, 180, 180, 3)
    img = img.astype('float32')
    img = img / 255.0

    prediction = model.predict(img)
    return prediction

if file is None:
    st.text('Please upload an image file')
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    labels = ['Cat', 'Dog', 'Panda']
    string = "This image is belonging to : "+labels[np.argmax(predictions)]
    st.success(string)


