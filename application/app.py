"""Streamlit Fake Data Detector App"""

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Fake Detector Classifier", page_icon="✨")

st.title("Fake Data Detector")

st.markdown(
    "Welcome to this simple web application that detect AI generated data"
)

IMG_HEIGHT = 32
IMG_WIDTH = 32
IMG_CHANNELS = 3
TEXT_PATH = "text_model"
IMAGE_PATH = "image_model"

@st.cache_resource(show_spinner=False)
def load_and_cache_model(model_path):
    #model_path = "genai_detection_model"
    model = load_model(model_path)
    return model

def get_predictions_text(model, text):
    instances = pd.Series([text])
    prediction = model.predict(instances)
    if prediction[0][0] > prediction[0][1]:
        pred = round(prediction[0][0],2)
        #return f"Human: {pred:.2f}
        result = f"Human. Confidence: {pred:.2f}%"  
    else:
        pred = round(prediction[0][1],2)
        result = f"AI. Confidence: {pred:.2f}%"
                    
    return result

def read_image(img_bytes):
    img = tf.image.decode_jpeg(img_bytes, channels=IMG_CHANNELS)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img


def get_predictions_image(model, image):
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    image = np.expand_dims(image, axis=0)
    predictions = model.predict(image)
    if predictions[0][0] > 0.5:
        pred = round(predictions[0][0],2)
        #return f"Human: {pred:.2f}
        result = f"Human. Confidence: {pred:.2f}"
    else:
        pred = round(1 - predictions[0][0],2)
        #return f"Human: {pred:.2f}
        result = f"AI. Confidence: {pred:.2f}"
    return result

def main():
    text = st.text_area("Please enter the content that you want to verify:")
    class_btn = st.button("Classify text")
    st.markdown('Please upload the image that you want to verify')
    file_uploaded = st.file_uploader("Choose File", type=["png", "jpg", "jpeg"])
    
    if text is not None:
        st.markdown(text)
        if class_btn:
            with st.spinner("Model predicting...."):
                loaded_model = load_and_cache_model(TEXT_PATH)
                prediction_text = get_predictions_text(loaded_model, text)
                st.success(prediction_text)
    
    
    if file_uploaded is not None:
        
        image = read_image(file_uploaded.read())
        st.image(image.numpy(), caption="Uploaded Image", use_column_width=True)
        class_btn = st.button("Classify image")
        if class_btn:
            with st.spinner("Model predicting...."):
                loaded_model = load_and_cache_model(IMAGE_PATH)
                prediction_image = get_predictions_image(loaded_model, image)
                st.success(prediction_image)
    

if __name__ == "__main__":
    main()
    
