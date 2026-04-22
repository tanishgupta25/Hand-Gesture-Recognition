import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('gesture_model.h5')
    return model

model = load_model()

# Configuration
IMG_HEIGHT = 64
IMG_WIDTH = 64
class_names = [str(i) for i in range(10)]

st.set_page_config(page_title="Hand Gesture Recognition", layout="centered")

st.markdown("""
<style>
.big-font {
    font-size:100px !important;
    font-weight: bold;
    color: #4CAF50;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.title("Hand Gesture Recognition (Digits 0-9)")
st.write("Upload an image of a hand showing a digit from 0 to 9, or take a picture using your webcam.")

# Sidebar for input selection
option = st.sidebar.radio("Select Input Method:", ("Upload Image", "Webcam"))

image_file = None

if option == "Upload Image":
    image_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
else:
    image_file = st.camera_input("Take a picture")

if image_file is not None:
    # Display the uploaded image
    image = Image.open(image_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image
    st.write("Predicting...")
    
    # Resize and prepare the image for the model
    size = (IMG_HEIGHT, IMG_WIDTH)
    image_resized = ImageOps.fit(image, size, Image.LANCZOS)
    img_array = tf.keras.preprocessing.image.img_to_array(image_resized)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    
    # Predict
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    
    predicted_class = class_names[np.argmax(score)]
    
    # Output ONLY the predicted number
    st.markdown(f'<p class="big-font">{predicted_class}</p>', unsafe_allow_html=True)

