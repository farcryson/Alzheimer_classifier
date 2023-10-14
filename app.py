import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image, ImageOps
import numpy as np

@st.cache_data()
def load_model():
    model = tf.keras.models.load_model('model.hdf5')
    return model
with st.spinner('Model is being loaded...'):
    model = load_model()
st.title("Alzheimer's Classification")
file = st.file_uploader("Upload the image", type = ["jpg", "png"])
st.set_option('deprecation.showfileUploaderEncoding', False)

def upload_predict(img, model):
    test_image = tf.keras.preprocessing.image.load_img(img, target_size=(32, 32))
    test_image = tf.keras.preprocessing.image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    prediction = model.predict(test_image)
    ans = np.argmax(prediction, axis = 1)
    return ans

if file is None:
    st.text("Please upload an image file")
else:
    prediction = upload_predict(file, model)
    img = tf.keras.preprocessing.image.load_img(file)
    st.image(img, width = 300)
    if str(prediction[0] == 0):
        image_class = "MildDemented"
    elif str(prediction[0] == 0):
        image_class = "ModerateDemented"
    elif str(prediction[0] == 0):
        image_class = "NonDemented"
    else:
        image_class = "VeryMildDemented"
    st.write("This image is classified as ", image_class)
    
