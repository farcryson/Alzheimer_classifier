import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image, ImageOps
import numpy as np

@st.cache_data(allow_output_mutation = True)
def load_model():
    model = tf.keras.models.load_model('model.hdf5')
    return model
with st.spinner('Model is being loaded...'):
    model = load_model()
st.title("Alzheimer's Classification")
file = st.file_uploader("Upload the image", type = ["jpg", "png"])
st.set_option('deprecation.showfileUploaderEncoding', False)

def upload_predict(upload_image, model):
    size = (180, 180)
    image = ImageOps.fit(upload_image, size)
    image = np.asarray(image)
    imgage = cv2.resize(image, dsize = (32, 32, 3))
    
    prediction = model.predict(image)
    ans = np.argmax(prediction, axis = 1)
    return ans
if file is None:
    st.text("Please upload an image file")
else:
    image  = Image.open(file)
    st.image(image, use_column_width = True)
    predictions = upload_predict(image, model)
    if str(predictions[0] == 0):
        image_class = "MildDemented"
    elif str(predictions[0] == 0):
        image_class = "ModerateDemented"
    elif str(predictions[0] == 0):
        image_class = "NonDemented"
    else:
        image_class = "VeryMildDemented"
    st.write("The image is classified as ", image_class)
    
