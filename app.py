import tensorflow as tf
import streamlit as st
import cv2
from PIL import Image, ImageOps
import numpy as np

model = tf.keras.models.load_model('C:/Users/kumar/Desktop/kk.hdf5')

st.write("""
         # Knock Knees prediction
         """
         )
st.write("This is a simple image classification web app to predict if you have knock knees or not.")
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])



def import_and_predict(image_data, model):
    
    size = (200,200)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis,...]
    prediction = model.predict(img_reshape)
    return prediction
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    
    if np.argmax(prediction) == 0:
        st.write("You have Knock Knees")
    elif np.argmax(prediction) == 1:
        st.write("You don't have knock knees")
    
    st.text("Probability (0: Have Knock Knees, 1: Don't have knock knees")
    st.write(prediction)