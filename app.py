import tensorflow as tf
import streamlit as st


@st.cache(allow_output_mutation=True)
def main():
    model = tf.keras.models.load_model('/Documents/kk.hdf5')
    return model
model = main()
st.write(""" 
# Genu Valgum Detection
"""
        )
file = st.file_uploader("Please upload your legs Image", type=["jpg", "png"])
import cv2
from PIL import Image, ImageOps
import numpy as np
def import_and_predict(image_data, model):
    size = (128,128)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis,...]
    predition = moel.predict(img_reshape)
    return prediction
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    class_names = ['knockknees', 'normal']
    string = "This Image is Most likely is:" + class_name[np.argmax(predictions)]
    st.success(string)

if __name__ == '__main__':
    main()
