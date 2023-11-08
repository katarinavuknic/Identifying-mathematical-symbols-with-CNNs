import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from PIL import Image as pil_image
from streamlit_drawable_canvas import st_canvas
import cv2 as cv

label_encoder2 = LabelEncoder()
label_encoder2.classes_ = np.load('classes.npy')

def load_model():
    model = tf.keras.models.load_model("mathsymbols.model")
    return model

model = load_model()

realtime_update = st.sidebar.checkbox("Update in realtime", True)
STROKE_WIDTH=32
drawing_mode = "freedraw"

def show_predict_page():

    st.title("Mathemathics symbols identification")
    st.write("In order for us to recognize a symbol, you need to write it inside a white box.")

    canvas_result = st_canvas(
        # fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=STROKE_WIDTH,
        stroke_color="rgb(0,0,0)", 
        background_color="#ffffff",  
        # background_image=pil_image.open(bg_image) if bg_image else None,
        update_streamlit=realtime_update,
        height=300, 
        drawing_mode=drawing_mode, 
        key="full_app",
    )

    if canvas_result.image_data is not None:
        
        image = canvas_result.image_data
        image1 = image.copy()
        image1 = cv.resize(image1, (32,32))
        image1.resize(1, 32, 32, 3)
        prediction = model.predict(image1)
        inverted = label_encoder2.inverse_transform([np.argmax(prediction)])
        st.title(inverted[0])

show_predict_page()