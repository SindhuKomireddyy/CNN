import os
import streamlit as st
import tensorflow as tf
import numpy as np

from PIL import Image

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


st.title(
    "Cat vs Dog Classification"
)


if not os.path.exists(
    "models/cnn_model.keras"
):

    st.error(
        "Train the model first"
    )

    st.stop()


model = tf.keras.models.load_model(
    "models/cnn_model.keras"
)


uploaded_file = st.file_uploader(

    "Upload Image",

    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


if uploaded_file:


    image = Image.open(
        uploaded_file
    )


    image = image.convert(
        "RGB"
    )


    st.image(

        image,

        caption="Uploaded Image",

        width=300
    )


    img = image.resize(
        (224,224)
    )


    img = np.array(
        img
    )


    img = np.expand_dims(
        img,
        axis=0
    )


    img = preprocess_input(
        img
    )


    prediction = model.predict(
        img
    )[0][0]


    st.write(
        "Confidence:",
        prediction
    )


    if prediction > 0.5:

        st.success(
            "Prediction : Dog 🐶"
        )

    else:

        st.success(
            "Prediction : Cat 🐱"
        )