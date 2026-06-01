import os
import tensorflow as tf

from PIL import Image

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


dataset_path = r"C:\Users\sindh\Downloads\archive (20)\PetImages"


# create models folder

os.makedirs(
    "models",
    exist_ok=True
)


# remove corrupted images

for folder in ["Cat", "Dog"]:

    folder_path = os.path.join(
        dataset_path,
        folder
    )

    for file in os.listdir(folder_path):

        path = os.path.join(
            folder_path,
            file
        )

        try:

            img = Image.open(path)

            img = img.convert("RGB")

            img.save(path)

        except:

            try:
                os.remove(path)

            except:
                pass


print("Dataset cleaned")


# load images

train_data = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(224,224),
    batch_size=32
)


val_data = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(224,224),
    batch_size=32
)


print(train_data.class_names)


# preprocessing

train_data = train_data.map(
    lambda x,y: (
        preprocess_input(x),
        y
    )
)


val_data = val_data.map(
    lambda x,y: (
        preprocess_input(x),
        y
    )
)


train_data = train_data.prefetch(
    tf.data.AUTOTUNE
)


val_data = val_data.prefetch(
    tf.data.AUTOTUNE
)


# pretrained CNN

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)


base_model.trainable = False


model = Sequential([

    base_model,

    GlobalAveragePooling2D(),

    Dense(
        128,
        activation="relu"
    ),

    Dropout(
        0.3
    ),

    Dense(
        1,
        activation="sigmoid"
    )
])


model.compile(

    optimizer="adam",

    loss="binary_crossentropy",

    metrics=["accuracy"]

)


history = model.fit(

    train_data,

    validation_data=val_data,

    epochs=5

)


model.save(
    "models/cnn_model.keras"
)


print("Model saved successfully")