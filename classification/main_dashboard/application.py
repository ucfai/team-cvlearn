import io
# necessary imports
import matplotlib.pyplot as plt
import numpy as np
import requests
# open source app framework designed for use with data science and machine learning
import streamlit as st
from PIL import Image
from tensorflow import keras
from tensorflow.keras.applications.inception_resnet_v2 import (
    preprocess_input as inception_preprocess,
)
from tensorflow.keras.applications.nasnet import decode_predictions
from tensorflow.keras.applications.nasnet import preprocess_input as nasnet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.xception import (
    preprocess_input as xception_preprocess,
)
# We used these pretrained models because we didn't Streamlit run and application.py and it'll 
# Different architectures and unique capacities trained on imagenet, which was 14 million images
# Experience with tensorflow and Streamlit

# Uses request library to send an http request 
def get_internet_image(image_address):
  
    r = requests.get(image_address)
    bytes = io.BytesIO(r.content)
    image = Image.open(bytes)
    col1.image(image,use_column_width=True)
    return image

# Get from the file which is provided by streamlit
def get_file_image(image_file):
    image = Image.open(image_file)
    col1.image(image,use_column_width=True)

    return image

# Formats the image to the provided size by passing in the image and its size
def format_image(image, size):
    image = image.resize(size)
    image = np.array(image)
    image = np.expand_dims(image, axis=0)

    return image

<<<<<<< HEAD:cvlearn/main_dashboard/application.py
<<<<<<< HEAD
@st.cache
def get_predictions(model_name: str,image,num_pd):

=======
# Different prediction models that are already pre-trained on a set of images
=======

>>>>>>> b2c257520ba2d4a102f139e7022f0e9ca8cd324d:classification/main_dashboard/application.py
def get_predictions(model_name: str, image, num_pd):
    if model_name == "xception":

        predictions = xception(image, num_pd)

    elif model_name == "nasnet":

        predictions = nasnet(image, num_pd)

    elif model_name == "inception":

        predictions = inception(image, num_pd)

    elif model_name == "resnet":

        predictions = resnet(image, num_pd)

    return predictions

# function for the xception model
def xception(image, num_pd):
    image = format_image(image, (299, 299))

    model = keras.applications.Xception(weights="imagenet")

    image = xception_preprocess(image)

    predictions = model.predict(image)

    results = decode_predictions(predictions, top=num_pd)[0]

    return results

# 
def nasnet(image, num_pd):
    image = format_image(image, (224, 224))

    model = keras.applications.NASNetMobile(
        input_shape=None,
        include_top=True,
        weights="imagenet",
        input_tensor=None,
        pooling=None,
        classes=1000,
    )

    image = nasnet_preprocess(image)
    predictions = model.predict(image)

    results = decode_predictions(predictions, top=num_pd)[0]

    return results


def inception(image, num_pd):
    image = format_image(image, (299, 299))

    model = keras.applications.InceptionResNetV2(
        include_top=True,
        weights="imagenet",
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        classifier_activation="softmax",
    )

    image = inception_preprocess(image)
    predictions = model.predict(image)

    results = decode_predictions(predictions, top=num_pd)[0]

    return results


def resnet(image, num_pd):
    image = format_image(image, (224, 224))
    model = keras.applications.ResNet50(weights="imagenet")

    image = resnet_preprocess(image)
    predictions = model.predict(image)

    results = decode_predictions(predictions, top=num_pd)[0]

    return results


def graph(predictions):

  

    num_pd = len(predictions)

    prediction_classes = np.array(
        [predictions[i][1] for i in range(num_pd)]
    )  # Gets the top 3 classes
    prediction_probabilities = (
        np.array([predictions[i][2] for i in range(num_pd)]) * 100
    )  # Gets the top 3 probablities

    fig, ax = plt.subplots()

    ax.barh(0, prediction_probabilities[0], color="green")

    for x in range(1, num_pd):
        ax.barh(x, prediction_probabilities[x], color="gray")  # Adds the third bar

    # ax.set_yticks(y) #Creates ticks on graph
    ax.set_xlabel("Probability")  # adds the labels

    # Adds titles and labels
    ax.set_yticks(np.arange(num_pd))
    ax.set_yticklabels(list(prediction_classes))
    ax.set_title("Predictions")

    col2.write(fig)


if __name__ == "__main__":

    image_address = st.text_input("Enter image url")
    image_file = st.file_uploader("Upload an image")

    col1,col2 = st.beta_columns(2)
    selection = col1.selectbox("Select", ["xception", "nasnet", "inception", "resnet"])

    num_pd = col2.slider("Number of predictions", min_value=1, max_value=25)

    if image_address:
        image = get_internet_image(image_address)
        predictions = get_predictions(selection, image, num_pd)
        graph(predictions)

    if image_file:
        image = get_file_image(image_file)
        predictions = get_predictions(selection, image, num_pd)
        graph(predictions)
