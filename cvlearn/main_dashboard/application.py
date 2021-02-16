import numpy as np 
import streamlit as st 
from tensorflow import keras 
import matplotlib.pyplot as plt
from PIL import Image
import requests
import io
from tensorflow.keras.applications.nasnet import preprocess_input as nasnet_preprocess, decode_predictions
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as inception_preprocess


def get_internet_image(image_address):

    r = requests.get(image_address)
    bytes = io.BytesIO(r.content)
    image = Image.open(bytes)
    st.image(image)
    return image

def get_file_image(image_file):

    image = Image.open(image_file)
 
    return image

def format_image(image,size):

    image = image.resize(size)
    image = np.array(image)
    image = np.expand_dims(image,axis = 0)

    return image

@st.cache
def get_predictions(model_name: str,image,num_pd):

    if model_name == "xception":

        predictions = xception(image,num_pd)

    elif model_name == "nasnet":

        predictions = nasnet(image,num_pd)

    elif model_name == "inception":

        predictions = inception(image,num_pd)

    elif model_name == "resnet":

        predictions = resnet(image,num_pd)


    return predictions


def xception(image,num_pd):

     image = format_image(image, (299,299))
 
     model = keras.applications.Xception(weights='imagenet')

     image = xception_preprocess(image)

     predictions = model.predict(image)

     results = decode_predictions(predictions, top= num_pd)[0]

     return results


def nasnet(image,num_pd):

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
     

def inception(image,num_pd):
    
    image = format_image(image, (299,299))

    model= keras.applications.InceptionResNetV2(
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


def resnet(image,num_pd):

    image = format_image(image, (224,224))
    model = keras.applications.ResNet50(weights='imagenet')

    image = resnet_preprocess(image)
    predictions = model.predict(image)

    results = decode_predictions(predictions, top=num_pd)[0]

    return results


def graph(predictions):

    num_pd = len(predictions)

    prediction_classes = np.array([predictions[i][1] for i in range(num_pd)]) #Gets the top 3 classes
    prediction_probabilities = np.array([predictions[i][2] for i in range(num_pd)]) *100 #Gets the top 3 probablities    

    fig, ax = plt.subplots()

    ax.bar(0, prediction_probabilities[0], color="green")

    for x in range(1, num_pd):
        ax.bar(x, prediction_probabilities[x], color="gray") #Adds the third bar

    y = np.arange(num_pd) #Creates an array of [1,2,3] for the ticks
    # ax.set_yticks(y) #Creates ticks on graph 
    ax.set_ylabel("Probability") #adds the labels

    #Adds titles and labels
    ax.set_xticks(np.arange(num_pd))
    ax.set_xticklabels(list(prediction_classes)) 
    ax.set_title('Predictions')

    st.write(fig)

if __name__ == "__main__":

    image_address = st.text_input("Enter image url")
    image_file = st.file_uploader('Upload an image')

    selection = st.selectbox('Select', ["xception","nasnet","inception","resnet"])

    num_pd = st.slider('Number of predictions', min_value=1, max_value=25)

    if image_address:
        image = get_internet_image(image_address)
        predictions = get_predictions(selection,image,num_pd)
        graph(predictions)


    if image_file:
        image = get_file_image(image_file)
        predictions = get_predictions(selection,image,num_pd)
        graph(predictions)
