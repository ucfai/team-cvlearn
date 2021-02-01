from tensorflow.keras.applications.resnet50 import ResNet50 #Choose what ever moodel
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
from tensorflow import keras
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def graph(prediction_classes,prediction_probabilities):

    fig,ax = plt.subplots()

    ax.barh(0,prediction_probabilities[0])
    ax.barh(1,prediction_probabilities[1])
    ax.barh(2,prediction_probabilities[2])

    y = np.arange(3)
    ax.set_yticks(y)
    ax.set_yticklabels(prediction_classes)

    ax.set_xlabel('Probability')
    ax.set_title('Predictions')

    return fig

def get_model():
    model = ResNet50(weights='imagenet')
    return model

if __name__ == "__main__":  

    model = get_model()

    img_path = '../images/car.jpg' #path for hte image
    img = image.load_img(img_path, target_size=(224, 224)) #loads the image and sizes accordingly
    x = image.img_to_array(img) #turns the image to an array so the model can use
    x = np.expand_dims(x, axis=0) #Expands by adding a row.
    
    x = preprocess_input(x) #Formats image for the way that model requires  

    preds = model.predict(x) 

    # decode the results into a list of tuples (class, description, probability)
    # (one such list for each sample in the batch)

    predictions = decode_predictions(preds, top=3)[0] 

    prediction_classes = np.array([predictions[0][1],predictions[1][1],predictions[2][1]]) #Gets the top 3 classes
    prediction_probabilities = np.array([predictions[0][2],predictions[1][2],predictions[1][2]]) *100 #Gets the top 3 probablities

    fig = graph(prediction_classes,prediction_probabilities)
    st.write(fig)