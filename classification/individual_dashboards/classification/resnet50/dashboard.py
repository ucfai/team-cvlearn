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

    size = prediction_classes.size

    for i in range(size):
        ax.barh(i,prediction_probabilities[i]) #Adds the first bar
   #Adds the third bar

    y = np.arange(size) #Creates an array of [1,2,3] for the ticks
    ax.set_yticks(y) #Creates ticks on graph 
    ax.set_yticklabels(prediction_classes) #adds the labels

    #Adds titles and labels
    ax.set_xlabel('Probability') 
    ax.set_title('Predictions')

    return fig

@st.cache #Caches the model saves time when refreshing page
def get_model():

    model = ResNet50(weights='imagenet')
    return model

def predict(preds, num_predictions):

    predictions = decode_predictions(preds, top=num_predictions)[0] #only one sample in the batch [0]
    
    prediction_classes = np.array([predictions[i][1] for i in range(num_predictions)]) #Gets the top 3 classes
    prediction_probabilities = np.array([predictions[i][2] for i in range(num_predictions)]) *100 #Gets the top 3 probablities

    return prediction_classes, prediction_probabilities


if __name__ == "__main__":  

    model = get_model()

    img_path = '../images/catmask.jpg' #path for the image
    img = image.load_img(img_path, target_size=(224, 224)) #loads the image and sizes accordingly
    x = image.img_to_array(img) #turns the image to an array so the model can use
    x = np.expand_dims(x, axis=0) #Expands by a row to represent a batch size of 1. 
    #preprocess_input needs aNumpy array encoding a batch of predictions. 
    
    x = preprocess_input(x) #each Keras Application expects a specific kind of input preprocessing.
    #For Xception, call tf.keras.applications.xception.preprocess_input on your inputs before passing them to the model.

    num_predictions = st.slider('Slide me', min_value=0, max_value=50)

    preds = model.predict(x) 

    # decode the results into a list of tuples (class, description, probability)
    # (one such list for each sample in the batch)

    prediction_classes, prediction_probabilities = predict(preds,num_predictions)
  
    fig = graph(prediction_classes,prediction_probabilities)

    st.write(fig) #Writes the graph to streamlit