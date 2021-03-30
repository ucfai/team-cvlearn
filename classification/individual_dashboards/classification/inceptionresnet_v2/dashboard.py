from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input, decode_predictions

import streamlit as st
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

def graph(predictions_classes, predictions_probabilities, predict_num):

    fig,ax = plt.subplots()
    for x in range(predict_num):
        ax.barh(x, predictions_probabilities[x])


    y = np.arange(predict_num)

    ax.set_yticks(y) #ticks on graph
    ax.set_yticklabels(predictions_classes) #add labels

    ax.set_xlabel('Probability')
    ax.set_title('Predictions')

    
    return fig
    pass

@st.cache 
def get_model():

    model= keras.applications.InceptionResNetV2(
        include_top=True,
        weights="imagenet",
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        classifier_activation="softmax",
        
    )

    return model


if __name__ == "__main__":

    model = get_model()

    img_path =  '../images/car.jpg'     #image path
    img = image.load_img(img_path, target_size=(299,299))
    st.image('../images/car.jpg') #display image
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)

    preds = model.predict(x)

    predict_num = st.slider ('Number of Predictions', min_value=1, max_value=15)
    predictions = decode_predictions(preds, top=predict_num)[0] #only one sample in the batch [0]

    predictions_classes = np.array([predictions[i][1] for i in range(predict_num)]) #Gets the top 3 classes
    predictions_probabilities = np.array([predictions[i][2] for i in range(predict_num)]) *100 #Gets the top 3 probablities


    fig = graph(predictions_classes,predictions_probabilities, predict_num)

    st.write(fig)
    
    