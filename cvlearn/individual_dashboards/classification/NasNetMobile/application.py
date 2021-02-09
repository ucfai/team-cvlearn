import numpy as np 
import streamlit as st 
from tensorflow import keras 
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.nasnet import preprocess_input, decode_predictions

@st.cache
def get_model():

    model =  keras.applications.NASNetMobile(
        input_shape=None,
        include_top=True,
        weights="imagenet",
        input_tensor=None,
        pooling=None,
        classes=1000,
    )

    return model

def graph(classes, probabilities, predict_num):
    fig, ax = plt.subplots()

    ax.bar(0, prediction_probabilities[0], color="green")

    for x in range(1, predict_num):
        ax.bar(x, prediction_probabilities[x], color="gray") #Adds the third bar

    y = np.arange(predict_num) #Creates an array of [1,2,3] for the ticks
    # ax.set_yticks(y) #Creates ticks on graph 
    ax.set_ylabel("Probability") #adds the labels

    #Adds titles and labels
    ax.set_xticks(np.arange(predict_num))
    ax.set_xticklabels(list(prediction_classes)) 
    ax.set_title('Predictions')

    return fig

if __name__ == "__main__":
    model = get_model()

    path =  "../images/car.jpg"
    img = image.load_img(path, target_size=(224, 224, 3))
    arr_img = image.img_to_array(img)
    arr_img = np.array([arr_img])

    x = preprocess_input(arr_img) 

    preds = model.predict(x) 

    # decode the results into a list of tuples (class, description, probability)
    # (one such list for each sample in the batch)

    st.text("Hello!")
    num_pd = st.number_input("Number of predictions", min_value=1, max_value=10)
    st.image(path)

    predictions = decode_predictions(preds, top=num_pd)[0] #only one sample in the batch [0]
  
    prediction_classes = np.array([predictions[i][1] for i in range(num_pd)]) #Gets the top 3 classes
    prediction_probabilities = np.array([predictions[i][2] for i in range(num_pd)]) *100 #Gets the top 3 probablities

    fig = graph(prediction_classes,prediction_probabilities, num_pd)
    st.write(fig) #Writes the graph to streamlit
