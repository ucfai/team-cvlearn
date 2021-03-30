import streamlit as st
import numpy as np
from tensorflow import keras
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.xception import preprocess_input, decode_predictions
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image


def graph(prediction_classes, prediction_probabilities):

    fig, ax = plt.subplots()

    ax.barh(0, prediction_probabilities[0])
    ax.barh(1, prediction_probabilities[1])
    ax.barh(2, prediction_probabilities[2])

    y = np.arange(3)
    ax.set_yticks(y)
    ax.set_yticklabels(prediction_classes)

    # Adds titles and labels
    ax.set_xlabel('Probability')
    ax.set_title('Predictions')

    return fig


if __name__ == "__main__":

    model = Xception(weights='imagenet')

    picture = '../images/car.jpg'
    pic = image.load_img(picture, target_size=(299, 299))
    st.image(pic, caption='car', use_column_width=True)

    x = image.img_to_array(pic)
    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)

    prediction = model.predict(x)

    results = decode_predictions(prediction, top=3)[0]

    print(results)

    prediction_classes = np.array(
        [results[0][1], results[1][1], results[2][1]])
    prediction_probabilities = np.array(
        [results[0][2], results[1][2], results[1][2]]) * 100

    fig = graph(prediction_classes, prediction_probabilities)

    st.write(fig)  # Writes the graph to streamlit
