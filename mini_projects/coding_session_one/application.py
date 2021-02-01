import streamlit as st
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


def graph(predictions):

    fig,ax = plt.subplots()

    ax.barh(0,predictions[0])
    ax.barh(1,predictions[1])
    ax.barh(2,predictions[2])

    y = np.arange(3)

    ax.set_yticks(y)
    ax.set_yticklabels(['Iris-setosa','Iris-versicolor','Iris-virginica'])

    ax.set_xlabel('Probability')
    ax.set_title('Predictions')

    return fig


if __name__ == "__main__":

     model = keras.models.load_model("/Users/ryanpattillo/Documents/machineLearning/CvLearnNew/coding_session_live/saved_model")


     sepal_length = st.number_input("Sepal Length")
     sepal_width = st.number_input("Sepal Width")
     petal_length = st.number_input("Petal Length")
     petal_width = st.number_input("Petal Width")


     x = np.array([sepal_length,sepal_width,petal_length,petal_width]).reshape(1,-1)
  
     predictions = model.predict(x).reshape(3,1) * 100
  
     fig = graph(predictions)

     st.write(fig)
