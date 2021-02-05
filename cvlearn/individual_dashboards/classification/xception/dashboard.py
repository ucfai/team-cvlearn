import streamlit as st
import numpy as numpy
from tensorflow import keras
from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input, decode_predictions
import matplotlib.pyplot as plt
#import Image

if __name__ == "__main__":

    model = Xception(weights='imagenet')

    picture = '/Users/kenleyrodriguez12/OneDrive - Knights - University of Central Florida/cvlearn/team-cvlearn/cvlearn/individual_dashboards/classification/images/car.jpg'
    st.image(picture, caption='car', use_column_width=True)
