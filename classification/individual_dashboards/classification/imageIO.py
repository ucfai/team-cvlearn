import os
import streamlit as st
import io
from PIL import Image #Pillow is a libary for working with images
import requests


def get_internet_image(image_address,size):

    r = requests.get(image_address) #Will send a http request to the image address
    image = io.BytesIO(r.content) #Create a BytesIO object which turns the image into a file object in memory. It is just a buffer in memory
    image = Image.open(image) # Opens the image
    image = image.resize(size) #Resizes the image to appropriate format
    #np.array(image) will return the image as a numpy array.
    return image

def get_file_image(file, size):

    image = Image.open(file) #Keras image libary just convers to a PIL type. 
    return image

if __name__ == "__main__":


    image_address = st.text_input("Enter Image Address")
    file = st.file_uploader('File uploader')

    if bool(image_address): #Want to wait til given a value 
        image = get_internet_image(image_address,(50,50))
        st.image(image)

    if bool(file): #Want to wait tile given a value
        image = get_file_image(file,(50,50))
        st.image(image)


    


