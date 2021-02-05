import streamlit as st 
from tensorflow import keras 
import matplotlib.pyplot as plt
import numpy as np 

def show_predicts(predict):
    fig, ax = plt.subplots(1)

    for x in range(3):
        ax.barh(x, predict[x])

    y = np.arange(3)
    
    ax.set_yticks(y)
    ax.set_yticklabels(["Iris-setoya", "Iris-vericolor", "Iris-virginica"])

    ax.set_xlabel("Probability")
    ax.set_title("Predictions")

    return fig

if __name__ == "__main__":
    model =  keras.applications.NASNetMobile(
        input_shape=None,
        include_top=True,
        weights="imagenet",
        input_tensor=None,
        pooling=None,
        classes=1000,
    )

    st.text("Hello there!")

    sp_len = st.number_input("Sepal Length", min_value=1, max_value=10)
    sp_wid = st.number_input("Sepal Width", min_value=1, max_value=10)

    pt_len = st.number_input("Petal Length", min_value=1, max_value=10)
    pt_wid = st.number_input("Petal Width", min_value=1, max_value=10)

    # x = np.array([sp_len, sp_wid, pt_len, pt_wid]).reshape(1, -1)

    # prediction = model.predict(x).reshape(3, 1) * 100

    # print(np.argmax(prediction))

    # fig = show_predicts(prediction)

    # st.write(fig)

    pass