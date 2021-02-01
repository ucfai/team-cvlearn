import numpy as np
import pandas as pd
from sklearn import datasets
from tensorflow import keras
from tensorflow import one_hot
from sklearn.model_selection import train_test_split


if __name__ == "__main__":


    data = datasets.load_iris()


    #print(data)

    x = data["data"]
    y = data["target"]


    x_train, x_test , y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    y_train = one_hot(y_train,3)
    y_test = one_hot(y_test,3)

    print(y_train)



    #print(x)



    




