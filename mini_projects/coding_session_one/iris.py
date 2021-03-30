import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow import one_hot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


if __name__ == "__main__":

    dataset = pd.read_csv("Iris.csv")

    x = np.array(dataset[["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]])

    y = np.array(dataset["Species"]) #Gets the integer values


    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
        ) #Splits up the data set


    #One hot encodes, better for training. 
    y_train = one_hot(y_train,3)
    y_test = one_hot(y_test,3)


    #Creates Sequential Model
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(100,activation = "relu"))
    model.add(keras.layers.Dense(10,activation = "relu"))
    model.add(keras.layers.Dense(3,activation = "softmax"))


    model.compile(loss = "categorical_crossentropy",optimizer = "adam" , metrics = ["accuracy"])
    model.fit(x_train,y_train,epochs = 50 , batch_size = 12) 

    model.summary()

    model.evaluate(x_test,y_test)  #Tests the model

    model.save("/Users/ryanpattillo/Documents/machineLearning/CvLearnNew/coding_session") #Saves the model





