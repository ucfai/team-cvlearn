# %%
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

EPOCHS = 5

# %%
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# %%
x_train.shape

# %%
# Normalize to [0, 1]
x_train = x_train / 255
x_test = x_test / 255

# One-hot encode the labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# %%
ann = Sequential()
ann.add(Flatten(input_shape=(28, 28)))
ann.add(Dense(64, activation="relu", name="hidden1"))
ann.add(Dense(32, activation="relu", name="hidden2"))
ann.add(Dense(10, activation="softmax"))

ann.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"])

# %%
ann.summary()

# %%
history = ann.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=EPOCHS)


# %%
def plot_history(history):
    """
    Plot train/test loss and accuracy curves.
    https://keras.io/visualization/

    # Arguments
        history: keras history
    """
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs = range(1, len(acc) + 1)

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    axs[0].plot(epochs, acc, label="Training acc")
    axs[0].plot(epochs, val_acc, label="Validation acc")
    axs[0].set_title("Training and Validation accuracy")
    axs[0].legend()

    axs[1].plot(epochs, loss, label="Training loss")
    axs[1].plot(epochs, val_loss, label="Validation loss")
    axs[1].set_title("Training and Validation loss")
    axs[1].legend()

    return fig, axs


# %%
fig, ax = plot_history(history)

# %%
ann.evaluate(x_test, y_test)

# %%
