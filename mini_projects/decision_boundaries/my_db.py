import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# here is to get dataset name
def get_dataset(name):
    if name == "iris":
        return datasets.load_iris()


def get_index(name):
    target_index = iris["target_names"].tolist().index(name)
    return target_index


def get_target(name, target_index):
    if name == "iris":
        return (iris["target"] == target_index).astype(np.int)


def get_features(name):
    if name == "iris":
        return iris["data"]


def get_desired_features_index(name, f1, f2):
    if name == "iris":
        return (iris["feature_names"].index(f1), iris["feature_names"].index(f2))


def get_desired_features(features, index_one, index_two):
    f1 = features[:, [index_one]]
    f2 = features[:, [index_two]]
    return np.concatenate((f1, f2), axis=1)


def get_min_max(features):
    return (
        features[:, [0]].min() - 1,
        features[:, [0]].max() + 1,
        features[:, [1]].min() - 1,
        features[:, [1]].max() + 1,
    )


def create_meshgrid(f_x, f_y):
    fx, fy = np.meshgrid(f_x, f_y)
    rx = fx.flatten()  # Return a copy of the array collapsed into one dimension
    ry = fy.flatten()
    rx, ry = rx.reshape((len(rx), 1)), ry.reshape((len(rx), 1))
    return (
        fx,
        fy,
        np.hstack((rx, ry)),
    )  # Stack arrays in sequence horizontally (column wise).


def create_graph(
    f1,
    f2,
    point_cmap,
    graph_cmap,
    target,
    titlelabel,
    xlabel,
    ylabel,
    figsize,
    fx,
    fy,
    fz,
):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(50, 40))
    ax.set_title("Decision Boundary " + titlelabel, fontsize=50)
    ax.contourf(fx, fy, fz, cmap=graph_cmap)
    ax.scatter(f1, f2, marker="^", s=300, c=target, cmap=point_cmap)
    ax.set_xlabel(xlabel, fontsize=50)
    ax.set_ylabel(ylabel, fontsize=50)
    st.write(fig)


if __name__ == "__main__":
    iris = datasets.load_iris()  # dictionary of numpy arrays

    st.title("Decision Boundaries Logistic Regression")

    dataset_name = st.sidebar.selectbox("Dataset ", ["iris"])
    target_name = st.sidebar.selectbox("Flower", ["virginica", "setosa", "versicolor"])
    f1_name = st.sidebar.selectbox(
        "Feature 1",
        [
            "petal length (cm)",
            "petal width (cm)",
            "sepal length (cm)",
            "sepal width (cm)",
        ],
    )
    f2_name = st.sidebar.selectbox(
        "Feature 2",
        [
            "petal length (cm)",
            "petal width (cm)",
            "sepal length (cm)",
            "sepal width (cm)",
        ],
    )
    graph_type = st.sidebar.selectbox(
        "Graph Type", ["Probability Classification", "Prediction Classification"]
    )
    cmap = st.sidebar.selectbox(
        "Graph Color Scheme", ["magma", "plasma", "Blues", "RdBu", "cool", "rainbow"]
    )
    point_cmap = st.sidebar.selectbox(
        "Point Color Scheme", ["magma", "plasma", "Blues", "RdBu", "cool", "rainbow"]
    )

    choice = 0

    if graph_type == "Probability Classification":
        choice = 1

    dataset = get_dataset(dataset_name)
    target_index = get_index(target_name)
    target = get_target(dataset_name, target_index)
    features = get_features(dataset_name)  # Gets the data

    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    f1, f2 = get_desired_features_index("iris", f1_name, f2_name)

    # Gets the data to train model
    desired_features = get_desired_features(x_train, f1, f2)
    first_entry = desired_features[:, [0]]
    second_entry = desired_features[:, [1]]

    log_model = LogisticRegression()

    log_model.fit(desired_features, y_train)

    f1_min, f1_max, f2_min, f2_max = get_min_max(desired_features)

    f_x = np.arange(f1_min, f1_max, 0.1)
    f_y = np.arange(f2_min, f2_max, 0.1)

    fx, fy, f_grid = create_meshgrid(f_x, f_y)

    f_predictions, f_probability = (
        log_model.predict(f_grid),
        log_model.predict_proba(f_grid)[:, 1],
    )

    if choice == 0:
        fz = f_predictions.reshape(fx.shape)

    if choice == 1:
        fz = f_probability.reshape(fx.shape)

    create_graph(
        first_entry,
        second_entry,
        point_cmap,
        cmap,
        y_train,
        target_name,
        f1_name,
        f2_name,
        (20, 30),
        fx,
        fy,
        fz,
    )

    # Testing Model:
    desired_features_test = get_desired_features(x_test, f1, f2)
    predictions = log_model.predict(desired_features_test)
    accuracy = accuracy_score(predictions, y_test)

    st.write("Testing Accuracy")
    st.write(accuracy)
