# %% [markdown]
# <h1 style="text-align:center;">Matplotlib and Seaborn Short Tutorial</h1>
#
# *Written by Calvin Yong*
#
# This notebook is a work in progress. Contributions are also welcome.
#
# ## The two matplotlib coding styles
#
# *Some example figures and text comes from here: https://matplotlib.org/tutorials/introductory/usage.html#sphx-glr-tutorials-introductory-usage-py*
#
# I've heard sometimes that the official matplotlib tutorials are not the best. Here I take some parts from the official matplotlib tutorial, along with giving my comments and suggestions regarding plotting in general
#
# There are two coding styles for making plots in matplotlib.
#
# - The pyplot style (I sometimes hear matlab style)
# - The object-oriented (OO) style
#
# *From https://matplotlib.org/3.1.0/api/index.html*
#
# Here is what the pyplot style is:
#
# > matplotlib.pyplot is a collection of command style functions that make Matplotlib work like MATLAB. Each pyplot function makes some change to a figure: e.g., creates a figure, creates a plotting area in a figure, plots some lines in a plotting area, decorates the plot with labels, etc. pyplot is mainly intended for interactive plots and simple cases of programmatic plot generation.
#
# And the OO style:
#
# > At its core, Matplotlib is object-oriented. We recommend directly working with the objects, if you need more control and customization of your plots. In many cases you will create a Figure and one or more Axes using pyplot.subplots and from then on only work on these objects. However, it's also possible to create Figures explicitly (e.g. when including them in GUI applications).
#
#
# Below is an example of pyplot style code

# %%
import numpy as np
import matplotlib.pyplot as plt

# For reproducibility
np.random.rand(42)

# %%
# Example plot from
# https://matplotlib.org/tutorials/introductory/usage.html#sphx-glr-tutorials-introductory-usage-py

x = np.linspace(0, 2, 100)

plt.plot(x, x, label="linear")  # Plot some data on the (implicit) axes.
plt.plot(x, x ** 2, label="quadratic")  # etc.
plt.plot(x, x ** 3, label="cubic")
plt.xlabel("x label")
plt.ylabel("y label")
plt.title("Simple Plot")
plt.legend()

# %% [markdown]
# Below shows the same plot as above, but using the OO style

# %%
x = np.linspace(0, 2, 100)

fig, ax = plt.subplots()
ax.plot(x, x, label="linear")
ax.plot(x, x ** 2, label="quadratic")
ax.plot(x, x ** 3, label="cubic")
ax.set_xlabel("x label")
ax.set_ylabel("y label")
ax.set_title("Simple Plot")
ax.legend()

# %% [markdown]
# For the simple examples above, it seems that there's not much difference between the two. But the differences start shining when we get to more sophisticated figures, where we need finer control between multiple subplots in the same figure.

# %%
# Example figure from
# https://matplotlib.org/tutorials/introductory/pyplot.html#sphx-glr-tutorials-introductory-pyplot-py

# Fixing random state for reproducibility
np.random.seed(42)

# make up some data in the open interval (0, 1)
y = np.random.normal(loc=0.5, scale=0.4, size=1000)
y = y[(y > 0) & (y < 1)]
y.sort()
x = np.arange(len(y))

# plot with various axes scales
plt.figure()

# linear
plt.subplot(221)
plt.plot(x, y)
plt.yscale("linear")
plt.title("linear")
plt.grid(True)

# log
plt.subplot(222)
plt.plot(x, y)
plt.yscale("log")
plt.title("log")
plt.grid(True)

# symmetric log
plt.subplot(223)
plt.plot(x, y - y.mean())
plt.yscale("symlog", linthresh=0.01)
plt.title("symlog")
plt.grid(True)

# logit
plt.subplot(224)
plt.plot(x, y)
plt.yscale("logit")
plt.title("logit")
plt.grid(True)
# Adjust the subplot layout, because the logit one may take more space
# than usual, due to y-tick labels like "1 - 10^{-3}"
plt.subplots_adjust(
    top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35
)

# %% [markdown]
# Now consider the below OO style example

# %%
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5), tight_layout=True)

# First plot
# Modified from http://scipy-lectures.org/intro/matplotlib/auto_examples/plot_scatter.html
x0 = np.random.normal(0, 1, 1024)
y0 = np.random.normal(0, 1, 1024)
color = np.arctan2(y0, x0)

ax[0].scatter(x0, y0, s=75, c=color, alpha=0.5)
ax[0].set_xlim(-1, 1)
ax[0].set_xticks([])
ax[0].set_ylim(-1, 1)
ax[0].set_yticks([])


# Second plot
# Modified from http://scipy-lectures.org/intro/matplotlib/index.html
x1 = np.linspace(-np.pi, np.pi, 256, endpoint=True)
c = np.cos(x1)
s = np.sin(x1)

ax[1].plot(x1, c, color="blue", linewidth=2.5, linestyle="-", label="cosine")
ax[1].plot(x1, s, color="red", linewidth=2.5, linestyle="-", label="sine")
ax[1].spines["right"].set_color("none")
ax[1].spines["top"].set_color("none")
ax[1].xaxis.set_ticks_position("bottom")
ax[1].spines["bottom"].set_position(("data", 0))
ax[1].yaxis.set_ticks_position("left")
ax[1].spines["left"].set_position(("data", 0))

ax[1].set_xlim(x1.min() * 1.1, x1.max() * 1.1)
ax[1].set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
ax[1].set_xticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$+\pi/2$", r"$+\pi$"])

ax[1].set_ylim(s.min() * 1.1, c.max() * 1.1)
ax[1].set_yticks([-1, 1])
ax[1].set_yticklabels([r"$-1$", r"$+1$"])

ax[1].legend(loc="upper left")

# %% [markdown]
# The pyplot style is hard to understand for me. It is not obvious to me what `plt.subplot(224)` means to me. While the OO style, through the use of keyword arguments (as opposed to positional arguments), makes lines like `plt.subplots(nrows=1, ncols=2)` readable. There is a clear separation between the subplots (self-explaining code, no need for comments), and all related code is grouped together. Figure related options are configured though the `fig` object, or even in the `plt.subplots()` call, while each axes are configured through their own element in the array of axes.
#
# I personally prefer the OO style. Splitting up the figure and axes makes the code more readable, and makes plotting easier and flexible. When learning matplotlib, I found using the pyplot style difficult. I couldn't understand subplots very easily, and felt that things did not connect with each other.
#
# Here is what matplotlib has to say about the two approaches
#
# *from https://matplotlib.org/tutorials/introductory/usage.html#sphx-glr-tutorials-introductory-usage-py*
#
# > Matplotlib's documentation and examples use both the OO and the pyplot approaches (which are equally powerful), and you should feel free to use either (however, it is preferable pick one of them and stick to it, instead of mixing them). In general, we suggest to restrict pyplot to interactive plotting (e.g., in a Jupyter notebook), and to prefer the OO-style for non-interactive plotting (in functions and scripts that are intended to be reused as part of a larger project).
#
# I think if pyplot has to be used, then it used be used in something like python. I still find the OO style useful in jupyter notebooks, where the code might eventually be moved into a script.

# %% [markdown]
# ## Some simple plots in matplotlib
#
# Below are some common plots that you might use. There are many other plots you can make, but this just gives an idea on common options along with the use of OO style plotting.
#
# If there's something don't know how to do, you can look it up with your favorite search engine, or read the matplotlib/seaborn documentation.
#
# ### Line plots
#
# Line plots are made with the `.plot()` function. To plot multiple lines on a single plot, just call `.plot()` on the same axes.

# %%
x = np.linspace(0, 2, 25)

fig, ax = plt.subplots()
ax.plot(x, x, label="line1")
ax.plot(x, x ** 2, label="line2")
ax.legend()

# %% [markdown]
# Another example with different line colors, markers, line styles, and line width. If you want only markers and no line, consider using scatterplots (`.scatter`) instead.

# %%
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].plot(x, x, c="red")
ax[1].plot(x, x ** 2, c="green", marker=".", linestyle="-.", markersize=2)
ax[2].plot(x, x ** 3, c="blue", linestyle="--", linewidth=3)

# %% [markdown]
# ### Scatter plots
#
# We can make a scatterplot given the x and y values. Below, we make an array of points. Each row contains a point. The first column are the x values, and the second column is are the y values.

# %%
points = np.random.rand(150, 2)

plt.scatter(points[:, 0], points[:, 1])

# %% [markdown]
# A trick that I really like is using `*array.T` as a shortcut to specifying `x` and `y`. Below is an example of that usage, along with using color, colormap, and size of points.

# %%
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(*points.T, c=points[:, 0], cmap="jet", s=100)


# %% [markdown]
# ### Showing heatmaps
#
# *Example based on http://scipy-lectures.org/intro/matplotlib/auto_examples/plot_imshow.html*
#
# You can use `plt.matshow()` to show heatmaps of matrices, which can be useful for visualizing them. You can also use `plt.imshow()`.

# %%
def make_example_img(n=10):
    def f(x, y):
        return (1 - x / 2 + x ** 5 + y ** 3) * np.exp(-(x ** 2) - y ** 2)

    x = np.linspace(-3, 3, 3 * n)
    y = np.linspace(-3, 3, 3 * n)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)

    return Z


# %%
img = make_example_img()
plt.matshow(img)

# %% [markdown]
# Below is an example with colorbar, cmap, interpolation, and different origin.

# %%
img = make_example_img(n=20)

fig, ax = plt.subplots(figsize=(6, 6))
cm = ax.matshow(img, cmap="bone", interpolation="nearest", origin="lower")
fig.colorbar(cm, shrink=0.80)

# %% [markdown]
# ### Showing images

# %%
from sklearn.datasets import load_sample_image

img = load_sample_image("china.jpg")
plt.imshow(img)

# %%
img.shape

# %% [markdown]
# Below is an example of removing the axis and making the image bigger.

# %%
fig, ax = plt.subplots(figsize=(10, 7))
ax.imshow(img)
ax.axis("off")

# %% [markdown]
# ## Seaborn
#
# From [seaborn](https://seaborn.pydata.org/)'s website:
#
# > Seaborn is a Python data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics.
#
# Seaborn is like a high-level API to the matplotlib interface. Since it's based on matplotlib, you can do matplotlib actions to seaborn generated figures. Below I will show some simple examples on how seaborn can cut down the amount of code you need to write, along with how you can finetune plots with matplotlib.

# %%
import seaborn as sns

# %% [markdown]
# ### Countplots
#
# Sometimes you need to visualize the distibution of the classes in a dataset. `sns.countplot()` can be useful for this task.

# %%
labels = np.random.randint(low=0, high=10, size=1000)

labels[:10]

# %%
sns.countplot(x=labels)

# %% [markdown]
# And below is an example of how you can modify the figure with matplotlib. Here we change the size of the figure, add a title, and change the font size of the tick labels.

# %%
fig, ax = plt.subplots(figsize=(12, 5))
sns.countplot(x=labels, ax=ax)
ax.set_title("Class Distribution", fontsize=16)
ax.set_ylabel("Count", fontsize=14)
ax.tick_params(axis="both", labelsize=12)

# %% [markdown]
# ### Histograms

# %%
values = np.random.randn(100)

sns.histplot(values)

# %% [markdown]
# ## Some do nots
#
# - You might encounter code bases that use `matplotlib.pylab`. As noted [here](https://matplotlib.org/tutorials/introductory/usage.html#sphx-glr-tutorials-introductory-usage-py), it imports both numpy and matplotlib at the same time, and is deprecated and is strongly discouraged.
