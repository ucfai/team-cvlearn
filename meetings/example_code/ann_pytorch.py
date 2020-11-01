# Pytorch Neural Network
#
# Just some sample code to train mnist on a multilayer perceptron.
# See https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html
# for a tutorial.


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import nn, optim
from torch.utils.data import DataLoader
from torchsummary import summary
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from utils.classification import fit, validate

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_ROOT = "/home/calvin/.cache/torch/datasets"
BATCH_SIZE = 128
EPOCHS = 5


###########################################################
# Load the data, and make dataloaders

dataloader_params = dict(batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)

train = MNIST(TORCH_ROOT, train=True, transform=ToTensor(), download=True)
train = DataLoader(train, shuffle=True, **dataloader_params)
test = MNIST(TORCH_ROOT, train=False, transform=ToTensor(), download=True)
test = DataLoader(test, shuffle=False, **dataloader_params)


###########################################################
# One way to make a model. Subclassing nn.Module gives
# more organization of of the model architecture
# and more control over the forward pass


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = MLP().to(DEVICE)

###########################################################
# Can also just use nn.Sequential if the model is simple

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 64),
    nn.ReLU(True),
    nn.Linear(64, 32),
    nn.ReLU(True),
    nn.Linear(32, 10),
)
model = model.to(DEVICE)

###########################################################
# Can print out a keras like summary like below
summary(model, (1, 28, 28), batch_size=BATCH_SIZE, device=DEVICE)


###########################################################
# Make the loss function and the optimizer.
# Here we are using cross entropy and Adam optimizer
#
# NOTE: We do not have to use softmax in the model.
# CrossEntropyLoss() does both softmax and cross entropy
# as the same time
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


###########################################################
# In pytorch, we code our own training loop
#
# Code taken from pytorch tutorials

print("Starting training")
# Loop over epochs
for epoch in range(EPOCHS):

    running_loss = 0.0
    for i, data in enumerate(train, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

    print(f"Epoch {epoch + 1} loss: {running_loss / len(train)}")

print("Finished Training")


###########################################################
# Validate/evaluate the model

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy of the network on the 10000 test images: {correct / total}")

###########################################################
# More sophisticated training loop
#
# We can make a more sophisticated training loop, with a
# progress bar and history that looks like keras.
#
# The code can be found here: https://github.com/calvinyong/pytorch_snippets)

model = MLP().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Using pretty training loop")
history = fit(model, train, test, EPOCHS, criterion, optimizer, DEVICE)

print(history.to_markdown())
print("Validation:", validate(model, test, criterion, DEVICE))
