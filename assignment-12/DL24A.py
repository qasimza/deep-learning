# DL24A.py CS5173/6073 cheng 2023
# Modernized LeNet5 on MNIST
# model restored from DL20A1000.zip
# find test samples with 1) the highest score,
# 2) the lowest score, and 3) the highest score but misclassified 
# for each of the ten categories
# Usage: python DL24A.py

import torch
import torchvision
import torch.nn as nn
import numpy as np

model = nn.Sequential(
    nn.LazyConv2d(6, kernel_size=5, padding=2), nn.ReLU(),
    nn.AvgPool2d(kernel_size=2),
    nn.LazyConv2d(16, kernel_size=5), nn.ReLU(),
    nn.AvgPool2d(kernel_size=2),
    nn.Flatten(),
    nn.LazyLinear(120), nn.ReLU(),
    nn.LazyLinear(84), nn.ReLU(),
    nn.LazyLinear(10))

saved = torch.load('./DL20A1000.zip')
model.load_state_dict(saved)

testset = torchvision.datasets.MNIST('/data/', train=False)
num_test = len(testset)
testImg = []
testTgt = []
for i in range(num_test):
    testImg.append(list(testset[i][0].getdata()))
    testTgt.append(testset[i][1])
Xtest = torch.tensor(testImg, dtype=torch.float32)
ytest = torch.tensor(testTgt, dtype=torch.long)
X2test = torch.reshape(Xtest, (len(Xtest), 1, 28, 28))
o = model(X2test)
ypred = torch.argmax(o, dim=1)
import matplotlib.pyplot as plt

# samples with the highest o scores in their predicted categories
topsamples = torch.argmax(o, dim=0)
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X2test[topsamples[i]][0], cmap='gray')
    plt.title(topsamples[i].item())
    plt.xticks([])
    plt.yticks([])
plt.show()

# samples with lowest o scores in their categories
originalIndex = torch.arange(num_test)
for i in range(10):
    mask = (ytest==i)
    subindex = torch.argmin(o[mask][:,i])
    worst = originalIndex[mask][subindex]
    print(worst, ytest[worst], ypred[worst])
    plt.subplot(2, 5, i + 1)
    plt.imshow(X2test[worst][0], cmap='gray')
    plt.title(worst.item())
    plt.xticks([])
    plt.yticks([])
plt.show()

# misclassified samples with highest o scores in their predicted categories
for i in range(10):
    mask = (ytest!=i)
    subindex = torch.argmax(o[mask][:,i])
    worst = originalIndex[mask][subindex]
    print(worst, ytest[worst], ypred[worst])
    plt.subplot(2, 5, i + 1)
    plt.imshow(X2test[worst][0], cmap='gray')
    plt.title(worst.item())
    plt.xticks([])
    plt.yticks([])
plt.show()
