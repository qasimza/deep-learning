# DL25A.py CS5173/6073 cheng 2023
# load saved Lenet5 model from DL20A1000.zip
# gradient for the input
# Usage: python DL25A.py

import torch
import torchvision
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

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
topsamples = torch.argmax(o, dim=0)

sample = X2test[topsamples[1]]
plt.imshow(sample[0], cmap='gray')
plt.xticks([])
plt.yticks([])
plt.show()

s = sample.view((1, 1, 28, 28)).clone().detach().requires_grad_(True)
s.retain_grad()
model.zero_grad()
p = model(s)
p[0][7].backward(retain_graph=True)
plt.imshow(s.grad[0][0], cmap='gray')
plt.xticks([])
plt.yticks([])
plt.show()
