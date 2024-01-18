# DL21A.py CS5173/6073 cheng 2023
# VGG on MNIST
# Usage: python DL21A.py

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import time

mnist = torchvision.datasets.MNIST('/data/')
num_samples = len(mnist)
x = []
targets = []
for i in range(num_samples):
    x.append(list(mnist[i][0].getdata()))
    targets.append(mnist[i][1])

X = torch.tensor(x, dtype=torch.float32)
X2 = torch.reshape(X, (len(X), 1, 28, 28))
y = torch.tensor(targets, dtype=torch.long)

def vgg_block(num_convs, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.LazyConv2d(out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)

class VGG(nn.Module):
    def __init__(self, arch):
        super(VGG, self).__init__()
        conv_blks = []
        in_channels = 1
        for (num_convs, out_channels) in arch:
            conv_blks.append(vgg_block(num_convs, out_channels))
        self.net = nn.Sequential(
            *conv_blks, nn.Flatten(),
            nn.LazyLinear(120), nn.ReLU(), nn.Dropout(0.5),
            nn.LazyLinear(84), nn.ReLU(), nn.Dropout(0.5),
            nn.LazyLinear(10))

    def forward(self, x):
        return self.net(x)

model = VGG(arch=((2, 6), (2, 16)))

loss_fun = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.1)

batch_size = 512
rounds = 1000
indices = list(range(num_samples))
model.train()
t1 = time.process_time()
for i in range(rounds):
    random.shuffle(indices)
    batch_indices = torch.tensor(indices[:batch_size])
    Xbatch = X2[batch_indices]
    ybatch = y[batch_indices]
    o = model(Xbatch)
    loss = loss_fun(o, ybatch)
    if i % 100 == 0:
        print(i, loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

t2 = time.process_time()
print('Training time', t2 - t1)

model.eval()
print(loss.item())
o = model(X2)
ypred = torch.argmax(o, dim=1)
misclassified = torch.sum((ypred != y))
print('Training misclassified =', misclassified.item(), 'out of', num_samples)

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
misclassified = torch.sum((ypred != ytest))
print('Test misclassified =', misclassified.item(), 'out of', num_test)