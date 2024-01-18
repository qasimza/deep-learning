# DL20F.py CS5173/6073 cheng 2023
# 2D LeNet5 on MNIST
# following d2l 7.6.1 with Sigmoid replaced with ReLU
# with dropout as in AlexNet between Linears
# add .eval()
# Usage: python DL20F.py

import torch
import torchvision
import torch.nn as nn
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

model = nn.Sequential(
    nn.LazyConv2d(6, kernel_size=5, padding=2), nn.ReLU(),
    nn.AvgPool2d(kernel_size=2),
    nn.LazyConv2d(16, kernel_size=5), nn.ReLU(),
    nn.AvgPool2d(kernel_size=2),
    nn.Flatten(),
    nn.LazyLinear(120), nn.ReLU(), nn.Dropout(p=0.5),
    nn.LazyLinear(84), nn.ReLU(), nn.Dropout(p=0.5),
    nn.LazyLinear(10))

loss_fun = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.1)

batch_size = 512
rounds = 1000
indices = list(range(num_samples))
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
