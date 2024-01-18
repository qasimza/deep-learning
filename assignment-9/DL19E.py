# DL19E.py CS5173/6073 cheng 2023
# 1D LeNet5 on MNIST as sequences
# model similar to that of DL16B.py
# Usage: python DL19E.py

import torch
import torchvision
import torch.nn.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt

mnist = torchvision.datasets.MNIST('/data/')
num_samples = len(mnist)
x = []
targets = []
for i in range(num_samples):
    x.append(list(mnist[i][0].getdata()))
    targets.append(mnist[i][1])

X = torch.tensor(x, dtype=torch.float32)
y = torch.tensor(targets, dtype=torch.long)

input_size = 28
seqlen = 28
kernel_size = 5
pool_size = 2
hidden_size_1 = 6
hidden_size_2 = 16
hidden_size_3 = 80
output_size = 10

class CNN(torch.nn.Module): 
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn1 = torch.nn.Conv1d(input_size, hidden_size_1, kernel_size, padding=2)
        self.cnn2 = torch.nn.Conv1d(hidden_size_1, hidden_size_2, kernel_size)
        self.linear = torch.nn.Linear(hidden_size_3, output_size)

    def forward(self, x):
        X2 = torch.reshape(x, (len(x), input_size, seqlen))
        conved1 = self.cnn1(X2)
        pooled1 = F.avg_pool1d(conved1, pool_size)
        conved2 = self.cnn2(pooled1)
        pooled2 = F.avg_pool1d(conved2, pool_size)
        x = pooled2.reshape(-1, hidden_size_3)
        return self.linear(x)

model = CNN()
loss_fun = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

batch_size = 512
rounds = 1000
indices = list(range(num_samples))
for i in range(rounds):
    print(i)
    random.shuffle(indices)
    batch_indices = torch.tensor(indices[:batch_size])
    Xbatch = X[batch_indices]
    ybatch = y[batch_indices]
    o = model(Xbatch)
    loss = loss_fun(o, ybatch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(loss.item())

testset = torchvision.datasets.MNIST('/data/', train=False)
num_test = len(testset)
testImg = []
testTgt = []
for i in range(num_test):
    testImg.append(list(testset[i][0].getdata()))
    testTgt.append(testset[i][1])
Xtest = torch.tensor(testImg, dtype=torch.float32)
ytest = torch.tensor(testTgt, dtype=torch.long)

o = model(Xtest)
ypred = torch.argmax(o, dim=1)
misclassified = torch.sum((ypred != ytest))
print('misclassified =', misclassified.item(), 'out of', num_test)
