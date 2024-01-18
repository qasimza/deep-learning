# DL19D.py CS5173/6073 cheng 2023
# GRU on MNIST
# Usage: python DL19D.py

import torch
import torchvision
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
hidden_size = 64
output_size = 10
class RNN(torch.nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = torch.nn.GRU(input_size, hidden_size, 2)
        self.linear = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.reshape(-1, seqlen, input_size)
        x = np.transpose(x, (1, 0, 2))
        _, rx = self.rnn(x)
        return self.linear(rx[1])

model = RNN()
loss_fun = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

batch_size = 256
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
