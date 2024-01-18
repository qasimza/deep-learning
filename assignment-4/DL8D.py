# DL8D.py CS5173/6073 cheng 2023
# softmax repression on a random sample of MNIST
# Usage: python DL8D.py

import torch
import torchvision
import torch.utils.data as D
import numpy as np
import random
import matplotlib.pyplot as plt

mnist = torchvision.datasets.MNIST('/data/')
num_samples = 4096
trainsample = D.RandomSampler(range(len(mnist)), num_samples=num_samples)
sampleiter = iter(trainsample)

x = []
targets = []
for i in range(num_samples):
    j = next(sampleiter)
    x.append(list(mnist[j][0].getdata()))
    targets.append(mnist[j][1])

X = torch.tensor(x, dtype=torch.float32)
y = torch.tensor(targets, dtype=torch.long)

d = 28 * 28
q = 10
class LinearRegression(torch.nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(d, q)

    def forward(self, x):
        return self.linear(x)

model = LinearRegression()
loss_fun = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

batch_size = 256
rounds = 1000
indices = list(range(num_samples))
for i in range(rounds):
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
o = model(X)
ypred = torch.argmax(o, dim=1)
misclassified = torch.sum((ypred != y))
print('misclassified =', misclassified.item())
