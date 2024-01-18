# DL7H.py CS5173/6073 cheng 2023
# iris linear classifier as softmax regression
# autograd and Linear are used
# CrossEntropyLoss and Adam used 
# all samples in minibatch training
# final loss and misclassification number printed
# Usage: python DL7H.py

import numpy as np
import torch
import random

d = 4
X = torch.tensor(np.genfromtxt('iris.data', delimiter=",")[:, :d], dtype=torch.float32)
m = len(X)

q = 3
y = torch.zeros(m, dtype=torch.long)
for i in range(50):
    y[50 + i] = 1
    y[100 + i] = 2

class LinearRegression(torch.nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(d, q)

    def forward(self, x):
        return self.linear(x)

model = LinearRegression()
loss_fun = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

batch_size = 32
rounds = 10000
indices = list(range(m))
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
for name, param in model.named_parameters():
    print(name, param)
o = model(X)
ypred = torch.argmax(o, dim=1)
print('ypred =', ypred)
misclassified = torch.sum((ypred != y))
print('misclassified =', misclassified.item())