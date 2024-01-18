# DL7E.py CS5173/6073 cheng 2023
# iris linear classifier as softmax regression
# autograd and Linear are used
# all samples in minibatch training
# final loss and misclassification number printed
# Usage: python DL7E.py

import numpy as np
import torch
import torch.nn.functional as F
import random

d = 4
X = torch.tensor(np.genfromtxt('iris.data', delimiter=",")[:, :d], dtype=torch.float32)
m = len(X)

q = 3
y = torch.zeros((m, q), dtype=torch.int)
for i in range(50):
    y[i][0] = 1
    y[50 + i][1] = 1
    y[100 + i][2] = 1

class SoftmaxRegression(torch.nn.Module):
    def __init__(self):
        super(SoftmaxRegression, self).__init__()
        self.linear = torch.nn.Linear(d, q)

    def forward(self, x):
        return F.softmax(self.linear(x), dim=1)

model = SoftmaxRegression()

batch_size = 32
rounds = 10000
eta = 0.001
indices = list(range(m))
for i in range(rounds):
    random.shuffle(indices)
    batch_indices = torch.tensor(indices[:batch_size])
    Xbatch = X[batch_indices]
    ybatch = y[batch_indices]
    yhat = model(Xbatch)
    nll = -torch.log(yhat)
    loss = torch.sum(nll * ybatch)
    model.zero_grad()
    loss.backward()
    for param in model.parameters():
        param.data -= eta * param.grad

print(loss.item())
for name, param in model.named_parameters():
    print(name, param)
yhat = model(X)
ypred = torch.argmax(yhat, dim=1)
print('ypred =', ypred)
ylabel = torch.argmax(y, dim=1)
misclassified = torch.sum((ypred != ylabel))
print('misclassified =', misclassified.item())
