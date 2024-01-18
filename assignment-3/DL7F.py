# DL7F.py CS5173/6073 cheng 2023
# iris linear classifier as softmax regression
# autograd and Linear are used
# LogSoftmax and NLLLoss
# all samples in minibatch training
# final loss and misclassification number printed
# Usage: python DL7F.py

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

class SoftmaxRegression(torch.nn.Module):
    def __init__(self):
        super(SoftmaxRegression, self).__init__()
        self.linear = torch.nn.Linear(d, q)
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, x):
        o = self.linear(x)
        return self.logsoftmax(o)

model = SoftmaxRegression()
loss_fun = torch.nn.NLLLoss()
batch_size = 32
rounds = 10000
eta = 0.001
indices = list(range(m))
for i in range(rounds):
    random.shuffle(indices)
    batch_indices = torch.tensor(indices[:batch_size])
    Xbatch = X[batch_indices]
    ybatch = y[batch_indices]
    p = model(Xbatch)
    loss = loss_fun(p, ybatch)
    model.zero_grad()
    loss.backward()
    for param in model.parameters():
        param.data -= eta * param.grad

print(loss.item())
for name, param in model.named_parameters():
    print(name, param)
p = model(X)
ypred = torch.argmax(p, dim=1)
print('ypred =', ypred)
misclassified = torch.sum((ypred != y))
print('misclassified =', misclassified.item())