# DL7C.py CS5173/6073 cheng 2023
# iris linear classifier as softmax regression
# b absorbed into w
# single batch all in training
# final loss and misclassification number printed
# Usage: python DL7C.py

import numpy as np
import torch
import torch.nn.functional as F
import random

d = 4
X = torch.tensor(np.genfromtxt('iris.data', delimiter=",")[:, :d], dtype=torch.float32)
m = len(X)
X_b = torch.cat((torch.ones(m, 1), X), 1)

q = 3
y = torch.zeros((m, q), dtype=torch.int)
for i in range(50):
    y[i][0] = 1
    y[50 + i][1] = 1
    y[100 + i][2] = 1

wb = torch.randn(d+1, q)
rounds = 10000
eta = 0.001
for i in range(rounds):
    o = X_b.matmul(wb)
    yhat = F.softmax(o, dim=1)
    nll = -torch.log(yhat)
    loss = torch.sum(nll * y)
    grad_o = yhat - y
    grad_wb = X_b.T.matmul(grad_o)
    wb -= eta * grad_wb

print(loss.item())
print('wb =', wb)
o = X_b.matmul(wb)
ypred = torch.argmax(o, dim=1)
print('ypred =', ypred)
ylabel = torch.argmax(y, dim=1)
misclassified = torch.sum((ypred != ylabel))
print('misclassified =', misclassified.item())
