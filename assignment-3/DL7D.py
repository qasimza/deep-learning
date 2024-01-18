# DL7D.py CS5173/6073 cheng 2023
# iris linear classifier as softmax regression
# autograd is used
# b absorbed into w
# all samples in minibatch training
# final loss and misclassification number printed
# Usage: python DL7D.py

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
wb.requires_grad_(True)
wb.grad = torch.zeros_like(wb)

batch_size = 32
rounds = 10000
eta = 0.001
indices = list(range(m))
for i in range(rounds):
    random.shuffle(indices)
    batch_indices = torch.tensor(indices[:batch_size])
    Xbatch = X_b[batch_indices]
    ybatch = y[batch_indices]
    o = Xbatch.matmul(wb)
    yhat = F.softmax(o, dim=1)
    nll = -torch.log(yhat)
    loss = torch.sum(nll * ybatch)
    wb.grad.zero_()
    loss.backward()
    wb.data = wb.data - eta * wb.grad

print(loss.item())
print('wb =', wb)
o = X_b.matmul(wb)
ypred = torch.argmax(o, dim=1)
print('ypred =', ypred)
ylabel = torch.argmax(y, dim=1)
misclassified = torch.sum((ypred != ylabel))
print('misclassified =', misclassified.item())
