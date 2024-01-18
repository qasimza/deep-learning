# DL17C.py CS5173/6073 cheng 2023
# following d2l 11.2.3, Nadaraya-Watson Gaussian kernel regression
# Usage:  python DL17C.py

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

n = 50
f = lambda x: 2 * torch.sin(x) + x**0.8
x_train, _ = torch.sort(torch.rand(n) * 5)
y_train = f(x_train) + torch.randn(n)
x_val = torch.arange(0, 5, 5.0/n)
y_val = f(x_val)

def diff(queries, keys):
    return queries.reshape((-1, 1)) - keys.reshape((1, -1))

D = diff(x_val, x_train)
W = F.softmax(- D**2 / 2, dim=1)

import seaborn
seaborn.heatmap(W)
plt.show()

y_hat = torch.matmul(W, y_train)
plt.plot(x_val, y_val)
plt.plot(x_train, y_train, "o")
plt.plot(y_hat)
plt.axis([0, 5, 0, 5])
plt.show()

