# DL17A.py CS5173/6073 cheng 2023
# following d2l 11.2.2, average pooling
# Usage:  python DL17A.py

import torch
import matplotlib.pyplot as plt

n = 50
f = lambda x: 2 * torch.sin(x) + x**0.8
x_train, _ = torch.sort(torch.rand(n) * 5)
y_train = f(x_train) + torch.randn(n)
x_val = torch.arange(0, 5, 5.0/n)
y_val = f(x_val)

y_hat = y_train.mean().repeat(n)
plt.plot(x_val, y_val)
plt.plot(x_train, y_train, "o")
plt.plot(y_hat)
plt.axis([0, 5, 0, 5])
plt.show()



