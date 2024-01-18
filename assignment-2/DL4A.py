# DL4A.py CS5173/6073 cheng 2023
# linear regression on randomly generated 1D data
# PyTorch neural network module Linear
# Usage:  python DL4A.py

import numpy as np
import torch
import matplotlib.pyplot as plt

X = 2 * torch.rand(100, 1) # X is 1D
y = 4 + 3 * X + torch.randn(100, 1) # w = 3, b = 4

linear = torch.nn.Linear(1, 1)
eta = 0.002 # learning rate
for i in range(4):
    for name, param in linear.named_parameters():
        print(name, param.data, param.grad)
    y_pred = linear(X)
    plt.plot(X, y, 'b.')
    plt.plot(X, y_pred.detach().numpy(), 'r.')
    plt.show()
    diff = y_pred - y
    loss = torch.sum(diff * diff)
    print('loss:', loss.item())
    linear.zero_grad()
    loss.backward()
    for param in linear.parameters():
        param.data -= eta * param.grad
