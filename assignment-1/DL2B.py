# DL2B.py CS5173/6073 cheng 2023
# linear regression on randomly generated 1D data
# to have both w and b estimated, data becomes 2D with a constant-one additional dimension
# gradient descent from randomly initialized w and b
# PyTorch is used to compute the gradient
# Usage:  python DL2B.py

import numpy as np
import torch
X = 2 * torch.rand(100, 1) # X is 1D
y = 4 + 3 * X + torch.randn(100, 1) # w = 3, b = 4

import matplotlib.pyplot as plt
X_b = torch.cat([torch.ones((100, 1)), X], 1)
wb = torch.randn(2, 1) 
wb.requires_grad_(True)
wb.grad = torch.zeros(2, 1)
eta = 0.002 # learning rate
for i in range(4):
    print(wb)
    y_pred = X_b.matmul(wb)
    plt.plot(X, y, "b.")
    plt.plot(X, y_pred.detach().numpy(), "r.")
    plt.show()
    diff = y_pred - y
    loss = torch.sum(diff * diff)
    print('loss:', loss.item())
    wb.grad.zero_()
    loss.backward()
    wb.data = wb.data - eta * wb.grad
