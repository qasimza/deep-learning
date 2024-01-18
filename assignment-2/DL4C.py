# DL4C.py CS5173/6073 cheng 2023
# linear regression on randomly generated 1D data
# making a new module with Linear a submodule
# using MSELoss
# Usage:  python DL4C.py

import numpy as np
import torch
import matplotlib.pyplot as plt

X = 2 * torch.rand(100, 1) # X is 1D
y = 4 + 3 * X + torch.randn(100, 1) # w = 3, b = 4

class LinearRegression(torch.nn.Module):
    def __init__(self, in_features=1):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(in_features, 1)

    def forward(self, x):
        return self.linear(x)

model = LinearRegression()
loss_fun = torch.nn.MSELoss()
eta = 0.2 # learning rate for reduction="mean"
for i in range(4):
    for name, param in model.named_parameters():
        print(name, param.data, param.grad)
    y_pred = model(X)
    plt.plot(X, y, "b.")
    plt.plot(X, y_pred.detach().numpy(), "r.")
    plt.show()
    loss = loss_fun(y_pred, y)
    print('loss:', loss.item())
    model.zero_grad()
    loss.backward()
    for param in model.parameters():
        param.data -= eta * param.grad
