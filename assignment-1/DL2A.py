# DL2A.py CS5173/6073 cheng 2023
# linear regression on randomly generated 1D data
# to have both w and b estimated, data becomes 2D with a constant-one additional dimension
# gradient descent from randomly initialized w and b
# Usage:  python DL2A.py

import numpy as np
X = 2 * np.random.rand(100, 1) # X is 1D
y = 4 + 3 * X + np.random.randn(100, 1) # w = 3, b = 4

import matplotlib.pyplot as plt

X_b = np.c_[np.ones((100, 1)), X] # X_b is 2D with an additional fixed feature 1
wb = np.random.randn(2, 1) # randomly initialized parameters
eta = 0.002 # learning rate
for i in range(4):
    print(wb)
    y_pred = X_b.dot(wb)
    plt.plot(X, y, "b.")
    plt.plot(X, y_pred, "r.")
    plt.show()
    gradient = X_b.T.dot(y_pred - y)
    wb -= eta * gradient
