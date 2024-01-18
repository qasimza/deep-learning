# DL1A.py CS5173/6073 cheng 2023
# linear regression on randomly generated 1D data
# to have both w and b estimated, data becomes 2D with a constant-one additional dimension
# closed form optimum solution is computed using slide 1/11/5 (5.12)
# Usage:  python DL1A.py

import numpy as np
X = 2 * np.random.rand(100, 1) # X is 1D
y = 4 + 3 * X + np.random.randn(100, 1) # w = 3, b = 4

import matplotlib.pyplot as plt

plt.plot(X, y, "b.")
plt.axis([0, 2, 0, 15])
plt.show()

X_b = np.c_[np.ones((100, 1)), X] # X_b is 2D with an additional fixed feature 1
wb_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y) 
print(wb_best) # optimum w and b (3.1.9)

y_predict = X_b.dot(wb_best)
plt.plot(X, y, "b.")
plt.axis([0, 2, 0, 15])
plt.plot(X, y_predict, "r.")
plt.show()
