# DL9E.py CS5173/6073 cheng 2023
# svd for iris
# 2D visualization
# Usage: python DL9E.py

import numpy as np
import torch
import random
import matplotlib.pyplot as plt

d = 4
X = torch.tensor(np.genfromtxt('iris.data', delimiter=",")[:, :d], dtype=torch.float32)
m = len(X)

q = 3
y = torch.zeros(m, dtype=torch.long)
for i in range(50):
    y[50 + i] = 1
    y[100 + i] = 2

u, _, _ = torch.linalg.svd(X)
plt.scatter(u[:, 0], u[:, 1], c=y)
plt.show()