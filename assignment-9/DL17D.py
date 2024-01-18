# DL17D.py CS5173/6073 cheng 2023
# autoregression on hospitalization
# Nadaraya-Watson Gaussian kernel regression
# using MSELoss and Adam
# with random sample of training data
# Usage: python DL17D.py

import numpy as np
import random
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

x = torch.tensor(np.genfromtxt('hamiltonCountyHospitalization.txt'), dtype=torch.float32) / 500.0

T = len(x)
tau = 10
w = 0.5
pos_diff = torch.zeros(tau)
for i in range(tau):
    pos_diff[i] = (tau - i) * w
W = F.softmax(- pos_diff**2 / 2, dim=0)
print(W)

features = [x[i: T-tau+i] for i in range(tau)]
X = torch.stack(features, 1)
y = x[tau:].reshape((-1, 1))

class Nadaraya(torch.nn.Module): 
    def __init__(self):
        super(Nadaraya, self).__init__()

    def forward(self, x):
        return x.matmul(W)

model = Nadaraya()
y2 = model(X)
plt.plot(y)
plt.plot(y2.detach().numpy())
plt.show()
