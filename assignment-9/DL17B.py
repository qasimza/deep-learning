# DL17B.py CS5173/6073 cheng 2023
# autoregression on hospitalization
# AvgPool1d for attention pooling
# using MSELoss and Adam
# with random sample of training data
# Usage: python DL17B.py

import numpy as np
import random
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

x = torch.tensor(np.genfromtxt('hamiltonCountyHospitalization.txt'), dtype=torch.float32) / 500.0

T = len(x)
num_train = T // 2
tau = 4
input_size = 1

features = [x[i: T-tau+i] for i in range(tau)]
X = torch.stack(features, 1)
y = x[tau:].reshape((-1, 1))
Xtrain = X[:num_train]
ytrain = y[:num_train]

class AvgPooling(torch.nn.Module): 
    def __init__(self):
        super(AvgPooling, self).__init__()

    def forward(self, x):
        X2 = torch.reshape(x, (len(x), input_size, tau))
        pooled = F.avg_pool1d(X2, tau)
        return pooled[:, 0, 0]

model = AvgPooling()
y2 = model(X)
plt.plot(y)
plt.plot(y2.detach().numpy())
plt.show()
