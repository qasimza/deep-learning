# DL6B.py CS5173/6073 cheng 2023
# autoregression with linear regression
# DL6A on hospitalization 
# Usage: python DL6B.py

import numpy as np
import torch
import matplotlib.pyplot as plt

x = torch.tensor(np.genfromtxt('hamiltonCountyHospitalization.txt'), dtype=torch.float32)
T = len(x)
up = torch.zeros_like(x)
for i in range(1, T):
    if x[i] > x[i-1]:
        up[i] = 1.0
plt.plot(x)
plt.plot(up * 400)
plt.show()

num_train = T // 2
tau = 4

features = [x[i: T-tau+i] for i in range(tau)]
X = torch.stack(features, 1)
X_b = torch.cat((torch.ones(len(X), 1), X), 1)
y = up[tau:].reshape((-1, 1))
Xtrain = X_b[:num_train]
ytrain = y[:num_train]
wb_best = torch.linalg.inv(Xtrain.T.matmul(Xtrain)).matmul(Xtrain.T).matmul(ytrain) 
print(wb_best)

y_pred = X_b.matmul(wb_best)
plt.plot(y)
plt.plot(y_pred)
plt.show()

diff = y_pred - y
train_loss = torch.mean(diff[:num_train] * diff[:num_train])
test_loss = torch.mean(diff[num_train:] * diff[num_train:])
print('training MSE', train_loss.item())
print('test MSE', test_loss.item())
