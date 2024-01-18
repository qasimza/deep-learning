# DL3A.py CS5173/6073 cheng 2023
# autoregression with linear regression
# following D2l 9.1
# using the closed form solution as in DL1A.py
# Usage: python DL3A.py

import torch
import matplotlib.pyplot as plt

T = 1000
time = torch.arange(1, T + 1, dtype=torch.float32)
x = torch.sin(0.01 * time) + torch.randn(T) * 0.2
plt.plot(time, x)
plt.show()

num_train = 600
tau = 4

features = [x[i: T-tau+i] for i in range(tau)]
X = torch.stack(features, 1)
X_b = torch.cat((torch.ones(len(X), 1), X), 1)
y = x[tau:].reshape((-1, 1))
Xtrain = X_b[:num_train]
ytrain = y[:num_train]
wb_best = torch.linalg.inv(Xtrain.T.matmul(Xtrain)).matmul(Xtrain.T).matmul(ytrain) 
print(wb_best)

y_pred = X_b.matmul(wb_best)
plt.plot(time[tau:], y)
plt.plot(time[tau:], y_pred)
plt.show()

diff = y_pred - y
train_loss = torch.mean(diff[:num_train] * diff[:num_train])
test_loss = torch.mean(diff[num_train:] * diff[num_train:])
print('training MSE', train_loss.item())
print('test MSE', test_loss.item())