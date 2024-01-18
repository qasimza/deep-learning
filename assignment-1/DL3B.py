# DL3B.py CS5173/6073 cheng 2023
# autoregression with linear regression
# following D2l 9.1
# gradient descent without backward()
# Usage: python DL3B.py

import torch
import matplotlib.pyplot as plt

T = 1000
time = torch.arange(1, T + 1, dtype=torch.float32)
x = torch.sin(0.01 * time) + torch.randn(T) * 0.2
num_train = 600
tau = 4

features = [x[i: T-tau+i] for i in range(tau)]
X = torch.stack(features, 1)
X_b = torch.cat((torch.ones(len(X), 1), X), 1)
y = x[tau:].reshape((-1, 1))
Xtrain = X_b[:num_train]
ytrain = y[:num_train]
wb = torch.randn(tau+1, 1)
eta = 0.001
for i in range(4):
    y_pred = X_b.matmul(wb)
    plt.plot(y)
    plt.plot(y_pred)
    plt.show()
    gradient = Xtrain.T.matmul(y_pred[:num_train] - ytrain)
    wb -= eta * gradient

    diff = y_pred[num_train:] - y[num_train:]
    test_loss = torch.mean(diff * diff)
    print('test MSE', test_loss.item())