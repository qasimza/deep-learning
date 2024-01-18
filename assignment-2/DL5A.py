# DL5A.py CS5173/6073 cheng 2023
# autoregression on hospitalization
# making a new module with Linear a submodule
# using MSELoss
# Goodfellow Algorithm 8.1 SGD
# with random sample of training data
# Usage:  python DL5A.py

import numpy as np
import random
import torch
import matplotlib.pyplot as plt

x = torch.tensor(np.genfromtxt('hamiltonCountyHospitalization.txt'), dtype=torch.float32)

T = len(x)
num_train = T // 2
tau = 4
batch_size = 32

features = [x[i: T-tau+i] for i in range(tau)]
X = torch.stack(features, 1)
y = x[tau:].reshape((-1, 1))
Xtrain = X[:num_train]
ytrain = y[:num_train]

class LinearRegression(torch.nn.Module):
    def __init__(self, in_features=tau):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(in_features, 1)

    def forward(self, x):
        return self.linear(x)

model = LinearRegression()
y2 = model(X)
plt.plot(y)
plt.plot(y2.detach().numpy())
plt.show()

loss_fun = torch.nn.MSELoss()
eta = 0.001 
rounds = 200
losses = np.zeros(rounds)
indices = list(range(num_train))
for i in range(rounds):
    random.shuffle(indices)
    batch_indices = torch.tensor(indices[:batch_size])
    y_pred = model(X[batch_indices])
    loss = torch.sqrt(loss_fun(y_pred, y[batch_indices])) / batch_size
    losses[i] = loss.item()
    model.zero_grad()
    loss.backward()
    for param in model.parameters():
        param.data -= eta * param.grad

y2 = model(X)
plt.plot(y)
plt.plot(y2.detach().numpy())
plt.show()

plt.plot(losses)
plt.show()