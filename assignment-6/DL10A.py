# DL10A.py CS5173/6073 cheng 2023
# autoregression on hospitalization
# using Linear
# using MSELoss and Adam
# with random sample of training data
# Usage:  python DL10A.py

import numpy as np
import random
import torch
import matplotlib.pyplot as plt

x = torch.tensor(np.genfromtxt('hamiltonCountyHospitalization.txt'), dtype=torch.float32) /500.0

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
optimizer = torch.optim.Adam(model.parameters())
rounds = 1000
losses = np.zeros(rounds)
indices = list(range(num_train))
for i in range(rounds):
    random.shuffle(indices)
    batch_indices = torch.tensor(indices[:batch_size])
    y_pred = model(X[batch_indices])
    loss = loss_fun(y_pred, y[batch_indices])
    losses[i] = loss.item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

y2 = model(X)
plt.plot(y)
plt.plot(y2.detach().numpy())
plt.show()

print(losses[rounds - 1])
plt.plot(losses)
plt.show()
