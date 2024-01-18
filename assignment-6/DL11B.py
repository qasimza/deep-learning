# DL11B.py CS5173/6073 cheng 2023
# autoregression on hospitalization
# implementing RNNCell/Elman from scratch
# following d2l 9.5.1
# using MSELoss and Adam
# with random sample of training data
# Usage: python DL11B.py

import numpy as np
import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

x = torch.tensor(np.genfromtxt('hamiltonCountyHospitalization.txt'), dtype=torch.float32) / 500.0

T = len(x)
num_train = T // 2
tau = 4
input_size = 1
hidden_size = 10
output_size = 1
batch_size = 32
sigma = 0.01

features = [x[i: T-tau+i] for i in range(tau)]
X = torch.stack(features, 1)
y = x[tau:].reshape((-1, 1))
Xtrain = X[:num_train]
ytrain = y[:num_train]

class RNN(torch.nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.W_xh = nn.Parameter(torch.randn(input_size, hidden_size) * sigma)
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size) * sigma)
        self.b_h = nn.Parameter(torch.zeros(hidden_size))
        self.W_hq = nn.Parameter(torch.randn(hidden_size, output_size) * sigma)
        self.b_q = nn.Parameter(torch.zeros(output_size))
        
    def forward(self, x):
        X2 = torch.reshape(x.T, (tau, len(x), input_size))
        hx = torch.randn(len(x), hidden_size)
        for X3 in X2:
            hx = torch.tanh(torch.matmul(X3, self.W_xh) + 
                 torch.matmul(hx, self.W_hh) + self.b_h)
        return torch.matmul(hx, self.W_hq) + self.b_q

model = RNN()
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
