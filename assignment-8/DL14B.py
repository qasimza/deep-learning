# DL14B.py CS5173/6073 cheng 2023
# autoregression on hospitalization
# using MGU from scratch
# using Linear
# using MSELoss and Adam
# with random sample of training data
# Usage: python DL14B.py

import numpy as np
import random
import torch
import matplotlib.pyplot as plt

x = torch.tensor(np.genfromtxt('hamiltonCountyHospitalization.txt'), dtype=torch.float32) / 500.0

T = len(x)
num_train = T // 2
tau = 20
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

class MGUScratch(torch.nn.Module): 
    def __init__(self):
        super(MGUScratch, self).__init__()
        self.forgetgate = torch.nn.Linear(input_size + hidden_size, hidden_size)
        self.candidate = torch.nn.Linear(input_size + hidden_size, hidden_size)
        self.linear = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        H = torch.randn(len(x), hidden_size)
        X2 = torch.reshape(x.T, (tau, len(x), input_size))
        for X in X2:
            input = torch.cat((X, H), 1)
            F = torch.sigmoid(self.forgetgate(input))
            input2 = torch.cat((X, F * H), 1)
            H_tilda = torch.tanh(self.candidate(input2))
            H = (1 - F) * H + F * H_tilda
        return self.linear(H)

model = MGUScratch()
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
