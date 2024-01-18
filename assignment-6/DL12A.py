# DL12A.py CS5173/6073 cheng 2023
# autoregression on hospitalization
# using RNN
# using MSELoss and Adam
# with random sample of training data
# Usage:  python DL12A.py

import numpy as np
import random
import torch
import matplotlib.pyplot as plt

x = torch.tensor(np.genfromtxt('hamiltonCountyHospitalization.txt'), dtype=torch.float32) / 500.0

T = len(x)
num_train = T // 2
tau = 4
input_size = 1
hidden_size = 10
output_size = 1
batch_size = 32

features = [x[i: T-tau+i] for i in range(tau)]
X = torch.stack(features, 1)
y = x[tau:].reshape((-1, 1))
Xtrain = X[:num_train]
ytrain = y[:num_train]

class RNN2(torch.nn.Module):
    def __init__(self):
        super(RNN2, self).__init__()
        self.rnn = torch.nn.RNN(input_size, hidden_size, 2)
        self.linear = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        X2 = torch.reshape(x.T, (tau, len(x), input_size))
        h0 = torch.randn(2, len(x), hidden_size)
        _, hn = self.rnn(X2, h0)
        return self.linear(hn[1])

model = RNN2()
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
