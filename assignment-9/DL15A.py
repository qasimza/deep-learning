# DL15A.py CS5173/6073 cheng 2023
# autoregression on hospitalization
# with conv1d
# using MSELoss and Adam
# with random sample of training data
# Usage: python DL15A.py

import numpy as np
import random
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

x = torch.tensor(np.genfromtxt('hamiltonCountyHospitalization.txt'), dtype=torch.float32) / 500.0

T = len(x)
num_train = T // 2
tau = 10
input_size = 1
kernel_size = 3
output_size = 1
batch_size = 32

features = [x[i: T-tau+i] for i in range(tau)]
X = torch.stack(features, 1)
y = x[tau:].reshape((-1, 1))
Xtrain = X[:num_train]
ytrain = y[:num_train]

Lout = tau - (kernel_size - 1)
class CNN(torch.nn.Module): 
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn = torch.nn.Conv1d(input_size, output_size, kernel_size)

    def forward(self, x):
        X2 = torch.reshape(x, (len(x), input_size, tau))
        output = self.cnn(X2)
        pooled = F.avg_pool1d(output, Lout)
        return pooled[:, 0, 0]

model = CNN()
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
    loss = loss_fun(y_pred, y[batch_indices][:, 0])
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
