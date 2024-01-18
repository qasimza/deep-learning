# DL16B.py CS5173/6073 cheng 2023
# autoregression on hospitalization
# with LeNet5 in 1D
# using MSELoss and Adam
# with random sample of training data
# Usage: python DL16B.py

import numpy as np
import random
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

x = torch.tensor(np.genfromtxt('hamiltonCountyHospitalization.txt'), dtype=torch.float32) / 500.0

T = len(x)
num_train = T // 2
tau = 28
input_size = 1
kernel_size = 5
pool_size = 2
hidden_size_1 = 6
hidden_size_2 = 16
hidden_size_3 = 80
hidden_size_4 = 20
output_size = 1
batch_size = 32

features = [x[i: T-tau+i] for i in range(tau)]
X = torch.stack(features, 1)
y = x[tau:].reshape((-1, 1))
Xtrain = X[:num_train]
ytrain = y[:num_train]

class CNN(torch.nn.Module): 
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn1 = torch.nn.Conv1d(input_size, hidden_size_1, kernel_size, padding=2)
        self.cnn2 = torch.nn.Conv1d(hidden_size_1, hidden_size_2, kernel_size)
        self.linear1 = torch.nn.Linear(hidden_size_3, hidden_size_4)
        self.linear2 = torch.nn.Linear(hidden_size_4, output_size)

    def forward(self, x):
        X2 = torch.reshape(x, (len(x), input_size, tau))
        conved1 = self.cnn1(X2)
        pooled1 = F.avg_pool1d(conved1, pool_size)
        conved2 = self.cnn2(pooled1)
        pooled2 = F.avg_pool1d(conved2, pool_size)
        x = pooled2.reshape(-1, hidden_size_3)
        z = self.linear1(x)
        output = self.linear2(z)
        return output[:, 0]

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
