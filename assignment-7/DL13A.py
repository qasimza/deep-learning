# DL13A.py CS5173/6073 cheng 2023
# autoregression on hospitalization
# using LSTM from scratch
# following d2l 10.2.2
# using MSELoss and Adam
# with random sample of training data
# Usage:  python DL13A.py

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
sigma = 0.01

features = [x[i: T-tau+i] for i in range(tau)]
X = torch.stack(features, 1)
y = x[tau:].reshape((-1, 1))
Xtrain = X[:num_train]
ytrain = y[:num_train]

class LSTMScratch(torch.nn.Module): 
    def __init__(self):
        super(LSTMScratch, self).__init__()
        init_weight = lambda *shape: torch.nn.Parameter(torch.randn(*shape) * sigma)
        triple = lambda: (init_weight(input_size, hidden_size),
                          init_weight(hidden_size, hidden_size),
                          torch.nn.Parameter(torch.zeros(hidden_size)))
        self.W_xi, self.W_hi, self.b_i = triple() # Input gate
        self.W_xf, self.W_hf, self.b_f = triple() # Forget gate
        self.W_xo, self.W_ho, self.b_o = triple() # Output gate
        self.W_xc, self.W_hc, self.b_c = triple() # Candidate memory cell
        self.linear = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        H = torch.randn(len(x), hidden_size)
        C = torch.randn(len(x), hidden_size)
        X2 = torch.reshape(x.T, (tau, len(x), input_size))
        for X in X2:
            I = torch.sigmoid(torch.matmul(X, self.W_xi) + 
                torch.matmul(H, self.W_hi) + self.b_i)
            F = torch.sigmoid(torch.matmul(X, self.W_xf) +
                              torch.matmul(H, self.W_hf) + self.b_f)
            O = torch.sigmoid(torch.matmul(X, self.W_xo) +
                              torch.matmul(H, self.W_ho) + self.b_o)
            C_tilda = torch.tanh(torch.matmul(X, self.W_xc) +
                                 torch.matmul(H, self.W_hc) + self.b_c)
            C = F * C + I * C_tilda
            H = O * torch.tanh(C)
        return self.linear(H)

model = LSTMScratch()
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
