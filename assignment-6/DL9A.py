# DL9A.py CS5173/6073 cheng 2023
# autoencoder for iris
# autograd and Linear are used
# MSELoss and Adam used 
# all samples in minibatch training
# hidden representation displayed
# Usage: python DL9A.py

import numpy as np
import torch
import random
import matplotlib.pyplot as plt

d = 4
X = torch.tensor(np.genfromtxt('iris.data', delimiter=",")[:, :d], dtype=torch.float32)
m = len(X)

q = 3
y = torch.zeros(m, dtype=torch.long)
for i in range(50):
    y[50 + i] = 1
    y[100 + i] = 2

c = 2
class Autoencoder(torch.nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = torch.nn.Linear(d, c)
        self.decoder = torch.nn.Linear(c, d)

    def forward(self, x, encode_only=False):
        code = self.encoder(x)
        if encode_only:
            return code.detach().numpy()
        else:
            return self.decoder(code)

model = Autoencoder()
loss_fun = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

batch_size = 32
rounds = 10000
indices = list(range(m))
for i in range(rounds):
    random.shuffle(indices)
    batch_indices = torch.tensor(indices[:batch_size])
    Xbatch = X[batch_indices]
    o = model(Xbatch)
    loss = loss_fun(o, Xbatch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

o = model(X)
loss = loss_fun(o, X)
print(loss.item())

C = model(X, encode_only=True)
plt.scatter(C[:, 0], C[:, 1], c=y)
plt.show()