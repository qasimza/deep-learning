# DL9C.py CS5173/6073 cheng 2023
# autoencoder for MNIST
# autograd and Linear are used
# MSELoss and Adam used 
# all samples in minibatch training
# hidden representation displayed
# Usage: python DL9C.py

import numpy as np
import torch
import torchvision
import torch.utils.data as D
import random
import matplotlib.pyplot as plt

mnist = torchvision.datasets.MNIST('/data/')
num_samples = 4096
trainsample = D.RandomSampler(range(len(mnist)), num_samples=num_samples)
sampleiter = iter(trainsample)

x = []
targets = []
for i in range(num_samples):
    j = next(sampleiter)
    x.append(list(mnist[j][0].getdata()))
    targets.append(mnist[j][1])

X = torch.tensor(x, dtype=torch.float32)

d = 784
q = 10
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

batch_size = 256
rounds = 10000
indices = list(range(num_samples))
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
plt.scatter(C[:, 0], C[:, 1], s=10, c=targets)
plt.show()