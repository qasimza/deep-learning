# DL30A.py CS5173/6073 cheng 2023
# graph convolutional network
# (7.39) of Hamilton
# initial random embedding is improved with Adam optimizer
# Usage: python DL30A.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

G = nx.gnm_random_graph(30, 50)
N = G.number_of_nodes()
A = torch.tensor(nx.to_numpy_array(G) + np.identity(N), dtype=torch.float)
one = torch.ones(len(A))
d = torch.matmul(one, A)
d2 = 1.0 / torch.sqrt(d)
D2 = torch.diag(d2)
Atilda = torch.matmul(torch.matmul(D2, A), D2)

class Model(nn.Module):
    def __init__(self, inputsize, hiddensize=16, outputsize=2):
        super(Model, self).__init__()
        self.h = nn.Parameter(torch.rand(inputsize, hiddensize))
        self.w = nn.Parameter(torch.randn(hiddensize, outputsize))
        self.z = torch.zeros(inputsize, outputsize)

    def forward(self):
        self.z = torch.matmul(torch.matmul(Atilda, self.h), self.w)
        return torch.matmul(self.z, self.z.T)

model = Model(N)
model()
z2 = model.z.detach().numpy()
pos = {}
for i in G.nodes:
    pos.update({i: z2[i]})
nx.draw(G, pos)
plt.show()

optimizer = optim.Adam(model.parameters())
loss_fun = nn.CrossEntropyLoss()

for i in range(5):
    for j in range(100):
        dec = model()
        loss = loss_fun(dec, A)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    z2 = model.z.detach().numpy()
    pos = {}
    for i in G.nodes:
        pos.update({i: z2[i]})
    nx.draw(G, pos)
    plt.show()