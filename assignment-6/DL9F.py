# DL9F.py CS5173/6073 cheng 2023
# svd for a sample of MNIST
# 2D representation displayed
# Usage: python DL9F.py

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

u, _, _ = torch.linalg.svd(X)
plt.scatter(u[:, 0], u[:, 1], s=10, c=targets)
plt.show()