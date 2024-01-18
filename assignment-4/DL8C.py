# DL8C.py CS5173/6073 cheng 2023
# MNIST as a dataset
# Usage: python DL8C.py

import torch
import torchvision
import torch.utils.data as D
import numpy as np
import matplotlib.pyplot as plt

mnist = torchvision.datasets.MNIST('/data/', download=False)
print(list(mnist[0][0].getdata()))
print(len(mnist))
print(mnist[0])
print(next(iter(mnist)))

plt.imshow(mnist[0][0], cmap='gray')
plt.title("{}: Truth: {}".format(0, mnist[0][1]))
plt.xticks([])
plt.yticks([])
plt.show()

batchsize = 256
trainsample = D.RandomSampler(range(len(mnist)), num_samples=batchsize)
sampleiter = iter(trainsample)

for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.tight_layout()    
    j = next(sampleiter)
    plt.imshow(mnist[j][0], cmap='gray')
    plt.title("{}: Truth: {}".format(j, mnist[j][1]))
    plt.xticks([])
    plt.yticks([])
plt.show()
