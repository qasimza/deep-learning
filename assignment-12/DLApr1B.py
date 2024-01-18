# DLMar30D.py CS5173/6073 cheng 2022
# load saved LeNet5 model from DLMar25B.py
# as the file DLMar25Bstate.pt
# gradient descent on input for the given target
# Usage: python DLMar30D.py

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

dataset = torchvision.datasets.MNIST('/data/', train=False,
		transform=torchvision.transforms.ToTensor())
images = torch.tensor([x[0].clone().detach().numpy() for x in dataset])

inputsize = 1
hiddensize1 = 6
hiddensize2 = 16
hiddensize3 = 400
hiddensize4 = 120
hiddensize5 = 84
outputsize = 10
kernelsize = 5
poolsize = 2

class Model(nn.Module):
    def __init__(self, inputsize, hiddensize1, hiddensize2, hiddensize3, 
            hiddensize4, hiddensize5, outputsize, kernelsize, poolsize):
        super(Model, self).__init__()
        self.cnn1 = nn.Conv2d(inputsize, hiddensize1, kernelsize, padding=2)
        self.cnn2 = nn.Conv2d(hiddensize1, hiddensize2, kernelsize)
        self.linear1 = nn.Linear(hiddensize3, hiddensize4)
        self.linear2 = nn.Linear(hiddensize4, hiddensize5)
        self.linear3 = nn.Linear(hiddensize5, outputsize)

    def forward(self, x):
        x = torch.sigmoid(self.cnn1(x))
        x = F.avg_pool2d(x, poolsize)
        x = torch.sigmoid(self.cnn2(x))
        x = F.avg_pool2d(x, poolsize)
        x = x.flatten(start_dim=1)
        x = torch.sigmoid(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        return self.linear3(x)

LeNet5 = Model(inputsize, hiddensize1, hiddensize2, hiddensize3, 
           hiddensize4, hiddensize5, outputsize, kernelsize, poolsize)

LeNet5.load_state_dict(torch.load("DLMar25Bstate.pt"))

p = LeNet5(images)
topsamples = torch.argmax(p, dim=0)

for i in range(10):
    sample, _ = dataset[topsamples[i]]
    s = sample.view((1, 1, 28, 28)).clone().detach().requires_grad_(True)
    for k in range(100):
        s.retain_grad()
        LeNet5.zero_grad()
        p = LeNet5(s)
        if torch.argmax(p) != i:
            break
        p[0][i].backward(retain_graph=True)
        s = s - 0.1 * s.grad # gradient descent
        s[s < 0] = 0
        s[s > 255] = 255.0

    j = torch.argmax(p).item()
    print(i, j, k)
    plt.subplot(2, 5, i + 1)  
    plt.imshow(s.detach().numpy()[0][0], cmap='gray')
    plt.title(j)
    plt.xticks([])
    plt.yticks([])
    plt.title(torch.argmax(p).item())
plt.show()  
