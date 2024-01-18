# DLApr4A.py CS5173/6073 cheng 2022
# load saved LeNet5 model from DLMar25B.py
# as the file DLMar25Bstate.pt
# random sample under gradient descent until 
# class changes
# compute attribution of each pixel to each of the two classes
# color map for displaying the attribution
# attribution based on gradient and integrated gradient
# You are asked to show the differences
# Usage: python DLApr4A.py

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

dataset = torchvision.datasets.MNIST('/data/', train=False,
    transform=torchvision.transforms.ToTensor())
test_size = len(dataset)
index = np.random.randint(test_size)
sample, t1 = dataset[index]

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

s = sample.view((1, 1, 28, 28)).clone().detach().requires_grad_(True)

plt.imshow(s.detach().numpy()[0][0], cmap='gray')
plt.title(t1)
plt.xticks([])
plt.yticks([])
plt.show()

for iter in range(100):
    s.retain_grad()
    LeNet5.zero_grad()
    p = LeNet5(s)
    if torch.argmax(p) != t1:
        break
    p[0][t1].backward(retain_graph=True)
    s = s - 0.1 * s.grad # gradient descent
    s[s < 0] = 0
    s[s > 255] = 255.0

t2 = torch.argmax(p).item()
print(t1, t2, iter)  

plt.subplot(1, 3, 2) 
plt.imshow(s.detach().numpy()[0][0], cmap='gray')
plt.xticks([])
plt.yticks([])
plt.title(t2)

s.retain_grad()
LeNet5.zero_grad()
p = LeNet5(s)
p[0][t1].backward(retain_graph=True)
s1 = s.grad

plt.subplot(1, 3, 1) 
plt.imshow(s1.detach().numpy()[0][0], cmap='Reds')
plt.xticks([])
plt.yticks([])
plt.title(t1)

s.retain_grad()
LeNet5.zero_grad()
p = LeNet5(s)
p[0][t2].backward(retain_graph=True)
s2 = s.grad

plt.subplot(1, 3, 3) 
plt.imshow(s2.detach().numpy()[0][0], cmap='Reds')
plt.xticks([])
plt.yticks([])
plt.title(t2)
plt.show()  

s3 = s * s1
s4 = s * s2

plt.subplot(1, 2, 1) 
plt.imshow(s3.detach().numpy()[0][0], cmap='Reds')
plt.xticks([])
plt.yticks([])
plt.title(t1)
plt.colorbar()

plt.subplot(1, 2, 2) 
plt.imshow(s4.detach().numpy()[0][0], cmap='Reds')
plt.xticks([])
plt.yticks([])
plt.title(t2)
plt.colorbar()
plt.show() 
