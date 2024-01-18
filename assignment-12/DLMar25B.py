# DLMar25B.py CS5173/6073 cheng 2022
# Conv2d for MNIST images
# LeNet5 (d2l) 
# Usage: python DLMar25B.py

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

num_inputs, seqlen, num_outputs = 28, 28, 10
batch_size_train = 256
n_epochs = 10

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('/data/', train=True, download=True,
        transform=torchvision.transforms.ToTensor()),
    batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('/data/', train=False, download=True,
        transform=torchvision.transforms.ToTensor()))
test_size = len(test_loader)

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

optimizer = optim.Adam(LeNet5.parameters())
loss_fn = nn.CrossEntropyLoss()
for epoch in range(n_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        p = LeNet5(data)
        train_loss = loss_fn(p, target)
        if batch_idx % 100 == 0:
            print('train', epoch, batch_idx, float(train_loss)) 
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
    m = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        p = LeNet5(data)
        if torch.argmax(p) == int(target[0]):
            m = m + 1
    print("test", epoch, m, "of", test_size, "correctly classified")

torch.save(LeNet5.state_dict(), "./DLMar25Bstate.pt")