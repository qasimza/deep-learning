# DL22B.py CS5173/6073 cheng 2023
# denseNet on MNIST
# Usage: python DL22B.py

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import time

mnist = torchvision.datasets.MNIST('/data/')
num_samples = len(mnist)
x = []
targets = []
for i in range(num_samples):
    x.append(list(mnist[i][0].getdata()))
    targets.append(mnist[i][1])

X = torch.tensor(x, dtype=torch.float32)
X2 = torch.reshape(X, (len(X), 1, 28, 28))
y = torch.tensor(targets, dtype=torch.long)

def conv_block(num_channels):
    return nn.Sequential(
        nn.LazyBatchNorm2d(), nn.ReLU(),
        nn.LazyConv2d(num_channels, kernel_size=3, padding=1))

class DenseBlock(nn.Module):
    def __init__(self, num_convs, num_channels):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_convs):
            layer.append(conv_block(num_channels))
        self.net = nn.Sequential(*layer)

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            X = torch.cat((X, Y), dim=1)
        return X

def transition_block(num_channels):
    return nn.Sequential(
        nn.LazyBatchNorm2d(), nn.ReLU(),
        nn.LazyConv2d(num_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2))

class DenseNet(nn.Module):
    def b1(self):
        return nn.Sequential(
            nn.LazyConv2d(16, kernel_size=5, stride=2, padding=2),
            nn.LazyBatchNorm2d(), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    def __init__(self, num_channels=16, growth_rate=32, arch=(4, 4)):
        super(DenseNet, self).__init__()
        self.net = nn.Sequential(self.b1())
        for i, num_convs in enumerate(arch):
            self.net.add_module(f'dense_blk{i+1}', DenseBlock(num_convs, 
                                                             growth_rate))
            # The number of output channels in the previous dense block
            num_channels += num_convs * growth_rate
            # A transition layer that halves the number of channels is added
            # between the dense blocks
            if i != len(arch) - 1:
                num_channels //= 2
                self.net.add_module(f'tran_blk{i+1}', transition_block(
                    num_channels))
        self.net.add_module('last', nn.Sequential(
            nn.LazyBatchNorm2d(), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.LazyLinear(10)))

    def forward(self, x):
        return self.net(x)

model = DenseNet()

loss_fun = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

batch_size = 512
rounds = 100
indices = list(range(num_samples))
model.train()
t1 = time.process_time()
for i in range(rounds):
    random.shuffle(indices)
    batch_indices = torch.tensor(indices[:batch_size])
    Xbatch = X2[batch_indices]
    ybatch = y[batch_indices]
    o = model(Xbatch)
    loss = loss_fun(o, ybatch)
    print(i, loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

t2 = time.process_time()
print('Training time', t2 - t1)

testset = torchvision.datasets.MNIST('/data/', train=False)
num_test = len(testset)
testImg = []
testTgt = []
for i in range(num_test):
    testImg.append(list(testset[i][0].getdata()))
    testTgt.append(testset[i][1])
Xtest = torch.tensor(testImg, dtype=torch.float32)
ytest = torch.tensor(testTgt, dtype=torch.long)
X2test = torch.reshape(Xtest, (len(Xtest), 1, 28, 28))
o = model(X2test)
ypred = torch.argmax(o, dim=1)
misclassified = torch.sum((ypred != ytest))
print('Test misclassified =', misclassified.item(), 'out of', num_test)
