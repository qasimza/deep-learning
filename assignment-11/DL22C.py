# DL22C.py CS5173/6073 cheng 2023
# ConvMixer on MNIST
# Usage: python DL22C.py

import torch
import torchvision
from torch.nn import *
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

def ConvMixer(h,d,k,p,n):
    S,C,A=Sequential,Conv2d,lambda x:S(x,GELU(),BatchNorm2d(h))
    R=type('',(S,),{'forward':lambda s,x:s[0](x)+x})
    return S(A(C(1,h,p,p)),
        *[S(R(A(C(h,h,k,groups=h,padding=k//2))),A(C(h,h,1))) for i in range(d)],
        AdaptiveAvgPool2d(1),Flatten(),Linear(h,n))

model = ConvMixer(96, 6, 9, 4, 10)

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
