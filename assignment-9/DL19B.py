# DL19B.py CS5173/6073 cheng 2023
# MLP on MNIST
# run batches all over 60000 training samples
# test on the 10000 test samples
# Usage: python DL19B.py

import torch
import torchvision
import numpy as np
import random
import matplotlib.pyplot as plt

mnist = torchvision.datasets.MNIST('/data/')
num_samples = len(mnist)

testset = torchvision.datasets.MNIST('/data/', train=False, download=True)
num_test = len(testset)
testImg = []
testTgt = []
for i in range(num_test):
    testImg.append(list(testset[i][0].getdata()))
    testTgt.append(testset[i][1])
Xtest = torch.tensor(testImg, dtype=torch.float32)
ytest = torch.tensor(testTgt, dtype=torch.long)

x = []
targets = []
for i in range(num_samples):
    x.append(list(mnist[i][0].getdata()))
    targets.append(mnist[i][1])

X = torch.tensor(x, dtype=torch.float32)
y = torch.tensor(targets, dtype=torch.long)

input_size = 784
hidden_size = 256
output_size = 10
class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear1 = torch.nn.Linear(input_size, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        return self.linear2(x)

model = MLP()
loss_fun = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

batch_size = 1024
rounds = 1000
indices = list(range(num_samples))
for i in range(rounds):
    random.shuffle(indices)
    batch_indices = torch.tensor(indices[:batch_size])
    Xbatch = X[batch_indices]
    ybatch = y[batch_indices]
    o = model(Xbatch)
    loss = loss_fun(o, ybatch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(loss.item())
o = model(X)
ypred = torch.argmax(o, dim=1)
misclassified = torch.sum((ypred != y))
print('misclassified =', misclassified.item(), 'out of', num_samples)
o = model(Xtest)
ypred = torch.argmax(o, dim=1)
misclassified = torch.sum((ypred != ytest))
print('misclassified =', misclassified.item(), 'out of', num_test)

plt.figure(figsize=(10, 10))
for i in range(256):
    plt.subplot(16, 16, i + 1)
    m = model.linear1.weight.detach().numpy()[i].reshape((28, 28)) 
    plt.imshow(m, cmap='gray')
    plt.xticks([])
    plt.yticks([])
plt.show()
