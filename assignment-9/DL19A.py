# DL19A.py CS5173/6073 cheng 2023
# softmax regression on MNIST
# DL8D.py plus weight display
# run batches all over 60000 training samples
# Usage: python DL19A.py

import torch
import torchvision
import numpy as np
import random
import matplotlib.pyplot as plt

mnist = torchvision.datasets.MNIST('/data/')
num_samples = len(mnist)
x = []
targets = []
for i in range(num_samples):
    x.append(list(mnist[i][0].getdata()))
    targets.append(mnist[i][1])

X = torch.tensor(x, dtype=torch.float32)
y = torch.tensor(targets, dtype=torch.long)

input_size = 784
output_size = 10
class LinearRegression(torch.nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)

model = LinearRegression()
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

for i in range(10):
    plt.subplot(3, 4, i + 1)
    plt.tight_layout() 
    m = model.linear.weight.detach().numpy()[i].reshape((28, 28)) 
    plt.imshow(m, cmap='gray')
    plt.title(i)
    plt.xticks([])
    plt.yticks([])
plt.show()