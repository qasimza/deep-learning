# DL29D.py CS5173/6073 2023 cheng
# MNIST denoising using autoencoder
# Usage: python DL29D.py

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

testset = torchvision.datasets.MNIST('/data/', train=False,
    transform=torchvision.transforms.ToTensor())
test_size = len(testset)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.cnn1 = nn.Conv2d(1, 64, 5, stride=2,  padding=2)
        self.cnn2 = nn.Conv2d(64, 128, 5, stride=2,  padding=2)
        self.conv1 = nn.ConvTranspose2d(128, 64, 5, stride=1, padding=2, bias=False)
        self.conv2 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, bias=False)
        self.conv3 = nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1, bias=False)

    def forward(self, x):
        x = F.relu(self.cnn1(x))
        x = self.cnn2(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return self.conv3(x)

model = Model()
optimizer = optim.Adam(model.parameters())
loss_fn = nn.MSELoss()

batch_size_train = 256
n_epochs = 1

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('/data/', train=True, download=True,
        transform=torchvision.transforms.ToTensor()),
    batch_size=batch_size_train, shuffle=True)

for epoch in range(n_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        p = model(data)
        train_loss = loss_fn(p, data)
        if batch_idx % 10 == 0:
            print('train', epoch, batch_idx, float(train_loss)) 
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

for i in range(5):
    index = np.random.randint(test_size)
    sample, t1 = testset[index]
    s = sample.view((1, 1, 28, 28))
    s2 = s.clone()
    for j in range(10):
      row = np.random.randint(28)
      col = np.random.randint(28)
      s2[0][0][row][col] = 1
    y = model(s2)
    plt.subplot(1, 3, 1)
    plt.imshow(s[0][0], cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1, 3, 2)
    plt.imshow(s2[0][0].detach().numpy(), cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1, 3, 3)
    plt.imshow(y[0][0].detach().numpy(), cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.show()
