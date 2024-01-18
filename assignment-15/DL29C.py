# DL29C.py CS5173/6073 cheng 2023
# following "Low-dose CT via convolutional neural network" by Chen et al.
# Biomedical Optics Express, vol.8, 2017, BOE.8.000679
# denoising or super-resolution image-to-image translation learning
# applied to MNIST with conv2d adding noise with a random kernal
# Usage: python DL29C.py

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

fuzzy = torch.randn(1, 1, 3, 3) # an arbitrary but fixed kernel for noise

testset = torchvision.datasets.MNIST('/data/', train=False,
    transform=torchvision.transforms.ToTensor())
test_size = len(testset)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.cnn1 = nn.Conv2d(1, 6, 5, padding=2)
        self.cnn2 = nn.Conv2d(6, 16, 5, padding=2)
        self.cnn3 = nn.Conv2d(16, 1, 5, padding=2)

    def forward(self, x):
        x = F.relu(self.cnn1(x))
        x = F.relu(self.cnn2(x))
        return self.cnn3(x)

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
        x = F.conv2d(data, fuzzy, padding=1)
        p = model(x)
        train_loss = loss_fn(p, data)
        if batch_idx % 10 == 0:
            print('train', epoch, batch_idx, float(train_loss)) 
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

for i in range(10):
    index = np.random.randint(test_size)
    sample, t1 = testset[index]
    s = sample.view((1, 1, 28, 28))
    s2 = F.conv2d(s, fuzzy, padding=1)
    p = model(s2)
    plt.subplot(1, 3, 1)
    plt.imshow(s[0][0], cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1, 3, 2)
    plt.imshow(s2[0][0], cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1, 3, 3)
    plt.imshow(p[0][0].detach().numpy(), cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.show()
