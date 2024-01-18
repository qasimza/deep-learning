# DL26A.py CS5173/6073 2023 cheng
# MNIST edge detection training with encoder-decoder
# ImageFilter.FIND_EDGES used to produce image-to-image translation
# in the training set and trained model is saved in DL26Aepoch1.zip
# The application of ImageFilter.FIND_EDGES to all training images may take a few minutes. 
# Usage: python DL26A.py

import torch
import torchvision
from PIL import Image, ImageFilter
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

dataset = torchvision.datasets.MNIST('/data/', train=True)
train_size = len(dataset)
totensor = transforms.ToTensor()
d2 = []
for i in range(train_size):
    data, target = dataset[i]
    target = data.filter(ImageFilter.FIND_EDGES)
    d2.append((totensor(data), totensor(target)))

batch_size_train = 256
n_epochs = 1

train_loader = torch.utils.data.DataLoader(
    d2,
    batch_size=batch_size_train, shuffle=True)

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

for epoch in range(n_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        y = model(data)
        train_loss = loss_fn(y, target)
        if batch_idx % 10 == 0:
            print('train', epoch, batch_idx, float(train_loss)) 
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

torch.save(model.state_dict(), "DL26Aepoch1.zip")
