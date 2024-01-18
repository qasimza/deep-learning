# DL26B.py CS5173/6073 2023 cheng
# MNIST edge detection using the loaded DL26Aepoch1.zip
# Usage: python DL26B.py

import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
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
model.load_state_dict(torch.load("DL26Aepoch1.zip"))

for i in range(5):
    index = np.random.randint(test_size)
    sample, t1 = testset[index]
    s = sample.view((1, 1, 28, 28))
    y = model(s)
    plt.subplot(1, 2, 1)
    plt.imshow(s[0][0], cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1, 2, 2)
    plt.imshow(y[0][0].detach().numpy(), cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.show()
