# DL29B.py CS5173/6073 2023 cheng
# MNIST deblurring using the loaded DL29Aepoch1.zip
# Usage: python DL29B.py

import torch
import torchvision
from PIL import Image, ImageFilter
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

testset = torchvision.datasets.MNIST('/data/', train=False)
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
model.load_state_dict(torch.load("DL29Aepoch1.zip"))
totensor = transforms.ToTensor()

for i in range(5):
    index = np.random.randint(test_size)
    sample, t1 = testset[index]
    blurred = totensor(sample.filter(ImageFilter.BLUR))
    s = blurred.view((1, 1, 28, 28))
    y = model(s)
    plt.subplot(1, 3, 1)
    plt.imshow(sample, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1, 3, 2)
    plt.imshow(s[0][0], cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1, 3, 3)
    plt.imshow(y[0][0].detach().numpy(), cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.show()
