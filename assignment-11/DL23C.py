# DL23C.py CS5173/6073 cheng 2023
# ConvMixer on MNIST
# read the saved DL23B100.zip
# Usage: python DL23C.py

import torch

saved = torch.load('./DL23B100.zip')
patch = saved.get('6.0.0.0.weight')

import matplotlib.pyplot as plt
for i in range(96):
    plt.subplot(8, 12, i + 1)
    plt.imshow(patch[i][0], cmap='gray')
    plt.xticks([])
    plt.yticks([])
plt.show()