# DL19C.py CS5173/6073 cheng 2023
# MNIST as sequences of rows
# Usage: python DL19C.py

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

mnist = torchvision.datasets.MNIST('/data/', download=True)

plt.imshow(mnist[0][0], cmap='gray')
plt.title("{}: Truth: {}".format(0, mnist[0][1]))
plt.xticks([])
plt.yticks([])
plt.show()

data = np.array(mnist[0][0].getdata()).reshape((28, 1, 28))

for i in range(27):
    plt.subplot(9, 3, i + 1)
    plt.tight_layout()    
    plt.imshow(data[i], cmap='gray')
    plt.xticks([])
    plt.yticks([])
plt.show()

