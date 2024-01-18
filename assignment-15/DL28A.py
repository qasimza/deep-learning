# DL28A.py CS5173/6073 cheng 2023
# deep dream with vgg19
# DL27C.py without the forward hook and unneccesary functions
# Usage: python DL28A.py

import numpy as np
import random
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt

vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
net = nn.Sequential(*[vgg.features[i] for i in range(27)])
channel = random.randrange(512)
print('channel', channel)

denorm = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.229, 1/0.224, 1/0.225 ]),                 
                              transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ]),                                                     
                              ])

img = Image.open('pink-sky-cover.jpg')
orig_size = np.array(img.size)
new_size = np.array(img.size)*0.5
img = img.resize(new_size.astype(int))
image_tensor = transforms.ToTensor()(img)
image_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image_tensor)
image_tensor = image_tensor.unsqueeze(0)
image_tensor.requires_grad = True
for i in range(20):
  print(i)
  net.zero_grad()
  net_out = net(image_tensor)
  loss = net_out[0][channel].norm()
  loss.backward()
  image_tensor.data = image_tensor.data + image_tensor.grad

img_out = image_tensor.squeeze().detach()
img_out = denorm(img_out)
img_out_np = img_out.numpy().transpose(1,2,0)
img_out_np = np.clip(img_out_np, 0, 1)
img_out_pil = Image.fromarray(np.uint8(img_out_np * 255))

img = img_out_pil.resize(orig_size)
fig = plt.figure(figsize = (10, 7))
plt.imshow(img)
plt.xticks([])
plt.yticks([])
plt.show()
