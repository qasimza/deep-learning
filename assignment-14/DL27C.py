# DL27C.py CS5173/6073 cheng 2023
# deep dream with vgg19
# following github.com/juanigp/Pytorch-Deep_Dream
# Usage: python DL27C.py

import numpy as np
import random
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.io.image import read_image
import torchvision.models as models

from PIL import Image, ImageFilter, ImageChops
import matplotlib.pyplot as plt

import requests
from io import BytesIO

vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
vgg.eval()

#Class to register a hook on the target layer (used to get the output channels of the layer)
class Hook():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output
    def close(self):
        self.hook.remove()
  
#Function to make gradients calculations from the output channels of the target layer  
def get_gradients(net_in, net, layer, channel):     
  net_in = net_in.unsqueeze(0)
  net_in.requires_grad = True
  net.zero_grad()
  hook = Hook(layer)
  net_out = net(net_in)
  loss = hook.output[0][channel].norm()
  loss.backward()
  return net_in.grad.data.squeeze()

#denormalization image transform
denorm = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.229, 1/0.224, 1/0.225 ]),                 
                              transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ]),                                                     
                              ])

#Function to run the dream.
def dream(image, net, layer, channel, iterations, lr):
  image_tensor = transforms.ToTensor()(image)
  image_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image_tensor)
  for i in range(iterations):
    print(i)
    gradients = get_gradients(image_tensor, net, layer, channel)
    image_tensor.data = image_tensor.data + lr * gradients.data

  img_out = image_tensor.detach()
  img_out = denorm(img_out)
  img_out_np = img_out.numpy().transpose(1,2,0)
  img_out_np = np.clip(img_out_np, 0, 1)
  img_out_pil = Image.fromarray(np.uint8(img_out_np * 255))
  return img_out_pil

#Input image
url = 'https://s3.amazonaws.com/pbblogassets/uploads/2018/10/22074923/pink-sky-cover.jpg'
response = requests.get(url)
img = Image.open(BytesIO(response.content))
orig_size = np.array(img.size)
new_size = np.array(img.size)*0.5
img = img.resize(new_size.astype(int))
layer = list( vgg.features.modules() )[27]
channel = random.randrange(512)
print('channel', channel)
img = dream(img, vgg, layer, channel, 20, 1)

img = img.resize(orig_size)
fig = plt.figure(figsize = (10, 7))
plt.imshow(img)
plt.xticks([])
plt.yticks([])
plt.show()
