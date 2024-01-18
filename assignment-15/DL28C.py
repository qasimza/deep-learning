# DL28C.py CS5173/6073 cheng 2023
# style transfer, following d2l 14.12
# Usage: python DL28C.py

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.models as models
import torch.nn as nn

image_shape = (300, 450) # PIL Image (h, w)
rgb_mean = torch.tensor([0.485, 0.456, 0.406])
rgb_std = torch.tensor([0.229, 0.224, 0.225])
transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(image_shape),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=rgb_mean, std=rgb_std)])

def postprocess(img):
    img = img[0]
    img = torch.clamp(img.permute(1, 2, 0) * rgb_std + rgb_mean, 0, 1)
    return torchvision.transforms.ToPILImage()(img.permute(2, 0, 1))

content_img = Image.open('rainier.jpg')
plt.imshow(content_img);
plt.xticks([])
plt.yticks([])
plt.show()

style_img = Image.open('autumn_oak.jpg')
plt.imshow(style_img);
plt.xticks([])
plt.yticks([])
plt.show()

vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
style_layers, content_layers = [0, 5, 10, 19, 28], [25]
net = nn.Sequential(*[vgg.features[i] for i in range(29)])

styles_Y = []
X = transforms(style_img).unsqueeze(0)
for i in range(len(net)):
    X = net[i](X)
    if i in style_layers:
        styles_Y.append(X)

contents_Y = []
content_X = transforms(content_img).unsqueeze(0)
X = content_X
for i in range(len(net)):
    X = net[i](X)
    if i in content_layers:
        contents_Y.append(X)

def extract_features(X, content_layers, style_layers):
    contents = []
    styles = []
    for i in range(len(net)):
        X = net[i](X)
        if i in style_layers:
            styles.append(X)
        if i in content_layers:
            contents.append(X)
    return contents, styles

def gram(X):
    c = X.shape[1]
    d = X.numel()
    X = X.reshape((c, d // c))
    return torch.matmul(X, X.T) / d

class SynthesizedImage(nn.Module):
    def __init__(self, img_shape):
        super(SynthesizedImage, self).__init__()
        self.weight = nn.Parameter(torch.rand(*img_shape))
    def forward(self):
        return self.weight

def get_inits():
    gen_img = SynthesizedImage(content_X.shape)
    gen_img.weight.data.copy_(content_X.data)
    trainer = torch.optim.Adam(gen_img.parameters())
    styles_Y_gram = [gram(Y) for Y in styles_Y]
    return gen_img(), styles_Y_gram, trainer

X, styles_Y_gram, trainer = get_inits()

def content_loss(Y_hat, Y):
    return torch.square(Y_hat - Y.detach()).mean()
def style_loss(Y_hat, gram_Y):
    return torch.square(gram(Y_hat) - gram_Y.detach()).mean()
def tv_loss(Y_hat):
    return 0.5 * (torch.abs(Y_hat[:, :, 1:, :] - Y_hat[:, :, :-1, :]).mean() +
                  torch.abs(Y_hat[:, :, :, 1:] - Y_hat[:, :, :, :-1]).mean())

for epoch in range(500):
    trainer.zero_grad()
    contents_Y_hat, styles_Y_hat = extract_features(
            X, content_layers, style_layers)
    contents_l = [content_loss(Y_hat, Y) for Y_hat, Y in zip(
        contents_Y_hat, contents_Y)]
    styles_l = [style_loss(Y_hat, Y) * 0.001 for Y_hat, Y in zip(
        styles_Y_hat, styles_Y_gram)]
    tv_l = tv_loss(X) * 10
    l = sum(10 * styles_l + contents_l + [tv_l])
    print(epoch, l.item())
    l.backward()
    trainer.step()

plt.imshow(postprocess(X));
plt.xticks([])
plt.yticks([])
plt.show()