# DL23A.py CS5173/6073 cheng 2023
# ConvMixer parameters
# Usage: python DL23A.py

from torch.nn import *

def ConvMixer(h,d,k,p,n):
    S,C,A=Sequential,Conv2d,lambda x:S(x,GELU(),BatchNorm2d(h))
    R=type('',(S,),{'forward':lambda s,x:s[0](x)+x})
    return S(A(C(1,h,p,p)),
        *[S(R(A(C(h,h,k,groups=h,padding=k//2))),A(C(h,h,1))) for i in range(d)],
        AdaptiveAvgPool2d(1),Flatten(),Linear(h,n))

model = ConvMixer(96, 6, 7, 7, 10)
patch = model.get_parameter('0.0.weight').detach().numpy()

import matplotlib.pyplot as plt
for i in range(96):
    plt.subplot(8, 12, i + 1)
    plt.imshow(patch[i][0], cmap='gray')
    plt.xticks([])
    plt.yticks([])
plt.show()
