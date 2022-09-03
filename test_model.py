import torch
from transunet import *
from torchsummary import summary

model = TransUNet()
img = torch.randn((1, 1, 224, 224))
print(model(img))
summary(model, (1, 224, 224))