import torch
from transunet import *
from torchsummary import summary
import matplotlib.pyplot as plt

model = TransUNet()
img = torch.randn((1, 1, 224, 224))
out = (model(img))
print(out)
summary(model, (1, 224, 224))

plt.imshow(out[0][0].detach().numpy())
plt.show()