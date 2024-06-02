import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
img = Image.open('./images/CERA_017_cy_614_cx_1475.png')
# img = img.resize((70,70))
# img = transforms.Pad(padding=215,fill=0,padding_mode='constant')(img)
# img = transforms.Resize([224,224])(img)
img = Image.open('./images/CERA_017_cy_614_cx_1475.png')
img = transforms.Pad(padding=672,fill=0,padding_mode='constant')(img)
img = transforms.Resize([224,224])(img)
img.save('./CrossoverDetection/tmpInput/padimg.png')
plt.imshow(img)
plt.show()