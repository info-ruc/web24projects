import torch as torch
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt
from PIL import Image
img = Image.open('F:/PostGraduate/TaskOne/images/CERA_101_cy_140_cx_1514.png').convert('RGB')
from torchvision import transforms
myTransform = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(90),
        transforms.ColorJitter(brightness=0.08,contrast=0.08,saturation=0.08,hue=0.03),
        transforms.Resize([224,224]),
        transforms.ToTensor()
    ]),
    'test': transforms.Compose([
        transforms.Resize([224,224]),
        transforms.ToTensor()
    ]),
}
print(img.size)
im = transforms.Compose([
    # transforms.RandomResizedCrop(size=(img.size[1],img.size[0]),scale=(0.6,1)),
    # transforms.ColorJitter(brightness=0.01,contrast=0.01,saturation=0.01,hue=0.3),
    transforms.ColorJitter(contrast=0.5)
])(img)
im.save("F:/PostGraduate/TaskOne/ttt.jpg")
print(im.size)
plt.imshow(im)
plt.show()