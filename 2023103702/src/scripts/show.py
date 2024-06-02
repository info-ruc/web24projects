from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
img = Image.open('./images/CERA_056_cy_752_cx_1684.png')
# img1 = transforms.RandomAffine(degrees=30,translate=(0.35,0.35))(img)

# img1 = transforms.ColorJitter(contrast=(0.5, 1.5), saturation=(0.5, 1.5),
#                            hue=(-0.5, 0.5),
#                            brightness=(0.875, 1.125))(img)

img1 = transforms.CenterCrop(200)(img)
plt.imshow(img1)
plt.show()