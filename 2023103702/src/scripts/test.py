# from PIL import Image
# import torch
# import matplotlib.pyplot as plt
# from torchvision import transforms
# import random
# path = 'F:/PostGraduate/TaskOne/SegSmallerROI/data/images/CERA_128_cy_70_cx_1026_crosspt_markedimg.png'
# img = Image.open(path).convert('RGB')
# img = img.resize((224,224))
# axes1 = plt.subplot(1, 2, 1)
# plt.imshow(img)
# axes2 = plt.subplot(1, 2, 2)
# # tmptransform = transforms.RandomCrop(100)
# torch.random.manual_seed(42)
# tmptransform = transforms.Compose([
#     transforms.RandomCrop(180),
#     transforms.Resize((224,224))
# ])
# out = tmptransform(img)
# plt.imshow(out)
# plt.savefig('1.jpg')
# plt.show()
#
# from PIL import Image
# import numpy as np
# import cv2
# grayIM = Image.open("F:/PostGraduate/TaskOne/SegSmallerROI/data/labels/CERA_097_cy_2004_cx_1758_crosspt_markedimg.png").convert('RGB')
# from torchvision import transforms
# # grayIM=cv2.cvtColor(grayIM,cv2.COLOR_BGR2GRAY)
# print(grayIM.size)
# grayIM = grayIM.convert('RGBA')
# grayIM=transforms.ToTensor()(grayIM)
# print(grayIM.shape)
list1 = [1,2,3]
list2 = [2,3,4]
list1.extend(list2)
print(list1)
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from scipy.io import loadmat
import numpy as np
path = 'F:/PostGraduate/TaskOne/DRIVE dataset/CossingBifurcation GT/JunctionsGTImagelabel/01_test_JunctionsPos.mat'
data = loadmat(path)
print(data.keys())
print(len(data['CrossPos']))
im_path = 'F:/PostGraduate/TaskOne/DRIVE dataset/CossingBifurcation GT/JunctionsGTonOriginalImage/01_test.png'
imm = cv2.imread(im_path)
# imm = cv2.cvtColor(imm,cv2.COLOR_BGR2RGB)
for i in range(data['CrossPos'].shape[0]):
    cv2.rectangle(imm, (data['CrossPos'][i][1]-2,data['CrossPos'][i][0]-2), (data['CrossPos'][i][1]+2,data['CrossPos'][i][0]+2), (0,255,0), thickness=1)
for i in range(data['BiffPos'].shape[0]):
    cv2.rectangle(imm, (data['BiffPos'][i][1]-2, data['BiffPos'][i][0]-2),  (data['BiffPos'][i][1]+2, data['BiffPos'][i][0]+2), (0,0,255) , thickness=1)

cv2.imshow('2.jpg',imm)
ppath = 'F:/PostGraduate/TaskOne/IOSTAR dataset/CossingBifurcation GT/JunctionsGTonOriginalImage/sSTAR 02_ODC_Merged.png'
iim = cv2.imread(ppath)
ii = cv2.imshow('3.jpg',iim)
cv2.waitKey(0)
# from skimage import feature as ft
# img = cv2.imread('F:/PostGraduate/TaskOne/images/CERA_015_cy_582_cx_1436.png', cv2.IMREAD_GRAYSCALE)
# features = ft.hog(img,orientations=6,pixels_per_cell=[20,20],cells_per_block=[2,2],visualize=True)
# plt.imshow(features[1],cmap=plt.cm.gray)
# plt.show()


# from skimage.color import rgb2gray
# from skimage.feature import hog
# gray = rgb2gray(img) / 255.0
# fd = hog(gray, orientations=12, block_norm='L1', pixels_per_cell=[10, 10], cells_per_block=[4, 4], visualize=False, transform_sqrt=True)
