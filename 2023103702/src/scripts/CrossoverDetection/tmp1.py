# from skimage import io
# from PIL import Image
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.io import loadmat
# import os
# # data = loadmat("../IOSTAR dataset/CossingBifurcation GT/JunctionsGTImagelabel/sSTAR 36_OSC_Merged_JunctionsPos.mat")
# # print(data['CrossPos'].shape[0])
# # ii = Image.open("../IOSTAR dataset/MaskRCNNSegLabel/sSTAR 36_OSC_Merged.png")
# # ii = np.array(ii)
# # print('unique',np.unique(ii))
# # # plt.imshow(ii)
# # # plt.show()
# #
# # # mm = Image.open('../IOSTAR dataset/VesselSegmentationGT/sSTAR 05_ODC_Merged.tif')
# # mm = Image.open('../DRIVE dataset/manual1/11_manual1.gif')
# # mm = np.array(mm)
# # # mm = cv2.imread('../IOSTAR dataset/VesselSegmentationGT/sSTAR 05_ODC_Merged.tif')
# # # mm = cv2.imread('../DRIVE dataset/manual1/11_manual1.gif')
# # print(mm.shape)
# # print('max',np.max(mm))
# # # mm = cv2.cvtColor(mm,cv2.COLOR_BGR2GRAY)
# # # print(mm.shape)
#
#
# # mm1 = np.array(Image.open('F:/PostGraduate/TaskOne/DRIVE dataset/MaskRCNNSegLabel/28_training.png'))
# # print('drive : ',np.unique(mm1))
# # plt.imshow(mm1)
# # plt.show()
# # data = loadmat("../DRIVE dataset/CossingBifurcation GT/JunctionsGTImagelabel/28_training_JunctionsPos.mat")
# # print('another:',data['CrossPos'].shape[0])
#
#
# import torch
# # import torchvision.transforms
# # from torchvision.transforms import functional as F
# # iimm = Image.open('../DRIVE dataset/MaskRCNNSegLabel/25_training.png')
# # # iimm = np.array(iimm)
# # # print('np.array is',iimm.shape)
# # iimm = torchvision.transforms.ToTensor()(iimm)
# # iimm = iimm.flip([-1,-2])
# # print('torch.tensor is',iimm.shape)
# # plt.imshow(iimm.squeeze())
# # print('after squeeze',iimm.squeeze().shape)
# # plt.show()
#
#
#
#
#
# # # 实现图像长短边等比例缩放。并补白色
# #
# # def img_pad(pil_file):
# #     # h,w 先后不要写错，不然图片会变形
# #     h, w, c = pil_file.shape
# #     # print(h, w, c)
# #     fixed_size = 3000  # 输出正方形图片的尺寸
# #
# #     if h >= w:
# #         factor = h / float(fixed_size)
# #         new_w = int(w / factor)
# #         if new_w % 2 != 0:
# #             new_w -= 1
# #         pil_file = cv2.resize(pil_file, (new_w, fixed_size))
# #         pad_w = int((fixed_size - new_w) / 2)
# #         array_file = np.array(pil_file)
# #         # array_file = np.pad(array_file, ((0, 0), (pad_w, fixed_size-pad_w)), 'constant') #实现黑白图缩放
# #         array_file = cv2.copyMakeBorder(array_file, 0, 0, pad_w, fixed_size - new_w - pad_w, cv2.BORDER_CONSTANT,
# #                                         value=(255, 255, 255))  # 255是白色，0是黑色
# #     else:
# #         factor = w / float(fixed_size)
# #         new_h = int(h / factor)
# #         if new_h % 2 != 0:
# #             new_h -= 1
# #         pil_file = cv2.resize(pil_file, (fixed_size, new_h))
# #         pad_h = int((fixed_size - new_h) / 2)
# #         array_file = np.array(pil_file)
# #         # array_file = np.pad(array_file, ((pad_h, fixed_size-pad_h), (0, 0)), 'constant')
# #         array_file = cv2.copyMakeBorder(array_file, pad_h, fixed_size - new_h - pad_h, 0, 0, cv2.BORDER_CONSTANT,
# #                                         value=(255, 255, 255))
# #
# #     plt.imshow(array_file)
# #     plt.show()
# #     output_file = Image.fromarray(array_file)
# #     return output_file
# #
# #
# # if __name__ == "__main__":
# #     dir_image = './tmpInput'  # 图片所在文件夹
# #     dir_output = './tmpOutput'  # 输出结果文件夹
# #     if not os.path.exists(dir_output):
# #         os.makedirs(dir_output)
# #     i = 0
# #     list_image = os.listdir(dir_image)
# #     for file in list_image:
# #         path_image = os.path.join(dir_image, file)
# #         path_output = os.path.join(dir_output, file)
# #         pil_image = cv2.imread(path_image)
# #         b, g, r = cv2.split(pil_image)  # 通道分离,再重新合并操作
# #         pil_image = cv2.merge([r, g, b])
# #         # print(pil_image)
# #         # pil_image = pil_image.load()
# #         output_image = img_pad(pil_image)
# #         output_image.save(path_output)
# #         i += 1
# #         if i % 1000 == 0:
# #             print('The num of processed images:', i)  #
#
#
# import imutils
# import cv2
# from PIL import Image
# # image = Image.open('./tmpInput/CERA_009_cy_575_cx_2051.png')
# # image = Image.open('../images/CERA_101_cy_140_cx_1514.png')
# # image=  Image.open('../images/CERA_017_cy_614_cx_1475.png')
# # image = Image.open('../images/CERA_123_cy_1696_cx_1898.png')
# # image = Image.open('../images/CERA_056_cy_752_cx_1684.png')
# image=  Image.open('../images/CERA_079_cy_228_cx_1104.png')
# image = image.resize((70,70))
# image = np.array(image)
# print(image.shape)
# # new_img = np.zeros((3000,3000,3),dtype=np.uint8)
# # for i in range(image.shape[0]):
# #     for j in range(image.shape[1]):
# #         for channel in range(3):
# #             new_img[i+1300][j+1300][channel]=image[i][j][channel]
#
# new_img = np.zeros((500,500,3),dtype=np.uint8)
# for i in range(image.shape[0]):
#     for j in range(image.shape[1]):
#         for channel in range(3):
#             new_img[i+240][j+240][channel]=image[i][j][channel]
#
#
# # translated = imutils.translate(image, -300,-300)
# # plt.imshow(image)
# # plt.imshow(translated)
# new_img = Image.fromarray(new_img)
# new_img.save("./tmpOutput/CERA_009_cy_575_cx_2051.png")
# plt.imshow(new_img)
# plt.show()
# # cv2.imshow('translated', translated)
# # cv2.waitKey(0)

from PIL import Image
import numpy as np
# imm1 = Image.open('./smaller_roi_junc_coco_data/train/000_05_test.jpg')
# imm1 = Image.open('../../TaskOne/images/CERA_056_cy_752_cx_1684.png')
imm1 = Image.open('../../TaskOne/segmentation_images/CERA_104_cy_664_cx_913.png')
# imm1 = Image.open('../../TaskOne/images/CERA_049_cy_2176_cx_1878.png')
# imm1 = Image.open('../../TaskOne/images/CERA_049_cy_2282_cx_1404.png')
# imm1 = Image.open('../../TaskOne/images/CERA_027_cy_582_cx_2384.png')
# imm1 = Image.open("../../TaskOne/images/CERA_123_cy_1696_cx_1898.png")
# imm1 = Image.open("../../TaskOne/images/CERA_106_cy_259_cx_1968.png")
imm1 = imm1.resize((224,224))
# for ii in range(np.array(imm1).shape[0]):
#         for jj in range(np.array(imm1).shape[1]):
#                 print(np.array(imm1)[ii][jj][0],np.array(imm1)[ii][jj][1],np.array(imm1)[ii][jj][2])

import torchvision
# imm1 = torchvision.transforms.CenterCrop(70)(imm1)
# imm1 = imm1.resize((224,224))
import torchvision
imm1 = torchvision.transforms.CenterCrop(100)(imm1)
imm1 = torchvision.transforms.Pad(padding=62,fill=0,padding_mode='constant')(imm1)
imm1 = torchvision.transforms.RandomAffine(degrees=0, translate=(0.35, 0.35))(imm1)
# imm1 = torchvision.transforms.RandomResizedCrop(224,scale=(0.6,1))(imm1)
# imm1 = imm1.resize((224,224))
# imm1 = Image.open("../../TaskOne/images/CERA_069_cy_1716_cx_1104.png")
# imm1 = Image.open('./roi_junc_coco_data/train/023_01_test.jpg')
# imm1 = Image.open('./roi_junc_coco_data/trainMask/023_01_test.png')
imm = Image.open('./smaller_roi_junc_coco_data/trainMask/000_05_test.png')
imm = torchvision.transforms.Resize([800,800],interpolation=torchvision.transforms.functional._interpolation_modes_from_int(0))(imm)
# imm = np.array(imm)
# print(np.unique(imm))
from torchvision import transforms
import torchvision.transforms.functional as F
# imm1 = transforms.ColorJitter(contrast = (0.5, 1.5),saturation = (0.5, 1.5),
#         hue = (-0.5, 0.5),
#         brightness = (0.875, 1.125))(imm1)
# imm1 = transforms.Resize([800,800],interpolation=F._interpolation_modes_from_int(2))(imm1)
# imm1 = transforms.CenterCrop(300)(imm1)
# imm1 = transforms.Resize([800,800])(imm1)
# imm1 = transforms.ColorJitter(contrast=(0.5,1.5),saturation=(0.5,1.5),hue=(-0.5,0.5),brightness=(0.875,1.125))(imm1)
imm1 = np.array(imm1)
imm = np.array(imm)
import matplotlib.pyplot as plt
import imageio
imageio.imwrite('./hhh.jpg',imm1)
plt.imshow(imm1)
# plt.imshow(imm)
plt.show()
# import os
# for ii in list(os.listdir('./smaller_roi_junc_coco_data/valMask')):
#     mask = Image.open(os.path.join('./smaller_roi_junc_coco_data/valMask',ii))
#     mask = np.array(mask)
#     if len(np.unique(mask))==1:
#         print(np.unique(mask))
#         print(ii)
