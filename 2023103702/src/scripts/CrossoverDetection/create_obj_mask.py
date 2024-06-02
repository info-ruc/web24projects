import os
import torch
import torch.nn as nn
from PIL import Image
from scipy.misc import imsave
from scipy.io import loadmat
import cv2
import matplotlib.pyplot as plt
import numpy as np

binary_mask_dir1 = 'F:/PostGraduate/TaskOne/IOSTAR dataset/VesselSegmentationGT'
annotation_dir1='F:/PostGraduate/TaskOne/IOSTAR dataset/CossingBifurcation GT/JunctionsGTImagelabel'
maskrcnn_label_dir1 = 'F:/PostGraduate/TaskOne/IOSTAR dataset/MaskRCNNSegLabel'

if not os.path.exists(maskrcnn_label_dir1):
    os.mkdir(maskrcnn_label_dir1)

for i, file in enumerate(os.listdir(binary_mask_dir1)):
    if file.endswith('.tif'):
        maskPath = os.path.join(binary_mask_dir1, file)
        mask_img = cv2.imread(maskPath)
        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
        H, W = mask_img.shape
        junc_anno = os.path.join(annotation_dir1, file.replace('.tif', '_JunctionsPos.mat'))
        data = loadmat(junc_anno)
        junction_classes=['CrossPos']
        cnt=1
        new_mask_img = np.zeros_like(mask_img)
        # new_mask_img = np.zeros((H,W),dtype=np.uint8)
        for class_id, junction in enumerate(junction_classes):
            for j in range(data[junction].shape[0]):
                xmin = int(data[junction][j][1])-10
                ymin = int(data[junction][j][0])-10
                xmax = int(data[junction][j][1])+10
                ymax = int(data[junction][j][0])+10
                for k1 in range(xmin,xmax+1):
                    for k2 in range(ymin,ymax+1):
                        if mask_img[k2][k1]>0:
                            new_mask_img[k2][k1]=cnt

                cnt=cnt+1

        new_mask_path = os.path.join(maskrcnn_label_dir1,file.replace('.tif','.png'))
        imsave(new_mask_path, new_mask_img)
        # new_mask_img.save("../CrossoverDetection/ttt.jpg")
        # new_mask_img = cv2.cvtColor(new_mask_img,cv2.COLOR_GRAY2RGB)
        # print("num of color is:",np.unique(new_mask_img))
        # cv2.imshow('ttt',new_mask_img)
        # cv2.imwrite('../CrossoverDetection/ttt.jpg',new_mask_img)
        # print(new_mask_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


binary_mask_dir2 = 'F:/PostGraduate/TaskOne/DRIVE dataset/manual1'
annotation_dir2 = 'F:/PostGraduate/TaskOne/DRIVE dataset/CossingBifurcation GT/JunctionsGTImagelabel'
maskrcnn_label_dir2 = 'F:/PostGraduate/TaskOne/DRIVE dataset/MaskRCNNSegLabel'

if not os.path.exists(maskrcnn_label_dir2):
    os.mkdir(maskrcnn_label_dir2)

for i, file in enumerate(os.listdir(binary_mask_dir2)):
    if file.endswith('.gif'):
        maskPath = os.path.join(binary_mask_dir2, file)
        # mask_img = cv2.imread(maskPath)
        mask_img = Image.open(maskPath)
        mask_img = np.array(mask_img)
        # mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
        H, W = mask_img.shape
        keyword = 'test' if int(file[:2])<=20 else 'training'
        junc_anno = os.path.join(annotation_dir2, file.replace('manual1.gif', keyword+'_JunctionsPos.mat'))
        print(junc_anno)
        data = loadmat(junc_anno)
        junction_classes = ['CrossPos']
        cnt = 1
        new_mask_img = np.zeros_like(mask_img)
        # new_mask_img = np.zeros((H,W),dtype=np.uint8)
        for class_id, junction in enumerate(junction_classes):
            for j in range(data[junction].shape[0]):
                xmin = int(data[junction][j][1]) - 10
                ymin = int(data[junction][j][0]) - 10
                xmax = int(data[junction][j][1]) + 10
                ymax = int(data[junction][j][0]) + 10
                for k1 in range(xmin, xmax + 1):
                    for k2 in range(ymin, ymax + 1):
                        if mask_img[k2][k1] > 0:
                            new_mask_img[k2][k1] = cnt

                cnt = cnt + 1

        new_mask_path = os.path.join(maskrcnn_label_dir2, file.replace('manual1.gif', keyword+'.png'))
        imsave(new_mask_path, new_mask_img)
