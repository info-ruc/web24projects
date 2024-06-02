import os
import torch
import torch.nn as nn
from PIL import Image
import imageio
from imageio import imwrite
from scipy.io import loadmat
import cv2
import matplotlib.pyplot as plt
import numpy as np

anno_dir1 = '../IOSTAR dataset/CossingBifurcation GT/JunctionsGTImagelabel'
anno_dir2 = '../DRIVE dataset/CossingBifurcation GT/JunctionsGTImagelabel'

train_roi_image_path = './smaller_roi_junc_coco_data/train'
train_roi_mask_path = './smaller_roi_junc_coco_data/trainMask'
val_roi_image_path = './smaller_roi_junc_coco_data/val'
val_roi_mask_path = './smaller_roi_junc_coco_data/valMask'
test_roi_image_path = './smaller_roi_junc_coco_data/test'
test_roi_mask_path = './smaller_roi_junc_coco_data/testMask'

if not os.path.exists(train_roi_image_path):
    os.mkdir(train_roi_image_path)
if not os.path.exists(train_roi_mask_path):
    os.mkdir(train_roi_mask_path)
if not os.path.exists(val_roi_image_path):
    os.mkdir(val_roi_image_path)
if not os.path.exists(val_roi_mask_path):
    os.mkdir(val_roi_mask_path)
if not os.path.exists(test_roi_image_path):
    os.mkdir(test_roi_image_path)
if not os.path.exists(test_roi_mask_path):
    os.mkdir(test_roi_mask_path)

train_image_path = './junc_coco_data/train'
train_mask_path = './junc_coco_data/trainMask'
for imgfile in os.listdir(train_image_path):
    img = Image.open(os.path.join(train_image_path,imgfile))
    img = np.array(img)
    mask = Image.open(os.path.join(train_mask_path,imgfile.replace('.jpg','.png')))
    mask = np.array(mask)
    if 'sSTAR' in imgfile:
        ann_path = os.path.join(anno_dir1,imgfile.replace('.jpg', '_JunctionsPos.mat'))
    else:
        ann_path = os.path.join(anno_dir2,imgfile.replace('.jpg', '_JunctionsPos.mat'))

    data = loadmat(ann_path)
    junction_classes = ['CrossPos']

    for class_id, junction in enumerate(junction_classes):
        for j in range(data[junction].shape[0]):
            xmin = max(int(data[junction][j][1]) - 15,0)
            ymin = max(int(data[junction][j][0]) - 15,0)
            xmax = min(int(data[junction][j][1]) + 15,img.shape[1])
            ymax = min(int(data[junction][j][0]) + 15,img.shape[0])

            new_img = img[ymin:ymax,xmin:xmax]
            new_mask = mask[ymin:ymax,xmin:xmax]

            # plt.imshow(new_mask)
            # plt.show()
            # new_img = Image.fromarray(new_img).resize((224,224))
            # new_mask = Image.fromarray(new_mask).resize((224,224))
            # new_img = np.array(new_img)
            # new_mask = np.array(new_mask)

            new_img_path = os.path.join(train_roi_image_path,'%03d_'%j + imgfile)
            new_mask_path = os.path.join(train_roi_mask_path,'%03d_'%j + imgfile.replace('.jpg','.png'))
            imwrite(new_img_path, new_img)
            imwrite(new_mask_path, new_mask)





val_image_path = './junc_coco_data/val'
val_mask_path = './junc_coco_data/valMask'
for imgfile in os.listdir(val_image_path):
    img = Image.open(os.path.join(val_image_path, imgfile))
    img = np.array(img)
    mask = Image.open(os.path.join(val_mask_path, imgfile.replace('.jpg', '.png')))
    mask = np.array(mask)
    if 'sSTAR' in imgfile:
        ann_path = os.path.join(anno_dir1, imgfile.replace('.jpg', '_JunctionsPos.mat'))
    else:
        ann_path = os.path.join(anno_dir2, imgfile.replace('.jpg', '_JunctionsPos.mat'))

    data = loadmat(ann_path)
    junction_classes = ['CrossPos']

    for class_id, junction in enumerate(junction_classes):
        for j in range(data[junction].shape[0]):
            xmin = max(int(data[junction][j][1]) - 15,0)
            ymin = max(int(data[junction][j][0]) - 15,0)
            xmax = min(int(data[junction][j][1]) + 15,img.shape[1])
            ymax = min(int(data[junction][j][0]) + 15,img.shape[0])

            new_img = img[ymin:ymax, xmin:xmax]
            new_mask = mask[ymin:ymax, xmin:xmax]
            # new_img = Image.fromarray(new_img).resize((224, 224))
            # new_mask = Image.fromarray(new_mask).resize((224, 224))
            # new_img = np.array(new_img)
            # new_mask = np.array(new_mask)

            new_img_path = os.path.join(val_roi_image_path, '%03d_' % j + imgfile)
            new_mask_path = os.path.join(val_roi_mask_path, '%03d_' % j + imgfile.replace('.jpg', '.png'))
            imwrite(new_img_path, new_img)
            imwrite(new_mask_path, new_mask)






test_image_path = './junc_coco_data/test'
test_mask_path = './junc_coco_data/testMask'
for imgfile in os.listdir(test_image_path):
    img = Image.open(os.path.join(test_image_path, imgfile))
    img = np.array(img)
    mask = Image.open(os.path.join(test_mask_path, imgfile.replace('.jpg', '.png')))
    mask = np.array(mask)
    if 'sSTAR' in imgfile:
        ann_path = os.path.join(anno_dir1, imgfile.replace('.jpg', '_JunctionsPos.mat'))
    else:
        ann_path = os.path.join(anno_dir2, imgfile.replace('.jpg', '_JunctionsPos.mat'))

    data = loadmat(ann_path)
    junction_classes = ['CrossPos']

    for class_id, junction in enumerate(junction_classes):
        for j in range(data[junction].shape[0]):
            xmin = max(int(data[junction][j][1]) - 15,0)
            ymin = max(int(data[junction][j][0]) - 15,0)
            xmax = min(int(data[junction][j][1]) + 15,img.shape[1])
            ymax = min(int(data[junction][j][0]) + 15,img.shape[0])

            new_img = img[ymin:ymax, xmin:xmax]
            new_mask = mask[ymin:ymax, xmin:xmax]
            # new_img = Image.fromarray(new_img).resize((224, 224))
            # new_mask = Image.fromarray(new_mask).resize((224, 224))
            # new_img = np.array(new_img)
            # new_mask = np.array(new_mask)

            new_img_path = os.path.join(test_roi_image_path, '%03d_' % j + imgfile)
            new_mask_path = os.path.join(test_roi_mask_path, '%03d_' % j + imgfile.replace('.jpg', '.png'))
            imwrite(new_img_path, new_img)
            imwrite(new_mask_path, new_mask)


