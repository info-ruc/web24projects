import os,sys
import cv2
import numpy as np
from skimage import io#使用IO库读取tif图片


def tif_jpg_transform(file_path_name, bgr_savepath_name):
    img = io.imread(file_path_name)#读取文件名
    img = img / img.max()#使其所有值不大于一
    img = img * 255 - 0.001  # 减去0.001防止变成负整型
    img = img.astype(np.uint8)#强制转换成8位整型
    # img = np.array([img,img,img])
    # img = img.transpose(1,2,0)
    print(img.shape)  # 显示图片大小和深度
    b = img[:, :, 0]  # 读取蓝通道
    g = img[:, :, 1]  # 读取绿通道
    r = img[:, :, 2]  # 读取红通道
    bgr = cv2.merge([r, g, b])  # 通道拼接
    cv2.imwrite(bgr_savepath_name, bgr)#图片存储


tif_file_path = r'F:/PostGraduate/TaskOne/DRIVE dataset/training/images'# 为tif图片的文件夹路径
tif_fileList = os.listdir(tif_file_path)
for tif_file in tif_fileList:
    file_path_name = tif_file_path + '/' + tif_file
    jpg_path = r'F:/PostGraduate/TaskOne/DRIVE dataset/images' + '/' + tif_file.split('.')[0] + '.jpg' #.jpg图片的保存路径
    tif_jpg_transform(file_path_name, jpg_path)


tif_file_path = r'F:/PostGraduate/TaskOne/DRIVE dataset/test/images'# 为tif图片的文件夹路径
tif_fileList = os.listdir(tif_file_path)
for tif_file in tif_fileList:
    file_path_name = tif_file_path + '/' + tif_file
    jpg_path = r'F:/PostGraduate/TaskOne/DRIVE dataset/images' + '/' + tif_file.split('.')[0] + '.jpg' #.jpg图片的保存路径
    tif_jpg_transform(file_path_name, jpg_path)