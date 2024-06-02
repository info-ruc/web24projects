from PIL import Image
import cv2
import numpy as np
import random

import cv2
import os
import numpy as np

def img_contrast_bright(img,a,b,g):
    h,w,c=img.shape
    blank=np.zeros([h,w,c],img.dtype)
    dst=cv2.addWeighted(img,a,blank,b,g)
    return dst

img = cv2.imread("../ttt.jpg")
cv2.imshow('aa',img)

a=1.2
b=1-a
g=10
img2 = img_contrast_bright(img,a,b,g)
cv2.imshow('bb',img2)

img3 = np.uint8(np.clip((a * img + g), 0, 255))
cv2.imshow('cc',img3)
cv2.waitKey()




# # 修改图像的对比度,coefficent>0, <1降低对比度,>1提升对比度 建议0-2
# def change_contrast(img, coefficent):
#     imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     m = cv2.mean(img)[0]
#     graynew = m + coefficent * (imggray - m)
#     img1 = np.zeros(img.shape, np.float32)
#     k = np.divide(graynew, imggray, out=np.zeros_like(graynew), where=imggray != 0)
#     img1[:, :, 0] = img[:, :, 0] * k
#     img1[:, :, 1] = img[:, :, 1] * k
#     img1[:, :, 2] = img[:, :, 2] * k
#     img1[img1 > 255] = 255
#     img1[img1 < 0] = 0
#     return img1.astype(np.uint8)
#
# img = cv2.imread("../ttt.jpg")
# img = change_contrast(img,2)
# cv2.imwrite("../ttt2.jpg",img)
# cv2.imshow("change_contrast",img)
# cv2.waitKey()

# hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
# h,s,v = cv2.split(hsv)
#
# h_new= h
# s_new = s
# v_new = v
#
# hsv = cv2.merge((h_new, s_new, v_new))


# 将新的 H 和 S 值写入 HSV 图像中
# cv2.putText(hsv, "New H: {} S: {} V: {}".format(h_new, s_new, v_new), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
# cv2.imshow('image', hsv)
# cv2.imwrite('../ttt.jpg',hsv)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.imshow('123',img)

# cv2.waitKey()