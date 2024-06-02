from pprint import pprint

import cv2
import numpy as np
import os

def find_specific_color(img, pixel_rgb):
    # 找到指定颜色，其他颜色设置为背景
    mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)

    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            r = img[row, col, 0]
            g = img[row, col, 1]
            b = img[row, col, 2]

            if ([r, g, b] == pixel_rgb):
                mask[row][col] = 255
            else:
                mask[row][col] = 0

    return mask


def flood_fill(img_closed_loop):
    # 找出图像中的闭合区间，对内部进行填充
    im_floodfill = img_closed_loop.copy()
    mask = np.zeros((img_closed_loop.shape[0] + 2, img_closed_loop.shape[1] + 2), np.uint8)

    cv2.floodFill(im_floodfill, mask, (0, 0), 255)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    im_out = img_closed_loop | im_floodfill_inv

    return im_out


if __name__ == '__main__':

    folder_path = "F:/PostGraduate/TaskOne/SegSmallerROI/data/images"
    files = os.listdir(folder_path)

    for file in files:
        if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg"):
            if "crosspt_markedimg" in file:
                file_path = os.path.join(folder_path, file)
                print(file_path)
                img = cv2.imread(file_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv2读取图像的通道为BGR需要进行转换
                mask = find_specific_color(img, [255, 0, 0])  # 设置需要找到的颜色RGB值，[255,0,0]表示红色
                mask = flood_fill(mask)
                new_file_path = file_path.replace("images","labels")
                print(new_file_path)
                cv2.imwrite(new_file_path,mask)
                # mask_rgb = np.stack((mask,)*3, axis=-1)
                # result = cv2.bitwise_and(img,mask_rgb)
                # cv2.imshow('result',result)
                # cv2.waitKey(0)



# 另一种得到mask的方法
# import cv2
#
# path='F:/PostGraduate/TaskOne/images/CERA_009_cy_575_cx_2051_crosspt_markedimg.png'
# ori_img = cv2.imread(path)
# print(ori_img.shape)
#
# # r , g , b = cv2.split(img)
# im_height , im_width = ori_img.shape[:2]
# img = ori_img.copy()
# for i in range(im_height):
#     for j in range(im_width):
#         if img[i][j][0]==0 and img[i][j][1]==0 and img[i][j][2]==255:
#             img[i][j][0]=img[i][j][1]=img[i][j][2]=255
#         else:
#             img[i][j][0]=img[i][j][1]=img[i][j][2]=0
#
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow('img',img)
#
# binary, contours, _ = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# # print('contours', contours)
#
# mask = img.copy()
#
# for contour in contours:
#     cv2.fillPoly(mask, [contour], (255,255,255))
#
# cv2.imshow('mask',mask)
# cv2.waitKey(0)






# # 圆形mask的构建
# x=int(img.shape[0]/2)
# y=int(img.shape[1]/2)
# r=70
# mask = np.zeros(img.shape[:2], dtype=np.uint8)
# mask = cv2.circle(mask, (x,y), r, (255,255,255), -1)
# image = cv2.add(img, np.zeros(np.shape(img), dtype=np.uint8), mask=mask)
# cv2.imshow('image',image)
# cv2.imshow('ori_image',img)
# cv2.waitKey()