import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import random
import os
import numpy as np

def setup_seed(seed=42):
    torch.manual_seed(seed)  # 为CPU设置随机种子
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    if torch.cuda.is_available():
        # torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
        #os.environ['PYTHONHASHSEED'] = str(seed)


setup_seed(42)
fh = open('F:/PostGraduate/TaskOne/grading.txt', 'r')
full_imgs = []
flag = 0

for line in fh:
    if flag == 0:
        flag = flag + 1
        continue
    line = line.rstrip()
    words = line.split(", ")
    full_imgs.append(("F:/PostGraduate/TaskOne/images/" + words[1] + ".png", int(words[2])))


numOfData = len(full_imgs)
rate = 0.1

cls0_imgs=[]
cls1_imgs=[]
cls2_imgs=[]
cls3_imgs=[]
for i in range(len(full_imgs)):
    if full_imgs[i][1]==0:
        cls0_imgs.append(full_imgs[i])
    elif full_imgs[i][1]==1:
        cls1_imgs.append(full_imgs[i])
    elif full_imgs[i][1]==2:
        cls2_imgs.append(full_imgs[i])
    elif full_imgs[i][1]==3:
        cls3_imgs.append(full_imgs[i])

cls0_imgs_copy=cls0_imgs.copy()
cls1_imgs_copy=cls1_imgs.copy()
cls2_imgs_copy=cls2_imgs.copy()
cls3_imgs_copy=cls3_imgs.copy()


aug_img_path = "F:/PostGraduate/TaskOne/augment"
if not os.path.exists(aug_img_path):
    os.mkdir(aug_img_path)


# 108-48=60
for i in range(60):
    ori_img=Image.open(cls0_imgs[i%48][0])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=(ori_img.size[1], ori_img.size[0]), scale=(0.6, 1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(90),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.01),
    ])
    aug_img = train_transform(ori_img)
    # axes1 = plt.subplot(1,2,1)
    # plt.imshow(ori_img)
    # axes2 = plt.subplot(1,2,2)
    # plt.imshow(aug_img)
    # plt.show()
    ori_img_path = cls0_imgs[i%48][0]
    # print(ori_img_path[0:-3])
    path_token=ori_img_path.split("/")
    new_img_path = aug_img_path + "/cls0_aug_" + '%03d' % i + "_" + path_token[-1]
    cls0_imgs.append((new_img_path,cls0_imgs[i%48][1]))
    aug_img.save(new_img_path)




# 108-16=92
for i in range(92):
    ori_img=Image.open(cls1_imgs[i%16][0])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=(ori_img.size[1], ori_img.size[0]), scale=(0.6, 1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(90),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.01),
    ])
    aug_img = train_transform(ori_img)
    # axes1 = plt.subplot(1,2,1)
    # plt.imshow(ori_img)
    # axes2 = plt.subplot(1,2,2)
    # plt.imshow(aug_img)
    # plt.show()
    ori_img_path = cls1_imgs[i%16][0]
    # print(ori_img_path[0:-3])
    path_token=ori_img_path.split("/")
    new_img_path = aug_img_path + "/cls1_aug_" + '%03d' % i + "_" + path_token[-1]
    cls1_imgs.append((new_img_path,cls1_imgs[i%16][1]))
    aug_img.save(new_img_path)


# 108-14=94
for i in range(94):
    ori_img=Image.open(cls2_imgs[i%14][0])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=(ori_img.size[1], ori_img.size[0]), scale=(0.6, 1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(90),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.01),
    ])
    aug_img = train_transform(ori_img)
    # axes1 = plt.subplot(1,2,1)
    # plt.imshow(ori_img)
    # axes2 = plt.subplot(1,2,2)
    # plt.imshow(aug_img)
    # plt.show()
    ori_img_path = cls2_imgs[i%14][0]
    # print(ori_img_path[0:-3])
    path_token=ori_img_path.split("/")
    new_img_path = aug_img_path + "/cls2_aug_" + '%03d' % i + "_" + path_token[-1]
    cls2_imgs.append((new_img_path,cls2_imgs[i%14][1]))
    aug_img.save(new_img_path)



# 108-12=96
for i in range(96):
    ori_img=Image.open(cls3_imgs[i%12][0])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=(ori_img.size[1], ori_img.size[0]), scale=(0.6, 1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(90),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.01),
    ])
    aug_img = train_transform(ori_img)
    # axes1 = plt.subplot(1,2,1)
    # plt.imshow(ori_img)
    # axes2 = plt.subplot(1,2,2)
    # plt.imshow(aug_img)
    # plt.show()
    ori_img_path = cls3_imgs[i%12][0]
    # print(ori_img_path[0:-3])
    path_token=ori_img_path.split("/")
    new_img_path = aug_img_path + "/cls3_aug_" + '%03d' % i + "_" + path_token[-1]
    cls3_imgs.append((new_img_path,cls3_imgs[i%12][1]))
    aug_img.save(new_img_path)







# random.seed(42)
# test_imgs = random.sample(full_imgs, int(numOfData * rate))
# ori_train_val_imgs = full_imgs.copy()
# for i in range(len(test_imgs)):
#     ori_train_val_imgs.remove(test_imgs[i])
# #
# # aug_img_path = "F:/PostGraduate/TaskOne/augment"
# # if not os.path.exists(aug_img_path):
# #     os.mkdir(aug_img_path)
#
# train_val_imgs = ori_train_val_imgs.copy()
# #
# # for i in range(len(ori_train_val_imgs) * 10):
# #     ori_img = Image.open(ori_train_val_imgs[i % len(ori_train_val_imgs)][0])
# #     train_transform = transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.02)
# #     aug_img = train_transform(ori_img)
# #     # axes1 = plt.subplot(1,2,1)
# #     # plt.imshow(ori_img)
# #     # axes2 = plt.subplot(1,2,2)
# #     # plt.imshow(aug_img)
# #     # plt.show()
# #     ori_img_path = ori_train_val_imgs[i % len(ori_train_val_imgs)][0]
# #     # print(ori_img_path[0:-3])
# #     path_token = ori_img_path.split("/")
# #     new_img_path = aug_img_path + "/aug_" + '%03d' % i + "_" + path_token[-1]
# #     train_val_imgs.append((new_img_path, ori_train_val_imgs[i % len(ori_train_val_imgs)][1]))
# #     aug_img.save(new_img_path)
#
#
# train_val_dataset = MyDataset(train_val_imgs,transform=myTransform['train'])
# test_dataset = MyDataset(test_imgs,transform=myTransform['test'])
#
# train_size = int(0.9 * len(train_val_dataset))
# val_size = len(train_val_dataset) - train_size
# train_dataset, val_dataset = torch.utils.data.random_split(train_val_dataset, [train_size, val_size],generator=torch.Generator().manual_seed(42))
# train_dataLoader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True, num_workers=0)
# val_dataLoader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=0)
