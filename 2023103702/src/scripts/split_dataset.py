import torch
import torch.nn as nn
import os
import random
from torch.utils.data import Dataset,DataLoader
from PIL import Image

from torchvision import transforms
myTransform = {
    'train': transforms.Compose([
        # transforms.Resize([800,800]),
        # transforms.CenterCrop(230),
        # transforms.RandomResizedCrop(size=(800,800), scale=(0.6, 1)),

        # transforms.Resize([224, 224]), #smaller_roi成功
        # transforms.CenterCrop(70),  #smaller_roi成功

        transforms.Resize([224,224]),
        # transforms.CenterCrop(100),
        # transforms.Pad(padding=62,padding_mode='constant'),

        transforms.RandomAffine(degrees=0, translate=(0.35, 0.35)),
        # transforms.RandomResizedCrop(224,scale=(0.75,1)),

        # transforms.RandomResizedCrop(size=(400,400), scale=(0.6, 1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(90),
        transforms.ColorJitter(contrast=(0.5,1.5),saturation=(0.5,1.5),hue=(-0.5,0.5),brightness=(0.875,1.125)),
        # transforms.ColorJitter(brightness=0.1,contrast=0.1,saturation=0.1,hue=0.1),
        # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        # transforms.Resize([224,224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ]),
    'test': transforms.Compose([
        # transforms.Resize([224, 224]),#smaller_roi成功
        # transforms.CenterCrop(70),#smaller_roi成功

        transforms.Resize([224,224]),
        # transforms.CenterCrop(100),   #move_resize_smaller_patch成功
        # transforms.Pad(padding=62, padding_mode='constant'),  #move_resize_smaller_patch成功
        # transforms.Resize([224,224]),
        # transforms.CenterCrop(100),
        # transforms.Resize([224,224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ]),
}

# myTransform = {
#     'train': transforms.Compose([
#         # transforms.Pad(padding=672,fill=0,padding_mode='constant'),
#         transforms.RandomHorizontalFlip(p=0.5),
#         transforms.RandomVerticalFlip(p=0.5),
#         transforms.RandomRotation(90),
#         # transforms.ColorJitter(brightness=0.08,contrast=0.08,saturation=0.08,hue=0.03),
#         transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
#         transforms.Resize([224,224]),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=(0.485, 0.456, 0.406),
#                              std=(0.229, 0.224, 0.225))
#     ]),
#     'test': transforms.Compose([
#         # transforms.Pad(padding=672, fill=0, padding_mode='constant'),
#         transforms.Resize([224,224]),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=(0.485, 0.456, 0.406),
#                              std=(0.229, 0.224, 0.225))
#     ]),
# }


class MyDataset(Dataset):

    def __init__(self, imgs, transform=None):
        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, index):
        # fn, segfn, label = self.imgs[index]
        fn, label = self.imgs[index]
        img = Image.open(fn).convert('RGB')
        # img1 = Image.open(fn).convert('RGB')
        # img2 = Image.open(segfn).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        # img2 = transforms.Compose([
        #     transforms.Resize([224,224]),
        #     transforms.ToTensor()
        #     ])(img2)

        # return img1, img2, label
        return img, label


    def __len__(self):
        return len(self.imgs)



ori_img_dir = './images'
# aug_img_dir = 'F:/PostGraduate/TaskOne/augment'
full_imgs = []

fh = open('./grading.txt', 'r')
full_imgs = []
flag = 0

for line in fh:
    if flag == 0:
        flag = flag + 1
        continue
    line = line.rstrip()
    words = line.split(", ")
    full_imgs.append(("./images/" + words[1] + ".png", int(words[2])))


# for file in sorted(os.listdir(aug_img_dir)):
#     if 'cls0' in file:
#         full_imgs.append((os.path.join(aug_img_dir,file),0))
#     elif 'cls1' in file:
#         full_imgs.append((os.path.join(aug_img_dir,file),1))
#     elif 'cls2' in file:
#         full_imgs.append((os.path.join(aug_img_dir,file),2))
#     elif 'cls3' in file:
#         full_imgs.append((os.path.join(aug_img_dir,file),3))
#
random.seed(0)
numOfData = len(full_imgs)
test_imgs = random.sample(full_imgs, int(numOfData * 0.2))
print('test nums:',len(test_imgs))
train_val_imgs = full_imgs.copy()
for i in range(len(test_imgs)):
    train_val_imgs.remove(test_imgs[i])

print('train_val nums:',len(train_val_imgs))
train_val_dataset = MyDataset(train_val_imgs,transform=myTransform['train'])
test_dataset = MyDataset(test_imgs,transform=myTransform['test'])

train_size = int(0.9 * len(train_val_dataset))
val_size = len(train_val_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_val_dataset, [train_size, val_size],generator=torch.Generator().manual_seed(0))

if __name__ == '__main__':
    print(test_imgs)