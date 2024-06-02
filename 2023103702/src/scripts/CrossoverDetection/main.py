# import torch
# import os
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# from torchvision import datasets, transforms
# from PIL import Image
# from xml.dom.minidom import parse
# # %matplotlib inline
#
# class MarkDataset(torch.utils.data.Dataset):
#     def __init__(self, root, transforms=None):
#         self.root = root
#         self.transforms = transforms
#         # load all image files, sorting them to ensure that they are aligned
#         self.imgs = list(sorted(os.listdir(os.path.join(root, "JPEGImages"))))
#         self.bbox_xml = list(sorted(os.listdir(os.path.join(root, "Annotations"))))
#
#     def __getitem__(self, idx):
#         # load images and bbox
#         img_path = os.path.join(self.root, "JPEGImages", self.imgs[idx])
#         bbox_xml_path = os.path.join(self.root, "Annotations", self.bbox_xml[idx])
#         img = Image.open(img_path).convert("RGB")
#
#         # 读取文件，VOC格式的数据集的标注是xml格式的文件
#         dom = parse(bbox_xml_path)
#         # 获取文档元素对象
#         data = dom.documentElement
#         # 获取 objects
#         objects = data.getElementsByTagName('object')
#         # get bounding box coordinates
#         boxes = []
#         labels = []
#         for object_ in objects:
#             # 获取标签中内容
#             name = object_.getElementsByTagName('name')[0].childNodes[0].nodeValue  # 就是label，mark_type_1或mark_type_2
#             labels.append(np.int(name[-1]))  # 背景的label是0，mark_type_1和mark_type_2的label分别是1和2
#
#             bndbox = object_.getElementsByTagName('bndbox')[0]
#             xmin = np.float(bndbox.getElementsByTagName('xmin')[0].childNodes[0].nodeValue)
#             ymin = np.float(bndbox.getElementsByTagName('ymin')[0].childNodes[0].nodeValue)
#             xmax = np.float(bndbox.getElementsByTagName('xmax')[0].childNodes[0].nodeValue)
#             ymax = np.float(bndbox.getElementsByTagName('ymax')[0].childNodes[0].nodeValue)
#             boxes.append([xmin, ymin, xmax, ymax])
#
#         boxes = torch.as_tensor(boxes, dtype=torch.float32)
#         # there is only one class
#         labels = torch.as_tensor(labels, dtype=torch.int64)
#
#         image_id = torch.tensor([idx])
#         area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
#         # suppose all instances are not crowd
#         iscrowd = torch.zeros((len(objects),), dtype=torch.int64)
#
#         target = {}
#         target["boxes"] = boxes
#         target["labels"] = labels
#         # 由于训练的是目标检测网络，因此没有教程中的target["masks"] = masks
#         target["image_id"] = image_id
#         target["area"] = area
#         target["iscrowd"] = iscrowd
#
#         if self.transforms is not None:
#             # 注意这里target(包括bbox)也转换\增强了，和from torchvision import的transforms的不同
#             # https://github.com/pytorch/vision/tree/master/references/detection 的 transforms.py里就有RandomHorizontalFlip时target变换的示例
#             img, target = self.transforms(img, target)
#
#         return img, target
#
#     def __len__(self):
#         return len(self.imgs)

import torch
import torchvision
import torchvision.transforms as T
import torchvision.datasets as datasets
from torchvision.transforms import functional as F
import cv2
import random

def collate_fn_coco(batch):
    print('before ',batch)
    print('after ',tuple(zip(*batch)))
    return tuple(zip(*batch))

if __name__ == '__main__':

    font = cv2.FONT_HERSHEY_SIMPLEX

    root = './junc_coco_data/train'
    annFile='./junc_coco_data/annotations/train.json'

    coco_det = datasets.CocoDetection(root, annFile, transform=T.ToTensor())
    sampler = torch.utils.data.RandomSampler(coco_det)
    batch_sampler=torch.utils.data.BatchSampler(sampler, 1, drop_last=True)


    data_loader=torch.utils.data.DataLoader(
        coco_det, batch_sampler=batch_sampler, num_workers=1,
        collate_fn=collate_fn_coco
    )

    print(len(data_loader))
    for imgs, labels in data_loader:
        print(len(imgs))
        for i in range(len(imgs)):
            bboxes=[]
            ids=[]
            img=imgs[i]
            labels_=labels[i]
            for label in labels_:
                if label['category_id']!=1:
                    continue
                bboxes.append([label['bbox'][0],
                               label['bbox'][1],
                               label['bbox'][0]+label['bbox'][2],
                               label['bbox'][1]+label['bbox'][3]
                               ])
                ids.append(label['category_id'])
                print('bbox ',label['bbox'])
                print('category_id ',label['category_id'])


            img = img.permute(1,2,0).numpy()
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            for box, id_ in zip(bboxes, ids):
                x1 = int(box[0])
                y1 = int(box[1])
                x2 = int(box[2])
                y2 = int(box[3])
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),thickness=2)
                cv2.putText(img, text='c',org=(x1+5,y1+5),fontFace=font,fontScale=0.3,
                            thickness=1,lineType=cv2.LINE_AA,color=(0,255,0))

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            import matplotlib.pyplot as plt
            plt.imshow(img)
            plt.show()
            # cv2.imshow('ttt',img)
            # cv2.waitKey()