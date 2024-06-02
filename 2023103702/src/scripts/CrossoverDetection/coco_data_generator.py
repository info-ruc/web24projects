import json
import os
import torch
import cv2
import numpy as np
import time
from PIL import Image
from scipy.io import loadmat

coco_format_save_path = 'F:/PostGraduate/TaskOne/CrossoverDetection/junc_coco_data/annotations'


categories=[]
categories.append({'id':1,'name':'CrossPos','supercategory':'None'})
# categories.append({'id':2,'name':'BiffPos','supercategory':'None'})
# categories.append({'id':3,'name':'EndpointPos','supercategory':'None'})


write_json_context=dict()
write_json_context['info']={'description': '', 'url': '', 'version': '', 'year': 2023, 'contributor': '', 'date_created': '2023-04-01'}
write_json_context['licenses']=[{'id':1,'name':None,'url':None}]
write_json_context['categories']=categories
write_json_context['images']=[]
write_json_context['annotations']=[]


img_dir1 = 'F:/PostGraduate/TaskOne/IOSTAR dataset/images'
annotation_dir1='F:/PostGraduate/TaskOne/IOSTAR dataset/CossingBifurcation GT/JunctionsGTImagelabel'
for i, file in enumerate(os.listdir(img_dir1)):
    if file.endswith('.jpg'):
        imagePath = os.path.join(img_dir1, file)
        image = Image.open(imagePath)
        W, H = image.size
        img_context = {}
        img_context['file_name'] = file
        img_context['height'] = H
        img_context['width'] = W
        img_context['date_captured'] = '2023-04-01'
        img_context['id'] = i
        img_context['license'] = 1
        img_context['coco_url'] = ''
        img_context['flickr_url'] = ''
        write_json_context['images'].append(img_context)

        junc_anno = os.path.join(annotation_dir1,file.replace('.jpg','_JunctionsPos.mat'))
        data = loadmat(junc_anno)
        # junction_classes=['CrossPos','BiffPos','EndpointPos']
        junction_classes=['CrossPos']
        for class_id, junction in enumerate(junction_classes):
            for j in range(data[junction].shape[0]):
                bbox_dict = {}
                bbox_dict['id']=i*10000+class_id*1000+j
                bbox_dict['image_id']=i
                bbox_dict['category_id']=class_id+1
                bbox_dict['iscrowd']=0
                bbox_dict['area']=20*20
                xmin = int(data[junction][j][1])-10
                ymin = int(data[junction][j][0])-10
                xmax = int(data[junction][j][1])+10
                ymax = int(data[junction][j][0])+10
                bbox_dict['bbox']=[xmin,ymin,20,20]
                bbox_dict['segmentation']=[[xmin,ymin,xmax,ymin,xmax,ymax,xmin,ymax]]
                write_json_context['annotations'].append(bbox_dict)


img_dir2 = 'F:/PostGraduate/TaskOne/DRIVE dataset/images'
annotation_dir2='F:/PostGraduate/TaskOne/DRIVE dataset/CossingBifurcation GT/JunctionsGTImagelabel'
for i, file in enumerate(os.listdir(img_dir2)):
    if file.endswith('.jpg'):
        imagePath = os.path.join(img_dir2, file)
        image = Image.open(imagePath)
        W, H = image.size
        img_context = {}
        img_context['file_name'] = file
        img_context['height'] = H
        img_context['width'] = W
        img_context['date_captured'] = '2023-04-01'
        img_context['id'] = i+len(os.listdir(img_dir1))
        img_context['license'] = 1
        img_context['coco_url'] = ''
        img_context['flickr_url'] = ''
        write_json_context['images'].append(img_context)

        junc_anno = os.path.join(annotation_dir2,file.replace('.jpg','_JunctionsPos.mat'))
        data = loadmat(junc_anno)
        # junction_classes=['CrossPos','BiffPos','EndpointPos']
        junction_classes=['CrossPos']
        for class_id, junction in enumerate(junction_classes):
            for j in range(data[junction].shape[0]):
                bbox_dict = {}
                bbox_dict['id']=(i+len(os.listdir(img_dir1)))*10000+class_id*1000+j
                bbox_dict['image_id']=i+len(os.listdir(img_dir1))
                bbox_dict['category_id']=class_id+1
                bbox_dict['iscrowd']=0
                bbox_dict['area']=20*20
                xmin = int(data[junction][j][1])-10
                ymin = int(data[junction][j][0])-10
                xmax = int(data[junction][j][1])+10
                ymax = int(data[junction][j][0])+10
                bbox_dict['bbox']=[xmin,ymin,20,20]
                bbox_dict['segmentation']=[[xmin,ymin,xmax,ymin,xmax,ymax,xmin,ymax]]
                write_json_context['annotations'].append(bbox_dict)


name = os.path.join(coco_format_save_path,'all.json')
with open(name, 'w') as f:
    f.write(json.dumps(write_json_context, indent=1, separators=(',',':')))


# path = 'F:/PostGraduate/TaskOne/IOSTAR dataset/CossingBifurcation GT/JunctionsGTImagelabel/sSTAR 02_ODC_Merged_JunctionsPos.mat'
# data = loadmat(path)
# print(data.keys())

# im_path = 'F:/PostGraduate/TaskOne/IOSTAR dataset/images/sSTAR 02_ODC_Merged.jpg'
# imm = cv2.imread(im_path)
# imm = cv2.cvtColor(imm,cv2.COLOR_BGR2RGB)
# for i in range(data['CrossPos'].shape[0]):
#     cv2.rectangle(imm, (data['CrossPos'][i][1] - 5, data['CrossPos'][i][0] - 5),
#                   (data['CrossPos'][i][1] + 5, data['CrossPos'][i][0] + 5), (0, 255, 0), thickness=2)
# for i in range(data['BiffPos'].shape[0]):
#     cv2.rectangle(imm, (data['BiffPos'][i][1] - 5, data['BiffPos'][i][0] - 5),
#                   (data['BiffPos'][i][1] + 5, data['BiffPos'][i][0] + 5), (0, 0, 255), thickness=2)