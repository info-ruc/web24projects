import os
import random
import shutil
from shutil import copy2
import time
import json

if __name__ == '__main__':

    img_path1 = 'F:/PostGraduate/TaskOne/IOSTAR dataset/images'
    img_path2 = 'F:/PostGraduate/TaskOne/DRIVE dataset/images'

    mask_path1 = 'F:/PostGraduate/TaskOne/IOSTAR dataset/MaskRCNNSegLabel'
    mask_path2 = 'F:/PostGraduate/TaskOne/DRIVE dataset/MaskRCNNSegLabel'

    img_list = os.listdir(img_path1)
    img_list.extend(os.listdir(img_path2))
    # all_imgs=sorted(img_list)
    all_imgs = img_list
    num_all_imgs = len(all_imgs)
    index_list = list(range(num_all_imgs))
    random.seed(42)
    random.shuffle(index_list)

    print(index_list)

    new_img_path = 'F:/PostGraduate/TaskOne/CrossoverDetection/junc_coco_data'
    trainDir = os.path.join(new_img_path,'train')
    if not os.path.exists(trainDir):
        os.mkdir(trainDir)

    valDir = os.path.join(new_img_path,'val')
    if not os.path.exists(valDir):
        os.mkdir(valDir)

    testDir = os.path.join(new_img_path,'test')
    if not os.path.exists(testDir):
        os.mkdir(testDir)


    trainMaskDir = os.path.join(new_img_path,'trainMask')
    if not os.path.exists(trainMaskDir):
        os.mkdir(trainMaskDir)

    valMaskDir = os.path.join(new_img_path,'valMask')
    if not os.path.exists(valMaskDir):
        os.mkdir(valMaskDir)

    testMaskDir = os.path.join(new_img_path,'testMask')
    if not os.path.exists(testMaskDir):
        os.mkdir(testMaskDir)



    train_list=[]
    val_list=[]
    test_list=[]
    num=0
    for i in index_list:

        if 'sSTAR' in all_imgs[i]:
            filePath = os.path.join(img_path1,all_imgs[i])
            maskfilePath = os.path.join(mask_path1,all_imgs[i].replace('.jpg','.png'))
        else:
            filePath = os.path.join(img_path2,all_imgs[i])
            maskfilePath = os.path.join(mask_path2,all_imgs[i].replace('.jpg','.png'))

        if num < num_all_imgs*0.7:
            train_list.append(i)
            copy2(filePath, os.path.join(trainDir,all_imgs[i]))
            copy2(maskfilePath, os.path.join(trainMaskDir,all_imgs[i].replace('.jpg','.png')))
        elif num < num_all_imgs*0.9:
            val_list.append(i)
            copy2(filePath, os.path.join(valDir,all_imgs[i]))
            copy2(maskfilePath, os.path.join(valMaskDir,all_imgs[i].replace('.jpg','.png')))
        else:
            test_list.append(i)
            copy2(filePath, os.path.join(testDir,all_imgs[i]))
            copy2(maskfilePath, os.path.join(testMaskDir,all_imgs[i].replace('.jpg','.png')))

        num+=1

    print('train_nums ',len(train_list))
    print('train_list ',train_list)
    print('val_nums ',len(val_list))
    print('val_list ',val_list)
    print('test_nums ',len(test_list))
    print('test_list ',test_list)

    all_jsons = json.load(open("F:/PostGraduate/TaskOne/CrossoverDetection/junc_coco_data/annotations/all.json",'r'))

    train_json_dict={}
    val_json_dict={}

    train_json_dict['info']=all_jsons['info']
    val_json_dict['info']=all_jsons['info']

    train_json_dict['licenses']=all_jsons['licenses']
    val_json_dict['licenses']=all_jsons['licenses']

    train_json_dict['categories']=all_jsons['categories']
    val_json_dict['categories']=all_jsons['categories']


    #images
    train_json_dict['images']=list()
    val_json_dict['images']=list()
    for i in all_jsons['images']:
        if i['id'] in train_list:
            # print(i['id'], i['file_name'])
            train_json_dict['images'].append(i)
        elif i['id'] in val_list:
            val_json_dict['images'].append(i)

    #annotations
    train_json_dict['annotations']=list()
    val_json_dict['annotations']=list()
    for j in all_jsons['annotations']:
        if j['image_id'] in train_list:
            train_json_dict['annotations'].append(j)
        elif j['image_id'] in val_list:
            # print(j['image_id'])
            val_json_dict['annotations'].append(j)



    output_dir = 'F:/PostGraduate/TaskOne/CrossoverDetection/junc_coco_data/annotations'

    with open(os.path.join(output_dir,'train.json'),'w') as f:
        json.dump(train_json_dict, f, indent=2)

    with open(os.path.join(output_dir,'val.json'),'w') as f:
        json.dump(val_json_dict, f, indent=2)

