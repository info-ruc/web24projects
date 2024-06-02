import os
import time
import datetime
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
from references_detection import utils
from references_detection.group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
from references_detection.coco_utils import CocoDetection, ConvertCocoToBox
from references_detection.engine import train_one_epoch, evaluate
from references_detection import transforms as T
from pycocotools.coco import COCO
coco = COCO('F:/PostGraduate/TaskOne/CrossoverDetection/junc_coco_data/annotations/val.json')

def get_transform(train):
    transforms=[]
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.RandomVerticalFlip(0.5))
    return T.Compose(transforms)

if __name__ == '__main__':

    train_img_folder = './junc_coco_data/train'
    val_img_folder = './junc_coco_data/val'
    train_annFile='./junc_coco_data/annotations/train.json'
    val_annFile = './junc_coco_data/annotations/val.json'

    train_t = [ConvertCocoToBox()]
    train_t.append(get_transform(train=True))
    train_transforms = T.Compose(train_t)
    train_dataset = CocoDetection(img_folder=train_img_folder,ann_file=train_annFile,transforms=train_transforms)

    val_t = [ConvertCocoToBox()]
    val_t.append(get_transform(train=False))
    val_transforms = T.Compose(val_t)
    val_dataset = CocoDetection(img_folder=val_img_folder,ann_file=val_annFile,transforms=val_transforms)

    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    val_sampler = torch.utils.data.SequentialSampler(val_dataset)
    group_ids = create_aspect_ratio_groups(train_dataset,k=3)
    train_batch_sampler = GroupedBatchSampler(train_sampler,group_ids,batch_size=2)

    train_dataLoader = DataLoader(
        train_dataset, batch_sampler=train_batch_sampler, num_workers=1,
        collate_fn=utils.collate_fn
    )
    val_dataLoader = DataLoader(
        val_dataset, batch_size=2,sampler=val_sampler,num_workers=1,
        collate_fn=utils.collate_fn
    )
    print('Creating model')
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes = 2
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor=FastRCNNPredictor(in_features,num_classes)

    device = torch.device("cuda")
    model.to(device)

    #Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.0001, momentum=0.9, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[16,22],gamma=0.33)


    #training
    print('Start training')
    start_time = time.time()
    outputDir = './result'
    for epoch in range(50):
        train_one_epoch(model,optimizer,train_dataLoader,device,epoch,print_freq=20)
        lr_scheduler.step()
        utils.save_on_master({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict()
        },
        os.path.join(outputDir, 'model_{}.pth'.format(epoch))
        )
        evaluate(model, val_dataLoader, device=device)

    total_time = time.time()-start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    # print(list(sorted(coco.imgs.keys())))

