from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import os
import time
import datetime
from PIL import Image
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset,DataLoader
from references_detection import transforms as T
from references_detection.engine import train_one_epoch, evaluate
from references_detection import utils


def get_instance_segmentation_model(num_classes):
    model = maskrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor=FastRCNNPredictor(in_features,num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer=256
    model.roi_heads.mask_predictor=MaskRCNNPredictor(in_features_mask,hidden_layer,num_classes)

    return model



class CrossoverVesselMaskRCNNDataset(Dataset):
    def __init__(self, root, transforms=None, keyword=None):
        self.root = root
        self.tranforms = transforms
        self.keyword = keyword
        self.imgs = os.listdir(os.path.join(root, keyword))
        # self.masks = os.listdir(os.path.join(root, keyword+"Mask"))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.keyword, self.imgs[idx])
        mask_path = os.path.join(self.root, self.keyword+"Mask", self.imgs[idx].replace(".jpg",".png"))
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)
        mask = np.array(mask)
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]
        masks = mask==obj_ids[:,None,None]
        #get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin,ymin,xmax,ymax])

        boxes = torch.as_tensor(boxes,dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,),dtype=torch.int64)
        masks = torch.as_tensor(masks,dtype=torch.uint8)
        image_id = torch.tensor([idx])
        area = (boxes[:,3]-boxes[:,1])*(boxes[:,2]-boxes[:,0])
        #suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,),dtype=torch.int64)

        target={}
        target["boxes"]=boxes
        target["labels"]=labels
        target["masks"]=masks
        target["image_id"]=image_id
        target["area"]=area
        target["iscrowd"]=iscrowd

        if self.tranforms is not None:
            img, target = self.tranforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


def get_transform(train):
    transforms=[]
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.RandomVerticalFlip(0.5))
    return T.Compose(transforms)


train_dataset=CrossoverVesselMaskRCNNDataset("./junc_coco_data",get_transform(train=True),keyword="train")
val_dataset=CrossoverVesselMaskRCNNDataset("./junc_coco_data",get_transform(train=False),keyword="val")

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=utils.collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=utils.collate_fn)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_instance_segmentation_model(num_classes=2)
model.to(device)
model.load_state_dict(torch.load('./maskrcnnOutput/model_29.pth')['model'])

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=5e-3, momentum=0.9, weight_decay=5e-4)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

LOSSES={
    'loss_classifier':[],
    'loss_box_reg':[],
    'loss_mask':[],
    'loss_objectness':[],
    'loss_rpn_box_reg':[],
    'loss_sum':[],
    'val_loss_classifier':[],
    'val_loss_box_reg':[],
    'val_loss_mask':[],
    'val_loss_objectness':[],
    'val_loss_rpn_box_reg':[],
    'val_loss_sum':[]
}


num_epochs=30
outputDir="./maskrcnnOutput"
start_time = time.time()
for epoch in range(num_epochs):
    epoch = epoch+30
    print(f"epoch {epoch} is training - learning rate = {lr_scheduler.get_last_lr()[0]}")
    train_one_epoch(model,optimizer,train_dataloader,device,epoch,print_freq=10)
    lr_scheduler.step()
    print(f"epoch {epoch} is validating")
    evaluate(model, val_dataloader, device=device)

    if (epoch+1)%10==0:
        utils.save_on_master({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict()
        },
        os.path.join(outputDir, 'model_{}.pth'.format(epoch))
        )

total_time = time.time() - start_time
total_time_str = str(datetime.timedelta(seconds=int(total_time)))
print('Training time {}'.format(total_time_str))
