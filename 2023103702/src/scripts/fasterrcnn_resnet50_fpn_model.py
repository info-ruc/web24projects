import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

ori_model = fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 2
in_features = ori_model.roi_heads.box_predictor.cls_score.in_features
ori_model.roi_heads.box_predictor=FastRCNNPredictor(in_features,num_classes)

ori_model.load_state_dict(torch.load('./CrossoverDetection/result/model_49.pth')['model'])
# ori_model.roi_heads.box_predictor.cls_score=nn.Linear(in_features=1024,out_features=4,bias=True)
# ori_model.roi_heads.box_predictor.bbox_pred=nn.Sequential()
# print(ori_model.backbone)

class FasterRCNN_Backbone_Grading(nn.Module):
    def __init__(self, backbone) -> None:
        super(FasterRCNN_Backbone_Grading, self).__init__()
        self.backbone=backbone
        self.pooling=nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.linear=nn.Linear(256,4)


    def forward(self, x):
        features = self.backbone(x)
        features = self.pooling(features['0'])
        features = features.view(features.shape[0],-1)
        # print(features.shape)
        # features = torch.flatten(features['0'],start_dim=1)
        out = self.linear(features)
        return out

# new_model = FasterRCNN_Backbone_Grading(ori_model.backbone)
# input = torch.rand(16,3,1024,1024)
# output = new_model(input)