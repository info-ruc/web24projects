from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torch
import torch.nn as nn

ori_model = maskrcnn_resnet50_fpn(pretrained=True)
num_classes = 2
in_features = ori_model.roi_heads.box_predictor.cls_score.in_features
ori_model.roi_heads.box_predictor=FastRCNNPredictor(in_features,num_classes)

in_features_mask = ori_model.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer=256
ori_model.roi_heads.mask_predictor=MaskRCNNPredictor(in_features_mask,hidden_layer,num_classes)

ori_model.load_state_dict(torch.load('./CrossoverDetection/maskrcnnROIOutput/roi_model_14.pth')['model'])


# for param in ori_model.parameters():
#     param.requires_grad = False


class MaskRCNN_Backbone_Grading(nn.Module):
    def __init__(self, backbone) -> None:
        super(MaskRCNN_Backbone_Grading, self).__init__()
        self.backbone=backbone
        self.pooling=nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.linear=nn.Linear(256,4)
        # self.linear1=nn.Linear(256,100)
        # self.relu=nn.ReLU()
        # self.linear2=nn.Linear(100,4)

    def forward(self, x):
        features = self.backbone(x)
        # print('0',features['0'].shape)
        # print('1',features['1'].shape)
        # print('2',features['2'].shape)
        # print('3',features['3'].shape)
        # print('pool',features['pool'].shape)
        features = self.pooling(features['0'])
        features = features.view(features.shape[0],-1)
        out = self.linear(features)
        # print(features.shape)
        # features = torch.flatten(features['0'],start_dim=1)
        # out = self.linear1(features)
        # out = self.relu(out)
        # out = self.linear2(out)
        return out

# if __name__ == '__main__':
#     xx = torch.randn(1,3,224,224)
#     model = MaskRCNN_Backbone_Grading(backbone=ori_model.backbone)
#     model.eval()
#     print(model(xx))


