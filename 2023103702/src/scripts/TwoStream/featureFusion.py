import torch
import torch.nn as nn
import torch.nn.functional as F
from MyResNet import ResNet50_FPN_backbone

class TwoStream_MultiModal_Bilinear_Model(nn.Module):
    def __init__(self, backbone1,backbone2) -> None:
        super(TwoStream_MultiModal_Bilinear_Model, self).__init__()
        self.backbone1=backbone1
        self.backbone2=backbone2
        self.pooling=nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.linear=nn.Linear(256*256,4)
        # self.linear1=nn.Linear(256,100)
        # self.relu=nn.ReLU()
        # self.linear2=nn.Linear(100,4)

    def forward(self, x):
        oriImg_feature = self.backbone1(x)
        segImg_feature = self.backbone2(x)
        fusion_feature = torch.einsum('imjk,injk->imn',oriImg_feature,segImg_feature)

        fusion_feature = fusion_feature.view(-1,256*256)
        fusion_feature = torch.mul(torch.sign(fusion_feature),torch.sqrt(torch.abs(fusion_feature)+1e-12))
        fusion_feature = F.normalize(fusion_feature, p=2, dim=1)
        fusion_feature = self.linear(fusion_feature)

        return fusion_feature