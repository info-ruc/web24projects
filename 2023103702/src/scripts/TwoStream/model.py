import torch
import torch.nn as nn
from torchvision import transforms

class TwoStreamMultimodal(nn.Module):
    def __init__(self, backbone) -> None:
        super(TwoStreamMultimodal, self).__init__()
        self.backbone1=backbone
        self.backbone2=backbone
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