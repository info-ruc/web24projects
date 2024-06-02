import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class FPN(nn.Module):
    def __init__(self, args,num_feature_levels=3):
        super(FPN, self).__init__()
        # num_feature_levels = args['num_feature_levels']
        self.resnet = models.resnet50()
        # Lateral layers
        self.latlayer1 = nn.Conv2d(2048, 256, kernel_size=1)
        self.latlayer2 = nn.Conv2d(1024, 256, kernel_size=1)
        self.latlayer3 = nn.Conv2d(512, 256, kernel_size=1)

        # Top layer
        self.toplayer = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.num_feature_levels = num_feature_levels

    def forward(self, x):
        # Bottom-up, 使用ResNet的层
        c1 = self.resnet.conv1(x)
        c1 = self.resnet.bn1(c1)
        c1 = self.resnet.relu(c1)
        c1 = self.resnet.maxpool(c1)

        # res block
        c2 = self.resnet.layer1(c1)
        c3 = self.resnet.layer2(c2)
        c4 = self.resnet.layer3(c3)
        c5 = self.resnet.layer4(c4)

        # Top-down
        p5 = self.latlayer1(c5)
        p4 = self._upsample_add(p5, self.latlayer2(c4))
        p3 = self._upsample_add(p4, self.latlayer3(c3))

        # Final output based on num_feature_levels
        output = []
        if self.num_feature_levels >= 3:
            p4 = self.toplayer(p4)
            p3 = self.toplayer(p3)
            output.extend([p3, p4, p5])
        elif self.num_feature_levels == 2:
            p4 = self.toplayer(p4)
            output.extend([p4, p5])
        elif self.num_feature_levels == 1:
            output.extend([p5])


        return tuple(output)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.'''
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y