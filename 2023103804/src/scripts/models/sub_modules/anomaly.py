import torch
import torch.nn as nn
import torch.nn.functional as F


class AdvancedAnomalyDetectionModule(nn.Module):
    def __init__(self):
        super(AdvancedAnomalyDetectionModule, self).__init__()
        self.conv1 = nn.Conv2d(512+1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, lidar_depth, image_features, is_train=True):
        n_agent,n_cam,_,H,W = image_features.shape
        lidar_depth = lidar_depth.view(n_cam*n_agent,1,H,W)
        image_features = image_features.view(n_cam*n_agent,-1,H,W)
        if is_train:
            combined_input = torch.cat([lidar_depth, image_features], dim=1)
            # 通过神经网络生成异常标签
            x = F.relu(self.conv1(combined_input))
            x = F.relu(self.conv2(x))
            anomaly_label = torch.sigmoid(self.conv3(x))
            anomaly_label = anomaly_label.view(n_agent,n_cam,1,H,W)
            return anomaly_label
        else:
            # 在测试阶段，我们不能使用lidar_depth，所以返回None
            return None