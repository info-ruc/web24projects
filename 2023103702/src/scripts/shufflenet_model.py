import torch
import torch.nn as nn
from torchvision.models import shufflenet_v2_x1_0
model = shufflenet_v2_x1_0(pretrained=True)
model.fc=nn.Sequential(
    nn.Linear(in_features=1024,out_features=1000,bias=True),
    nn.Linear(in_features=1000,out_features=50,bias=True),
    # nn.Linear(in_features=784,out_features=500,bias=True),
    # nn.Linear(in_features=500,out_features=100,bias=True),
    # nn.Linear(in_features=500,out_features=50,bias=True),
    nn.Linear(in_features=50,out_features=5,bias=True),
    # nn.Linear(in_features=5,out_features=1,bias=True)
)