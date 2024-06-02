import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small
model=mobilenet_v3_small(pretrained=True)
model.classifier[3]=nn.Sequential(
    nn.Linear(in_features=1024,out_features=1000,bias=True),
    nn.Linear(in_features=1000,out_features=50,bias=True),
    # nn.Linear(in_features=784,out_features=500,bias=True),
    # nn.Linear(in_features=500,out_features=100,bias=True),
    # nn.Linear(in_features=500,out_features=50,bias=True),
    nn.Linear(in_features=50,out_features=5,bias=True),
    # nn.Linear(in_features=5,out_features=1,bias=True)
)
# print(model)