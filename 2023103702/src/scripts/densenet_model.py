import torch
import torch.nn as nn
from torchvision.models import densenet169
model=densenet169(pretrained=True)
model.classifier=nn.Sequential(
    nn.Linear(in_features=1664,out_features=1000,bias=True),
    nn.Linear(in_features=1000,out_features=50,bias=True),
    # nn.Linear(in_features=784,out_features=500,bias=True),
    # nn.Linear(in_features=500,out_features=100,bias=True),
    # nn.Linear(in_features=500,out_features=50,bias=True),
    nn.Linear(in_features=50,out_features=5,bias=True),
    # nn.Linear(in_features=5,out_features=1,bias=True)
)