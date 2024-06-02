import timm
import torch
import torch.nn as nn
from pprint import pprint
model=timm.models.swin_tiny_patch4_window7_224(pretrained=True)
model.head=nn.Sequential(
    # nn.Linear(in_features=768,out_features=5,bias=True),
    nn.Linear(in_features=768,out_features=1000,bias=True),
    # nn.ReLU(True),
    nn.Linear(in_features=1000,out_features=50,bias=True),
    # nn.ReLU(True),
    # nn.Linear(in_features=784,out_features=500,bias=True),
    # nn.Linear(in_features=500,out_features=100,bias=True),
    # nn.Linear(in_features=500,out_features=50,bias=True),
    nn.Linear(in_features=50,out_features=5,bias=True),
    # nn.Linear(in_features=5,out_features=1,bias=True)
)

pprint(model)