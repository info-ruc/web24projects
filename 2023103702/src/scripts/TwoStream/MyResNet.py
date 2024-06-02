import os
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F


class IntermediateLayerGetter(nn.ModuleDict):
    def __init__(self, model, return_layers):
        # 首先判断 return_layer中的key 是否在model中
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")

        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}
        layers = OrderedDict()

        # 遍历模型子模块按顺序存入有序字典
        # 只保存layer4及其之前的结构，舍去之后不用的结构
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        # 依次遍历模型的所有子模块，并进行正向传播，
        # 收集layer1, layer2, layer3, layer4的输出
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, norm_layer=nn.BatchNorm2d):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)

        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)), #全局平均池化
            nn.Conv2d(out_channel*self.expansion,out_channel*self.expansion//16,kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(out_channel*self.expansion//16,out_channel*self.expansion,kernel_size=1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identify = x
        if self.downsample is not None:
            identify = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        tmp = self.se(out)
        out = out * tmp

        out += identify
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    # 仅包含分类器前面的模型结构
    def __init__(self, block, blocks_num, num_classes=1000):
        super(ResNet, self).__init__()

        self._norm_layer = nn.BatchNorm2d

        self.in_channel = 64

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel, channel, downsample=downsample,
                            stride=stride, norm_layer=norm_layer))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


class MaxPool_Block(nn.Module):

    def forward(self, x, names):
        names.append('pool')
        # input, kernel_size, stride, padding
        x.append(F.max_pool2d(x[-1], 1, 2, 0))
        return x, names


def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition  # 这⾥应⽤了⼴播机制


class FPN(nn.Module):
    def __init__(self, backbone_output_channels_list, out_channels, maxpool_blocks=True):
        super(FPN, self).__init__()

        # 用来调整resnet输出特征矩阵(layer1,2,3,4)的channel（kernel_size=1）
        self.inner_blocks = nn.ModuleList()
        # 对调整后的特征矩阵使用3x3的卷积核来得到对应的预测特征矩阵
        self.layer_blocks = nn.ModuleList()

        for backbone_channels in backbone_output_channels_list:
            inner_block_module = nn.Conv2d(backbone_channels, out_channels, 1)
            layer_block_module = nn.Conv2d(out_channels, out_channels, 3, padding=1)

            self.inner_blocks.append(inner_block_module)
            self.layer_blocks.append(layer_block_module)
        # 初始化权重参数
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

        self.extra_blocks = maxpool_blocks


    def forward(self, x):
        # 对应backbone中layer1、2、3、4特征图的输出的key
        names = list(x.keys())
        # 对应backbone中layer1、2、3、4特征图的输出的value
        x = list(x.values())


        #计算ResNet不同特征层权重
        safpn_channel_weight=[]
        for i in len(x):
            safpn_channel_weight.append(torch.sqrt(torch.sum(torch.square(x))))

        safpn_channel_weight = softmax(safpn_channel_weight)



        # backbone中layer4的输出调整到指定维度上 channels: 2048->256
        last_inner = self.inner_blocks[-1](x[-1])
        # results中保存着经过3*3conv后的每个预测特征层
        results = []
        # layer4
        results.append(self.layer_blocks[-1](last_inner))
        # layer4+layer3
        layer3_inner = self.inner_blocks[2](x[2])
        layer3_shape = layer3_inner.shape[-2:]
        last_layer3 = F.interpolate(last_inner, size=layer3_shape, mode="nearest")
        #此处使用SAFPN结构赋予权重
        last_layer3_sum = safpn_channel_weight[-1]*layer3_inner + (1-safpn_channel_weight[-1])*last_layer3
        results.insert(0, self.layer_blocks[2](last_layer3_sum))
        # layer3+layer2
        layer2_inner = self.inner_blocks[1](x[1])
        layer2_shape = layer2_inner.shape[-2:]
        layer3_layer2 = F.interpolate(last_layer3_sum, size=layer2_shape, mode="nearest")
        # 此处使用SAFPN结构赋予权重
        layer3_layer2_sum = safpn_channel_weight[-2]*layer2_inner + (1-safpn_channel_weight[-2])*layer3_layer2
        results.insert(0, self.layer_blocks[1](layer3_layer2_sum))
        # layer2+layer1
        layer1_inner = self.inner_blocks[0](x[0])
        layer1_shape = layer1_inner.shape[-2:]
        layer2_layer1 = F.interpolate(layer3_layer2_sum, size=layer1_shape, mode="nearest")
        # 此处使用SAFPN结构赋予权重
        layer2_layer1_sum = safpn_channel_weight[-3]*layer1_inner + (1-safpn_channel_weight[-3])*layer2_layer1
        results.insert(0, self.layer_blocks[0](layer2_layer1_sum))
        # results 存储着FPN后特征图从大到小的 key:0,1,2,3,4. value(H W shape):[1/4, 1/8, 1/16, 1/32, 1/64]
        if self.extra_blocks:
            results, names = self.extra_blocks(results, names)
        # out: key: 0,1,2,3,pool
        out = OrderedDict([(k, v) for k, v in zip(names, results)])

        return out


class Backbone_FPN(nn.Module):
    def __init__(self, backbone, return_layers, in_channels_list, out_channels):
        super(Backbone_FPN, self).__init__()

        maxpool_block = MaxPool_Block()
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.fpn = FPN(
            backbone_output_channels_list=in_channels_list,
            out_channels=out_channels,
            maxpool_blocks=maxpool_block,
        )

        self.out_channels = out_channels

    def forward(self, x):
        x = self.body(x)
        x = self.fpn(x)

        return x


def ResNet50_FPN_backbone():
    resnet_backbone = ResNet(Bottleneck, [3, 4, 6, 3])
    return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}
    in_channels_list = [256, 512, 1024, 2048]
    out_channels = 256

    return Backbone_FPN(resnet_backbone, return_layers, in_channels_list, out_channels)


if __name__ == '__main__':
    model = ResNet50_FPN_backbone()
    print(model)