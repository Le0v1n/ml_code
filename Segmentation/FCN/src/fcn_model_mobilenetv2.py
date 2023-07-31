from collections import OrderedDict

from typing import Dict

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from .mobilenet_v2_backbone import MobileNetV2


class FCN(nn.Module):
    def __init__(self, backbone, classifier):
        super(FCN, self).__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        input_shape = x.shape[-2:]  # weight, height
        # contract: features is a dict of tensors
        x = self.backbone(x)  # 返回的是一个OrderedDict
        x = self.classifier(x)  # 主干上的FCN Head
        
        # 原论文中虽然使用的是ConvTranspose2d，但权重是冻结的，所以就是一个bilinear插值
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x


class FCNHead(nn.Sequential):
    def __init__(self, in_channels, channels):
        inter_channels = in_channels // 4
        layers = [
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1)
        ]

        super(FCNHead, self).__init__(*layers)


def fcn_mobilenetv2(num_classes=21, pretrain_backbone=False):
    backbone = MobileNetV2()

    if pretrain_backbone:
        # 载入预训练权重
        backbone.load_state_dict(torch.load("pretrained_models/mobilenet_v2.pth", map_location='cpu'))

    out_inplanes = 2048

    classifier = FCNHead(out_inplanes, num_classes)

    model = FCN(backbone, classifier)

    return model