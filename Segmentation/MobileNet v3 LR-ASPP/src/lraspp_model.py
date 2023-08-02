from collections import OrderedDict

from typing import Dict

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from .mobilenet_backbone import mobilenet_v3_large


class IntermediateLayerGetter(nn.ModuleDict):
    """用于从一个模型中提取指定的中间层，并以字典形式返回这些中间层的输出。

    Module wrapper that returns intermediate layers from a model

    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.

    Args:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
            
    这是一个模块包装器，从模型中返回中间层。

    这个包装器的一个重要假设是，模块必须按照使用的顺序注册到模型中。这意味着如果要使用这个包装器，不应该在前向传播中重复使用相同的nn.Module两次。
    此外，它只能查询直接分配给模型的子模块。因此，如果传入了model，可以返回model.feature1，但不能返回model.feature1.layer2。
    
    参数:
        1. model (nn.Module): 要从中提取特征的模型。
        2. return_layers (Dict[name, new_name]): 包含模块名称和将作为返回值的激活名称的字典。
    """
    _version = 2
    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        """
            set()：用于创建一个集合（set）对象的内置函数
            set.issubset(): 用于检查一个集合是否是另一个集合的子集
        """
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        
        # 保留原始的 return_layers
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}

        # 重新构建backbone，将没有使用到的模块全部删掉
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


class LRASPP(nn.Module):
    """
    Implements a Lite R-ASPP Network for semantic segmentation from
    `"Searching for MobileNetV3"
    <https://arxiv.org/abs/1905.02244>`_.

    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "high" for the high level feature map and "low" for the low level feature map.
        low_channels (int): the number of channels of the low level features.
        high_channels (int): the number of channels of the high level features.
        num_classes (int): number of output classes of the model (including the background).
        inter_channels (int, optional): the number of channels for intermediate computations.

    一个实现了 Lite R-ASPP（轻量级的空洞空间金字塔池化）网络用于语义分割的模型。该模型由论文 “Searching for MobileNetV3” 提出。

    参数:
        backbone (nn.Module): 用于计算模型特征的网络。该网络应该返回一个 OrderedDict[Tensor]，其中键为 “high” 对应高级特征图，“low” 对应低级特征图。
        low_channels (int): 低级特征的通道数。
        high_channels (int): 高级特征的通道数。
        num_classes (int): 模型的输出类别数（包括背景）。
        inter_channels (int, optional): 中间计算的通道数。
    """
    __constants__ = ['aux_classifier']

    def __init__(self,
                 backbone: nn.Module,
                 low_channels: int,
                 high_channels: int,
                 num_classes: int,
                 inter_channels: int = 128) -> None:
        super(LRASPP, self).__init__()
        
        # 构建 MobileNet v3 LR-ASPP 的 Backbone
        self.backbone = backbone

        # 构建 MobileNet v3 LR-ASPP 的 LR-ASPP 部分
        self.classifier = LRASPPHead(low_channels, high_channels, num_classes, inter_channels)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        input_shape = x.shape[-2:]  # [H, W]
        features = self.backbone(x)
        out = self.classifier(features)  # F5
        out = F.interpolate(out, size=input_shape, mode="bilinear", align_corners=False)  # F_out

        result = OrderedDict()
        result["out"] = out

        return result


class LRASPPHead(nn.Module):
    def __init__(self,
                 low_channels: int,
                 high_channels: int,
                 num_classes: int,
                 inter_channels: int) -> None:
        super(LRASPPHead, self).__init__()
        
        # 右分支（1x1卷积）
        self.cbr = nn.Sequential(
            nn.Conv2d(high_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True)
        )
        
        # 左分支（AvgPool） —— 类似于 SE 注意力模块
        self.scale = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(high_channels, inter_channels, 1, bias=False),
            nn.Sigmoid()
        )
        
        self.low_classifier = nn.Conv2d(low_channels, num_classes, 1)
        self.high_classifier = nn.Conv2d(inter_channels, num_classes, 1)

    def forward(self, inputs: Dict[str, Tensor]) -> Tensor:
        low = inputs["low"]
        high = inputs["high"]

        x = self.cbr(high)  # F1
        s = self.scale(high)  # F2
        x = x * s
        x = F.interpolate(x, size=low.shape[-2:], mode="bilinear", align_corners=False)  # F3

        # F5
        return self.low_classifier(low) + self.high_classifier(x)


def lraspp_mobilenetv3_large(num_classes=21, pretrain_backbone=False):
    # 'mobilenetv3_large_imagenet': 'https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth'
    # 'lraspp_mobilenet_v3_large_coco': 'https://download.pytorch.org/models/lraspp_mobilenet_v3_large-d234d4ea.pth'
    backbone = mobilenet_v3_large(dilated=True)

    if pretrain_backbone:
        # 载入mobilenetv3 large backbone预训练权重
        backbone.load_state_dict(torch.load("mobilenet_v3_large.pth", map_location='cpu'))

    backbone = backbone.features

    # Gather the indices of blocks which are strided. These are the locations of C1, ..., Cn-1 blocks.
    # The first and last blocks are always included because they are the C0 (conv1) and Cn. 第一个和最后一个块始终被包括在内，因为它们分别是C0（conv1）和Cn
    """
        记录一下会进行下采样层的索引：
            [0]：第一个是3x3卷积层
            [i for i, b in enumerate(backbone) if getattr(b, "is_strided", False)]：Bneck中会进行下采样的Bneck索引
            [len(backbone) - 1]：最后一层（Conv2d）
    """
    stage_indices = [0] + [i for i, b in enumerate(backbone) if getattr(b, "is_strided", False)] + [len(backbone) - 1]  # getattr()中，False表示当属性不存在时的默认返回值。

    # 取出倒数第四个会进行下采样的层的索引给 low_pos （这里是以原始MobileNet v3结构为基准的） —— 8 倍下采样
    low_pos = stage_indices[-4]  # use C2 here which has output_stride = 8
    # 取出倒数第一个会进行下采样的层的索引给 high_pos（这里是以原始MobileNet v3结构为基准的） —— 16 倍下采样（Backbone的输出）
    high_pos = stage_indices[-1]  # use C5 which has output_stride = 16
    
    # 将 low_pos 和 high_pos 传入 Backbone中就可以获取对应的层结构，并继续获取它们的输出通道数
    low_channels = backbone[low_pos].out_channels
    high_channels = backbone[high_pos].out_channels

    return_layers = {str(low_pos): "low", str(high_pos): "high"}
    
    # 重构Backbone
    """
        此时，backbone为 Dict[str, Tensor]，即有两个输出特征图：
            1. 8 倍下采样的特征图 —— low
            2. Backbone主分支的输出特征图（16 倍下采样） —— high
    """
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = LRASPP(backbone, low_channels, high_channels, num_classes)
    return model
