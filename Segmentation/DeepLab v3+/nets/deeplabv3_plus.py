import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.xception import xception
from nets.mobilenetv2 import mobilenetv2


class MobileNetV2(nn.Module):
    def __init__(self, downsample_factor=8, pretrained=True):
        super(MobileNetV2, self).__init__()
        from functools import partial

        model = mobilenetv2(pretrained)
        
        # 去除逆残差结构后面的1×1卷积
        self.features = model.features[:-1]

        self.total_idx = len(self.features)  # Block的数量
        self.down_idx = [2, 4, 7, 14]  # 每个进行下采样逆残差结构在模型中所处的idx

        
        """
            partial 函数说明：
                from functools import partial

                # 原始函数
                def add(x, y):
                    return x + y

                # 使用 partial 部分应用 add 函数的第一个参数为 5
                add_5 = partial(add, 5)

                # 调用新的函数 add_5 只需要提供第二个参数
                result = add_5(10)  # 实际调用 add(5, 10)

                print(result)  # 输出: 15

            在上面的例子中，partial 函数将 add 函数的第一个参数固定为 5，然后返回一个新的函数 add_5。
            当我们调用 add_5(10) 时，它实际上等同于调用 add(5, 10)，返回结果为 15。
            这个功能在某些情况下可以使代码更加简洁和易读，并且使得函数的复用更加方便。
        """
        if downsample_factor == 8:
            """
                如果下采样倍数为8，则先对倒数两个需要下采样的Block进行参数修改，使其stride=1、dilate=2
                如果下采样倍数为16，则先对最后一个Block进行参数修改，使其stride=1、dilate=2
            """
            # 修改倒数两个Block
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2))  # 修改stride=1, dilate=2
            # 修改剩下的所有block，使其都使用膨胀卷积
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=4))
                
        elif downsample_factor == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )

    def _nostride_dilate(self, m, dilate):
        """修改返回的block，使其stride=1

        Args:
            m (str): 模块的名称
            dilate (int): 膨胀系数
        """
        classname = m.__class__.__name__  # 获取模块名称
        if classname.find('Conv') != -1:  # 如果有卷积层
            if m.stride == (2, 2):  # 如果卷积层的步长为2
                m.stride = (1, 1)  # 修改步长为1
                if m.kernel_size == (3, 3):  # 修改对应的膨胀系数和padding
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            else:  # 如果卷积层步长本来就是1
                if m.kernel_size == (3, 3):  # 修改对应的膨胀系数和padding
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        """前向推理

        Args:
            x (tensor): 输入特征图

        Returns:
            (tensor, tensor): 输出特征图（两个）
        """
        low_level_features = self.features[:4](x)  # 浅层的特征图
        x = self.features[4:](low_level_features)  # 经过完整Backbone的特征图
        return low_level_features, x


class ASPP(nn.Module):
    """ASPP特征提取模块：利用不同膨胀率的膨胀卷积进行特征提取

    Args:
        nn (_type_): _description_
    """
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super(ASPP, self).__init__()
        self.branch1 = nn.Sequential(  # 1×1 普通卷积
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0,
                      dilation=rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(  # 3×3膨胀卷积(r=6)
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=6 *
                      rate, dilation=6*rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(  # 3×3膨胀卷积(r=12)
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=12 *
                      rate, dilation=12*rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(  # 3×3膨胀卷积(r=18)
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=18 *
                      rate, dilation=18*rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        
        """
            在论文中，这里应该是池化层，但这里定义为普通卷积层，
            但莫慌，在forward函数中先进行池化再进行卷积（相当于增加了一个后置卷积）
        """
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
        self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)

        self.conv_cat = nn.Sequential(  # concat之后需要用到的1×1卷积
            nn.Conv2d(dim_out*5, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        [b, c, row, col] = x.size()  # [BS, C, H, W]
        
        # 先进行前4个分支
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        
        # 第五个分支：全局平均池化+卷积
        global_feature = torch.mean(input=x, dim=2, keepdim=True)  # 沿着H进行mean
        global_feature = torch.mean(input=global_feature, dim=3, keepdim=True)  # 再沿着W进行mean
        
        # 经典汉堡包卷积结构
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        
        # 双线性插值使其回复到输入特征图的shape
        global_feature = F.interpolate(
            input=global_feature, size=(row, col), scale_factor=None, mode='bilinear', align_corners=True)

        # 沿通道方向将五个分支的内容堆叠起来
        feature_cat = torch.cat(
            [conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        
        # 最后经过1×1卷积调整通道数
        result = self.conv_cat(feature_cat)
        return result


class DeepLab(nn.Module):
    def __init__(self, num_classes, backbone="mobilenet", pretrained=True, downsample_factor=16):
        super(DeepLab, self).__init__()
        if backbone == "xception":
            """
            获得两个特征层
                1. 浅层特征    [128,128,256]
                2. 主干部分    [30,30,2048]
            """
            self.backbone = xception(
                downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 2048
            low_level_channels = 256
        elif backbone == "mobilenet":
            """
            获得两个特征层
                1. 浅层特征    [128,128,24
                2. 主干部分    [30,30,320]
            """
            self.backbone = MobileNetV2(
                downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 320
            low_level_channels = 24
        else:
            raise ValueError(
                'Unsupported backbone - `{}`, Use mobilenet, xception.'.format(backbone))

        # ASPP特征提取模块：利用不同膨胀率的膨胀卷积进行特征提取
        self.aspp = ASPP(dim_in=in_channels, dim_out=256,
                         rate=16//downsample_factor)

        # 浅层特征图
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        self.cat_conv = nn.Sequential(
            nn.Conv2d(48+256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Dropout(0.1),
        )
        
        # 最后的1×1卷积，目的是调整输出特征图的通道数，调整为 num_classes
        self.cls_conv = nn.Conv2d(256, num_classes, 1, stride=1)

    def forward(self, x):
        H, W = x.size(2), x.size(3)
        
        low_level_features, x = self.backbone(x)  # 浅层特征-进行卷积处理
        x = self.aspp(x)  # 主干部分-利用ASPP结构进行加强特征提取
        
        # 先利用1×1卷积对浅层特征图进行通道数的调整
        low_level_features = self.shortcut_conv(low_level_features)

        # 先对深层特征图进行上采样
        x = F.interpolate(x, size=(low_level_features.size(
            2), low_level_features.size(3)), mode='bilinear', align_corners=True)
        # 再进行堆叠
        x = self.cat_conv(torch.cat((x, low_level_features), dim=1))

        # 最后使用3×3卷积进行特征提取
        x = self.cls_conv(x)
        
        # 上采样得到和原图一样大小的特征图
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x
