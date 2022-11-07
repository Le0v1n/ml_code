"""
original code from apple:
https://github.com/apple/ml-cvnets/blob/main/cvnets/models/classification/mobilevit.py
"""

from typing import Optional, Tuple, Union, Dict
import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from transformer import TransformerEncoder
from model_config import get_config


def make_divisible(
        v: Union[float, int],
        divisor: Optional[int] = 8,
        min_value: Optional[Union[float, int]] = None,
) -> Union[float, int]:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvLayer(nn.Module):
    """
    Applies a 2D convolution over an input

    Args:
        in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H_{in}, W_{in})`
        out_channels (int): :math:`C_{out}` from an expected output of size :math:`(N, C_{out}, H_{out}, W_{out})`
        kernel_size (Union[int, Tuple[int, int]]): Kernel size for convolution.
        stride (Union[int, Tuple[int, int]]): Stride for convolution. Default: 1
        groups (Optional[int]): Number of groups in convolution. Default: 1
        bias (Optional[bool]): Use bias. Default: ``False``
        use_norm (Optional[bool]): Use normalization layer after convolution. Default: ``True``
        use_act (Optional[bool]): Use activation layer after convolution (or convolution and normalization).
                                Default: ``True``

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})`

    .. note::
        For depth-wise convolution, `groups=C_{in}=C_{out}`.
    """

    def __init__(
            self,
            in_channels: int,  # 输入特征图通道数
            out_channels: int,  # 输出特征图通道数
            kernel_size: Union[int, Tuple[int, int]],  # 卷积核大小(k, k)
            stride: Optional[Union[int, Tuple[int, int]]] = 1,  # 步长(s, s)
            groups: Optional[int] = 1,  # 组卷积为1（默认为普通卷积）
            bias: Optional[bool] = False,  # 卷积中是否使用偏置
            use_norm: Optional[bool] = True,  # 是否使用BN
            use_act: Optional[bool] = True,  # 是否使用激活函数
            se: Optional[bool] = False  # 是否使用SE注意力
    ) -> None:  # 无返回值
        super().__init__()

        # 检查kernel_size和stride是否为int，如果是则将其扩充为tuple
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        if isinstance(stride, int):
            stride = (stride, stride)

        # 判断kernel_size和stride是否为tuple
        assert isinstance(kernel_size, Tuple)
        assert isinstance(stride, Tuple)

        # 当kernel_size为奇数时，不进行下采样
        padding = (
            int((kernel_size[0] - 1) / 2),
            int((kernel_size[1] - 1) / 2),
        )

        # 创建block容器
        block = nn.Sequential()

        # 构建卷积层
        conv_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
            padding=padding,
            bias=bias,
        )

        # 将卷积层添加到block容器中
        block.add_module(name="conv", module=conv_layer)

        # 判断是否使用BN和act，如果使用则往block中添加对应的模块
        if use_norm:
            norm_layer = nn.BatchNorm2d(num_features=out_channels, momentum=0.1)  # 创建BN层
            block.add_module(name="norm", module=norm_layer)  # 添加BN层

        if se:
            block.add_module(name="se attention", module=SEModule(channel=out_channels, reduction=4))

        if use_act:
            act_layer = nn.SiLU()  # 创建act层
            block.add_module(name="act", module=act_layer)  # 添加act层

        # 将block赋值给实例
        self.block = block

    # 前向推理函数
    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Hardsigmoid()
            # nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Identity(nn.Module):
    def __init__(self, channel):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class InvertedResidual(nn.Module):
    """
    This class implements the inverted residual block, as described in `MobileNetv2 <https://arxiv.org/abs/1801.04381>`_ paper

    Args:
        in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H_{in}, W_{in})`
        out_channels (int): :math:`C_{out}` from an expected output of size :math:`(N, C_{out}, H_{out}, W_{out)`
        stride (int): Use convolutions with a stride. Default: 1
        expand_ratio (Union[int, float]): Expand the input channels by this factor in depth-wise conv
        skip_connection (Optional[bool]): Use skip-connection. Default: True

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})`

    .. note::
        If `in_channels =! out_channels` and `stride > 1`, we set `skip_connection=False`

    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int,
            expand_ratio: Union[int, float],  # 逆残差结构的第一个点卷积升维的倍数
            skip_connection: Optional[bool] = True,  # 是否使用shortcut
    ) -> None:

        # 判断步长是否为 1 or 2（1表示不进行下采样，2表示进行2倍下采样）
        assert stride in [1, 2]

        # 计算1×1卷积具体的升维数（通道数必须可以被8整除）
        hidden_dim = make_divisible(int(round(in_channels * expand_ratio)), 8)

        super().__init__()

        block = nn.Sequential()

        # 如果expand_ratio ≠ 1时，说明需要进行通道升维，那么将1×1卷积添加到block中
        if expand_ratio != 1:
            block.add_module(
                name="exp_1x1",
                module=ConvLayer(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    kernel_size=1,  # 1×1卷积核
                    groups=1,
                    bias=False,
                    use_norm=True,
                    use_act=True
                ),
            )
        if stride == 1:
            block.add_module(
                name="conv_3x3",
                module=ConvLayer(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    stride=stride,
                    kernel_size=3,
                    groups=hidden_dim,  # 输入通道数和分组数相同，说明使用的是深度卷积（Depth-wise Convolution）
                    bias=False,
                    use_norm=True,
                    use_act=True,
                    se=True
                ),
            )
        else:
            block.add_module(
                name="conv_3x3",
                module=ConvLayer(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    stride=stride,
                    kernel_size=3,
                    groups=hidden_dim,  # 输入通道数和分组数相同，说明使用的是深度卷积（Depth-wise Convolution）
                    bias=False,
                    use_norm=True,
                    use_act=True,
                    se=False
                ),
            )

        block.add_module(
            name="red_1x1",
            module=ConvLayer(
                in_channels=hidden_dim,
                out_channels=out_channels,
                kernel_size=1,
                use_act=False,  # 不使用激活函数
                use_norm=True,  # 使用BN
            ),
        )

        # 为实例添加attribute
        self.block = block
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.exp = expand_ratio
        self.stride = stride
        # 判断是否使用shortcut（步长为1且输入输出通道数相同）
        self.use_res_connect = (
                self.stride == 1 and in_channels == out_channels and skip_connection
        )

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)


class MobileViTBlock(nn.Module):
    """
    This class defines the `MobileViT block <https://arxiv.org/abs/2110.02178?context=cs.LG>`_

    Args:
        opts: command line arguments
        in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H, W)`
        transformer_dim (int): Input dimension to the transformer unit
        ffn_dim (int): Dimension of the FFN block
        n_transformer_blocks (int): Number of transformer blocks. Default: 2
        head_dim (int): Head dimension in the multi-head attention. Default: 32
        attn_dropout (float): Dropout in multi-head attention. Default: 0.0
        dropout (float): Dropout rate. Default: 0.0
        ffn_dropout (float): Dropout between FFN layers in transformer. Default: 0.0
        patch_h (int): Patch height for unfolding operation. Default: 8
        patch_w (int): Patch width for unfolding operation. Default: 8
        transformer_norm_layer (Optional[str]): Normalization layer in the transformer block. Default: layer_norm
        conv_ksize (int): Kernel size to learn local representations in MobileViT block. Default: 3
        no_fusion (Optional[bool]): Do not combine the input and output feature maps. Default: False
    """

    def __init__(
            self,
            in_channels: int,  # 输入block的通道数
            transformer_dim: int,  # 输入Transformer Encoder中每个token的序列长度
            ffn_dim: int,  # Transformer Encoder中的MLP结构的第一个全连接层节点的个数
            n_transformer_blocks: int = 2,  # Transformer Encoder的重复次数
            head_dim: int = 32,  # Transformer Encoder中的Multi-head Self-Attention时，每个head所对应的维度
            attn_dropout: float = 0.0,
            dropout: float = 0.0,
            ffn_dropout: float = 0.0,
            patch_h: int = 8,  # patch的高度
            patch_w: int = 8,  # patch的宽度
            conv_ksize: Optional[int] = 3,  # 局部表征建模所用的卷积核大小
            *args,
            **kwargs
    ) -> None:
        super().__init__()

        """
            构建Local representation时所用到的3×3卷积和1×1卷积
        """
        conv_3x3_in = ConvLayer(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=conv_ksize,
            stride=1
        )
        conv_1x1_in = ConvLayer(
            in_channels=in_channels,
            out_channels=transformer_dim,
            kernel_size=1,
            stride=1,
            use_norm=False,
            use_act=False
        )

        """
            构建Fusion中所用到的卷积
        """
        conv_1x1_out = ConvLayer(
            in_channels=transformer_dim,
            out_channels=in_channels,
            kernel_size=1,
            stride=1
        )
        conv_3x3_out = ConvLayer(
            in_channels=2 * in_channels,
            out_channels=in_channels,
            kernel_size=conv_ksize,
            stride=1
        )

        # 构建Local representation
        self.local_rep = nn.Sequential()
        self.local_rep.add_module(name="conv_3x3", module=conv_3x3_in)
        self.local_rep.add_module(name="conv_1x1", module=conv_1x1_in)

        # 判断输入Transformer Encoder的通道数是否可以被每个head对应维度整除
        assert transformer_dim % head_dim == 0
        # 计算得到MSA中head的个数
        num_heads = transformer_dim // head_dim

        # 构建Global Representation
        global_rep = [  # 并非Sequential模块，而是list
            TransformerEncoder(
                embed_dim=transformer_dim,
                ffn_latent_dim=ffn_dim,
                num_heads=num_heads,
                attn_dropout=attn_dropout,
                dropout=dropout,
                ffn_dropout=ffn_dropout
            )
            for _ in range(n_transformer_blocks)  # 构建多个Transformer Encoder模块
        ]
        # 添加LayerNorm层
        global_rep.append(nn.LayerNorm(transformer_dim))
        # 将list解包转换为Sequential模块
        self.global_rep = nn.Sequential(*global_rep)

        self.conv_proj = conv_1x1_out
        self.fusion = conv_3x3_out

        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_area = self.patch_w * self.patch_h  # patch的面积

        self.cnn_in_dim = in_channels
        self.cnn_out_dim = transformer_dim
        self.n_heads = num_heads
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.ffn_dropout = ffn_dropout
        self.n_blocks = n_transformer_blocks
        self.conv_ksize = conv_ksize

    # Global Representation中的Unfold：按照不同的颜色将token组成新的序列
    def unfolding(self, x: Tensor) -> Tuple[Tensor, Dict]:
        # 获取patch信息并计算patch的面积
        patch_w, patch_h = self.patch_w, self.patch_h
        patch_area = patch_w * patch_h

        # 获取特征图不同维度的信息
        batch_size, in_channels, orig_h, orig_w = x.shape

        # 重新计算特征图的h和w以确保特征图可以被完成地划分为patches(ceil: 向上取整)
        new_h = int(math.ceil(orig_h / self.patch_h) * self.patch_h)
        new_w = int(math.ceil(orig_w / self.patch_w) * self.patch_w)

        interpolate = False  # 是否进行插值

        # 如果计算出来的h,w和原本的h,w不相同，则进行插值，以保证特征图可以被完成地划分为patches
        if new_w != orig_w or new_h != orig_h:
            # Note: Padding can be done, but then it needs to be handled in attention function.
            # x: 输入大小; size:输出大小
            x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
            interpolate = True  # 因为进行了插值，所以将这个标志位设置为True

        # 计算宽度和高度方向的patch个数
        # number of patches along width and height
        num_patch_w = new_w // patch_w  # n_w
        num_patch_h = new_h // patch_h  # n_h

        # 计算出总共的patch个数
        num_patches = num_patch_h * num_patch_w  # N

        """
            将相同位置的token抽离出来组成新的序列
                n_h: 高度方向上patch的个数
                patch_h: 高度方向上patch的大小
                n_w: 宽度方向上patch的个数
                patch_w: 宽度方向上patch的大小
                
                P: patch的面积
                N: patch的总数（不同序列token的个数）
                C: 每个token的向量长度
        """
        # [B, C, H, W] -> [B * C * n_h, p_h, n_w, p_w] 将H和W按照patch进行拆分
        x = x.reshape(batch_size * in_channels * num_patch_h, patch_h, num_patch_w, patch_w)

        # [B * C * n_h, p_h, n_w, p_w] -> [B * C * n_h, n_w, p_h, p_w] 将n_h和n_w放在一起，p_h和p_w放在一起
        x = x.transpose(1, 2)

        # [B * C * n_h, n_w, p_h, p_w] -> [B, C, N, P] where P = p_h * p_w and N = n_h * n_w
        # n_h * n_w = N; p_h * p_w = P
        x = x.reshape(batch_size, in_channels, num_patches, patch_area)

        # [B, C, N, P] -> [B, P, N, C]
        x = x.transpose(1, 3)

        # [B, P, N, C] -> [BP, N, C] | BP维度相对于是并行计算
        x = x.reshape(batch_size * patch_area, num_patches, -1)

        # 创建字典记录一些信息
        info_dict = {
            "orig_size": (orig_h, orig_w),  # 原始特征图的size
            "batch_size": batch_size,  # batch的大小
            "interpolate": interpolate,  # 是否进行了插值
            "total_patches": num_patches,  # patch的总个数
            "num_patches_w": num_patch_w,  # 宽度方向上patch的个数
            "num_patches_h": num_patch_h,  # 高度方向上patch的个数
        }

        return x, info_dict

    # Global Representation中的Fold：将序列返回为特征图
    def folding(self, x: Tensor, info_dict: Dict) -> Tensor:
        n_dim = x.dim()  # 获取tensor的维度个数
        # 判断是否为BP*N*C这样的三维度分布
        assert n_dim == 3, "Tensor should be of shape BPxNxC. Got: {}".format(x.shape)

        # [BP, N, C] --> [B, P, N, C] 拆解tensor维度
        x = x.contiguous().view(
            info_dict["batch_size"], self.patch_area, info_dict["total_patches"], -1
        )

        batch_size, pixels, num_patches, channels = x.size()
        num_patch_h = info_dict["num_patches_h"]
        num_patch_w = info_dict["num_patches_w"]

        # Unfold中的逆操作
        # [B, P, N, C] -> [B, C, N, P]
        x = x.transpose(1, 3)

        # [B, C, N, P] -> [B*C*n_h, n_w, p_h, p_w]
        x = x.reshape(batch_size * channels * num_patch_h, num_patch_w, self.patch_h, self.patch_w)

        # [B*C*n_h, n_w, p_h, p_w] -> [B*C*n_h, p_h, n_w, p_w]
        x = x.transpose(1, 2)

        # [B*C*n_h, p_h, n_w, p_w] -> [B, C, H, W]
        x = x.reshape(batch_size, channels, num_patch_h * self.patch_h, num_patch_w * self.patch_w)

        # 如果在Unfold中进行插值，那么需要还原
        if info_dict["interpolate"]:
            x = F.interpolate(
                x,
                size=info_dict["orig_size"],
                mode="bilinear",
                align_corners=False,
            )
        return x

    def forward(self, x: Tensor) -> Tensor:
        res = x

        fm = self.local_rep(x)  # fm: feature map

        # 进行Unfold操作
        # convert feature map to patches
        patches, info_dict = self.unfolding(fm)

        # 进行Global Representation操作
        # learn global representations
        for transformer_layer in self.global_rep:
            patches = transformer_layer(patches)

        # 再进行Fold操作，变为原特征图形式
        # [B x Patch x Patches x C] -> [B x C x Patches x Patch]
        fm = self.folding(x=patches, info_dict=info_dict)

        # 经过Fusion中的1×1卷积
        fm = self.conv_proj(fm)

        # 先经过Fusion中shortcut，再经过3×3卷积，得到MobileViT block的输出
        fm = self.fusion(torch.cat((res, fm), dim=1))
        return fm


class MobileViT(nn.Module):
    """
    This class implements the `MobileViT architecture <https://arxiv.org/abs/2110.02178?context=cs.LG>`_
    """

    def __init__(self, model_cfg: Dict, num_classes: int = 1000):
        super().__init__()

        image_channels = 3  # 输入图片的通道数
        out_channels = 16  # stem（conv_1）过后特征图的通道数

        self.conv_1 = ConvLayer(
            in_channels=image_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2  # 进行两倍下采样
        )

        # 构建所有的Layer
        self.layer_1, out_channels = self._make_layer(input_channel=out_channels, cfg=model_cfg["layer1"])
        self.layer_2, out_channels = self._make_layer(input_channel=out_channels, cfg=model_cfg["layer2"])
        self.layer_3, out_channels = self._make_layer(input_channel=out_channels, cfg=model_cfg["layer3"])
        self.layer_4, out_channels = self._make_layer(input_channel=out_channels, cfg=model_cfg["layer4"])
        self.layer_5, out_channels = self._make_layer(input_channel=out_channels, cfg=model_cfg["layer5"])

        exp_channels = min(model_cfg["last_layer_exp_factor"] * out_channels, 960)
        # MobileViT模型结构中的最后一个1x1卷积层
        self.conv_1x1_exp = ConvLayer(
            in_channels=out_channels,
            out_channels=exp_channels,
            kernel_size=1
        )

        # MobileViT模型结构中的最后一个层: 全局池化和FC层
        self.classifier = nn.Sequential()
        self.classifier.add_module(name="global_pool", module=nn.AdaptiveAvgPool2d(1))
        self.classifier.add_module(name="flatten", module=nn.Flatten())

        # 根据模型配置文件决定是否添加Dropout层
        if 0.0 < model_cfg["cls_dropout"] < 1.0:
            self.classifier.add_module(name="dropout", module=nn.Dropout(p=model_cfg["cls_dropout"]))

        # 添加全连接层
        self.classifier.add_module(name="fc", module=nn.Linear(in_features=exp_channels, out_features=num_classes))

        # 参数初始化
        # weight init
        self.apply(self.init_parameters)

    def _make_layer(self, input_channel, cfg: Dict) -> Tuple[nn.Sequential, int]:
        """
            input_channel: 输入通道数
            cfg: 详细配置(输入的是一个dict)

            return: 一个tuple: (容器, int)

            Note:
                dict.get(key, default=None)
        """
        # 如果有block_type则返回，没有返回mobilevit
        block_type = cfg.get("block_type", "mobilevit")

        if block_type.lower() == "mobilevit":  # 创建MobileViT block
            return self._make_mit_layer(input_channel=input_channel, cfg=cfg)
        else:  # 创建MV2模块
            return self._make_mobilenet_layer(input_channel=input_channel, cfg=cfg)

    @staticmethod
    def _make_mobilenet_layer(input_channel: int, cfg: Dict) -> Tuple[nn.Sequential, int]:
        output_channels = cfg.get("out_channels")
        num_blocks = cfg.get("num_blocks", 2)
        expand_ratio = cfg.get("expand_ratio", 4)
        block = []

        for i in range(num_blocks):
            # 对于MV2模块而言，只有第一个MV2模块会考虑下采样，其他的MV2模块的stride=1，不会进行下采样
            stride = cfg.get("stride", 1) if i == 0 else 1  # 当i == 0时，stride = cfg.get("stride", 1)，否则stride=1

            layer = InvertedResidual(
                in_channels=input_channel,
                out_channels=output_channels,
                stride=stride,
                expand_ratio=expand_ratio
            )
            block.append(layer)
            input_channel = output_channels

        return nn.Sequential(*block), input_channel

    @staticmethod
    def _make_mit_layer(input_channel: int, cfg: Dict) -> [nn.Sequential, int]:
        stride = cfg.get("stride", 1)  # 如果cfg中有stride, 则返回对应值，否则返回1
        block = []

        if stride == 2:  # 如果进行下采样，则会创建一个MV2模块进行下采样
            layer = InvertedResidual(
                in_channels=input_channel,
                out_channels=cfg.get("out_channels"),  # 不存在返回None
                stride=stride,
                expand_ratio=cfg.get("mv_expand_ratio", 4)  # 不存在返回4
            )

            block.append(layer)
            input_channel = cfg.get("out_channels")  # 不存在返回None

        # 读取构建MobileViT block所用的信息
        transformer_dim = cfg["transformer_channels"]  # 不存在返回None
        ffn_dim = cfg.get("ffn_dim")  # 不存在返回None
        num_heads = cfg.get("num_heads", 4)  # 不存在返回4
        head_dim = transformer_dim // num_heads

        if transformer_dim % head_dim != 0:
            raise ValueError("Transformer input dimension should be divisible by head dimension. "
                             "Got {} and {}.".format(transformer_dim, head_dim))

        # 构建MobileViT block模块
        block.append(MobileViTBlock(
            in_channels=input_channel,
            transformer_dim=transformer_dim,
            ffn_dim=ffn_dim,
            n_transformer_blocks=cfg.get("transformer_blocks", 1),
            patch_h=cfg.get("patch_h", 2),
            patch_w=cfg.get("patch_w", 2),
            dropout=cfg.get("dropout", 0.1),
            ffn_dropout=cfg.get("ffn_dropout", 0.0),
            attn_dropout=cfg.get("attn_dropout", 0.1),
            head_dim=head_dim,
            conv_ksize=3
        ))

        return nn.Sequential(*block), input_channel

    @staticmethod
    def init_parameters(m):
        if isinstance(m, nn.Conv2d):
            if m.weight is not None:
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.Linear,)):
            if m.weight is not None:
                nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        else:
            pass

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_1(x)

        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)

        x = self.conv_1x1_exp(x)
        x = self.classifier(x)
        return x


def mobile_vit_xx_small(num_classes: int = 1000):
    # pretrain weight link
    # https://docs-assets.developer.apple.com/ml-research/models/cvnets/classification/mobilevit_xxs.pt
    config = get_config("xx_small")
    m = MobileViT(config, num_classes=num_classes)
    return m


def mobile_vit_x_small(num_classes: int = 1000):
    # pretrain weight link
    # https://docs-assets.developer.apple.com/ml-research/models/cvnets/classification/mobilevit_xs.pt
    config = get_config("x_small")
    m = MobileViT(config, num_classes=num_classes)
    return m


def mobile_vit_small(num_classes: int = 1000):
    # pretrain weight link
    # https://docs-assets.developer.apple.com/ml-research/models/cvnets/classification/mobilevit_s.pt
    config = get_config("small")
    m = MobileViT(config, num_classes=num_classes)
    return m


if __name__ == '__main__':
    import time
    from torch.autograd import Variable
    import numpy as np

    model = mobile_vit_xx_small(num_classes=27)
    print(model)
    model.eval()

    count = 0
    use_cuda = True
    evaluate_fps = True
    count_num = 500 if use_cuda else 200

    if evaluate_fps:
        device = "cuda" if use_cuda else "cpu"
        model.to(device)
        fps_ls = []
        while True:
            t1 = time.time()
            input_var = Variable(torch.randn(1, 3, 224, 224)).to(device)
            output = model(input_var)
            fps_ls.append(1 / (time.time() - t1))
            print(f"[{device}] Average FPS: {np.average(fps_ls):.4f}")
            count += 1
            if count == count_num:
                break
