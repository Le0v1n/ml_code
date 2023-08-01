import numpy as np
from PIL import Image

#---------------------------------------------------------#
#   将图像转换成RGB图像，防止灰度图在预测时报错。
#   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
#---------------------------------------------------------#
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 


def resize_image(image, size):
    """将给定的图像进行调整大小并居中放置在新的画布上

    Args:
        image (_type_): 输入图片
        size (_type_): 目标大小

    Returns:
        (elem1, elem2, elem3): (裁剪后的图片, 缩放后的图像宽度, 缩放后的图像高度)
    """
    iw, ih  = image.size  # 获取图片大小
    w, h    = size  # 获取目标大小

    scale   = min(w/iw, h/ih)  # 计算了原始图像与目标大小之间的缩放比例。
    nw      = int(iw*scale)  # 计算了缩放后的图像宽度
    nh      = int(ih*scale)  # 计算了缩放后的图像高度

    # 使用 Pillow 的 resize() 方法将图像调整为缩放后的大小。
    # 使用 BICUBIC 插值方法进行图像的重采样，以获得更平滑的结果。
    image   = image.resize((nw,nh), Image.BICUBIC)
    
    # 创建了一个新的画布，用于放置调整后的图像。
    # 画布大小与目标大小相同，并且以灰色 (128, 128, 128) 作为默认背景色。
    new_image = Image.new('RGB', size, (128,128,128))
    
    # 将调整后的图像粘贴到新的画布上。图像会居中放置在画布上
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))

    return new_image, nw, nh
    
#---------------------------------------------------#
#   获得学习率
#---------------------------------------------------#
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def preprocess_input(image):
    image /= 255.0
    return image

def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)

def download_weights(backbone, model_dir="./model_data"):
    import os
    from torch.hub import load_state_dict_from_url
    
    download_urls = {
        'mobilenet' : 'https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/mobilenet_v2.pth.tar',
        'xception'  : 'https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/xception_pytorch_imagenet.pth',
    }
    url = download_urls[backbone]
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    load_state_dict_from_url(url, model_dir)