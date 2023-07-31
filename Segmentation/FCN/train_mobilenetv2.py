import os
import time
import datetime
import torch
import re
import copy

from src.fcn_model_mobilenetv2 import fcn_mobilenetv2
from train_utils import train_one_epoch_for_mobilenet, evaluate_for_mobilenetv2, create_lr_scheduler
from my_dataset import VOCSegmentation
import transforms as T
from torch.utils.tensorboard import SummaryWriter

import numpy as np

# 设置PyTorch的随机数种子
seed = 42
torch.manual_seed(seed)

# 如果使用了GPU，还需要设置相应的随机数种子
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# 设置Numpy的随机数种子
np.random.seed(seed)


def get_current_time():

    # 获取当前时间
    current_time = datetime.datetime.now()

    # 将时间格式化为年月日时分秒格式
    formatted_time_a = current_time.strftime("%Y-%m-%d %H:%M:%S")
    return formatted_time_a


# 设置Tensorboard
save_folder_path = f"runs/{get_current_time()}"
if not os.path.exists(save_folder_path):
    os.mkdir(save_folder_path)

tb_writer = SummaryWriter(log_dir=save_folder_path, flush_secs=10)


class SegmentationPresetTrain:
    """
        在[训练]过程中的图像预处理方法
    """
    def __init__(self, base_size, crop_size, hflip_prob=0.5, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        """
        Args:
            base_size: 输入图片大小
            crop_size: 裁剪后的大小
            hflip_prob: 水平翻转的概率
            mean: 均值
            std: 方差
        """
        min_size = int(0.5 * base_size)
        max_size = int(2.0 * base_size)
        # 将图片按照最小边大小等比例缩放
        # trans是一个list，里面存放各种数据增强方法
        trans = [T.RandomResize(min_size, max_size)]

        # 随机水平翻转
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))

        # 进行其他数据增强方式
        trans.extend([
            T.RandomCrop(crop_size),  # 随机裁剪
            T.ToTensor(),  # Tensor化转换
            T.Normalize(mean=mean, std=std),  # 标准化 [0, 255] -> [0, 1]
        ])

        # 将上面的各种预处理方法进行打包，赋值给self.transforms
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetEval:
    """
        在[验证]过程中的图像预处理方法
    """
    def __init__(self, base_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.RandomResize(base_size, base_size),  # 传入的都是base_size
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def get_transform(train):
    base_size = 520
    crop_size = 480

    # 两种数据增强的选择（判断是否处于训练）
    return SegmentationPresetTrain(base_size, crop_size) if train else SegmentationPresetEval(base_size)


def create_model(num_classes, pretrain=True):
    """
    创建模型
    Args:
        aux: 是否使用辅助分类器
        num_classes: 类别数
        pretrain: 是否使用预训练权重

    Returns:
        model：定义好的模型

    """
    model = fcn_mobilenetv2(num_classes=num_classes)

    if pretrain:
        weights_dict = torch.load("pretrained_models/mobilenet_v2.pth", map_location='cpu')
        
        if num_classes != 21:
            # 官方提供的预训练权重是21类(包括背景)
            # 如果训练自己的数据集，将和类别相关的权重删除，防止权重shape不一致报错
            for k in list(weights_dict.keys()):
                if "classifier.4" in k:
                    del weights_dict[k]
        
        # 新建一个空字典用于存放加上前缀后的键值对
        new_weights_dict = {}

        # 遍历原始的weights_dict，并给键加上前缀后，添加到新的字典中
        for key, value in weights_dict.items():
            new_key = "backbone." + key  # 在键前面添加"backbone."前缀
            new_weights_dict[new_key] = value
            
        # 创建一个新的预训练模型字典，仅包含模型自带的权重键值对
        filtered_pretrained_state_dict = copy.deepcopy(new_weights_dict)
        filtered_pretrained_state_dict = {k: v for k, v in filtered_pretrained_state_dict.items() if k in model.state_dict()}
        print(filtered_pretrained_state_dict.keys())
        
        key_to_remove = "backbone.classifier.1.weight", "backbone.classifier.1.bias"
        for key_name in key_to_remove:
            if key_name in filtered_pretrained_state_dict:
                del filtered_pretrained_state_dict[key_name]
                print(F"要删除的键为：{key_name}")

        missing_keys, unexpected_keys = model.load_state_dict(filtered_pretrained_state_dict, strict=False)
        if len(missing_keys) != 0 or len(unexpected_keys) != 0:
            print("missing_keys: ", missing_keys)
            print("-" * 50)
            print("unexpected_keys: ", unexpected_keys)
            print("-" * 50)

        print("-------- Summary ---------")
        # print(model.state_dict()['backbone.features.0.0.weight'] == filtered_pretrained_state_dict['backbone.features.0.0.weight'])
        # raise KeyError
    return model
        


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    # segmentation nun_classes + background
    num_classes = args.num_classes + 1  # 加上背景类别

    # 用来保存训练以及验证过程中信息
    # results_file = "runs/[mobilenetv2]results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    results_file = f"{save_folder_path}/records.txt"

    # VOCdevkit -> VOC2012 -> ImageSets -> Segmentation -> train.txt
    train_dataset = VOCSegmentation(args.data_path,
                                    year="2012",
                                    transforms=get_transform(train=True),
                                    txt_name="train.txt")

    # VOCdevkit -> VOC2012 -> ImageSets -> Segmentation -> val.txt
    val_dataset = VOCSegmentation(args.data_path,
                                  year="2012",
                                  transforms=get_transform(train=False),
                                  txt_name="val.txt")

    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=True,
                                               pin_memory=True,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             num_workers=num_workers,
                                             pin_memory=True,
                                             collate_fn=val_dataset.collate_fn)

    model = create_model(num_classes=num_classes, pretrain=args.pretrained)
    model.to(device)

    """
        backbone：MobileNetv2的backbone
        classifier：FCN Head
    """
    params_to_optimize = [
        {"params": [p for p in model.backbone.parameters() if p.requires_grad]},
        {"params": [p for p in model.classifier.parameters() if p.requires_grad]}
    ]

    optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # 创建学习率更新策略，这里是每个step更新一次(不是每个epoch)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)

    if args.resume:  # 断点续训
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        # 训练
        mean_loss, lr = train_one_epoch_for_mobilenet(model, optimizer, train_loader, device, epoch,
                                        lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler)

        # 验证
        confmat = evaluate_for_mobilenetv2(model, val_loader, device=device, num_classes=num_classes)
        val_info = str(confmat)
        print(val_info)
        
        # write into txt
        with open(results_file, "a") as f:
            # 记录每个epoch对应的train_loss、lr以及验证集各指标
            train_info = f"[epoch: {epoch}]\n" \
                         f"train_loss: {mean_loss:.4f}\n" \
                         f"lr: {lr:.6f}\n"
            f.write(train_info + val_info + "\n\n")
            
            # 使用正则表达式提取每个Epoch的mean IoU和Loss值
            pattern_iou = r"mean IoU: ([\d.]+)"
            mean_iou_values = re.findall(pattern_iou, val_info)
            
        # 添加Tensorboard
        tb_writer.add_scalar("train_loss", mean_loss, epoch)
        tb_writer.add_scalar("val_mean_IoU", float(mean_iou_values[0]), epoch)

        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}
        if args.amp:
            save_file["scaler"] = scaler.state_dict()
        torch.save(save_file, "save_weights/model_{}.pth".format(epoch))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch fcn training")

    parser.add_argument("--data-path", default="/Datasets/", help="VOCdevkit root")
    parser.add_argument("--num-classes", default=20, type=int)  # 不包含背景
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch-size", default=16, type=int)
    parser.add_argument("--epochs", default=100, type=int, metavar="N", help="number of total epochs to train")
    parser.add_argument("--pretrained", default=True, help="use pretrained model to speed up the training")

    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate, default=0.0001')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--resume', default='', help='resume from checkpoint(指向最后一次的权值文件)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='start epoch')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=False, type=bool, help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists("./save_weights"):
        os.mkdir("./save_weights")

    main(args)
