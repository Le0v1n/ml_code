import os
import random
import paddle
import numpy as np
import matplotlib.pyplot as plt
import gzip
import json
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.io as io
import paddle.optimizer as opt
import paddle.metric as metric
import argparse
import PIL
import PIL.Image as Image


# 定义数据集读取器
def load_data(mode="train", batch_size=4):
    print("Loading MNIST dataset form {}...".format(args.dataset_path))
    data = json.load(gzip.open(args.dataset_path))
    print("MNIST Dataset has been loaded!")

    # 对数据集进行划分
    train_set, val_set, test_set = data
    
    img_rows = 28
    img_cols = 28
    
    if mode == "train":
        imgs, labels = train_set[0], train_set[1]
    elif mode == "valid":
        imgs, labels = val_set[0], val_set[1]
    elif mode == "eval":
        imgs, labels = test_set[0], test_set[1]
    else:
        raise Exception("mode can only be one of ['train', 'valid', 'eval']")
    
    # 校验数据
    imgs_length = len(imgs)
    assert len(imgs) == len(labels), "length of train_imgs({}) should be the same as train_labels({})".format(len(imgs), len(labels))
    
    # 定义数据集每个数据的序号，根据序号读取数据
    index_lst = list(range(imgs_length))
    
    
    # 定义数据生成器
    def data_generator():
        if mode == "train":
            random.shuffle(index_lst)
        imgs_lst = []
        labels_lst = []
        
        for i in index_lst:
            # 在深度学习中，常见的数据类型是32位浮点数（float32），因为这种数据类型在数值计算中具有较好的精度和效率
            # 并且在常见的深度学习框架中也是默认的数据类型
            img = np.array(imgs[i]).astype("float32")
            label = np.array(labels[i]).astype("float32")
            
            img = np.reshape(imgs[i], newshape=[1, img_rows, img_cols]).astype("float32")  # [H, W] -> [C, H, W]
            label = np.reshape(labels[i], newshape=[1]).astype("float32")
            
            imgs_lst.append(img)
            labels_lst.append(label)
            
            if len(imgs_lst) == batch_size:
                yield np.array(imgs_lst), np.array(labels_lst)  # 返回一个迭代器
                imgs_lst = []
                labels_lst = []
                
        # 如果剩余数据的数目小于batch size，则剩余数据一起构成一个大小为len(imgs_list)的mini-batch
        if len(imgs_lst) > 0:
            yield np.array(imgs_lst), np.array(labels_lst)
            
    return data_generator


class MNIST_Dataset(io.Dataset):
    """创建一个类MnistDataset，继承paddle.io.Dataset 这个类
        MnistDataset的作用和上面load_data()函数的作用相同，均是构建一个迭代器

    Args:
        io (_type_): _description_
    """
    def __init__(self, mode="train"):
        data = json.load(gzip.open(args.dataset_path))
        
        train_set, val_set, test_set = data
    
        if mode == "train":
            self.imgs, self.labels = train_set[0], train_set[1]
        elif mode == "valid":
            self.imgs, self.labels = val_set[0], val_set[1]
        elif mode == "eval":
            self.imgs, self.labels = test_set[0], test_set[1]
        else:
            raise Exception("mode can only be one of ['train', 'valid', 'eval']")
    
        # 校验数据
        assert len(self.imgs) == len(self.labels), "length of train_imgs({}) should be the same as train_labels({})".format(len(self.imgs), len(self.labels))
        
    def __getitem__(self, idx):
        # img = np.array(self.imgs[idx]).astype('float32')
        # label = np.array(self.labels[idx]).astype('int64')
        img = np.reshape(self.imgs[idx], newshape=[1, 28, 28]).astype("float32")
        label = np.reshape(self.labels[idx], newshape=[1]).astype("int64")
        
        return img, label
    
    def __len__(self):
        return len(self.imgs)


# 全连接层神经网络实现
class MNIST_FC_Model(nn.Layer):  
    def __init__(self):  
        super(MNIST_FC_Model, self).__init__()  
          
        # 定义两层全连接隐含层，输出维度是10，当前设定隐含节点数为10，可根据任务调整  
        self.classifier = nn.Sequential(nn.Linear(in_features=784, out_features=256),
                                        nn.Sigmoid(),
                                        nn.Linear(in_features=256, out_features=64),
                                        nn.Sigmoid())

        # 定义一层全连接输出层，输出维度是1  
        self.head = nn.Linear(in_features=64, out_features=10)  
          
    def forward(self, x):  
        # x.shape: [bath size, 1, 28, 28]
        x = paddle.flatten(x, start_axis=1)  # [bath size, 784]
        x = self.classifier(x)  
        y = self.head(x)
        return y
    
    
# 多层卷积神经网络实现
class MNIST_CNN_Model(nn.Layer):
     def __init__(self):
         super(MNIST_CNN_Model, self).__init__()
         
         self.classifier = nn.Sequential(
             nn.Conv2D( in_channels=1, out_channels=20, kernel_size=5, stride=1, padding=2),
             nn.ReLU(),
             nn.MaxPool2D(kernel_size=2, stride=2),
             nn.Conv2D(in_channels=20, out_channels=20, kernel_size=5, stride=1, padding=2),
             nn.ReLU(),
             nn.MaxPool2D(kernel_size=2, stride=2))
         
         self.head = nn.Linear(in_features=980, out_features=args.num_classes)
         
         
     def forward(self, x):
         # x.shape: [10, 1, 28, 28]
         x = self.classifier(x)  # [bath size, 20, 7, 7]
         x = x.flatten(1)  # [batch size, 980]
         x = self.head(x)  # [batch size, num_classes]
         return x
    
    
def plot_loss_curve(loss_list):
    plt.figure(figsize=(10,5))
    
    freqs = [i for i in range(1, len(loss_list) + 1)]
    # 绘制训练损失变化曲线
    plt.plot(freqs, loss_list, color='#e4007f', label="Train loss")
    
    # 绘制坐标轴和图例
    plt.ylabel("loss", fontsize='large')
    plt.xlabel("epoch", fontsize='large')
    plt.legend(loc='upper right', fontsize='x-large')
    
    plt.savefig(f"train_loss_curve for {args.model_name}_{args.optimizer}.png")
    

class ModelNameError(Exception):
    pass


def evaluation(model: nn.Layer, datasets):
    model.eval()
    
    acc_list = []
    for batch_idx, data in enumerate(datasets()):
        imgs, labels = data
        imgs = paddle.to_tensor(imgs)
        labels = paddle.to_tensor(labels)
        
        pred = model(imgs)
        acc = metric.accuracy(input=pred, label=labels)
        acc_list.append(acc.item()) # type: ignore
        
    # 计算多个batch的平均准确率
    acc_val_mean = np.array(acc_list).mean()
    return acc_val_mean
    
    
def train():
    # 定义模型
    if args.model_name == "FC":
        model = MNIST_FC_Model()
    elif args.model_name == "CNN":
        model = MNIST_CNN_Model()
    else:
        raise ModelNameError("请选择正确的模型(CNN或FC)!")
        
    model.train()
    
    # 加载数据，获取 MNIST 训练数据集
    train_dataset = MNIST_Dataset(mode="train")
    val_dataset = MNIST_Dataset(mode="valid")
    # 使用paddle.io.DataLoader 定义DataLoader对象用于加载Python生成器产生的数据，
    # DataLoader 返回的是一个批次数据迭代器，并且是异步的；
    train_loader = io.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = io.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    
    # 定义 SGD 优化器
    if args.optimizer == "sgd" or "SGD":
        optimizer = opt.SGD(learning_rate=args.lr, parameters=model.parameters())
    elif args.optimizer == "momentum" or "Momentum":
        optimizer = opt.Momentum(learning_rate=args.lr, parameters=model.parameters())
    elif args.optimizer == "adagrad" or "Adagrad":
        optimizer = opt.Adagrad(learning_rate=args.lr, parameters=model.parameters())
    elif args.optimizer == "adam" or "Adam":
        optimizer = opt.Adam(learning_rate=args.lr, parameters=model.parameters())
    else:
        raise KeyError("Please select correct optimizer in [sgd, momentum, adagrad, adam]!")
    
    # 保存loss
    loss_list = []
    acc_list = []
    
    for epoch in range(1, args.epochs+1):
        epoch_loss = []
        
        for data in train_loader():
            imgs, labels = data
            imgs = paddle.to_tensor(imgs)
            labels = paddle.to_tensor(labels)
            
            # 前向推理
            preds = model(imgs)
            
            # 计算损失
            loss = F.cross_entropy(preds, labels)
            avg_loss = paddle.mean(loss)
            
            # 反向传播
            avg_loss.backward()
            
            # 保存每次迭代的损失
            epoch_loss.append(avg_loss.item()) # type: ignore
        
            """
            Note: 
                对于一个0-D的Tensor而言，直接使用tensor.item()就行，别用tensor.numpy()
                0-D其实就是一个list, shape为 (165, )
            print(f"epoch_loss: {np.shape(epoch_loss)}")  # epoch_loss: (254,)
            print(f"type: {type(epoch_loss)}")  # type: <class 'list'>
            """

            # 优化器
            optimizer.step()

            # 清空梯度
            optimizer.clear_grad()
            
        # 保存模型和优化器参数
        if epoch % 10 == 0:
            paddle.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()
            }, path=f"{args.save_path}/model_{args.model_name}_{epoch}_{args.optimizer}.pdparams")
        
        # 保存每个epoch的loss
        current_epoch_loss = np.mean(epoch_loss)
        loss_list.append(current_epoch_loss)
        epoch_loss.clear()
        acc_epoch = evaluation(model, val_loader)
        acc_list.append(acc_epoch)
        
        print(f"Epoch: {epoch}\tLoss: {current_epoch_loss:.4f}\tacc: {acc_epoch*100:.2f}%")
        
    print(f"模型最终loss为: {loss_list[-1]:.4f}")
    print(f"模型最终accuracy为: {acc_list[-1]*100:.2f}%")
    
    # 绘制Loss-Epoch曲线图
    plot_loss_curve(loss_list)
    
    print(model)
    
    
def load_one_img():
    img = Image.open(args.img_path).convert("L")  # 转为灰度图
    img = img.resize((28, 28))
    img = np.array(img).reshape(1, 1, 28, 28).astype(np.float32)
    
    # 归一化
    img = 1.0 - img / 255
    return img


def predict():
    # 读取要预测的图片
    img = load_one_img()
    img = paddle.to_tensor(img)
    
    # 定义模型
    if args.model_name == "FC":
        model = MNIST_FC_Model()
    elif args.model_name == "CNN":
        model = MNIST_CNN_Model()
    else:
        raise ModelNameError("请选择正确的模型(CNN或FC)!")
        
    # 加载模型权重
    param_state_dict = paddle.load(args.weights_path)
    model.load_dict(param_state_dict["model_state_dict"])

    # 声明模型状态
    model.eval()
    
    # 前向推理
    pred = model(img)
    """
    推理结果为: Tensor(shape=[1, 10], dtype=float32, place=Place(gpu:0), stop_gradient=False,
       [[0.00000163, 0.00267692, 0.00088234, 0.04414432, 0.00028779, 0.00000287,
         0.00000097, 0.95190734, 0.00004345, 0.00005248]])
    推理结果.shape为: [1, 10]
    推理结果.type为: <class 'paddle.Tensor'>
    """
    
    # 取概率最大的位置
    max_class = paddle.argmax(pred).item()  # type: ignore
    
    # 画出这张图片并给出相关信息
    # 将图片数据转换为 PIL 图像对象
    img_data = img.numpy()[0][0] * 255  # type: ignore
    img_data = img_data.astype(np.uint8)
    img_pil = Image.fromarray(img_data, "L")

    # 显示图片
    plt.imshow(img_data, cmap='gray')
    plt.title(f"Predicted Image -> class: {max_class} | prob: {pred[:, max_class].item() * 100:.2f}%")
    plt.axis('off')  # 去除坐标轴
    plt.savefig("predict_res.png")
    
    print(f"预测值的数字为: {max_class}\t预测概率为: {pred[:, max_class].item() * 100:.2f}%")

    
def main(args):
    if args.mode == "train":
        train()
    elif args.mode == "predict" or "eval":
        predict()
    else:
        raise KeyError("train or predict or eval")
    

def parse_args():
    parser = argparse.ArgumentParser()
    
    # 超参数
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.09, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size")
    parser.add_argument("--dataset_path", type=str, default="/data/data_01/lijiandong/Datasets/MNIST/mnist.json.gz", help="Path to the dataset file")
    parser.add_argument("--save_path", type=str, default="results/", help="The path of saving model & params")
    parser.add_argument("--device", type=str, default="gpu", help="cpu or cuda")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of classes")
    parser.add_argument("--model_name", type=str, default="CNN", help="The name of saving model (CNN or FC)")
    parser.add_argument("--img_path", type=str, default="test.png", help="The path of the image predicted")
    parser.add_argument("--weights_path", type=str, default="results/model_CNN_10.pdparams", help="The path of the model's weights")
    parser.add_argument("--mode", type=str, default="train", help="train / predict")
    parser.add_argument("--optimizer", type=str, default="sgd", help="sgd, momentum, adagrad, adam")
    
    # 解析命令行参数  
    args = parser.parse_args()
    
    return args


if __name__ == "__main__":
    # 固定随机种子
    seed = 10010
    paddle.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    args = parse_args()
    
    # 设置使用CPU还是GPU训练
    paddle.set_device(args.device)
    
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
        
    main(args)