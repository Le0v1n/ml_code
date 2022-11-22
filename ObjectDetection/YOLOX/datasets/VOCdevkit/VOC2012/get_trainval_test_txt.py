import os
import random


def main():
    # 保证随机可复现
    random.seed(1)

    # 将数据集中10%的数据划分到测试集中
    split_rate = 0.1

    # 指向mask_dataset文件夹
    cwd = os.getcwd()
    origin_images_path = os.path.join(cwd, "JPEGImages")
    assert os.path.exists(origin_images_path), "path '{}' does not exist.".format(origin_images_path)

    filenames = os.listdir(origin_images_path)  # list
    file_nums = len(filenames)

    # 随机采样测试集的索引
    test_filename = random.sample(filenames, k=int(file_nums * split_rate))  # list

    trainval_filename = []
    for elem in filenames:
        if elem not in test_filename:
            trainval_filename.append(elem)

    # 检查划分样本数是否一致
    nums_total = len(filenames)
    nums_trainval = len(trainval_filename)
    nums_test = len(test_filename)
    print("划分前后样本数量一致!") if nums_total == nums_trainval + nums_test else print(
        "训练集和测试集数量不等于总样本数，请检查")

    # 检查ImageSets/Main文件夹是否存在，若不存在则创建
    label_folder_path = os.path.join(cwd, "ImageSets/Main")
    if not os.path.exists(label_folder_path):
        print("存放训练集和测试集的文件夹不存在，开始创建...")
        os.makedirs(os.path.join(cwd, "ImageSets/Main"))
        print("创建成功!")
    else:
        print("标签文件夹已存在!")

    # 创建trainval.txt和test.txt文件并写入
    with open(label_folder_path + "/trainval.txt", 'w') as f:
        for elem in trainval_filename:
            f.write(elem.split(".")[0] + "\n")  # 去除文件后缀

    with open(label_folder_path + "/test.txt", 'w') as f:
        for elem in test_filename:
            f.write(elem.split(".")[0] + "\n")  # 去除文件后缀

    print("Processing Done!")


if __name__ == '__main__':
    main()
