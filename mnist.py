import torch
import torchvision
import torchvision.transforms as transforms
from IPython import display
import matplotlib.pyplot as plt
import numpy as np


# @Class  : MNIST数据集识别相关函数
# @Author  : xwh
# @Time    : 2020/4/8 20:41
def load_dataset(data_dir='resources/dataset', batch_size=100):
    """
    获取数据管道
    :param data_dir:
    :param batch_size:
    :returns: train_loader, test_loader 训练集，测试集的数据管道
    """
    # 载入手写体数据 如果不存在则下载
    train_dataset = torchvision.datasets.MNIST(root=data_dir,
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)

    test_dataset = torchvision.datasets.MNIST(root=data_dir,
                                              train=False,
                                              transform=transforms.ToTensor())

    # 构建数据管道
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)
    return train_loader, test_loader


def labels_to_one_hot(labels):
    """
    将传入的标签转换为one-hot编码
    :param labels: 标签tensor list
    :return: one_hot_label one-hot编码的tensor list
    """
    one_hot_label = torch.zeros([len(labels), 10], dtype=torch.float)
    for i in range(len(labels)):
        one_hot_label[i][labels[i]] = 1
    return one_hot_label


def show_mnist_image(data_loader=None, size=10):
    """
    随机展示传入的MNIST数据集的图片
    :param data_loader: 数据管道, 训练集或测试集
    :param size: 默认显示张数
    :return:
    """
    # 图片数据, 标签数据
    images, labels = data_loader.dataset.data, data_loader.dataset.targets
    # tart为第一张待展示图片的下标
    start = np.random.randint(0, 60000 - size * 2)

    # 随机在60000张图片内选择size张
    # img: 待展示图片  img_label: img对应的标签
    img, img_label = [], []
    for i in range(start, start + size):
        img.append(images[i])
        img_label.append(labels[i].item())

    # 设置为svg格式显示, 可缩放矢量图形(Scalable Vector Graphics)
    display.set_matplotlib_formats('svg')

    # _: 忽略的变量
    _, figs = plt.subplots(1, len(img), figsize=(10, 3))

    # 将每一张图片和对应的标签在子图中展示
    for f, img, lbl in zip(figs, img, img_label):
        # 将784转为28*28展示
        f.imshow(img.view((28, 28)).numpy())
        # 在图片上方设置对应的label
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()
