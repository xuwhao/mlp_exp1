import torch
import torchvision
import torchvision.transforms as transforms
from IPython import display
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import os


# @Class   : MNIST数据集识别相关函数
# @Author  : xwh
# @Time    : 2020/4/8 20:41
def load_dataset(data_dir='resources/dataset', batch_size=100):
    """
    获取数据管道
    :param data_dir: 数据集的文件地址
    :param batch_size: 批量大小
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
    :return: one-hot编码的tensor list
    """
    # n个标签*10列的tensor list
    one_hot_label = torch.zeros([len(labels), 10], dtype=torch.float)
    # 对于每个label 对应位置置为1
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
    _, figs = plt.subplots(1, len(img), figsize=(5, 2))

    # 将每一张图片和对应的标签在子图中展示
    for f, img, lbl in zip(figs, img, img_label):
        # 将784的vector转为28*28展示
        f.imshow(img.view((28, 28)).numpy())
        # 在图片上方设置对应的label
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()


def train(train_loader, model, num_epochs, learning_rate, criterion_name, weight_decay=0, test_loader=None):
    """
    训练神经网络
    :param train_loader: 训练集数据管道
    :param model: 待训练的网络模型
    :param num_epochs: 每个batch的训练次数
    :param learning_rate: 学习率
    :param criterion_name: 损失函数名, "mse": MSE, "cross": 交叉熵
    :param weight_decay: 权重衰减值, 默认不衰减
    :return: exp_data 实验数据dict 需要的数据自行记录, 不要更改他人记录的key,
                    只返回一个dict, key在下方列出
    :key:  learning_rate 学习率, num_epochs 迭代周期, batch_size batch大小,
            loss_x 当前迭代次数, loss_y 对应的loss值, loss_y_test 测试集loss值
            accuracy_train 训练集精度, accuracy_test 测试集精度
    """
    # 待记录的数据初始化
    exp_data = {"learning_rate": learning_rate, "num_epochs": num_epochs,
                "batch_size": train_loader.batch_size}
    loss_x, loss_y, loss_y_test = [], [], []
    correct = 0
    total = 0

    # 判断设备是cpu还是gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Adam优化器, weight_decay 权重衰减
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate, weight_decay=weight_decay)

    # 根据传入的criterion_name设置损失函数
    criterion = nn.MSELoss()
    if criterion_name == "cross":
        criterion = nn.CrossEntropyLoss()

    total_step = len(train_loader)
    # 开始训练
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):

            # 如果是MSE, label转换为one-hot编码
            # 为计算精确度方便，并未直接替换原标签tensor
            one_hot = labels
            if criterion_name == "mse":
                one_hot = labels_to_one_hot(labels=labels)

            # gpu加速
            images = images.reshape(-1, 28 * 28).to(device)
            labels = labels.to(device)
            one_hot = one_hot.to(device)

            # 前向传播和计算loss
            outputs = model(images)
            loss = criterion(outputs, one_hot)

            # 后向传播和调整参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 数据记录
            loss_x.append(int((epoch + 1) * i))  # 迭代次数
            loss_y.append(loss.item())  # 对应loss

            # 精度
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 训练集损失
            loss_test = test_loss(test_loader=test_loader, model=model,
                                  criterion_name=criterion_name, criterion=criterion)
            loss_y_test.append(loss_test.item())

            if (i + 1) % 10 == 0:  # 每十次打印一下
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

    # 保存数据到dict
    accuracy = 100 * correct / total
    print('Train Accuracy of the network: {} %'.format(accuracy))
    exp_data["loss_x"] = loss_x
    exp_data["loss_y"] = loss_y
    exp_data["loss_y_test"] = loss_y_test
    exp_data["accuracy_train"] = accuracy
    return exp_data


def test_accuracy(test_loader, model):
    """
    在测试集上计算精度
    :param test_loader: 测试集数据管道
    :param model: 训练好的模型
    :return: accuracy 精度
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    accuracy = 0

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.reshape(-1, 28 * 28).to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            # print(_, predicted)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print('Test Accuracy of the network: {} %'.format(accuracy))

    return accuracy


def train_and_test(train_loader, test_loader, model, num_epochs, learning_rate, criterion_name, weight_decay=0):
    """
    训练并计算精度
    :param train_loader: 训练集数据管道
    :param test_loader: 测试集数据管道
    :param model: 待训练的网络模型
    :param num_epochs: 每个batch的训练次数
    :param learning_rate: 学习率
    :param criterion_name: 损失函数名, "mse": MSE, "cross": 交叉熵
    :param weight_decay: 权重衰减值, 默认不衰减
    :return: exp_data 实验数据dict 需要的数据自行记录, 不要更改他人记录的key, 只返回一个dict, key在下方列出
    :key:  learning_rate 学习率, num_epochs 迭代周期, batch_size batch大小,
            loss_x 当前迭代次数, loss_y 对应的loss值, loss_x_test 测试集迭代次数, loss_y_test 测试集loss值
            accuracy_train 训练集精度, accuracy_test 测试集精度
    """
    exp_data = train(train_loader=train_loader, model=model, num_epochs=num_epochs, learning_rate=learning_rate,
                     criterion_name=criterion_name, weight_decay=weight_decay, test_loader=test_loader)
    accuracy_test = test_accuracy(test_loader, model)
    exp_data['accuracy_test'] = accuracy_test
    return exp_data


def save_exp_data(exp_data, file_name, data_dir):
    """
    保存实验数据
    :param exp_data 实验数据dict
    :param file_name 文件名
    :param data_dir 保存路径
    """
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    fp = open(data_dir + file_name + ".txt", 'w', encoding='utf-8')
    result = []
    for key in exp_data.keys():
        result.append(key + ":" + str(exp_data[key]))
    fp.writelines([line + '\n' for line in result])
    fp.close()


def test_loss(test_loader, model, criterion_name, criterion):
    """
    在测试集上计算损失函数
    :param test_loader: 测试集数据管道
    :param model: 训练好的模型
    :return: accuracy 精度
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss = 0.0
    i = 0
    with torch.no_grad():
        for images, labels in test_loader:
            one_hot = labels
            if criterion_name == "mse":
                one_hot = labels_to_one_hot(labels=labels)
            images = images.reshape(-1, 28 * 28).to(device)
            labels = labels.to(device)
            one_hot = one_hot.to(device)
            outputs = model(images)
            loss += criterion(outputs, one_hot)
            i += 1
    return loss / i
