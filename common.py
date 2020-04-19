import mnist
import painter
from MyModule import *
import analysis
import numpy as np
import os
# 载入训练集和测试集的数据管道
# train_loader, test_loader = mnist.load_dataset(batch_size=100)
#
# # 任务一: 将标签转换为one-hot编码, 验证前10个标签
# one_hot_labels = mnist.labels_to_one_hot(labels=train_loader.dataset.targets)
# print(">>> 转换one_hot编码，检验是否正确:\n")
# for i in range(10):
#     print("  ", train_loader.dataset.targets[i].item(), one_hot_labels[i].numpy())
# print("\n>>> 检验完毕！")
# #
# # print(one_hot_labels.size())
# #
# print(len(one_hot_labels))
# # 任务二: 数据可视化, 随机显示size张data_loader中的图片
# mnist.show_mnist_image(data_loader=train_loader, size=5)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# model = Net2Layers(hidden_size=300, activation='sigmoid', out_activate=False).to(device)
#
#
# exp_data = mnist.train_and_test(train_loader=train_loader, test_loader=test_loader, model=model, num_epochs=5,
#                                 learning_rate=1e-3, criterion_name="mse", weight_decay=0)
#

# print(get_best_accuracy_module("resources/ans/exp_data/module2/"))

dir = "resources/ans/exp_data/module6/"

exp_data_dict = analysis.get_all_exp_data(dir)
key = "0.001_100_relu_False"
exp_data = exp_data_dict[key]
x, y = analysis.partial_arr(exp_data["loss_x"], exp_data["loss_y"])
length = len(y)
x = []
for i in range(length):
    x.append(i+5)
data = [{'x': x, 'y': y}]
painter.line_chart(data, "iter", "loss", "模型2_754+800+10_loss分析", img_size=(12, 8), x_gap=20)
# exp_data["loss_x"]
# x = [0., 1., 2., 3.,4.,5.,6.,7.,8.,9., 0., 1., 2.]
# y = analysis.partial_arr(x)
# print(y)
