import mnist
import numpy as np

# 载入训练集和测试集的数据管道
train_loader, test_loader = mnist.load_dataset()


# 将data_loader中的标签转换为one-hot编码, 验证前10个标签
one_hot_labels = mnist.labels_to_one_hot(data_loader=train_loader)
print(">>> 检验one-hot编码是否正确:\n")
for i in range(10):
    print("  ", train_loader.dataset.targets[i].item(), one_hot_labels[i].numpy())
print("\n>>> 检验完毕！")

# 显示size张data_loader中的图片
mnist.show_mnist_image(data_loader=train_loader, size=10)