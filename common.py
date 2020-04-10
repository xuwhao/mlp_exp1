import mnist
import painter
from MyModule import *

# 载入训练集和测试集的数据管道
train_loader, test_loader = mnist.load_dataset()

# 任务一: 将标签转换为one-hot编码, 验证前10个标签
# one_hot_labels = mnist.labels_to_one_hot(labels=train_loader.dataset.targets)
# print(">>> 检验one-hot编码是否正确:\n")
# for i in range(10):
#     print("  ", train_loader.dataset.targets[i].item(), one_hot_labels[i].numpy())
# print("\n>>> 检验完毕！")

# 任务二: 数据可视化, 随机显示size张data_loader中的图片
# mnist.show_mnist_image(data_loader=train_loader, size=10)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net2Layers(hidden_size=300, activation='relu').to(device)
exp_data = mnist.train_and_test(train_loader, test_loader, model, num_epochs=1,
                                learning_rate=1e-3, criterion_name="mse", weight_decay=0)
data = [{'x': exp_data["loss_x"], 'y': exp_data["loss_y"]}]
painter.line_chart(data, "iter", "loss", "loss分析", img_size=(12, 8), x_gap=100)
