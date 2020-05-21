# @Author  : xwh
# @Time    : 2020/5/21 19:15
import mnist
from CNN import *
import os
import time

data_loader, test_loader = mnist.load_CIFAR10(batch_size=4)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
exp_data_dir = 'resources/ans/exp_data/alexnet/'
lrs, epoch, batch = [1e-3], 1, 256
train_loader, test_loader = mnist.load_CIFAR10(batch_size=batch, resize=224)
for lr in lrs:

    # 保存的文件名 卷积核_卷积步长_池化核_池化步长_学习率_batch_初始化
    file_name = "alex" + "_" + batch + " " + lr
    # 如果对应名字的文件已存在, 说明这组参数训练过了, 跳过
    if os.path.isfile(exp_data_dir + file_name + ".txt"):
        print("参数组合 [" + file_name + "] 已训练, 跳过该组超参数...")
        continue
    else:
        print("参数组合 [" + file_name + "] 开始训练...")
    # 模型创建
    model = AlexNet()
    start = time.time()
    # 训练
    exp_data = mnist.train_and_test(train_loader=train_loader,
                                    test_loader=test_loader, model=model,
                                    num_epochs=epoch, learning_rate=lr,
                                    criterion_name="cross", weight_decay=0)
    duration = time.time() - start
    exp_data["duration"] = duration
    # 保存数据
    mnist.save_exp_data(exp_data, file_name, exp_data_dir)
