# @Author  : xwh
# @Time    : 2020/5/2 21:08
from LeNet5 import LeNet5
import time
import mnist
import torch
import os
# net = LeNet5(5, 1, 2, 2, True)
#
# x = torch.randn(1, 1, 28, 28)
# out = net(x)
# print(out)

# 调试大小
# net2 = LeNet5(4, 1, 2, 2)
# print(net2)
# exit()
# [4, 1, 2, 1], [4, 2, 2, 1], [2, 1, 2, 1] , [3, 1, 2, 1], [5, 1, 2, 1]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
exp_data_dir = 'resources/ans/exp_data/LeNet5_xwh/'
kernels = [[5, 1, 2, 2], [3, 1, 2, 1], [5, 1, 2, 1]]
lrs, epoch, batches = [1e-3, 1e-4, 1e-5], 20, [100, 256, 600]
# kernels = [[5, 1, 2, 2]]
# lrs, epoch, batches = [1e-3], 20, [256]
for kernel in kernels:
    for lr in lrs:
        for batch in batches:
            for init in [True, False]:
                # 载入训练集和测试集的数据管道
                train_loader, test_loader = mnist.load_dataset(batch_size=batch)
                conv_kn, conv_step, pool_kn, pool_step = kernel[0], kernel[1], kernel[2], kernel[3]
                # 保存的文件名 卷积核_卷积步长_池化核_池化步长_学习率_batch_初始化
                file_name = str(conv_kn) + '_' + str(conv_step) + '_' + str(pool_kn) + '_' + str(pool_step) + '_'
                file_name += str(lr) + '_' + str(batch) + '_' + str(init)

                # 如果对应名字的文件已存在, 说明这组参数训练过了, 跳过
                if os.path.isfile(exp_data_dir + file_name + ".txt"):
                    print("参数组合 [" + file_name + "] 已训练, 跳过该组超参数...")
                    continue
                else:
                    print("参数组合 [" + file_name + "] 开始训练...")

                # 模型创建
                model = LeNet5(conv_kn, conv_step, pool_kn, pool_step, init).to(device)

                start = time.time()

                # 训练
                exp_data = mnist.train_and_test(train_loader=train_loader, test_loader=test_loader, model=model,
                                                num_epochs=epoch, learning_rate=lr, criterion_name="cross",
                                                weight_decay=0)

                duration = time.time() - start
                exp_data["duration"] = duration

                # 保存数据
                mnist.save_exp_data(exp_data, file_name, exp_data_dir)
