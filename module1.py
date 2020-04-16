from MyModule import *
import time
import mnist
import os

# @Author  : xwh
# @Time    : 2020/4/16 19:10

# 载入训练集和测试集的数据管道
train_loader, test_loader = mnist.load_dataset(batch_size=5000)

# 训练位置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 学习率、迭代周期的所有待训练可能
learning_rate, num_epochs, activation = [1e-3, 1e-4, 1e-5], [100, 200, 300], ["sigmoid", "relu"]

exp_data_dir = 'resources/ans/exp_data/module1/'

for lr in learning_rate:  # 每一个学习率
    for epoch in num_epochs:  # 每一个迭代周期
        for activate in activation:  # 每一种激活函数
            for out_activation in [True, False]:  # 输出层是否调用激活函数

                # 保存的文件名 学习率_迭代周期_激活函数_输出层是否使用.txt
                file_name = str(lr) + '_' + str(epoch) + '_' + activate + '_' + str(out_activation)

                # 如果对应名字的文件已存在, 说明这组参数训练过了, 跳过
                if os.path.isfile(exp_data_dir + file_name + ".txt"):
                    print("参数组合 [" + file_name + "] 已训练, 跳过该组超参数...")
                    continue
                else:
                    print("参数组合 [" + file_name + "] 开始训练...")

                # 模型创建
                model = Net2Layers(hidden_size=300, activation=activate, out_activate=out_activation).to(device)

                start = time.time()

                # 训练
                exp_data = mnist.train_and_test(train_loader=train_loader, test_loader=test_loader, model=model,
                                                num_epochs=epoch, learning_rate=lr, criterion_name="mse",
                                                weight_decay=0)

                duration = time.time() - start
                exp_data["duration"] = duration

                # 保存数据
                mnist.save_exp_data(exp_data, file_name, exp_data_dir)
