import torch.nn as nn
import torch


# @Class   : 自定义神经网络类
# @Author  : xwh
# @Time    : 2020/4/9 18:28
class Net2Layers(nn.Module):
    """
    结构：2-layer Neutral Net, 28*28-hidden_size-10
    :param activation 激活函数类型 "sigmoid" or "relu"
    :param out_activate 输出层是否使用激活函数
    """

    def __init__(self, hidden_size, activation="sigmoid", out_activate=False):
        # 输入层, 隐藏层, 输出层
        input_size, output_size = 28 * 28, 10
        activation_func = nn.Sigmoid()
        if activation == "relu":
            activation_func = nn.ReLU(True)
        super(Net2Layers, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(input_size, hidden_size), activation_func)
        if out_activate:
            self.layer2 = nn.Sequential(nn.Linear(hidden_size, output_size), activation_func)
        else:
            self.layer2 = nn.Sequential(nn.Linear(hidden_size, output_size))
        for name, param in self.named_parameters():
            if 'weight' in name:
                if activation == "relu":
                    nn.init.kaiming_uniform_(param, nonlinearity='relu')
                else:
                    nn.init.xavier_uniform_(param, gain=1.)
            if 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class Net3Layers(nn.Module):
    """
    结构：3-layer Neutral Net, 784-hidden_size[0]-hidden_size[1]-10
    :param hidden_size 数组
    :param activation 激活函数类型 "sigmoid" or "relu"
    :param out_activate 输出层是否使用激活函数
    """

    def __init__(self, hidden_size, activation="sigmoid", out_activate=False):
        # 输入层, 输出层
        input_size, output_size = 28 * 28, 10
        activation_func = nn.Sigmoid()
        if activation == "relu":
            activation_func = nn.ReLU(True)
        super(Net2Layers, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(input_size, hidden_size[0]), activation_func)
        self.layer2 = nn.Sequential(nn.Linear(hidden_size[0], hidden_size[1]), activation_func)
        if out_activate:
            self.layer3 = nn.Sequential(nn.Linear(hidden_size, output_size), activation_func)
        else:
            self.layer3 = nn.Sequential(nn.Linear(hidden_size, output_size))
        for name, param in self.named_parameters():
            if 'weight' in name:
                if activation == "relu":
                    nn.init.kaiming_uniform_(param, nonlinearity='relu')
                else:
                    nn.init.xavier_uniform_(param, gain=1.)
            if 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
