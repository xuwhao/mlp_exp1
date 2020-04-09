import torch.nn as nn


# @Author  : xwh
# @Time    : 2020/4/8 20:41
class NeuralNet(nn.Module):
    """
        自定义的含有隐藏层的神经网络包装类
        :param input_size 输入层size
        :param hidden_size 隐藏层_size
        :param output_size 输出层size
    """

    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)


    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out



