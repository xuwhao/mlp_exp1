# @Author  : xwh
# @Time    : 2020/4/9 18:28
import torch.nn as nn


class MyModule(nn.Module):
    """

    """

    def __init__(self, input_size, hidden_size, output_size):
        super(MyModule, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(hidden_size, output_size))
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_uniform_(param, nonlinearity='relu')
            if 'bias' in name:
                nn.init.zeros_(param)
        """
        这里的Sequential()函数的功能是将网络的层组合到一起。
        """

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x
