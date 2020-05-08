import torch.nn as nn
import torch.nn.functional as F


# @Author  : xwh
# @Time    : 2020/5/1 17:02
def calc_conv_params(nh, nw, kh, kw, out_channel, stride, ph=0, pw=0):
    oh, ow = (nh - kh + ph + stride) / stride, (nw - kw + pw + stride) / stride
    param_size = out_channel * (kh * kw + 1)
    flops = oh * ow * param_size
    return oh, ow, param_size, flops


def calc_pooling_params(nh, nw, kh, kw, stride):
    oh, ow = (nh - kh + stride) / stride, (nw - kw + stride) / stride
    return oh, ow


class LeNet5(nn.Module):

    def __init__(self, kernel_size, kernel_stride, pooling_size, pooling_step, init_weight=False):
        super(LeNet5, self).__init__()
        print("开始实例化模型, 参数计算中...")

        out_channel1, img_size, out_channel2 = 6, 28, 16

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=out_channel1, kernel_size=kernel_size, stride=kernel_stride)
        oh, ow, param_size,flops = calc_conv_params(nh=img_size, nw=img_size,
                                                    kh=kernel_size, kw=kernel_size, stride=kernel_stride,
                                                    out_channel=out_channel1)
        print("卷积层1 - shape - 参数个数 - 连接数: ", oh, "*", ow, param_size, flops)

        self.maxPool1 = nn.MaxPool2d(pooling_size, stride=pooling_step, padding=0)
        oh, ow = calc_pooling_params(nh=oh, nw=ow, kh=pooling_size, kw=pooling_size, stride=pooling_step)
        print("最大池化1 - shape: ", oh, "*", ow)

        self.conv2 = nn.Conv2d(out_channel1, out_channel2, kernel_size, kernel_stride)
        oh, ow, param_size, flops = calc_conv_params(nh=oh, nw=ow, kh=kernel_size, stride=kernel_stride,
                                                     kw=kernel_size, out_channel=out_channel2)
        print("卷积层2 - shape - 参数个数 - 连接数: ", oh, "*", ow, param_size, flops)

        self.maxPool2 = nn.MaxPool2d(pooling_size, stride=pooling_step, padding=0)
        oh, ow = calc_pooling_params(nh=oh, nw=ow, kh=pooling_size, kw=pooling_size, stride=pooling_step)
        print("最大池化2 - shape: ", oh, "*", ow)

        self.fc1 = nn.Linear(int(out_channel2 * oh * ow), 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        if init_weight:
            self._init_weight()

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.conv1(x)
        x = F.relu(x)
        x = self.maxPool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxPool2(x)
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

    def _init_weight(self):
        for m in self.modules():  # 继承nn.Module的方法
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0.)
