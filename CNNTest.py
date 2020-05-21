# @Author  : xwh
# @Time    : 2020/5/21 19:15
import mnist
from CNN import *

data_loader, test_loader = mnist.load_CIFAR10(batch_size=4)

print(data_loader, test_loader)