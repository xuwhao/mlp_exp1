from __future__ import print_function
import sys, os, argparse
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt

print(sys.path[0])
data_dir = 'resources/dataset'

cuda = torch.cuda.is_available()


parse = argparse.ArgumentParser(description='Pytorch MNIST Example')
parse.add_argument('--batchSize', type=int, default=64, metavar='input batch size')
parse.add_argument('--testBatchSize', type=int, default=100, metavar='input batch size for testing')
parse.add_argument('--trainSize', type=int, default=10000, metavar='input dataset size(max=60000).Default=1000')
parse.add_argument('--nEpochs', type=int, default=2, metavar='number of epochs to train')
parse.add_argument('--lr', type=float, default=0.01, metavar='Learning rate.Deafault=0.01')
parse.add_argument('--momentum', type=float, default=0.5, metavar='Default=0.5', )
parse.add_argument('--seed', type=int, default=123, metavar='Romdom Seed to use.Default=123')

opt = parse.parse_args()

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        root=data_dir,
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True),
    batch_size=opt.batchSize,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        root=data_dir,
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=True),
    batch_size=opt.batchSize,
    shuffle=True
)

# print('===>Loading data')
# # with open(data_dir + "/MNIST/processed/training.pt", 'rb') as f:
# #     training_set = torch.load(f)
# #
# # with open(data_dir + "/MNIST/processed/test.pt", 'rb') as f:
# #     test_set = torch.load(f)
# # print('<===Done')
# #
# # # reshape image to 60000*1*28*28
# # training_data = training_set[0].view(-1, 1, 28, 28)
# # training_data = training_data[:opt.trainSize]
# # training_labels = training_set[1]
# # test_data = test_set[0].view(-1, 1, 28, 28)
# # test_labels = test_set[1]
# #
# # print(training_labels.shape)
# for y in training_labels:
#     print()
print(train_loader.dataset.data)
print(train_loader.dataset.targets)