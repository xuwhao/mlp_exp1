import os
import numpy as np

# @Author  : xwh
# @Time    : 2020/4/17 20:49
def get_all_exp_data(exp_data_dir):
    """
    获取传入的路径下的所有试验数据
    :param exp_data_dir:
    :return: 一个dict key是文件名, value是exp_data 也是个dict
    """
    exp_data_dict = {}
    file_urls = os.listdir(exp_data_dir)
    for url in file_urls:
        record = {}
        f = open(exp_data_dir + url, 'r')
        for line in f:
            key, value = line.strip().split(':')
            if key == "loss_x" or key == "loss_y":
                value = value.strip('[')
                value = value.strip(']')
                value = list(map(float, value.split(',')))
            elif key == "num_epochs" or key == "batch_size":
                value = int(value)
            else:
                value = float(value)
            record[key] = value
        f.close()
        exp_data_dict[url[0:-4]] = record
    return exp_data_dict


def get_best_accuracy_module(exp_data_dir=None, exp_data_dict=None):
    """
    获取一组数据里测试集精度最好的那个
    :param exp_data_dict: 所有数据的字典 和路径必须传入一个
    :param exp_data_dir:  数据的存储位置, 两个参数必须传入一个
    :return: exp_data
    """
    if exp_data_dict is None:
        exp_data_dict = get_all_exp_data(exp_data_dir)
    accuracy = 0.0
    name = ""
    best_exp_data = {}
    for key, exp_data in exp_data_dict.items():
        if exp_data["accuracy_test"] > accuracy:
            accuracy = exp_data["accuracy_test"]
            best_exp_data = exp_data
            name = key
    print(accuracy)
    return name, best_exp_data


def partial_arr(arr, arr_y):
    arr = np.array(arr)
    count, x, pos = 0, 5, 0
    for i in range(len(arr)):
        if arr[i] - 0 < 1e-6:
            count += 1
        if count == 20:
            pos = i
            break
    return arr[5:pos], arr_y[5:pos]
