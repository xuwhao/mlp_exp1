# @Author  : xwh
# @Time    : 2020/4/19 16:15
import analysis
import painter

dir = "resources/ans/exp_data/module2/"
all_time = 0.0
exp_data_dict = analysis.get_all_exp_data(exp_data_dir=dir)

for key, exp_data in exp_data_dict.items():
    all_time+=exp_data["duration"]
print(all_time, all_time/3600)
key, exp_data = analysis.get_best_accuracy_module(exp_data_dir=dir)
print(key)

dir = "resources/ans/exp_data/module5/"
exit()
# name, exp_data = analysis.get_best_accuracy_module(exp_data_dir=dir)
# print(name, exp_data)

# exp_data_dict = analysis.get_all_exp_data(dir)
# key = "0.0001_200_relu_True"
# exp_data = exp_data_dict[key]
# x, y = analysis.partial_arr(exp_data["loss_x"], exp_data["loss_y"])
# x1, y1 = analysis.partial_arr(exp_data["loss_x"], exp_data["loss_y_test"])
# length = len(y)
# x = []
# for i in range(length):
#     x.append(i+5)
# length1 = len(y1)
# x1 = []
# for i in range(length1):
#     x1.append(i+5)
#
# data = [{'x': x, 'y': y, "label": "训练集loss"}, {'x': x1, 'y': y1, "label": "测试集loss"}]
# painter.line_chart(data, "iter", "loss", "模型2-训练集与测试集损失函数趋势图", img_size=(12, 8), x_gap=200)

# 过拟合的参数
# exp_data_dict = analysis.get_all_exp_data(dir)
# for key, exp_data in exp_data_dict.items():
#     if exp_data["accuracy_train"] > exp_data["accuracy_test"]:
#         print(key)
