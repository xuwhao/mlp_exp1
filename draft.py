# @Author  : xwh
# @Time    : 2020/4/19 16:15
import analysis
import painter

dir = "resources/ans/exp_data/module3/"
# all_time = 0.0
# exp_data_dict = analysis.get_all_exp_data(exp_data_dir=dir)
#
# for key, exp_data in exp_data_dict.items():
#     all_time+=exp_data["duration"]
# print(all_time, all_time/3600)
# key, exp_data = analysis.get_best_accuracy_module(exp_data_dir=dir)
# print(key)
#
# dir = "resources/ans/exp_data/module5/"
# exit()
# name, exp_data = analysis.get_best_accuracy_module(exp_data_dir=dir)
# print(name, exp_data)

# exp_data_dict = analysis.get_all_exp_data(dir)
# key = "0.001_100_relu_False"
# exp_data = exp_data_dict[key]
# x, y = analysis.partial_arr(exp_data["loss_x"], exp_data["loss_y"])
# x1, y1 = analysis.partial_arr(exp_data["loss_x"], exp_data["loss_y_test"])
# length = len(y)
# x = []
# for i in range(length):
#     x.append(i + 5)
# length1 = len(y1)
# x1 = []
# for i in range(length1):
#     x1.append(i + 5)
# print(exp_data["loss_y_test"])
# data = [{'x': x, 'y': y, "label": "训练集loss"}, {'x': x1, 'y': y1, "label": "测试集loss"}]
# painter.line_chart(data, "iter", "loss", "模型3-训练集与测试集损失函数趋势图", img_size=(12, 8), x_gap=200)

# 过拟合的参数
dir1 = "resources/ans/exp_data/module1/"
dir2 = "resources/ans/exp_data/module2/"
dir3 = "resources/ans/exp_data/module3/"
dir4 = "resources/ans/exp_data/module4/"
dir5 = "resources/ans/exp_data/module5/"
dir6 = "resources/ans/exp_data/module6/"
i=1
for dir in [dir1, dir2, dir3, dir4, dir5, dir6]:
    exp_data_dict = analysis.get_all_exp_data(dir)
    print("模型" + str(i) + "过拟合的参数组合")
    i += 1
    for key, exp_data in exp_data_dict.items():
        if exp_data["accuracy_train"] > exp_data["accuracy_test"]:
            print(key)

#
# learning_rate, num_epochs, activation = [1e-3, 1e-4, 1e-5], [100, 200, 300], ["sigmoid", "relu"]
# t_cnt, f_cnt, cnt = 0, 0, 0
# for i in [1, 2, 3, 4]:
#     dir = "resources/ans/exp_data/module" + str(i) + "/"
#     exp_data_dict = analysis.get_all_exp_data(dir)
#     t_cnt, f_cnt, cnt = 0, 0, 0
#     for lr in learning_rate:
#         for epoch in num_epochs:
#             for activate in activation:
#
#                 key_true = str(lr) + '_' + str(epoch) + '_' + activate + '_' + "True"
#                 key_false = str(lr) + '_' + str(epoch) + '_' + activate + '_' + "False"
#                 if not exp_data_dict.get(key_true) or not exp_data_dict.get(key_false):
#                     break
#                 cnt += 1
#                 if exp_data_dict[key_true]["accuracy_test"] > exp_data_dict[key_false]["accuracy_test"]:
#                     t_cnt += 1
#                 else:
#                     f_cnt += 1
#     data = [t_cnt / cnt, f_cnt / cnt]
#     label = ["使用激活函数精度更高", "不使用激活函数精度更高"]
#     f_name = "模型" + str(i) + "输出层是否使用激活函数对测试集精度的影响"
#     painter.pie_chart(data=data, label=label, f_name=f_name, img_size=(8, 6), save=False)

# print(t_cnt, f_cnt, cnt)
# data = [t_cnt / cnt, f_cnt / cnt]
# label = ["使用激活函数精度更高", "不使用激活函数精度更高"]
# f_name = "输出层是否使用激活函数对测试集精度的影响"
# painter.pie_chart(data=data, label=label, f_name=f_name, img_size=(8, 6))

# learning_rate, num_epochs, activation = [1e-3, 1e-4, 1e-5], [100, 200, 300], ["sigmoid", "relu"]
# y1 = []
# y2 = []
# y3 = []
# y4 = []
# y5 = []
# y6 = []
# x1 = [0]
# x2 = [0]
# x3 = [0]
# x4 = [0]
# x5 = [0]
# x6 = [0]
# cnt = 0
#
# dir1 = "resources/ans/exp_data/module1/"
# dir2 = "resources/ans/exp_data/module2/"
# dir3 = "resources/ans/exp_data/module3/"
# dir4 = "resources/ans/exp_data/module4/"
# dir5 = "resources/ans/exp_data/module5/"
# dir6 = "resources/ans/exp_data/module6/"
# exp_data_dict1 = analysis.get_all_exp_data(dir1)
# exp_data_dict2 = analysis.get_all_exp_data(dir2)
# exp_data_dict3 = analysis.get_all_exp_data(dir3)
# exp_data_dict4 = analysis.get_all_exp_data(dir4)
# exp_data_dict5 = analysis.get_all_exp_data(dir5)
# exp_data_dict6 = analysis.get_all_exp_data(dir6)
#
# for lr in learning_rate:
#     for epoch in num_epochs:
#         for activate in activation:
#             for out in [True, False]:
#                 key = str(lr) + '_' + str(epoch) + '_' + activate + '_' + str(out)
#                 if exp_data_dict1.get(key):
#                     x1.append(x1[-1] + 1)
#                     y1.append(exp_data_dict1[key]["accuracy_test"])
#                 if exp_data_dict2.get(key):
#                     x2.append(x2[-1] + 1)
#                     y2.append(exp_data_dict2[key]["accuracy_test"])
#                 if exp_data_dict3.get(key):
#                     x3.append(x3[-1] + 1)
#                     y3.append(exp_data_dict3[key]["accuracy_test"])
#                 if exp_data_dict4.get(key):
#                     x4.append(x4[-1] + 1)
#                     y4.append(exp_data_dict4[key]["accuracy_test"])
#                 if exp_data_dict5.get(key):
#                     x5.append(x5[-1] + 1)
#                     y5.append(exp_data_dict5[key]["accuracy_test"])
#                 if exp_data_dict6.get(key):
#                     x6.append(x6[-1] + 1)
#                     y6.append(exp_data_dict6[key]["accuracy_test"])
# x1 = x1[1:]
# x2 = x2[1:]
# x3 = x3[1:]
# x4 = x4[1:]
# x5 = x5[1:]
# x6 = x6[1:]
# data = [
#     {'x': x1, 'y': y1, "label": "模型1"},
#     {'x': x2, 'y': y2, "label": "模型2"},
#     {'x': x3, 'y': y3, "label": "模型3"},
#     {'x': x4, 'y': y4, "label": "模型4"},
#     {'x': x5, 'y': y5, "label": "模型5"},
#     {'x': x6, 'y': y6, "label": "模型6"}
# ]
# painter.line_chart(data, "参数组合序号", "精度", "6种模型同一参数时测试集精度图", img_size=(12, 8), x_gap=1)

# learning_rate, num_epochs, activation, out_a = [1e-3, 1e-4, 1e-5], [100, 200, 300], ["sigmoid", "relu"], [True, False]
# s_cnt, r_cnt, cnt = 0, 0, 0
# for i in [1, 2, 3, 4, 5, 6]:
#     dir = "resources/ans/exp_data/module" + str(i) + "/"
#     exp_data_dict = analysis.get_all_exp_data(dir)
#     for lr in learning_rate:
#         for epoch in num_epochs:
#             for out in out_a:
#                 key_s = str(lr) + '_' + str(epoch) + '_' + "sigmoid" + '_' + "True"
#                 key_r = str(lr) + '_' + str(epoch) + '_' + "relu" + '_' + "False"
#                 if not exp_data_dict.get(key_s) or not exp_data_dict.get(key_r):
#                     continue
#                 cnt += 1
#                 if exp_data_dict[key_s]["accuracy_test"] > exp_data_dict[key_r]["accuracy_test"]:
#                     s_cnt += 1
#                 else:
#                     r_cnt += 1
#
# print(s_cnt, r_cnt, cnt)
# data = [s_cnt / cnt, r_cnt / cnt]
# label = ["sigmoid函数测试集精度高", "relu函数测试集精度高"]
# f_name = "激活函数对测试集精度的影响"
# painter.pie_chart(data=data, label=label, f_name=f_name, img_size=(12, 8))

# dir1 = "resources/ans/exp_data/module1/"
# dir2 = "resources/ans/exp_data/module2/"
# dir3 = "resources/ans/exp_data/module3/"
# dir4 = "resources/ans/exp_data/module4/"
# dir5 = "resources/ans/exp_data/module5/"
# dir6 = "resources/ans/exp_data/module6/"
# for dir in [dir1, dir2, dir3, dir4, dir5, dir6]:
#     key, exp_data = analysis.get_best_accuracy_module(exp_data_dir=dir)
#     print((100-exp_data["accuracy_test"]))
#
