import numpy as np #(activate this if CPU is used)
# import cupy as np #(activate this if GPU is used)

from mlxtend.data import loadlocal_mnist
import json

import numpy as np
import matplotlib.pyplot as plt

x_train, y_train = loadlocal_mnist(images_path='./Inputs/train-images-idx3-ubyte', labels_path='./Inputs/train-labels-idx1-ubyte')
print(x_train.shape)

x_train, x_test = x_train.reshape([-1, 784]), x_test.reshape([-1, 784])

print(x_train.shape)
first_image_pixels = x_train[3555]
print(first_image_pixels)
#
# x_train_agent = np.split(x_train, 10)  ##  例如，如果 par.split_number 是 5，那么 x_train_agent 将会是一个包含 5 个子数组的列表，每个子数组都包含了 x_train 中的一部分数据
# y_train_agent = np.split(y_train, 10)
# print(x_train_agent.shape)
# x_train_agent, y_train_agent = np.array(x_train_agent), np.array(y_train_agent)
#
#
# x_list = []; y_list = [];
# for p in range(10):
#     x_list.append(x_train_agent[p])
#     y_list.append(y_train_agent[p])
#
#   x_train_new = np.concatenate( np.array(x_list) )
#   y_train_new = np.concatenate( np.array(y_list) )
#
#
# # 假设你的784维向量为 image_vector
# image_vector = first_image_pixels
#
# # 将784维向量转换为28x28的图像矩阵
# image_matrix = image_vector.reshape(28, 28)
#
# # 显示图像
# plt.imshow(image_matrix, cmap='gray')  # cmap='gray' 表示以灰度图像显示
# # plt.axis('off')  # 不显示坐标轴
# plt.show()