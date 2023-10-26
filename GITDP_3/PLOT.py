import matplotlib.pyplot as plt
from Structure import *

import numpy as np

def PLOT(par, plot_what, step):

    what_can_be_plotted = \
        ['ITERATION',
         'COST',
         'ACCURACY_TEST',
         'RESIDUAL',
         'ELAPSED_TIME',
         'ITER_TIME',
         'RUNTIME_1',
         'RUNTIME_2',
         'GRAD_TIME',
         'NOISE_TIME',
         'AVG_NOISE_MAG',
         'Z_CHANGE_MEAN',
         'XIK']

    # 创建一个图形窗口
    plt.figure()

    for i in plot_what:
        if i in what_can_be_plotted:
            plt.subplot(len(plot_what), 1, plot_what.index(i) + 1)
            temp = getattr(par, i)  # 使用循环中的当前元素 i

            ylabel = i

            if step == "0":
                plt.plot(range(par.training_steps + 1), temp[:par.training_steps + 1], label='XXX Data')
            else:
                plt.plot(range(int(step) + 1), temp[0:int(step) + 1], label='XXX Data')

            plt.xlabel('Iteration $k$')
            plt.ylabel(ylabel)
            plt.legend()

    plt.show()  # 一次性显示所有子图



def PLOT_PP_ADMM(par):

    # 创建一个图形窗口
    plt.figure()

    plt.subplot(3, 1, 1)
    plt.plot(range(par.training_steps), [np.linalg.norm(par.centralized_x)]*(par.training_steps), label='')
    temp = np.zeros((par.training_steps ,par.split_number))
    for k in range(par.training_steps):
        for i in range(par.split_number):
            temp[k,i] = np.linalg.norm(par.xik[k,i])

    for p in range(par.split_number):
        plt.plot(range(par.training_steps), temp[:,p], label='')

    plt.subplot(3, 1, 2)
    plt.plot(range(par.training_steps + 1), [par.centralized_cost]*(par.training_steps + 1), label='')
    plt.plot(range(par.training_steps + 1),  par.COST[:par.training_steps + 1], label='')

    plt.subplot(3, 1, 3)
    plt.plot(range(par.training_steps + 1), [par.centralized_accuracy]*(par.training_steps + 1), label='')
    plt.plot(range(par.training_steps + 1), par.ACCURACY_TEST[:par.training_steps + 1], label='')

    plt.xlabel('Iteration $k$')

    plt.show()  # 一次性显示所有子图


