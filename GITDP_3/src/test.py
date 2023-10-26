import sys

import matplotlib.pyplot as plt

sys.path.append("./src")
from Main import *
from Structure import *
from PLOT import *
import numpy as np
from scipy.optimize import fsolve

# 创建变量
import cvxpy as cp
import numpy as np
import time

from mlxtend.data import loadlocal_mnist
import json

import numpy as np  # (activate this if CPU is used)
# import cupy as np #(activate this if GPU is used)

from mlxtend.data import loadlocal_mnist
import json

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

par = Parameters()
p = 1
Agent = 3
par.gamma = 1e-3 ## regularizer parameter 其实应该是beta

x_test, y_test, x_train_new, y_train_new, x_train_agent, y_train_agent = Read(par, Agent)
par.total_data = x_train_new.shape[0]

y_train_Bin = (np.arange(y_train_new.max() + 1) == y_train_new[..., None]).astype(int)  ## I X K
num_data = y_train_Bin.shape[0]

## Variables
par.W_val = np.zeros((par.num_features, par.num_classes))  ## Global model parameters 768*10维的模型参数，对应w
par.Z_val = np.zeros(
    (par.split_number, par.num_features, par.num_classes));  ## Local model parameters defined for every agent
# 因此，这行代码创建了一个大小为 (10, 784, 62) 的三维数组，用于存储每个代理的本地模型参数。每个代理都有一个自己的参数矩阵，大小为 (784, 62)，并且这些参数都被初始化为全零

par.Lambdas_val = np.zeros((par.split_number, par.num_features, par.num_classes));  ## Dual variables

## Matrix Normal Distribution used for Gaussian Mechanism for "Base" algorithm
par.M = np.zeros((par.num_features, par.num_classes))
par.U = np.zeros((par.num_features, par.num_features))
par.V = np.zeros((par.num_classes, par.num_classes))

num_data = []
y_train_Bin = []
for p in range(par.split_number):
    num_data.append(y_train_agent[p].shape[0])  ## 6000*10
    Temp = (np.arange(par.num_classes) == y_train_agent[p][..., None]).astype(int)  ## Ip X K 10元素数组与y_train_agent[p][..., None]的每一个元素比较
    y_train_Bin.append(Temp)  ##长度10 的列表，每个元素是 6000个数据的二维标签值

######9.16

def calculate_Fixi(xi, par, x_train, y_train_Bin, num_data):
    Temp_H = np.exp(x_train @ xi)
    H = Temp_H/Temp_H.sum(axis=1)[:,None]  ##没错，但容易出错
    # H = np.clip(H,1e-9, 1.) ## I X K matrix
    # grad = np.zeros((par.num_features, par.num_classes))    ## J X K matrix
    H_hat = np.subtract(H, y_train_Bin)
    grad  = np.matmul(x_train.transpose(), H_hat) / num_data + 2. * par.gamma * xi  ## J X K matrix 相当于，f对w J*K的每个元素求导。
    return grad
grad = calculate_Fixi(par.Z_val[p], par, x_train_agent[p], y_train_Bin[p], par.total_data)

# 定义变量和参数

########9/17
##可用的
X = cp.Variable((4,3))
def fp(xi, par, x_train, y_train_Bin, num_data):
    temp = 0
    for i in range(x_train.shape[0]):
        temp = temp +  cp.log_sum_exp(x_train[i] @ xi)

    fpw = cp.sum(- cp.multiply(y_train_Bin, x_train @ xi) ) / num_data + temp/num_data    \
          + par.gamma / par.split_number * cp.sum_squares(xi) \
        ## J X K matrix 相当于，f对w J*K的每个元素求导

    return fpw

xi = cp.Variable((4,3))
xi = np.random.rand(4,3)
for i in range(x_train_agent[p].shape[0]):
    temp = temp + cp.log_sum_exp(x_train_agent[p,i] @ xi)

# 创建目标函数的CVXPY表达式
start_time = time.time()
objective_expr = fp(X, par, x_train_agent[p], y_train_Bin[p], par.total_data)
objective = cp.Minimize(objective_expr)
problem = cp.Problem(objective)
problem.solve(solver=cp.SCS)
end_time = time.time()
# problem.solve(solver=cp.SCS,verbose=True, warm_start = True)
# problem.solve(solver=cp.ECOS,verbose=True, max_iters=20)
# 获取结果
print("最优值：", problem.value)
print("最优解：", X.value)
print("Time: ", end_time - start_time)






###918
X = cp.Variable((784,10))

def fp(xi, par, x_train, y_train_Bin, num_data):
    fpw = 0
    for i in range(len(x_train)):
        fpw += - x_train[i] @ xi @ y_train_Bin[i].transpose() + cp.log_sum_exp(x_train[i] @ xi)
    # for i in range(H.shape[0]):
    #     Q += cp.log_sum_exp(H[i, :])
    fpw = fpw/num_data + par.gamma/ Agent * cp.sum_squares(xi)## J X K matrix 相当于，f对w J*K的每个元素求导。
    return fpw

# 创建目标函数的CVXPY表达式
objective_expr = fp(X, par, x_train_agent[p], y_train_Bin[p], par.total_data)
objective = cp.Minimize(objective_expr)
problem = cp.Problem(objective)
start_time = time.time()
# problem.solve(solver=cp.SCS,verbose=True, warm_start = True,eps=1e-2)
problem.solve(solver=cp.ECOS,verbose=True, max_iters = 45)

end_time = time.time()
# 获取结果
print("最优值：", problem.value)
print("最优解：", X.value)
print("Lzy: Using Time: ", end_time - start_time)



# 创建一个示例的 NumPy 数组
arr = X.value

# 使用 numpy.nonzero() 获取非零元素的索引
non_zero_indices = np.nonzero(arr)
non_zero_values = arr[non_zero_indices]

# 输出非零元素的索引和值
print("索引:", non_zero_indices)
print("值:", non_zero_values)


start_time = time.time()
for i in range(6000):
    x = cp.Variable((784,10))
    A = np.random.rand(784,10)
    objective = cp.Minimize(cp.sum(x - A))
    problem = cp.Problem(objective)
    # problem.solve(solver=cp.SCS,verbose=True, warm_start = True,eps=1e-2)
    problem.solve()

end_time = time.time()
print("Lzy: Using Time: ", end_time - start_time)



import cvxpy as cp

# 定义变量和目标函数
x = cp.Variable()
objective = cp.Minimize((x - 2)**2 + 1)

# 定义优化问题
problem = cp.Problem(objective)

# 求解优化问题
problem.solve()

# 获取变量x的解
optimal_x = x.value
print("Optimal x:", optimal_x)

# 原始函数
def original_function(x):
    return cp.sum(x)

# 代入最优解
result = original_function(optimal_x)
print("Result of the original function at the optimal x:", result)





###########df(x_i^k+1)

for p in range(par.split_number):
    start_grad = time.time()
    H = calculate_hypothesis(par.Z_val[p], x_train_agent[p])  ## I_p x K matrix  (see Functions.py)
    grad = calculate_gradient(par, par.total_data, H, x_train_agent[p],
                              y_train_Bin[p])  ## (see Functions.py) 对应Appendix G求梯度
    end_grad = time.time()
    Grad_Time += end_grad - start_grad

qq = np.zeros((66,1,3))
for k in range(65):
    p = 0
    i = 0
    def calculate_hypothesis(W_val, x_train):
        Temp_H = np.exp( np.matmul(x_train, W_val) )
        H = Temp_H/Temp_H.sum(axis=1)[:,None]    ##没错，但容易出错
        H = np.clip(H,1e-9, 1.) ## I X K matrix

        return H

    H = calculate_hypothesis(par.xik[k+1,p], x_train_agent[p])
    grad = np.zeros((par.num_features, par.num_classes))  ## J X K matrix
    H_hat = np.subtract(H, y_train_Bin[p])  ## I X K matrix
    grad = np.matmul(x_train_agent[p].transpose(), H_hat) / par.total_data + 2. * par.gamma * par.xik[k+1,p]  ## J X K matrix 相当于，f对w J*K的每个元素求导。

    shape_to_broadcast = (par.num_features, par.num_classes)
    broadcasted_array = np.tile(par.Augmented_incidence_matrix[:, i][:, np.newaxis, np.newaxis], shape_to_broadcast)
    temp2 = np.sum(par.tilde_lambda_ij[k] * broadcasted_array, axis=0)
    Ni = 2
    Nj = np.array(([1, 2], [0, 2], [0, 1]))

    # objective_expr = fp(X, par, x_train_agent[i], y_train_Bin[i], par.total_data) + cp.sum(cp.multiply(temp2, X)) \
    #                  + par.theta * Ni * cp.sum_squares(X) \
    #                  + par.theta * cp.sum(cp.multiply(Ni * (- par.xik[k, i] + par.zeta[k, i]) - np.sum(par.yik[k, Nj[i]]), X)) \
    #                  + par.theta * cp.sum(cp.multiply(par.aik[k + 1, i], X))


    gxik1 = temp2 + (2 * (2*par.xik[k+1,i] - par.xik[k,i] + par.zeta[k,i]) -  np.sum(par.yik[k, Nj[i]])  )   \
            + par.aik[k+1,i]

    qq[k] = (gxik1 + grad)

plt.plot(par.xik[:,0])




