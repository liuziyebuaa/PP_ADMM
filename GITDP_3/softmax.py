import cvxpy as cp
import numpy as np

def fp(xi, par, x_train, y_train_Bin, num_data):
    temp = 0
    for i in range(x_train.shape[0]):
        temp = temp +  cp.log_sum_exp(x_train[i] @ xi)

    fpw = cp.sum(- cp.multiply(y_train_Bin, x_train @ xi) ) / num_data + temp/num_data    \
          + par.gamma / par.split_number * cp.sum_squares(xi) \
        ## J X K matrix 相当于，f对w J*K的每个元素求导


    return fpw


def f_test(xi, x_train_agent):
    temp = 0
    for i in range(x_train_agent.shape[0]):
        temp = temp + cp.sum_squares( xi - x_train_agent[0])

    return temp

def f_test2(xi, x_train_agent):
    temp = 0
    for i in range(x_train_agent.shape[0]):
        temp = temp + np.sum(( xi - x_train_agent[0])**2)

    return temp


# ###
#
# wine = load_wine()
# xx = wine.data  # 特征数据
# yy = wine.target  # 标签数据
#
# x_train, x_test, y_train, y_test = train_test_split(xx, yy, test_size=0.2, random_state=42)
#
# par.split_number = 1
# num_data = x_train.shape[0]
# y_train_Bin = []
# Temp = (np.arange(par.num_classes) == y_train[..., None]).astype(int)  ## Ip X K 10元素数组与y_train_agent[p][..., None]的每一个元素比较
# y_train_Bin= Temp  ##长度10 的列表，每个元素是 6000个数据的二维标签值
#
# xi = cp.Variable((13, 3))
#
# objective_expr = fp(xi, par, x_train, y_train_Bin, num_data)
#
#
# objective = cp.Minimize(objective_expr)
# problem = cp.Problem(objective)
# problem.solve(solver=cp.SCS)
# print("X = " + str(xi.value) )
#
#
# y_train_Bin_plus = (np.arange(y_train.max() + 1) == y_train[..., None]).astype(int)  ##60000维变成60000 *10  I X K
# num_data = y_train_Bin_plus.shape[0]
# # H = calculate_hypothesis(par.W_val, x_train_new) ## I x K matrix (see Functions.py)\
# H = calculate_hypothesis(par.xik[k + 1, 0], x_train)  ## I x K matrix (see Functions.py)
# cost = calculate_cost(par, num_data, H, y_train_Bin_plus, k)  ## Compute the objective function value (see Functions.py)
#
# accuracy_test = calculate_accuracy(par, xi.value, x_test,  y_test)  ## Compute testing accuracy (see Functions.py)
#
# residual = calculate_sum_residual(par, k)  ##   Compute concensus violation (see Functions.py)