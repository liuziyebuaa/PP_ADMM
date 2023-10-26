##本部分之有一个DP_ADMM算法
import numpy as np
#(activate this if CPU is used)
# import cupy as np #(activate this if GPU is used)
import time
from Models import *
from Functions import *
import cvxpy as cp
from softmax import *
from itertools import product

def centralized_solution(par,y_train_new,x_train_new,x_test, y_test):

    y_train_Bin_plus = (np.arange(y_train_new.max() + 1) == y_train_new[..., None]).astype(
        int)  ##60000维变成60000 *10  I X K
    num_data = y_train_Bin_plus.shape[0]


####
    X = cp.Variable((par.num_features, par.num_classes))
    objective_expr = fp(X, par, x_train_new, y_train_Bin_plus, par.total_data)

    objective = cp.Minimize(objective_expr)
    problem = cp.Problem(objective)
    problem.solve(solver=cp.SCS)

    par.centralized_x = X.value

    H = calculate_hypothesis(par.centralized_x, x_train_new)  ## I x K matrix (see Functions.py)
    par.centralized_cost = calculate_centralized_cost(par, par.centralized_x, num_data, H, y_train_Bin_plus)
    par.centralized_accuracy = calculate_accuracy(par, par.centralized_x, x_test, y_test)  ## Compute testing accuracy (see Functions.py)

    return par

def centralized_solution_optimal(par,y_train_new,x_train_new,x_test, y_test):

    y_train_Bin_plus = (np.arange(y_train_new.max() + 1) == y_train_new[..., None]).astype(int)  ##60000维变成60000 *10  I X K
    num_data = y_train_Bin_plus.shape[0]

    ####
    proximal_parameter = [1e-8, 1e-7,1e-6,1e-5,1e-4,1e-3,0.01,0.1,1,10]
    beta = [1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9,1e-10,0]
    proximal_parameter_beta =  list(product(proximal_parameter, beta))
    set_of_accuracy = []
    set_of_cost = []

    for proximal_parameter, beta in proximal_parameter_beta:

        par.theta = proximal_parameter
        par.gamma = float(beta)

        X = cp.Variable((par.num_features, par.num_classes))
        objective_expr = fp(X, par, x_train_new, y_train_Bin_plus, par.total_data)

        objective = cp.Minimize(objective_expr)
        problem = cp.Problem(objective)
        problem.solve(solver=cp.SCS)

        temp_centralized_x = X.value

        H = calculate_hypothesis(temp_centralized_x, x_train_new)  ## I x K matrix (see Functions.py)
        temp_centralized_cost = calculate_centralized_cost(par, temp_centralized_x, num_data, H, y_train_Bin_plus)
        temp_centralized_accuracy = calculate_accuracy(par, temp_centralized_x, x_test, y_test)   ## Compute testing accuracy (see Functions.py)

        set_of_cost.append([proximal_parameter, beta, temp_centralized_cost])
        set_of_accuracy.append([proximal_parameter, beta, temp_centralized_accuracy])

    set_of_accuracy = sorted(set_of_accuracy, key=lambda x: x[2])
    min_centralized_cost = min(set_of_cost, key=lambda x: x[2])
    max_centralized_accuracy =max(set_of_accuracy, key=lambda x: x[2])

    max_accuracy = max(set_of_accuracy, key=lambda x: x[2])[2]
    max_elements = [element for element in set_of_accuracy if element[2] == max_accuracy]


    # 使用自定义比较函数来找到第三个数最小的元素
    # print("min_centralized_cost: " + str(min_centralized_cost))
    for i in range(len(max_elements)):
        print(max_elements[i])
    print("min_centralized_accuracy: " + str(max_centralized_accuracy))

    return par



