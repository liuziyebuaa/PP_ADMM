import numpy as np
from scipy.optimize import fsolve



X = cp.Variable((4,3))
def fp(xi, par, x_train, y_train_Bin, num_data):
    fpw = cp.sum(- cp.multiply(y_train_Bin , x_train @ xi)  + cp.log_sum_exp(x_train @ xi))/ par.split_number   \
          +  par.gamma/ Agent * cp.sum_squares(xi)## J X K matrix 相当于，f对w J*K的每个元素求导。
    return fpw

# 创建目标函数的CVXPY表达式
objective_expr = fp(X, par, x_train_agent[p], y_train_Bin[p], par.total_data)
objective = cp.Minimize(objective_expr)
problem = cp.Problem(objective)
problem.solve(solver=cp.SCS)

temp = calculate_Fixi(xi,par, x_train, y_train_Bin, 10)

start = time.time()
initial_guess = np.zeros((4, 2)).flatten(order='F')    ## J X K matrix
solution = fsolve(calculate_Fixi, initial_guess, args = (par, x_train, y_train_Bin, 10))
end = time.time()
print("Running for " + str(end - start) + 's')

##虽然xi必须是向量，但是在函数中可以用先reshape的方式然后。

start = time.time()
initial_guess = np.zeros((a, b)).flatten(order='F')    ## J X K matrix
ss = fsolve(calculate_Fixi, initial_guess, args=(a, b, C),method ='lm')
end = time.time()
print("Running for " + str(end - start) + 's')

##


