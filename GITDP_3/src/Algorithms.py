##本部分之有一个DP_ADMM算法
import numpy as np
#(activate this if CPU is used)
# import cupy as np #(activate this if GPU is used)

import math
import time
from Models import *
from Functions import *
import cvxpy as cp
from softmax import *
from centralized_operation import *

def DP_IADMM(par, x_train_agent, y_train_agent, x_train_new, y_train_new, x_test, y_test, file1):
    ## Data Label Reformatting: Integer -> Binary (train_data) for calculating the objective function value
    y_train_Bin = ( np.arange(y_train_new.max()+1) == y_train_new[...,None] ).astype(int) ## I X K
    num_data = y_train_Bin.shape[0]

    # ITERATION, COST,  ACCURACY_TEST, RESIDUAL, ELAPSED_TIME, ITER_TIME, RUNTIME_1, RUNTIME_2, GRAD_TIME, NOISE_TIME, AVG_NOISE_MAG, Z_CHANGE_MEAN \
    # = [],      [],         [],        [],          [],        [],         [],         [],       [],         [],         [],              []

    ### [0] Initialization
    ## Variables
    par.W_val = np.zeros((par.num_features, par.num_classes))  ## Global model parameters 768*10维的模型参数，对应w
    par.Z_val = np.zeros((par.split_number, par.num_features, par.num_classes)); ## Local model parameters defined for every agent
    #因此，这行代码创建了一个大小为 (10, 784, 62) 的三维数组，用于存储每个代理的本地模型参数。每个代理都有一个自己的参数矩阵，大小为 (784, 62)，并且这些参数都被初始化为全零

    par.Lambdas_val = np.zeros((par.split_number, par.num_features, par.num_classes)); ## Dual variables

    ## Matrix Normal Distribution used for Gaussian Mechanism for "Base" algorithm
    par.M = np.zeros((par.num_features, par.num_classes  ))
    par.U = np.zeros((par.num_features, par.num_features ))
    par.V = np.zeros((par.num_classes , par.num_classes  ))

    start_time_initial = time.time()
    title=" Iter    Train_Cost     Test_Acc     Violation    Elapsed(s)     IterT(s)   Solve_1(s)   Solve_2(s)    GradT(s)   NoiseT(s)  AbsNoiseMag    Z_change     AdapRho \n"
    print(title)
    file1.write(title)

    for iteration in range(par.training_steps + 1):
        start_time_iter = time.time()
        ### Hyperparameter Rho
        hyperparameter_rho(par, iteration)  ## see Functions.py

        ### [1] First Block Problem
        if par.Algorithm == "OutP":
            par, Runtime_1, Avg_Noise_Mag, z_change_mean, Noise_Time, Grad_Time  = Base_First_Block_Problem_ClosedForm(par, x_train_agent, y_train_agent, iteration) ## see Models.py
        else:
            par, Runtime_1 = First_Block_Problem_ClosedForm(par) ## see Models.py  一般是运行这个

        ### [2] Second Block Problem
        if par.Algorithm == "OutP":
            par, Runtime_2 = Base_Second_Block_Problem_ClosedForm(par) ## see Models.py
        else:
            par, Runtime_2, Avg_Noise_Mag, z_change_mean, Noise_Time, Grad_Time = Second_Block_Problem_ClosedForm(par, x_train_agent, y_train_agent, iteration) ## see Models.py


        ### [3] Dual update
        par.Lambdas_val += par.rho*(par.W_val - par.Z_val)

        end_time = time.time()
        iter_time = end_time - start_time_iter
        elapsed_time = end_time - start_time_initial

        H = calculate_hypothesis(par.W_val, x_train_new) ## I x K matrix (see Functions.py)
        cost = calculate_cost(par, num_data, H, y_train_Bin,iteration)  ## Compute the objective function value (see Functions.py)
        accuracy_test = calculate_accuracy(par, par.W_val, x_test, y_test)   ## Compute testing accuracy (see Functions.py)
        residual = calculate_residual(par) ## Compute concensus violation (see Functions.py)

        ##record the data and to print
        par.ITERATION.append(iteration)
        par.COST.append(cost)
        par.ACCURACY_TEST.append(accuracy_test)
        par.RESIDUAL.append(residual)
        par.ELAPSED_TIME.append(elapsed_time)
        par.ITER_TIME.append(iter_time)
        par.RUNTIME_1.append(Runtime_1)
        par.RUNTIME_2.append(Runtime_2)
        par.GRAD_TIME.append(Grad_Time)
        par.NOISE_TIME.append(Noise_Time)
        par.AVG_NOISE_MAG.append(Avg_Noise_Mag)
        par.Z_CHANGE_MEAN.append(z_change_mean)

        results = '%4d %12.6e %12.6e %12.3e %12.2f %12.2f %12.2f %12.2f %12.2f %12.2f %12.6e %12.6e %12.6e \n' %(iteration, cost, accuracy_test, residual, elapsed_time, iter_time, Runtime_1,  Runtime_2, Grad_Time, Noise_Time, Avg_Noise_Mag, z_change_mean, par.rho)
        print(results)
        file1.write(results)

    return par.W_val, cost, file1

def PP_ADMM(par, x_train_agent, y_train_agent, x_train_new, y_train_new, x_test, y_test, file1):
    ## Data Label Reformatting: Integer -> Binary (train_data) for calculating the objective function value

    ## Variables
    par = PP_ADMM_initialization(par)

    num_data = []
    y_train_Bin = []
    for p in range(par.split_number):
        num_data.append(y_train_agent[p].shape[0])  ## 6000*10
        Temp = (np.arange(par.num_classes) == y_train_agent[p][..., None]).astype(int)  ## Ip X K 10元素数组与y_train_agent[p][..., None]的每一个元素比较
        y_train_Bin.append(Temp)  ##长度10 的列表，每个元素是 6000个数据的二维标签值

    start_time_initial = time.time()
    title=" Iter    Train_Cost     Test_Acc     Violation    Elapsed(s)     IterT(s)   Solve_1(s)   Solve_2(s)    GradT(s)   NoiseT(s)  AbsNoiseMag    Z_change     AdapRho \n"
    print(title)
    file1.write(title)

    temp_y = np.zeros((2 * par.edge, par.num_features, par.num_classes))    ## 5b 6 中的omega，6个，each agent holds 2
    temp_z = np.zeros((2 * par.edge, par.num_features, par.num_classes))

    Ni = 2
    Nj = np.array(([1,2],[0,2],[0,1]))

    shape_to_broadcast = (par.num_features, par.num_classes)

    for k in range(par.training_steps + 1):
    for k in range(4998,10000):
        start_time_iter = time.time()

        ##updating yik
        par.yik[k] = par.xik[k] + par.zeta[k]

        for i in range(par.split_number):
            ##updating aik 4a
            par.aik[k+1,i] = par.aik[k,i] - 2* Ni * par.yik[k,i] + 2*np.sum( par.yik[ k, Nj[i] ],axis = 0)

            ###argmin, keystep 4b

            # stack = np.sum( par.tilde_lambda_ij[k, par.Augmented_incidence_matrix[:,i] ], axis = 0)

            broadcasted_array = np.tile(par.Augmented_incidence_matrix[:, i][:, np.newaxis, np.newaxis],shape_to_broadcast)
            temp2 = np.sum(par.tilde_lambda_ij[k] * broadcasted_array, axis=0)

            X = cp.Variable((par.num_features, par.num_classes))
            objective_expr = fp(X, par, x_train_agent[i], y_train_Bin[i], par.total_data) +  cp.sum (cp.multiply( temp2 , X ) ) \
                              + par.theta * Ni * cp.sum_squares(X)   \
                              + par.theta * cp.sum(cp.multiply ( Ni * (- par.xik[k,i] + par.zeta[k,i] ) - np.sum( par.yik[ k, Nj[i] ],axis = 0) , X )  )   \
                              + par.theta * cp.sum(cp.multiply( par.aik[k+1,i] , X ) )



            objective = cp.Minimize(objective_expr)
            problem = cp.Problem(objective)
            problem.solve(solver=cp.SCS)

            par.xik[k+1,i] = X.value
            #####################


        stacking_index_i = np.where(par.Augmented_incidence_matrix == 1)[1]
        stacking_index_j = np.where(par.Augmented_incidence_matrix == -1)[1]

        par.zik[k] = par.xik[k+1] + 1/3 * par.zeta[k]

        temp_xik1 = par.xik[k + 1, stacking_index_i]
        temp_zetaik1 = par.zeta[k + 1, stacking_index_i]
        temp_zjk = par.zik[k, stacking_index_j]

        par.tilde_lambda_ij[k+1] = par.tilde_lambda_ij[k] + par.theta * (temp_xik1 + temp_zetaik1 - 3 * temp_zjk )

        ###chasing Vk

        end_time = time.time()
        iter_time = end_time - start_time_iter
        elapsed_time = end_time - start_time_initial


            ### Display intermediat results
        # if k % par.display_step == 0: #取模打印


        # H = calculate_hypothesis(par.W_val, x_train_new) ## I x K matrix (see Functions.py)\
        y_train_Bin_plus = (np.arange(y_train_new.max() + 1) == y_train_new[..., None]).astype(int)  ##60000维变成60000 *10  I X K

        H = calculate_hypothesis(par.xik[k+1,0], x_train_new)  ## I x K matrix (see Functions.py)
        cost = calculate_cost(par, x_train_new.shape[0], H, y_train_Bin_plus,k)  ## Compute the objective function value (see Functions.py)

        accuracy_test = calculate_accuracy(par, par.xik[k+1,i], x_test, y_test)   ## Compute testing accuracy (see Functions.py)
        residual = calculate_sum_residual(par,k) ##   Compute concensus violation (see Functions.py)

        ##record the data and to print _lzy
        par.ITERATION.append(k)
        par.COST.append(cost)
        par.ACCURACY_TEST.append(np.average(accuracy_test))
        par.RESIDUAL.append(0)
        par.ELAPSED_TIME.append(elapsed_time)
        par.ITER_TIME.append(iter_time)
        par.RUNTIME_1.append(0)
        par.RUNTIME_2.append(0)
        par.GRAD_TIME.append(0)
        par.NOISE_TIME.append(0)
        par.AVG_NOISE_MAG.append(0)
        par.Z_CHANGE_MEAN.append(0)

        results = '%5d  %.4f  %.4f %12.3e %12.2f %12.2f %12.2f %12.2f %12.2f %12.2f %12.3e %12.3e %12.3e \n' %(k, cost, accuracy_test, residual, 0, iter_time, 0,  0, 0, 0, 0, 0, par.rho)
        print(results)
        file1.write(results)


    return par, par.xik[-1,0], cost, file1



def PP_ADMM_TEST(par, x_train_agent, y_train_agent, x_train_new, y_train_new, x_test, y_test, file1):
    ## Data Label Reformatting: Integer -> Binary (train_data) for calculating the objective function value

    ## Variables
    par = PP_ADMM_initialization(par)

    num_data = []
    y_train_Bin = []
    for p in range(par.split_number):
        num_data.append(y_train_agent[p].shape[0])  ## 6000*10
        Temp = (np.arange(par.num_classes) == y_train_agent[p][..., None]).astype(int)  ## Ip X K 10元素数组与y_train_agent[p][..., None]的每一个元素比较
        y_train_Bin.append(Temp)  ##长度10 的列表，每个元素是 6000个数据的二维标签值

    start_time_initial = time.time()
    title=" Iter    Train_Cost    \n"
    print(title)
    file1.write(title)

    temp_y = np.zeros((2 * par.edge, par.num_features, par.num_classes))    ## 5b 6 中的omega，6个，each agent holds 2
    temp_z = np.zeros((2 * par.edge, par.num_features, par.num_classes))

    Ni = 2
    Nj = np.array(([1,2],[0,2],[0,1]))

    shape_to_broadcast = (par.num_features, par.num_classes)

    ## centralized

    ##

    for k in range(par.training_steps + 1):
        start_time_iter = time.time()

        ##updating yik
        par.yik[k] = par.xik[k] + par.zeta[k]

        for i in range(par.split_number):
            ##updating aik 4a
            par.aik[k+1,i] = par.aik[k,i] - 2* Ni * par.yik[k,i] + 2*np.sum( par.yik[ k, Nj[i] ],axis = 0)

            ###argmin, keystep 4b
            X = cp.Variable((par.num_features, par.num_classes))

            # stack = np.sum( par.tilde_lambda_ij[k, par.Augmented_incidence_matrix[:,i] ], axis = 0)

            broadcasted_array = np.tile(par.Augmented_incidence_matrix[:, i][:, np.newaxis, np.newaxis],shape_to_broadcast)
            temp2 = np.sum(par.tilde_lambda_ij[k] * broadcasted_array, axis=0)

            objective_expr = f_test(X, x_train_agent[i]) +  cp.sum (cp.multiply( temp2 , X ) )     \
                              + par.theta * Ni * cp.sum_squares(X)   \
                              + par.theta * cp.sum (cp.multiply (Ni * ( - par.xik[k,i] + par.zeta[k,i] ) - np.sum( par.yik[ k, Nj[i] ], axis = 0) , X )  )   \
                              + par.theta * cp.sum (cp.multiply( par.aik[k+1,i] , X ) )

            # H = calculate_hypothesis(par.xik[k,i], x_train_agent[i])
            # objective_expr =   cp.sum (cp.multiply( calculate_gradient(par, num_data[i], H, x_train_agent[i], y_train_Bin[i]) + temp2 , X ) )     \
            #                   + par.theta * Ni * cp.sum_squares(X) \
            #                   + par.theta * cp.sum (cp.multiply (Ni * ( - par.xik[k,i] + par.zeta[k,i] ) - np.sum( par.yik[ k, Nj[i] ]) , X )  )   \
            #                   + par.theta * cp.sum (cp.multiply( par.aik[k+1,i] , X ) )

            objective = cp.Minimize(objective_expr)
            problem = cp.Problem(objective)
            problem.solve(solver=cp.SCS)

            par.xik[k+1,i] = X.value
            #####################


        stacking_index_i = np.where(par.Augmented_incidence_matrix == 1)[1]
        stacking_index_j = np.where(par.Augmented_incidence_matrix == -1)[1]

        par.zik[k] = par.xik[k+1] + 1/3 * par.zeta[k]

        temp_xik1 = par.xik[k + 1, stacking_index_i]
        temp_zetaik1 = par.zeta[k + 1, stacking_index_i]
        temp_zjk = par.zik[k, stacking_index_j]

        par.tilde_lambda_ij[k+1] = par.tilde_lambda_ij[k] + par.theta * (temp_xik1 + temp_zetaik1 - 3 * temp_zjk )

        ###chasing Vk



        end_time = time.time()
        iter_time = end_time - start_time_iter
        elapsed_time = end_time - start_time_initial


            ### Display intermediat results
        # if k % par.display_step == 0: #取模打印

        cost = f_test2(par.xik[k,0], x_train_new)



        ##record the data and to print _lzy
        par.ITERATION.append(k)
        par.COST.append(cost)


        results = '%4d   %.2f   \n' %(k, cost)
        print(results)
        file1.write(results)


    return par, par.xik[-1,0], cost, file1
