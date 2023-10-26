import numpy as np  # (activate this if CPU is used)
# import cupy as np #(activate this if GPU is used)

import math
from Functions import *
import time
import numpy as np

############################################## CLOSED FORM
def First_Block_Problem_ClosedForm(par):
    ## Update W_val
    start = time.time()
    Temp = np.sum(par.Z_val - par.Lambdas_val / par.rho, axis=0) / par.split_number  ##algorithm 1 line 5.
    par.W_val = Temp
    end = time.time()

    return par, end - start


def Second_Block_Problem_ClosedForm(par, x_train_agent, y_train_agent, iteration):
    start = time.time()
    ## Data Label Reformatting: Integer -> Binary
    num_data = []
    y_train_Bin = []
    for p in range(par.split_number):
        num_data.append(y_train_agent[p].shape[0]) ## 6000*10
        Temp = (np.arange(par.num_classes) == y_train_agent[p][..., None]).astype(int)  ## Ip X K 10元素数组与y_train_agent[p][..., None]的每一个元素比较
        y_train_Bin.append(Temp) ##长度10 的列表，每个元素是 6000个数据的二维标签值


    Z_Change = 0
    Avg_Noise_Mag = 0
    Grad_Time = 0
    Noise_Time = 0

    for p in range(par.split_number):
        start_grad = time.time()
        H    = calculate_hypothesis(par.Z_val[p], x_train_agent[p])  ## I_p x K matrix  (see Functions.py)
        grad = calculate_gradient(par, par.total_data, H, x_train_agent[p], y_train_Bin[p])  ## (see Functions.py) 对应Appendix G求梯度
        end_grad = time.time()
        Grad_Time += end_grad - start_grad

        ## Generate Laplacian Noise
        tilde_xi = np.zeros((par.num_features, par.num_classes))
        if par.bar_eps_str != "infty":
            start_noise = time.time()
            tilde_xi = generate_laplacian_noise(par, H, par.total_data, x_train_agent[p], y_train_Bin[p],
                                                tilde_xi)  ## (see Functions.py)
            end_noise = time.time()
            Noise_Time += end_noise - start_noise
            Avg_Noise_Mag += np.mean(np.absolute(tilde_xi))

            ## Update Z_val
        if par.Algorithm == "ObjP":
            par.eta = float(par.a_str) / math.sqrt(iteration + 1)
            Z_Prev = par.Z_val[p]
            Temp = (1.0 / (par.rho + (1.0 / par.eta))) * (
                        -grad + par.rho * par.W_val + par.Lambdas_val[p] - tilde_xi + (1.0 / par.eta) * Z_Prev) ##极小化，KKT条件而已，已证明正确。
            Z_Change += np.absolute(Z_Prev - Temp)
            par.Z_val[p] = Temp

        if par.Algorithm == "ObjT":
            par.eta = float(par.a_str) / (iteration + 1) * (iteration + 1)
            Z_Prev = par.Z_val[p]
            Temp = (1.0 / par.rho) * (-grad + par.rho * par.W_val + par.Lambdas_val[p] - tilde_xi)
            Temp = np.clip(Temp, Z_Prev - par.eta, Z_Prev + par.eta)  ## Trust-Region using the infinity norm
            Z_Change += np.absolute(Z_Prev - Temp)
            par.Z_val[p] = Temp

    end = time.time()

    Avg_Noise_Mag = Avg_Noise_Mag / par.split_number
    return par, end - start, Avg_Noise_Mag, np.mean(Z_Change), Noise_Time, Grad_Time


def Base_First_Block_Problem_ClosedForm(par, x_train_agent, y_train_agent, Iteration):
    start = time.time()
    num_data = []
    y_train_Bin = []
    for p in range(par.split_number):
        num_data.append(y_train_agent[p].shape[0])
        Temp = (np.arange(par.num_classes) == y_train_agent[p][..., None]).astype(int)  ## Ip X K
        y_train_Bin.append(Temp)

    Z_Change = 0;
    Avg_Noise_Mag = 0;
    Grad_Time = 0;
    Noise_Time = 0
    for p in range(par.split_number):
        start_grad = time.time()
        ## Hypothesis
        H = calculate_hypothesis(par.Z_val[p], x_train_agent[p])  ## I_p x K matrix
        ## Gradient                                         
        grad = calculate_gradient(par, num_data[p], H, x_train_agent[p], y_train_Bin[p])
        end_grad = time.time()
        Grad_Time += end_grad - start_grad

        ## Decide eta        
        par.eta = calculate_eta_Base(par, num_data[p], Iteration)
        ## Update Z_val
        Temp = (1.0 / (par.rho + (1.0 / par.eta))) * (
                    -grad + par.rho * par.W_val + par.Lambdas_val[p] + (1.0 / par.eta) * par.Z_val[p])

        Z_Change += np.absolute(par.Z_val[p] - Temp)
        par.Z_val[p] = Temp

        ## Generate Matrix Normal Noise
        tilde_xi = np.zeros((par.num_features, par.num_classes))
        if par.bar_eps_str != "infty":
            start_noise = time.time()
            tilde_xi = generate_matrix_normal_noise(par, num_data[p], tilde_xi)
            end_noise = time.time()
            Noise_Time += end_noise - start_noise

            par.Z_val[p] += tilde_xi
        Avg_Noise_Mag += np.mean(np.absolute(tilde_xi))

    end = time.time()
    Avg_Noise_Mag = Avg_Noise_Mag / par.split_number
    return par, end - start, Avg_Noise_Mag, np.mean(Z_Change), Noise_Time, Grad_Time


def Base_Second_Block_Problem_ClosedForm(par):
    ## Update W_val
    start = time.time()
    Temp = np.sum(par.Z_val - par.Lambdas_val / par.rho, axis=0) / par.split_number
    par.W_val = Temp
    end = time.time()

    return par, end - start


def PP_ADMM_initialization(par):
    ##accumulative variable

    np.random.seed(1)
    temp1 = 10*np.random.rand(par.training_steps + 2, par.split_number, par.num_features, par.num_classes)
    temp2 = np.random.rand(par.training_steps + 2,2 * par.edge, par.num_features, par.num_classes)
    temp3 = np.random.rand(2 * par.edge, par.num_features, par.num_classes)
    temp4 = np.random.rand(par.split_number, par.num_features, par.num_classes)

    par.xik = temp1.copy()
    # par.xik[0] = np.array([par.qqq,par.qqq,par.qqq]) + temp1[0]
    par.aik = temp1.copy()
    par.zeta = temp1.copy()## random noise

    par.yik  = temp1.copy() ##1 masked variable y
    par.zik  = temp1.copy()##2 masked variable z
    ##pure x
    ##masked lambda
    par.tilde_lambda_ij = temp2.copy()

    par.gamma_ij0  = temp3.copy() ##rij0
    par.lambda_ij0 = temp3.copy()
    par.omega_ij0  = temp3.copy()   ## 5b 6 中的omega，6个，each agent holds 2

    # par.aik0  = temp4.copy()
    ## k = 0 initialization
    ##stacked xio for initlizing  wij0 (6)
    temp_x0  = temp3.copy() ## 5b 6 中的omega，6个，each agent holds 2
    temp_zeta0 = temp3.copy()

    stacking_index = np.where(par.Augmented_incidence_matrix == 1)[1] ## 12 13 23 21 31 32

    par.omega_ij0 = 2 * par.xik[0 ,stacking_index] + par.zeta[0, stacking_index] - par.theta * par.gamma_ij0  ## 6
##tilde lambda updating
    par.tilde_lambda_ij[0] = par.gamma_ij0 + par.lambda_ij0
##ai0 updating
    shape_to_broadcast = (par.num_features, par.num_classes)

    for i in range(par.split_number):
        broadcasted_array = np.tile(par.Augmented_incidence_matrix[:,i][:, np.newaxis, np.newaxis], shape_to_broadcast)
        par.aik[0,i] = np.sum( par.omega_ij0 * broadcasted_array, axis = 0 )


    return par



