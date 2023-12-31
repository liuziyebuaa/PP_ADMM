import numpy as np #(activate this if CPU is used)
# import cupy as np #(activate this if GPU is used)
import numpy as np
from scipy.optimize import minimize
import math

from scipy.stats import matrix_normal
import time


def calculate_hypothesis(W_val, x_train):
    Temp_H = np.exp( np.matmul(x_train, W_val) )
    H = Temp_H/Temp_H.sum(axis=1)[:,None]    ##没错，但容易出错
    H = np.clip(H,1e-9, 1.) ## I X K matrix

    return H

def calculate_gradient(par, num_data, H, x_train, y_train_Bin):
    grad = np.zeros((par.num_features, par.num_classes))   ## J X K matrix
    H_hat= np.subtract(H, y_train_Bin)    ## I X K matrix
    grad = np.matmul(x_train.transpose(), H_hat) / num_data + 2. * par.gamma * par.W_val ## J X K matrix 相当于，f对w J*K的每个元素求导。

    return grad

def calculate_cost(par, num_data, H, y_train_Bin, k):
    if par.Algorithm != "PP_ADMM":
        return -np.sum( np.multiply(y_train_Bin, np.log(H)) ) / num_data + par.gamma * np.sum( np.square(par.W_val) )
    else:
        return -np.sum( np.multiply(y_train_Bin, np.log(H)) ) / num_data + par.gamma * np.sum( np.square(par.xik[k+1]) )



def calculate_centralized_cost(par, X, num_data, H, y_train_Bin,):
        return -np.sum( np.multiply(y_train_Bin, np.log(H)) ) / num_data + par.gamma * np.sum( np.square(X) )



def calculate_accuracy(par: object, W: object, x_train: object, y_train: object) -> object:
    H = calculate_hypothesis(W, x_train)
    H_argmax = np.argmax(H, axis=1)
    H_argmax = H_argmax.astype(int)
    return np.mean( np.equal(H_argmax, y_train ) )

##贼几把合理，H最大值位置H_argmax代表分类，第三行比较分类


def calculate_residual(par):
    Temp = np.absolute(par.W_val - par.Z_val)
    residual = np.sum(Temp) / par.split_number
    return residual

def calculate_sum_residual(par,k):
    AVE = np.sum(  par.xik[k+1], axis = 0 ) / par.split_number
    B = np.tile(AVE, (par.split_number, 1))
    C = B.reshape(par.split_number, par.num_features, par.num_classes)
    residual = np.sum(  np.absolute(par.xik[k+1] - C) )

    return residual

def generate_laplacian_noise(par, H, num_data, x_train, y_train_Bin, tilde_xi):

    H_hat=np.subtract(H, y_train_Bin)    ## I_p X K matrix
    H_hat_abs = np.absolute(H_hat)

    x_train_sum = np.sum(x_train, axis = 1)
    H_hat_abs_sum = np.sum(H_hat_abs, axis = 1)
    x_train_H_hat_abs = np.multiply(x_train_sum,H_hat_abs_sum) / num_data
    bar_lambda = np.max(x_train_H_hat_abs)/float(par.bar_eps_str)

    tilde_xi_shape = par.M + bar_lambda
    tilde_xi = np.random.laplace( par.M, tilde_xi_shape, [par.num_features, par.num_classes])

    return tilde_xi

def calculate_eta_Base(par,num_data, Iteration):

    delta = 1e-6  ## (epsilon, delta)-differential privacy
    c1 = num_data*1
    c3 = num_data*0.25
    cw = math.sqrt( par.num_features*par.num_classes*4 )

    if par.bar_eps_str != "infty":
        par.eta = 1.0 / ( c3 + 4.0*c1*math.sqrt( par.num_features*par.num_classes*(Iteration+1)*math.log(1.25/delta)  )/(num_data*float(par.bar_eps_str)*cw)  )
    else:
        par.eta = 1.0 / c3

    par.eta = par.eta * float(par.a_str)

    return par.eta

def generate_matrix_normal_noise(par, num_data,tilde_xi):
    c1 = num_data*1
    delta = 1e-6  ## 1e-308, 1e-6

    sigma = 2*c1*math.sqrt(2*math.log(1.25/delta)) / (num_data*float(par.bar_eps_str)*(par.rho + 1.0/par.eta))

    tilde_xi_shape = par.M + sigma*sigma
    tilde_xi = np.random.normal( par.M, tilde_xi_shape, [par.num_features, par.num_classes])

    return tilde_xi

def hyperparameter_rho(par, iteration):
    if par.rho_str == "dynamic_1" or par.rho_str == "dynamic_2":
        if par.Instance == "MNIST":
            c1 = 2.0;
            c2 = 5.0;
            Tc = 10000.0;
            rhoC = 1.2
        if par.Instance == "FEMNIST":
            c1 = 0.005;
            c2 = 0.05;
            Tc = 2000.0;
            rhoC = 1.2
        else:
            c1 = 2.0;
            c2 = 5.0;
            Tc = 10000.0;
            rhoC = 1.2

        if par.bar_eps_str == "infty":
            par.rho = c1 * math.pow(rhoC, math.floor((iteration + 1) / Tc))
        else:
            par.rho = c1 * math.pow(rhoC, math.floor((iteration + 1) / Tc)) + c2 / float(par.bar_eps_str)
        if par.rho_str == "dynamic_2":
            par.rho = par.rho / 100.0
    else:
        par.rho = float(par.rho_str)

        # the parameter is bounded above
    if par.rho > 1e9:
        par.rho = 1e9







