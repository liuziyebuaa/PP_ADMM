from Read import *
from Algorithms import *
from Structure import *
import os
from gpuinfo import GPUInfo
from PLOT import *

def main(Instance, Agent, Algorithm, Hyperparameter, ScalingConst, Epsilon, TrainingSteps, DisplaySteps, plot_what, ylabel, beta, Augmented_incidence_matrix,seed):

    GPU = GPUInfo.check_empty()  # Check if GPU is used.
    print("**********************GPU=", GPU)

    ## Parameters
    par = Parameters()
    par.Instance = Instance
    par.Algorithm = Algorithm
    par.rho_str = Hyperparameter
    par.a_str = ScalingConst
    par.bar_eps_str = Epsilon
    par.training_steps = int(TrainingSteps)
    par.display_step = int(DisplaySteps)
    par.gamma = float(beta)  ## regularizer parameter 其实应该是beta
    par.edge = 3
    par.split_number = int(Agent)
    par.Augmented_incidence_matrix = Augmented_incidence_matrix
    par.theta = float(ScalingConst)
    par.qqq = np.array([[  493.75106794,   400.32100504,  -894.08104691],
       [ 1200.35690677,   148.63416856, -1348.99578433],
       [-1493.24735419,  -162.13543927,  1655.37952693],
       [ -863.29581903,  -624.84324801,  1488.13843389]])
    par.randomseed =seed

    ## Read Instance
    x_test, y_test, x_train_new, y_train_new, x_train_agent, y_train_agent = Read(par)
    par.total_data = x_train_new.shape[0]

    #### Write output
    foldername = "Outputs"
    filename = 'Results_%s_Agent_%s_%s_rho_%s_a_%s_eps_%s' % (
        par.Instance, par.split_number, par.Algorithm, par.rho_str, par.a_str, par.bar_eps_str)

    file_ext = '.txt'
    Path = './%s/%s%s' % (foldername, filename, file_ext)
    uniq = 1
    while os.path.exists(Path):
        Path = './%s/%s_%d%s' % (foldername, filename, uniq, file_ext)
        uniq += 1
    file1 = open(Path, "w")

    # ##centralized
    # temp1 = par.theta
    # temp2 = par.beta
    # par = centralized_solution_optimal(par, y_train_new, x_train_new, x_test, y_test)
    # par.theta =temp1
    # par.beta=temp

    par = centralized_solution(par, y_train_new, x_train_new, x_test, y_test)

    #### Training Process
    DP_ADMM_algorithm_name_set = ['OutP','ObjP','ObjT']

    if par.Algorithm in DP_ADMM_algorithm_name_set:
        W, cost, file1 = DP_IADMM(par, x_train_agent, y_train_agent, x_train_new, y_train_new, x_test, y_test, file1)
    elif par.Algorithm == 'PP_ADMM':
        par, W, cost, file1 = PP_ADMM(par, x_train_agent, y_train_agent, x_train_new, y_train_new, x_test, y_test, file1)
    else:
        print("Algorithm name is error, not PP-ADMM, not DP-ADMM")

    #### Testing Accuracy
    accuracy = calculate_accuracy(par, W, x_test, y_test)

    #### PRINT & WRITE
    # 生成報告
    # 设定需要打印的变量
    GPU_is = "GPU=%s \n" % (GPU)
    Instance_Name = "Instance=%s \n" % (par.Instance)
    Agent_num = "#Agents=%s \n" % (par.split_number)
    Feature_num = "#Features=%s \n" % (par.num_features)
    Class_num = "#Classes=%s \n" % (par.num_classes)
    Algorithm_Name = "Algorithm=%s \n" % (par.Algorithm)
    Hyperparameter_Name = "Hyperparameter_rho=%s \n" % (par.rho_str)
    ScalingConst_Name = "ScalingConst_a=%s \n" % (par.a_str)
    DP_Epsilon = "DP_Epsilon=%s \n" % (par.bar_eps_str)
    training_cost = "cost (training)=%s \n" % (cost)
    testing_accuracy = "accuracy (testing)=%s \n" % (accuracy)

    ##这一步是打印变量
    print(GPU_is)
    print(Instance_Name)
    print(Agent_num)
    print(Feature_num)
    print(Class_num)
    print(Algorithm_Name)
    print(Hyperparameter_Name)
    print(ScalingConst_Name)
    print(DP_Epsilon)
    print(training_cost)
    print(testing_accuracy)

    file1.write("\n \n")
    file1.write(GPU_is)
    file1.write(Instance_Name)
    file1.write(Agent_num)
    file1.write(Feature_num)
    file1.write(Class_num)
    file1.write(Algorithm_Name)
    file1.write(Hyperparameter_Name)
    file1.write(ScalingConst_Name)
    file1.write(DP_Epsilon)
    file1.write(training_cost)
    file1.write(testing_accuracy)
    file1.close()

    return par
