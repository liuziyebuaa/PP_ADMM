import sys
sys.path.append("./src")
from Main import *
from Structure import *
from PLOT import *
from itertools import product
import numpy as np
import os


Augmented_incidence_matrix= np.array(([1, 1, 0, -1, -1, 0], [-1, 0, 1, 1, 0, -1], [0, -1, -1, 0, 1, 1])).transpose()

proximal_parameter = "1"
beta = "1e-5"
par = main("CANCER","3", "PP_ADMM", "dynamic_1", proximal_parameter, "infty", "10000", "1", \
           'ACCURACY_TEST', 'y_label_test', beta, Augmented_incidence_matrix,3420)
# par = main("MNIST", "2", "ObjP", "dynamic_1", proximal_parameter, "infty", "1000", "1", 'ACCURACY_TEST', 'y_label_test', beta)
# Instance, Agent, Algorithm, Hyperparameter, ScalingConst, Epsilon, TrainingSteps, DisplaySteps, plot_what, ylabel, beta, Augmented_incidence_matrix, seed\
#    = "CANCER","3", "PP_ADMM", "dynamic_1", "1", "100", "100", "1", 'ACCURACY_TEST', 'y_label_test', beta, Augmented_incidence_matrix, 3420

plt.rcParams['figure.figsize'] = (8, 9)
PLOT(par, ['COST',"ACCURACY_TEST"],'0')
# PLOT_PP_ADMM(par)
column_number= 3
temp = []
file_path = os.path.abspath(r'C:\Users\Administrator\Desktop\GITDP_3\Outputs\Results_IRIS_Agent_3_ObjP_rho_dynamic_1_a_1_eps_0.1_2.txt')
with open(file_path, 'r') as file:
    for line in file:
        # 以空格分隔每一行的数据
        parts = line.strip().split()
        if len(parts) >= column_number:
            column2_data = parts[(column_number - 1)]
            temp.append(column2_data)

temp = temp[1:-1]

temp = [float(item) for item in temp]




    ##def main(Instance, Agent, Algorithm, Hyperparameter, ScalingConst, Epsilon, TrainingSteps, DisplaySteps, plot_what,

#     set_of_terminal_accuracy.append([proximal_parameter, beta, temp[-1]])
#
# max_element = max(set_of_terminal_accuracy, key=lambda x: x[2])
# print("proximal_parameter = " + str(max_element[0]) +", beta = " + str(max_element[1]) + ", max accuracy = " + str(max_element[2]))
#
# proximal_parameter, beta = (max_element[:2])
# par = main("IRIS","1", "ObjT", "dynamic_1", str(proximal_parameter), "10000", "3000", "1", 'ACCURACY_TEST', 'y_label_test',str(beta))
# PLOT(par, 'ACCURACY_TEST')

### [1] Instances:
## "MNIST": total number of training data => I = 60000; the number of agents => P \in \{5, 10, 50, 100, ... \}
## "FEMNIST":
##      (large)  Original FEMNIST data from leaf (https://leaf.cmu.edu)
##      (medium) Extract 25% of the original data
##      (small)  Extract  5% of the original data
## Note: the number of agents for FEMNIST is given, e.g., P=195 for small FEMNIST.
#Instance, Agent, Algorithm, Hyperparameter, ScalingConst, Epsilon, TrainingSteps, DisplaySteps, plot_what,ylabel = "IRIS","3", "ObjP", "dynamic_1", "1.0", "0.05", "300", "1", 'ACCURACY_TEST', 'y_label_test'
### [2] Algorithms:
## "OutP":  DP-IADMM-Prox  (output perturbation)
## "ObjP":  DP-IADMM-Prox  (objective perturbation)
## "ObjT":  DP-IADMM-Trust (objective perturbation)
## PPADMM :

### [3] Hyperparameter \rho^t
## "static":  \rho^t \in \{ "0.1", "1.0", "10.0" \}
## "dynamic":  "dynamic_1", "dynamic_2" (defined in "hyperparameter_rho" in "Functions.py")

### [4] Scaling constant a > 0 for the proximity parameters \eta^t=1/\sqrt{t} and \delta^t=1/t^2
## default: a = "1.0"

### [5] Privacy parameter \bar{\epsilon} \in \{"0.01", "0.05", "0.1", "1.0", "10.0", "infty"\}

### [6] Training_step (Total iteration) and Display_step (display intermediate results)
##   Training_step = \{ "20000", "500000", "1000000" \}
##   Display_step = \{ "200", "5000", "10000" \}

## Example:


##Variable par saved

# # what_can_be_plotted =
#                ['ITERATION',
#                 'COST',
#                 'ACCURACY_TEST',
#                 'RESIDUAL',
#                 'ELAPSED_TIME',
#                 'ITER_TIME',
#                 'RUNTIME_1',
#                 'RUNTIME_2',
#                 'GRAD_TIME',
#                 'NOISE_TIME',
#                 'AVG_NOISE_MAG',
#                 'Z_CHANGE_MEAN']

# main("MNIST","10", "ObjP", "dynamic_1", "1.0", "0.05", "100", "1")
# main("MNIST","10", "OutP", "dynamic_1", "1.0", "0.05", "100", "1")

# main("MNIST","10", "OutP", "dynamic_1", "1.0", "0.05", "20000", "200")
# main("MNIST","10", "ObjP", "dynamic_1", "1.0", "0.05", "20000", "200")
# main("MNIST","10", "ObjT", "dynamic_1", "1.0", "0.05", "20000", "200")
# main("FEMNIST","small", "OutP", "dynamic_1", "1.0", "0.05", "20000", "200")
# main("FEMNIST","small", "ObjP", "dynamic_1", "1.0", "0.05", "20000", "200")
# main("FEMNIST","small", "ObjT", "dynamic_1", "1.0", "0.05", "20000", "200")

# for eps in ["0.05", "0.1", "1.0"]:
#     main("MNIST","10", "ObjT", "dynamic_1", "1.0", eps, "20000", "200")
#     main("FEMNIST","small", "ObjT", "dynamic_1", "1.0", eps, "20000", "200")