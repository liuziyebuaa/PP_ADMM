****Dear readers:

This code is derived from the following project for DP-IADMM, which corresponds to the reference
'Differentially Private Federated Learning via Inexact ADMM with Multiple Local Update,' 2022.
We conducted a similar project but used different datasets and wrote our own algorithm, PP-ADMM.

Most of the code is based on the original one, We have added approximately 40% more content based on this foundation
you should mainly concentrate on the files 'Algorithms.py' and 'Models.py' 
for the updating roles are in the former, and the initialization is in the latter.

Note that the seed for randomization can be set as you like. However, 
our algorithm's performance is usually the best. The variables and parameters are set in the file 'run_1.py'


P.S. The author is a person with slow but steady improvement in output. He is interesting and is still single~

Lzy
22.10.2023
****

#####
The original text is as follows.

# Differentially Private Inexact ADMM for a Federated Learning model

In this open-source code, we implement inexact differentially private alternating direction method of multipliers (DP-IADMM) algorithms for solving a distributed empirical risk minimization problem with the multi-class logistic regression function.
In specific, the following three algorithms are implemented:

- "ObjT":  DP-IADMM with a trust-region of solutions incorporated with the objective perturbation method. 
- "ObjP":  DP-IADMM with a proximal function incorporated with the objective perturbation method.
- "OutP": DP-IADMM with a proximal function incorporated with the output perturbation method.
- "PP_ADMM": PP_ADMM.

 
## Install and Run 

```
git clone https://github.com/minseok-ryu/DP-IADMM-Multiclass-Logistic.git
```

After downloading the code, open the terminal and go to the directory where "run_1.py" exists.

1. Do the followings:

```
conda create -n DPFL	
conda activate DPFL	
conda install numpy	
conda install cupy
pip install GPUInfo
pip install mlxtend
```	

Packages `cupy` and `GPUInfo` are required for GPU computation. `mlxtend` is required for MNIST dataset.

2. Run:

```
python run_1.py
```	

3. Go to "Outputs" directory to see the results 

## Important Note

To use GPU, change from "import numpy as np" to  "import cupy as np" located in the first line of Algorithms.py, Models.py, Read.py, and Functions.py.

## Acknowledgements

This material is based upon work supported by the U.S. Department of Energy, Office of Science, under contract number DE-AC02-06CH11357.
