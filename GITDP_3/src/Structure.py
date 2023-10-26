import csv

class Parameters:
    def __init__(self):
        self.Instance = ""
        self.Algorithm = ""
        self.rho_str = ""
        self.a_str = ""
        self.bar_eps_str = ""

        self.total_data = 0
        self.parameter_size = 0
        self.split_number = 0
        self.gamma = 0
        self.rho_const = 0
        self.rho = 0
        self.beta_const = 0
        self.beta = 0
        self.eta = 0
        self.UB = 0
        self.LB = 0
        self.bar_epsilon = 0
        self.Iteration_Limit = 0
        self.print_iter = 0
        self.tilde_xi = []
        self.bar_lambda = 0
        self.edge = 0
        
        self.W_val = [] 
        self.Z_val = [] 
        self.Lambdas_val = [] 
        self.M = []
        self.U = []
        self.V = []


        ####
        self.learning_rate = 0
        self.training_steps = 0
        self.batch_size = 0
        self.display_step = 0        
        self.num_features = 0
        self.num_classes = 0


        ##
        self.ITERATION = []
        self.COST = []
        self.ACCURACY_TEST = []
        self.RESIDUAL = []
        self.ELAPSED_TIME = []
        self.ITER_TIME = []
        self.RUNTIME_1 = []
        self.RUNTIME_2 = []
        self.GRAD_TIME = []
        self.NOISE_TIME = []
        self.AVG_NOISE_MAG = []
        self.Z_CHANGE_MEAN = []
        self.Augmented_incidence_matrix= []
        self.V = []

        
        

            

    
