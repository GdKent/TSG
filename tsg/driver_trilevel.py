import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import functions as func
import trilevel_solver as tls



#--------------------------------------------------#
#-------------- Auxiliary Functions  --------------#
#--------------------------------------------------#

def run_experiment(exp_param_dict, num_rep_value=1):
    """
    Auxiliary function to run the experiments
    
    Args:
        exp_param_dict (dictionary):   Dictionary having some of the attributes of the class TrilevelProblem in functions.py as keys 
        num_rep_value (int, optional): Number of runs for each algorithm (default 10)    
    """
    
    run = tls.TrilevelSolver(exp_param_dict['prob'], 
                                  algo=exp_param_dict['algo'], 
                                  algo_full_name=exp_param_dict['algo_full_name'], 
                                  ul_lr=exp_param_dict['ul_lr'], 
                                  ml_lr=exp_param_dict['ml_lr'], 
                                  ll_lr=exp_param_dict['ll_lr'], 
                                  use_stopping_iter=exp_param_dict['use_stopping_iter'], 
                                  max_iter=exp_param_dict['max_iter'], 
                                  stopping_time=exp_param_dict['stopping_time'], 
                                  true_func=exp_param_dict['true_func'],  
                                  true_fbar=exp_param_dict['true_fbar'],
                                  plot_f2_fbar=exp_param_dict['plot_f2_fbar'],  
                                  plot_f3=exp_param_dict['plot_f3'],
                                  plot_grad_f=exp_param_dict['plot_grad_f'],  
                                  plot_grad_fbar=exp_param_dict['plot_grad_fbar'],
                                  plot_grad_f3=exp_param_dict['plot_grad_f3'],
                                  ul_stepsize_scheme=exp_param_dict['ul_stepsize_scheme'], 
                                  ml_stepsize_scheme=exp_param_dict['ml_stepsize_scheme'], 
                                  ll_stepsize_scheme=exp_param_dict['ll_stepsize_scheme'],
                                  ml_stepsize_scheme_true_funct_armijo=exp_param_dict['ml_stepsize_scheme_true_funct_armijo'],
                                  ll_stepsize_scheme_true_funct_armijo=exp_param_dict['ll_stepsize_scheme_true_funct_armijo'],
                                  ml_iters_true_funct_armijo=exp_param_dict['ml_iters_true_funct_armijo'],
                                  ll_iters_true_funct_armijo=exp_param_dict['ll_iters_true_funct_armijo'],
                                  mlp_iters_max=exp_param_dict['mlp_iters_max'],
                                  llp_iters_max=exp_param_dict['llp_iters_max'],                 
                                  mlp_iters_init=exp_param_dict['mlp_iters_init'], 
                                  llp_iters_init=exp_param_dict['llp_iters_init'],                                  
                                  inc_acc=exp_param_dict['inc_acc'],
                                  inc_acc_threshold_f1=exp_param_dict['inc_acc_threshold_f1'],
                                  inc_acc_threshold_f2=exp_param_dict['inc_acc_threshold_f2'],
                                  hess=exp_param_dict['hess'], 
                                  cg_fd_rtol=exp_param_dict['cg_fd_rtol'],
                                  cg_fd_maxiter=exp_param_dict['cg_fd_maxiter'],
                                  cg_fd_rtol_ml=exp_param_dict['cg_fd_rtol_ml'],
                                  cg_fd_maxiter_ml=exp_param_dict['cg_fd_maxiter_ml'],
                                  neumann_eta=exp_param_dict['neumann_eta'],
                                  neumann_hessian_q=exp_param_dict['neumann_hessian_q'],
                                  neumann_eta_ml=exp_param_dict['neumann_eta_ml'],
                                  neumann_hessian_q_ml=exp_param_dict['neumann_hessian_q_ml'],
                                  normalize=exp_param_dict['normalize'], 
                                  advlearn_noise=exp_param_dict['advlearn_noise'],
                                  advlearn_std_dev=exp_param_dict['advlearn_std_dev'],
                                  iprint=exp_param_dict['iprint'])

    run_out = run.main_algorithm_avg_ci(num_rep=num_rep_value)
    values_avg = run_out[0]
    values_ci = run_out[1]
    true_func_values_avg = run_out[2]
    true_func_values_ci = run_out[3]
    times = run_out[4]
    aux_lists = run_out[5]
    accuracy_values_avg = run_out[6] # November 2024 Update
    accuracy_values_ci = run_out[7]
    
    return run, values_avg, values_ci, true_func_values_avg, true_func_values_ci, times, aux_lists, accuracy_values_avg, accuracy_values_ci


def get_nparray(file_name):
    """
    Auxiliary function to obtain numpy arrays from csv files
    """
    values_avg = pd.read_csv(file_name)
    values_avg = [item for item_2 in values_avg.values.tolist() for item in item_2]
    values_avg = np.array(values_avg)
    return values_avg








#--------------------------------------------------------------------#
#------------------ Define the numerical experiments ----------------#
#--------------------------------------------------------------------#

## Create a dictionary with parameters for each experiment (defined by problem to solve and algorithm)
exp_param_dict = {}

## Dictionary of problems to solve
prob_dict = {}

## Dictionary of parameters common to all optimization algorithms
param_dict = {}










########################
## Problem Quadratic1 ##
########################

## Problem Quadratic1
prob_name_root = "Quadratic1"
prob_name = prob_name_root 
prob_dict[prob_name] = {'name_prob_to_run_val': prob_name_root}
prob = func.TrilevelProblem(name_prob_to_run=prob_dict[prob_name]['name_prob_to_run_val'])

#----------------------------------------------------------------------#
#-------------- Parameters common to all the algorithms  --------------#
#----------------------------------------------------------------------#

param_dict[prob_name] = {}

# A flag to use the total number of iterations as a stopping criterion
param_dict[prob_name]['use_stopping_iter'] = True
# Maximum number of iterations
param_dict[prob_name]['max_iter'] = 50 
# Maximum running time (in sec) used when use_stopping_iter is False
param_dict[prob_name]['stopping_time'] = 10 
# Use true function
param_dict[prob_name]['true_func'] = True  
param_dict[prob_name]['true_fbar'] = True  
# Create plots for fbar and f3
param_dict[prob_name]['plot_f2_fbar'] = True
param_dict[prob_name]['plot_f3'] = True
# Create plots for grad_f, grad_fbar, and grad_f3
param_dict[prob_name]['plot_grad_f'] = True
param_dict[prob_name]['plot_grad_fbar'] = True
param_dict[prob_name]['plot_grad_f3'] = True
# Compute test MSE with or without Gaussian noise (only for Adversarial Learning)
param_dict[prob_name]['advlearn_noise'] = None
param_dict[prob_name]['advlearn_std_dev'] = None
# Number of runs for each algorithm
param_dict[prob_name]['num_rep_value'] = 1 
# Sets the verbosity level for printing information (higher values correspond to more detailed output)
param_dict[prob_name]['iprint'] = 8

####################################################### OLD
# List of colors for the algorithms in the plots
# plot_color_list = ['#1f77b4','#2ca02c','#bcbd22','#CD5C5C','#ff7f0e']
# plot_legend_list = ['TSG-N-FD (inc. acc.)','TSG-1 (inc. acc.)','TSG-H (inc. acc.)','DARTS','StocBiO (inc. acc.)']
####################################################### OLD

#--------------------------------------------------------------------#
#-------------- Parameters specific for each algorithm --------------#
#--------------------------------------------------------------------#

exp_param_dict[prob_name] = {}

## TSG-H 
algo_name = 'TSG-H'
exp_param_dict[prob_name][algo_name] = {'algo': 'tsg', 
                     'algo_full_name': algo_name, 
                     'ul_lr': 0.3, 
                     'ml_lr': 0.2, 
                     'll_lr': 0.01, 
                     'use_stopping_iter': param_dict[prob_name]['use_stopping_iter'],
                     'max_iter': param_dict[prob_name]['max_iter'], 
                     'stopping_time': param_dict[prob_name]['stopping_time'], 
                     'true_func': param_dict[prob_name]['true_func'], 
                     'true_fbar': param_dict[prob_name]['true_fbar'],
                     'plot_f2_fbar': param_dict[prob_name]['plot_f2_fbar'], 
                     'plot_f3': param_dict[prob_name]['plot_f3'],
                     'plot_grad_f': param_dict[prob_name]['plot_grad_f'], 
                     'plot_grad_fbar': param_dict[prob_name]['plot_grad_fbar'],
                     'plot_grad_f3': param_dict[prob_name]['plot_grad_f3'],
                     'ul_stepsize_scheme': 2, 
                     'ml_stepsize_scheme': 2, 
                     'll_stepsize_scheme': 2,
                     'ml_stepsize_scheme_true_funct_armijo': 2,
                     'll_stepsize_scheme_true_funct_armijo': 2,
                     'ml_iters_true_funct_armijo': 1,
                     'll_iters_true_funct_armijo': 1,
                     'mlp_iters_max': 30,
                     'llp_iters_max': 60,                 
                     'mlp_iters_init': 1, 
                     'llp_iters_init': 1,
                     'inc_acc': True, 
                     'inc_acc_threshold_f1': 1e-2,
                     'inc_acc_threshold_f2': 1e-1,
                     'hess': True, 
                     'cg_fd_rtol': None,
                     'cg_fd_maxiter': None,
                     'cg_fd_rtol_ml': None,
                     'cg_fd_maxiter_ml': None,
                     'neumann_eta': None,
                     'neumann_hessian_q': None,
                     'neumann_eta_ml': None,
                     'neumann_hessian_q_ml': None,
                     'normalize': False, 
                     'advlearn_noise': param_dict[prob_name]['advlearn_noise'],
                     'advlearn_std_dev': param_dict[prob_name]['advlearn_std_dev'],
                     'iprint': param_dict[prob_name]['iprint'],
                     'plot_color': '#bcbd22',
                     'plot_legend': 'TSG-H',
                     'linestyle': 'dashed',
                     'prob': prob} 
    
## TSG-N-FD 
algo_name = 'TSG-N-FD'
exp_param_dict[prob_name][algo_name] = {'algo': 'tsg', 
                      'algo_full_name': algo_name, 
                      'ul_lr': 0.01, 
                      'ml_lr': 0.001, 
                      'll_lr': 0.05, 
                      'use_stopping_iter': param_dict[prob_name]['use_stopping_iter'], 
                      'max_iter': param_dict[prob_name]['max_iter'], 
                      'stopping_time': param_dict[prob_name]['stopping_time'], 
                      'true_func': param_dict[prob_name]['true_func'], 
                      'true_fbar': param_dict[prob_name]['true_fbar'],
                      'plot_f2_fbar': param_dict[prob_name]['plot_f2_fbar'], 
                      'plot_f3': param_dict[prob_name]['plot_f3'],
                      'plot_grad_f': param_dict[prob_name]['plot_grad_f'], 
                      'plot_grad_fbar': param_dict[prob_name]['plot_grad_fbar'],
                      'plot_grad_f3': param_dict[prob_name]['plot_grad_f3'],
                      'ul_stepsize_scheme': 2, 
                      'ml_stepsize_scheme': 2, 
                      'll_stepsize_scheme': 2,
                      'ml_stepsize_scheme_true_funct_armijo': 2,
                      'll_stepsize_scheme_true_funct_armijo': 2,
                      'ml_iters_true_funct_armijo': 1,
                      'll_iters_true_funct_armijo': 1,
                      'mlp_iters_max': 30,
                      'llp_iters_max': 60,                 
                      'mlp_iters_init': 1, 
                      'llp_iters_init': 1,
                      'inc_acc': True, 
                      'inc_acc_threshold_f1': 1e-2,
                      'inc_acc_threshold_f2': 1e-1,
                      'hess': 'CG-FD', 
                      'cg_fd_rtol': 1e-4,
                      'cg_fd_maxiter': 3,
                      'cg_fd_rtol_ml': 1e-4,
                      'cg_fd_maxiter_ml': 3,
                      'neumann_eta': None,
                      'neumann_hessian_q': None,
                      'neumann_eta_ml': None,
                      'neumann_hessian_q_ml': None,
                      'normalize': False, 
                      'advlearn_noise': param_dict[prob_name]['advlearn_noise'],
                      'advlearn_std_dev': param_dict[prob_name]['advlearn_std_dev'],
                      'iprint': param_dict[prob_name]['iprint'],  
                      'plot_color': '#1f77b4',
                      'plot_legend': 'TSG-N-FD',
                      'linestyle': 'dashdot',
                      'prob': prob}                        
                        
## TSG-AD 
algo_name = 'TSG-AD'
exp_param_dict[prob_name][algo_name] = {'algo': 'stoctio', 
                      'algo_full_name': algo_name, 
                      'ul_lr': 0.001, 
                      'ml_lr': 0.001, 
                      'll_lr': 0.001, 
                      'use_stopping_iter': param_dict[prob_name]['use_stopping_iter'], 
                      'max_iter': param_dict[prob_name]['max_iter'], 
                      'stopping_time': param_dict[prob_name]['stopping_time'], 
                      'true_func': param_dict[prob_name]['true_func'], 
                      'true_fbar': param_dict[prob_name]['true_fbar'],
                      'plot_f2_fbar': param_dict[prob_name]['plot_f2_fbar'], 
                      'plot_f3': param_dict[prob_name]['plot_f3'],
                      'plot_grad_f': param_dict[prob_name]['plot_grad_f'], 
                      'plot_grad_fbar': param_dict[prob_name]['plot_grad_fbar'],
                      'plot_grad_f3': param_dict[prob_name]['plot_grad_f3'],
                      'ul_stepsize_scheme': 2, 
                      'ml_stepsize_scheme': 2, 
                      'll_stepsize_scheme': 2,
                      'ml_stepsize_scheme_true_funct_armijo': 2,
                      'll_stepsize_scheme_true_funct_armijo': 2,
                      'ml_iters_true_funct_armijo': 1,
                      'll_iters_true_funct_armijo': 1,
                      'mlp_iters_max': 30,
                      'llp_iters_max': 60,                 
                      'mlp_iters_init': 1, 
                      'llp_iters_init': 1,
                      'inc_acc': True, 
                      'inc_acc_threshold_f1': 1e-2,
                      'inc_acc_threshold_f2': 1e-1,
                      'hess': 'autodiff', 
                      'cg_fd_rtol': None,
                      'cg_fd_maxiter': None,
                      'cg_fd_rtol_ml': None,
                      'cg_fd_maxiter_ml': None,
                      'neumann_eta': 0.05,
                      'neumann_hessian_q': 2,
                      'neumann_eta_ml': 2,
                      'neumann_hessian_q_ml': 2,
                      'normalize': False, 
                      'advlearn_noise': param_dict[prob_name]['advlearn_noise'],
                      'advlearn_std_dev': param_dict[prob_name]['advlearn_std_dev'],
                      'iprint': param_dict[prob_name]['iprint'],
                      'plot_color': '#CD5C5C',
                      'plot_legend': 'TSG-AD',
                      'linestyle': 'solid',
                      'prob': prob} 

# prob_name_list = ['Quadratic1']
 
# algo_name_list = ['TSG-H'] 
# algo_name_list = ['TSG-N-FD']
# algo_name_list = ['TSG-AD']
# algo_name_list = ['TSG-H', 'TSG-N-FD', 'TSG-AD'] 

###############################################################################











########################
## Problem Quadratic2 ##
########################

for use_stopping_iter_val, max_iter_val in [(True, 40), (False, 400)]:
        
        if use_stopping_iter_val:
            name_aux = "iter"
        else:
            name_aux = "time"
        
        ## Problem Quadratic2
        prob_name_root = "Quadratic2" 
        prob_name = prob_name_root + name_aux
        prob_dict[prob_name] = {'name_prob_to_run_val': prob_name_root}
        prob = func.TrilevelProblem(name_prob_to_run=prob_dict[prob_name]['name_prob_to_run_val'])
        
        #----------------------------------------------------------------------#
        #-------------- Parameters common to all the algorithms  --------------#
        #----------------------------------------------------------------------#
        
        param_dict[prob_name] = {}
        
        # A flag to use the total number of iterations as a stopping criterion
        param_dict[prob_name]['use_stopping_iter'] = use_stopping_iter_val
        # Maximum number of iterations
        param_dict[prob_name]['max_iter'] = max_iter_val
        # Maximum running time (in sec) used when use_stopping_iter is False
        param_dict[prob_name]['stopping_time'] = 10 
        # Use true function
        param_dict[prob_name]['true_func'] = True
        param_dict[prob_name]['true_fbar'] = True
        # Create plots for fbar and f3
        param_dict[prob_name]['plot_f2_fbar'] = True
        param_dict[prob_name]['plot_f3'] = True
        # Create plots for grad_f, grad_fbar, and grad_f3
        param_dict[prob_name]['plot_grad_f'] = True
        param_dict[prob_name]['plot_grad_fbar'] = True
        param_dict[prob_name]['plot_grad_f3'] = True
        # Compute test MSE with or without Gaussian noise (only for Adversarial Learning)
        param_dict[prob_name]['advlearn_noise'] = None
        param_dict[prob_name]['advlearn_std_dev'] = None
        # Number of runs for each algorithm
        param_dict[prob_name]['num_rep_value'] = 1 
        # Sets the verbosity level for printing information (higher values correspond to more detailed output)
        param_dict[prob_name]['iprint'] = 5
        
        ####################################################### OLD
        # List of colors for the algorithms in the plots
        # plot_color_list = ['#1f77b4','#2ca02c','#bcbd22','#CD5C5C','#ff7f0e']
        # plot_legend_list = ['TSG-N-FD (inc. acc.)','TSG-1 (inc. acc.)','TSG-H (inc. acc.)','DARTS','StocBiO (inc. acc.)']
        ####################################################### OLD
        
        #--------------------------------------------------------------------#
        #-------------- Parameters specific for each algorithm --------------#
        #--------------------------------------------------------------------#
        
        exp_param_dict[prob_name] = {}
        
        ## TSG-H 
        algo_name = 'TSG-H'
        exp_param_dict[prob_name][algo_name] = {'algo': 'tsg', 
                             'algo_full_name': algo_name, 
                             'ul_lr': 0.3, 
                             'ml_lr': 0.2, 
                             'll_lr': 0.1, 
                             'use_stopping_iter': param_dict[prob_name]['use_stopping_iter'],
                             'max_iter': param_dict[prob_name]['max_iter'], 
                             'stopping_time': param_dict[prob_name]['stopping_time'], 
                             'true_func': param_dict[prob_name]['true_func'], 
                             'true_fbar': param_dict[prob_name]['true_fbar'],
                             'plot_f2_fbar': param_dict[prob_name]['plot_f2_fbar'], 
                             'plot_f3': param_dict[prob_name]['plot_f3'],                     
                             'plot_grad_f': param_dict[prob_name]['plot_grad_f'], 
                             'plot_grad_fbar': param_dict[prob_name]['plot_grad_fbar'],
                             'plot_grad_f3': param_dict[prob_name]['plot_grad_f3'],
                             'ul_stepsize_scheme': 1, 
                             'ml_stepsize_scheme': 1, 
                             'll_stepsize_scheme': 1,
                             'ml_stepsize_scheme_true_funct_armijo': 2,
                             'll_stepsize_scheme_true_funct_armijo': 2,
                             'ml_iters_true_funct_armijo': 1,
                             'll_iters_true_funct_armijo': 1,
                             'mlp_iters_max': 30,
                             'llp_iters_max': 60,                 
                             'mlp_iters_init': 1, 
                             'llp_iters_init': 1,
                             'inc_acc': True, 
                             'inc_acc_threshold_f1': 1e-2,
                             'inc_acc_threshold_f2': 1e-1,
                             'hess': True, 
                             'cg_fd_rtol': None,
                             'cg_fd_maxiter': None,
                             'cg_fd_rtol_ml': None,
                             'cg_fd_maxiter_ml': None,
                             'neumann_eta': None,
                             'neumann_hessian_q': None,
                             'neumann_eta_ml': None,
                             'neumann_hessian_q_ml': None,
                             'normalize': False, 
                             'advlearn_noise': param_dict[prob_name]['advlearn_noise'],
                             'advlearn_std_dev': param_dict[prob_name]['advlearn_std_dev'],
                             'iprint': param_dict[prob_name]['iprint'],
                             'plot_color': '#bcbd22',
                             'plot_legend': 'TSG-H',
                             'linestyle': 'dashed',
                             'prob': prob} 
            
        ## TSG-N-FD 
        algo_name = 'TSG-N-FD'
        exp_param_dict[prob_name][algo_name] = {'algo': 'tsg', 
                              'algo_full_name': algo_name, 
                              'ul_lr': 0.01, 
                              'ml_lr': 0.1, 
                              'll_lr': 0.05, 
                              'use_stopping_iter': param_dict[prob_name]['use_stopping_iter'], 
                              'max_iter': param_dict[prob_name]['max_iter'], 
                              'stopping_time': param_dict[prob_name]['stopping_time'], 
                              'true_func': param_dict[prob_name]['true_func'], 
                              'true_fbar': param_dict[prob_name]['true_fbar'],
                              'plot_f2_fbar': param_dict[prob_name]['plot_f2_fbar'], 
                              'plot_f3': param_dict[prob_name]['plot_f3'],
                              'plot_grad_f': param_dict[prob_name]['plot_grad_f'], 
                              'plot_grad_fbar': param_dict[prob_name]['plot_grad_fbar'],
                              'plot_grad_f3': param_dict[prob_name]['plot_grad_f3'],
                              'ul_stepsize_scheme': 1, 
                              'ml_stepsize_scheme': 1, 
                              'll_stepsize_scheme': 1,
                              'ml_stepsize_scheme_true_funct_armijo': 2,
                              'll_stepsize_scheme_true_funct_armijo': 2,
                              'ml_iters_true_funct_armijo': 1,
                              'll_iters_true_funct_armijo': 1,
                              'mlp_iters_max': 30,
                              'llp_iters_max': 60,                 
                              'mlp_iters_init': 1, 
                              'llp_iters_init': 1,
                              'inc_acc': True, 
                              'inc_acc_threshold_f1': 1e-2,
                              'inc_acc_threshold_f2': 1e-1,
                              'hess': 'CG-FD', 
                              'cg_fd_rtol': 1e-4,
                              'cg_fd_maxiter': 3,
                              'cg_fd_rtol_ml': 1e-4,
                              'cg_fd_maxiter_ml': 3,
                              'neumann_eta': None,
                              'neumann_hessian_q': None,
                              'neumann_eta_ml': None,
                              'neumann_hessian_q_ml': None,
                              'normalize': False, 
                              'advlearn_noise': param_dict[prob_name]['advlearn_noise'],
                              'advlearn_std_dev': param_dict[prob_name]['advlearn_std_dev'],
                              'iprint': param_dict[prob_name]['iprint'],  
                              'plot_color': '#1f77b4',
                              'plot_legend': 'TSG-N-FD',
                              'linestyle': 'dashdot',
                              'prob': prob}                   
                                
        ## TSG-AD 
        algo_name = 'TSG-AD'
        exp_param_dict[prob_name][algo_name] = {'algo': 'stoctio', 
                              'algo_full_name': algo_name, 
                              'ul_lr': 0.01, 
                              'ml_lr': 0.1, 
                              'll_lr': 0.1, 
                              'use_stopping_iter': param_dict[prob_name]['use_stopping_iter'], 
                              'max_iter': param_dict[prob_name]['max_iter'], 
                              'stopping_time': param_dict[prob_name]['stopping_time'], 
                              'true_func': param_dict[prob_name]['true_func'], 
                              'true_fbar': param_dict[prob_name]['true_fbar'],
                              'plot_f2_fbar': param_dict[prob_name]['plot_f2_fbar'], 
                              'plot_f3': param_dict[prob_name]['plot_f3'],
                              'plot_grad_f': param_dict[prob_name]['plot_grad_f'], 
                              'plot_grad_fbar': param_dict[prob_name]['plot_grad_fbar'],
                              'plot_grad_f3': param_dict[prob_name]['plot_grad_f3'],
                              'ul_stepsize_scheme': 1, 
                              'ml_stepsize_scheme': 1, 
                              'll_stepsize_scheme': 1,
                              'ml_stepsize_scheme_true_funct_armijo': 2,
                              'll_stepsize_scheme_true_funct_armijo': 2,
                              'ml_iters_true_funct_armijo': 1,
                              'll_iters_true_funct_armijo': 1,
                              'mlp_iters_max': 30,
                              'llp_iters_max': 60,                 
                              'mlp_iters_init': 1, 
                              'llp_iters_init': 1,
                              'inc_acc': True, 
                              'inc_acc_threshold_f1': 1e-2,
                              'inc_acc_threshold_f2': 1e-1,
                              'hess': 'autodiff', 
                              'cg_fd_rtol': None,
                              'cg_fd_maxiter': None,
                              'cg_fd_rtol_ml': None,
                              'cg_fd_maxiter_ml': None,
                              'neumann_eta': 0.05,
                              'neumann_hessian_q': 2,
                              'neumann_eta_ml': 0.05,
                              'neumann_hessian_q_ml': 2,
                              'normalize': False, 
                              'advlearn_noise': param_dict[prob_name]['advlearn_noise'],
                              'advlearn_std_dev': param_dict[prob_name]['advlearn_std_dev'],
                              'iprint': param_dict[prob_name]['iprint'],
                              'plot_color': '#CD5C5C',
                              'plot_legend': 'TSG-AD',
                              'linestyle': 'solid',
                              'prob': prob} 


# prob_name_list = ['Quadratic2']

 
# algo_name_list = ['TSG-H'] 
# algo_name_list = ['TSG-N-FD']
# algo_name_list = ['TSG-AD']
# algo_name_list = ['TSG-H', 'TSG-N-FD', 'TSG-AD'] 

###############################################################################






########################
## Problem Quadratic3 ##
########################

for use_stopping_iter_val, max_iter_val in [(True, 80), (False, 10000)]:
        
        if use_stopping_iter_val:
            name_aux = "iter"
        else:
            name_aux = "time"
            
        ## Problem Quadratic3
        prob_name_root = "Quadratic3" 
        prob_name = prob_name_root + name_aux
        prob_dict[prob_name] = {'name_prob_to_run_val': prob_name_root}
        prob = func.TrilevelProblem(name_prob_to_run=prob_dict[prob_name]['name_prob_to_run_val'])
        
        #----------------------------------------------------------------------#
        #-------------- Parameters common to all the algorithms  --------------#
        #----------------------------------------------------------------------#
        
        param_dict[prob_name] = {}
        
        # A flag to use the total number of iterations as a stopping criterion
        param_dict[prob_name]['use_stopping_iter'] = use_stopping_iter_val
        # Maximum number of iterations
        param_dict[prob_name]['max_iter'] = max_iter_val
        # Maximum running time (in sec) used when use_stopping_iter is False
        param_dict[prob_name]['stopping_time'] = 10 
        # Use true function
        param_dict[prob_name]['true_func'] = True
        param_dict[prob_name]['true_fbar'] = True
        # Create plots for fbar and f3
        param_dict[prob_name]['plot_f2_fbar'] = True
        param_dict[prob_name]['plot_f3'] = True
        # Create plots for grad_f, grad_fbar, and grad_f3
        param_dict[prob_name]['plot_grad_f'] = True
        param_dict[prob_name]['plot_grad_fbar'] = True
        param_dict[prob_name]['plot_grad_f3'] = True
        # Compute test MSE with or without Gaussian noise (only for Adversarial Learning)
        param_dict[prob_name]['advlearn_noise'] = None
        param_dict[prob_name]['advlearn_std_dev'] = None
        # Number of runs for each algorithm
        param_dict[prob_name]['num_rep_value'] = 10 
        # Sets the verbosity level for printing information (higher values correspond to more detailed output)
        param_dict[prob_name]['iprint'] = 8
        
        ####################################################### OLD
        # List of colors for the algorithms in the plots
        # plot_color_list = ['#1f77b4','#2ca02c','#bcbd22','#CD5C5C','#ff7f0e']
        # plot_legend_list = ['TSG-N-FD (inc. acc.)','TSG-1 (inc. acc.)','TSG-H (inc. acc.)','DARTS','StocBiO (inc. acc.)']
        ####################################################### OLD
        
        #--------------------------------------------------------------------#
        #-------------- Parameters specific for each algorithm --------------#
        #--------------------------------------------------------------------#
        
        exp_param_dict[prob_name] = {}
        
        ## TSG-H 
        algo_name = 'TSG-H'
        exp_param_dict[prob_name][algo_name] = {'algo': 'tsg', 
                             'algo_full_name': algo_name, 
                             'ul_lr': 0.1, 
                             'ml_lr': 0.1, 
                             'll_lr': 0.1, 
                             'use_stopping_iter': param_dict[prob_name]['use_stopping_iter'],
                             'max_iter': param_dict[prob_name]['max_iter'], 
                             'stopping_time': param_dict[prob_name]['stopping_time'], 
                             'true_func': param_dict[prob_name]['true_func'], 
                             'true_fbar': param_dict[prob_name]['true_fbar'],
                             'plot_f2_fbar': param_dict[prob_name]['plot_f2_fbar'], 
                             'plot_f3': param_dict[prob_name]['plot_f3'],                     
                             'plot_grad_f': param_dict[prob_name]['plot_grad_f'], 
                             'plot_grad_fbar': param_dict[prob_name]['plot_grad_fbar'],
                             'plot_grad_f3': param_dict[prob_name]['plot_grad_f3'],
                             'ul_stepsize_scheme': 1, 
                             'ml_stepsize_scheme': 1, 
                             'll_stepsize_scheme': 1,
                             'ml_stepsize_scheme_true_funct_armijo': 2,
                             'll_stepsize_scheme_true_funct_armijo': 2,
                             'ml_iters_true_funct_armijo': 1,
                             'll_iters_true_funct_armijo': 1,
                             'mlp_iters_max': 30,
                             'llp_iters_max': 60,                 
                             'mlp_iters_init': 1, 
                             'llp_iters_init': 1,
                             'inc_acc': True, 
                             'inc_acc_threshold_f1': 1e-2,
                             'inc_acc_threshold_f2': 1e-1,
                             'hess': True, 
                             'cg_fd_rtol': None,
                             'cg_fd_maxiter': None,
                             'cg_fd_rtol_ml': None,
                             'cg_fd_maxiter_ml': None,
                             'neumann_eta': None,
                             'neumann_hessian_q': None,
                             'neumann_eta_ml': None,
                             'neumann_hessian_q_ml': None,
                             'normalize': False, 
                             'advlearn_noise': param_dict[prob_name]['advlearn_noise'],
                             'advlearn_std_dev': param_dict[prob_name]['advlearn_std_dev'],
                             'iprint': param_dict[prob_name]['iprint'],
                             'plot_color': '#bcbd22',
                             'plot_legend': 'TSG-H',
                             'linestyle': 'dashed',
                             'prob': prob} 
            
        ## TSG-N-FD 
        algo_name = 'TSG-N-FD'
        exp_param_dict[prob_name][algo_name] = {'algo': 'tsg', 
                              'algo_full_name': algo_name, 
                              'ul_lr': 0.01, #0.1
                              'ml_lr': 0.1, 
                              'll_lr': 0.1, 
                              'use_stopping_iter': param_dict[prob_name]['use_stopping_iter'], 
                              'max_iter': param_dict[prob_name]['max_iter'], 
                              'stopping_time': param_dict[prob_name]['stopping_time'], 
                              'true_func': param_dict[prob_name]['true_func'], 
                              'true_fbar': param_dict[prob_name]['true_fbar'],
                              'plot_f2_fbar': param_dict[prob_name]['plot_f2_fbar'], 
                              'plot_f3': param_dict[prob_name]['plot_f3'],
                              'plot_grad_f': param_dict[prob_name]['plot_grad_f'], 
                              'plot_grad_fbar': param_dict[prob_name]['plot_grad_fbar'],
                              'plot_grad_f3': param_dict[prob_name]['plot_grad_f3'],
                              'ul_stepsize_scheme': 1, 
                              'ml_stepsize_scheme': 1, 
                              'll_stepsize_scheme': 1,
                              'ml_stepsize_scheme_true_funct_armijo': 2,
                              'll_stepsize_scheme_true_funct_armijo': 2,
                              'ml_iters_true_funct_armijo': 1,
                              'll_iters_true_funct_armijo': 1,
                              'mlp_iters_max': 30,
                              'llp_iters_max': 60,                 
                              'mlp_iters_init': 1, 
                              'llp_iters_init': 1,
                              'inc_acc': True, 
                              'inc_acc_threshold_f1': 1e-2,
                              'inc_acc_threshold_f2': 1e-1,
                              'hess': 'CG-FD', 
                              'cg_fd_rtol': 1e-4,
                              'cg_fd_maxiter': 3,
                              'cg_fd_rtol_ml': 1e-4,
                              'cg_fd_maxiter_ml': 3,
                              'neumann_eta': None,
                              'neumann_hessian_q': None,
                              'neumann_eta_ml': None,
                              'neumann_hessian_q_ml': None,
                              'normalize': False, 
                              'advlearn_noise': param_dict[prob_name]['advlearn_noise'],
                              'advlearn_std_dev': param_dict[prob_name]['advlearn_std_dev'],
                              'iprint': param_dict[prob_name]['iprint'],  
                              'plot_color': '#1f77b4',
                              'plot_legend': 'TSG-N-FD',
                              'linestyle': 'dashdot',
                              'prob': prob}                   
                                
        ## TSG-AD 
        algo_name = 'TSG-AD'
        exp_param_dict[prob_name][algo_name] = {'algo': 'stoctio', 
                              'algo_full_name': algo_name, 
                              'ul_lr': 0.01, 
                              'ml_lr': 0.1, 
                              'll_lr': 0.1, 
                              'use_stopping_iter': param_dict[prob_name]['use_stopping_iter'], 
                              'max_iter': param_dict[prob_name]['max_iter'], 
                              'stopping_time': param_dict[prob_name]['stopping_time'], 
                              'true_func': param_dict[prob_name]['true_func'], 
                              'true_fbar': param_dict[prob_name]['true_fbar'],
                              'plot_f2_fbar': param_dict[prob_name]['plot_f2_fbar'], 
                              'plot_f3': param_dict[prob_name]['plot_f3'],
                              'plot_grad_f': param_dict[prob_name]['plot_grad_f'], 
                              'plot_grad_fbar': param_dict[prob_name]['plot_grad_fbar'],
                              'plot_grad_f3': param_dict[prob_name]['plot_grad_f3'],
                              'ul_stepsize_scheme': 1, 
                              'ml_stepsize_scheme': 1, 
                              'll_stepsize_scheme': 1,
                              'ml_stepsize_scheme_true_funct_armijo': 2,
                              'll_stepsize_scheme_true_funct_armijo': 2,
                              'ml_iters_true_funct_armijo': 1,
                              'll_iters_true_funct_armijo': 1,
                              'mlp_iters_max': 30,
                              'llp_iters_max': 60,                 
                              'mlp_iters_init': 1, 
                              'llp_iters_init': 1,
                              'inc_acc': True, 
                              'inc_acc_threshold_f1': 1e-2,
                              'inc_acc_threshold_f2': 1e-1,
                              'hess': 'autodiff', 
                              'cg_fd_rtol': None,
                              'cg_fd_maxiter': None,
                              'cg_fd_rtol_ml': None,
                              'cg_fd_maxiter_ml': None,
                              'neumann_eta': 0.05,
                              'neumann_hessian_q': 2,
                              'neumann_eta_ml': 0.05,
                              'neumann_hessian_q_ml': 5,
                              'normalize': False, 
                              'advlearn_noise': param_dict[prob_name]['advlearn_noise'],
                              'advlearn_std_dev': param_dict[prob_name]['advlearn_std_dev'],
                              'iprint': param_dict[prob_name]['iprint'],
                              'plot_color': '#CD5C5C',
                              'plot_legend': 'TSG-AD',
                              'linestyle': 'solid',
                              'prob': prob} 


# prob_name_list = ['Quadratic3']

 
# algo_name_list = ['TSG-H'] 
# algo_name_list = ['TSG-N-FD']
# algo_name_list = ['TSG-AD']
# algo_name_list = ['TSG-N-FD', 'TSG-AD'] 
# algo_name_list = ['TSG-H', 'TSG-N-FD', 'TSG-AD'] 

###############################################################################






########################
## Problem Quadratic4 ##
########################

for use_stopping_iter_val, max_iter_val in [(True, 80), (False, 400)]:
        
        if use_stopping_iter_val:
            name_aux = "iter"
        else:
            name_aux = "time"
            
        ## Problem Quadratic3
        prob_name_root = "Quadratic4" 
        prob_name = prob_name_root + name_aux
        prob_dict[prob_name] = {'name_prob_to_run_val': prob_name_root}
        prob = func.TrilevelProblem(name_prob_to_run=prob_dict[prob_name]['name_prob_to_run_val'])
        
        #----------------------------------------------------------------------#
        #-------------- Parameters common to all the algorithms  --------------#
        #----------------------------------------------------------------------#
        
        param_dict[prob_name] = {}
        
        # A flag to use the total number of iterations as a stopping criterion
        param_dict[prob_name]['use_stopping_iter'] = use_stopping_iter_val
        # Maximum number of iterations
        param_dict[prob_name]['max_iter'] = max_iter_val
        # Maximum running time (in sec) used when use_stopping_iter is False
        param_dict[prob_name]['stopping_time'] = 10 
        # Use true function
        param_dict[prob_name]['true_func'] = True
        param_dict[prob_name]['true_fbar'] = True
        # Create plots for fbar and f3
        param_dict[prob_name]['plot_f2_fbar'] = True
        param_dict[prob_name]['plot_f3'] = True
        # Create plots for grad_f, grad_fbar, and grad_f3
        param_dict[prob_name]['plot_grad_f'] = True
        param_dict[prob_name]['plot_grad_fbar'] = True
        param_dict[prob_name]['plot_grad_f3'] = True
        # Compute test MSE with or without Gaussian noise (only for Adversarial Learning)
        param_dict[prob_name]['advlearn_noise'] = None
        param_dict[prob_name]['advlearn_std_dev'] = None
        # Number of runs for each algorithm
        param_dict[prob_name]['num_rep_value'] = 10 
        # Sets the verbosity level for printing information (higher values correspond to more detailed output)
        param_dict[prob_name]['iprint'] = 8
        
        ####################################################### OLD
        # List of colors for the algorithms in the plots
        # plot_color_list = ['#1f77b4','#2ca02c','#bcbd22','#CD5C5C','#ff7f0e']
        # plot_legend_list = ['TSG-N-FD (inc. acc.)','TSG-1 (inc. acc.)','TSG-H (inc. acc.)','DARTS','StocBiO (inc. acc.)']
        ####################################################### OLD
        
        #--------------------------------------------------------------------#
        #-------------- Parameters specific for each algorithm --------------#
        #--------------------------------------------------------------------#
        
        exp_param_dict[prob_name] = {}
        
        ## TSG-H 
        algo_name = 'TSG-H'
        exp_param_dict[prob_name][algo_name] = {'algo': 'tsg', 
                             'algo_full_name': algo_name, 
                             'ul_lr': 0.1, 
                             'ml_lr': 0.1, 
                             'll_lr': 0.1, 
                             'use_stopping_iter': param_dict[prob_name]['use_stopping_iter'],
                             'max_iter': param_dict[prob_name]['max_iter'], 
                             'stopping_time': param_dict[prob_name]['stopping_time'], 
                             'true_func': param_dict[prob_name]['true_func'], 
                             'true_fbar': param_dict[prob_name]['true_fbar'],
                             'plot_f2_fbar': param_dict[prob_name]['plot_f2_fbar'], 
                             'plot_f3': param_dict[prob_name]['plot_f3'],                     
                             'plot_grad_f': param_dict[prob_name]['plot_grad_f'], 
                             'plot_grad_fbar': param_dict[prob_name]['plot_grad_fbar'],
                             'plot_grad_f3': param_dict[prob_name]['plot_grad_f3'],
                             'ul_stepsize_scheme': 1, 
                             'ml_stepsize_scheme': 1, 
                             'll_stepsize_scheme': 1,
                             'ml_stepsize_scheme_true_funct_armijo': 2,
                             'll_stepsize_scheme_true_funct_armijo': 2,
                             'ml_iters_true_funct_armijo': 1,
                             'll_iters_true_funct_armijo': 1,
                             'mlp_iters_max': 30,
                             'llp_iters_max': 60,                 
                             'mlp_iters_init': 1, 
                             'llp_iters_init': 1,
                             'inc_acc': True, 
                             'inc_acc_threshold_f1': 1e-2,
                             'inc_acc_threshold_f2': 1e-1,
                             'hess': True, 
                             'cg_fd_rtol': None,
                             'cg_fd_maxiter': None,
                             'cg_fd_rtol_ml': None,
                             'cg_fd_maxiter_ml': None,
                             'neumann_eta': None,
                             'neumann_hessian_q': None,
                             'neumann_eta_ml': None,
                             'neumann_hessian_q_ml': None,
                             'normalize': False, 
                             'advlearn_noise': param_dict[prob_name]['advlearn_noise'],
                             'advlearn_std_dev': param_dict[prob_name]['advlearn_std_dev'],
                             'iprint': param_dict[prob_name]['iprint'],
                             'plot_color': '#bcbd22',
                             'plot_legend': 'TSG-H',
                             'linestyle': 'dashed',
                             'prob': prob} 
            
        ## TSG-N-FD 
        algo_name = 'TSG-N-FD'
        exp_param_dict[prob_name][algo_name] = {'algo': 'tsg', 
                              'algo_full_name': algo_name, 
                              'ul_lr': 0.01, #0.1
                              'ml_lr': 0.1, 
                              'll_lr': 0.1, 
                              'use_stopping_iter': param_dict[prob_name]['use_stopping_iter'], 
                              'max_iter': param_dict[prob_name]['max_iter'], 
                              'stopping_time': param_dict[prob_name]['stopping_time'], 
                              'true_func': param_dict[prob_name]['true_func'], 
                              'true_fbar': param_dict[prob_name]['true_fbar'],
                              'plot_f2_fbar': param_dict[prob_name]['plot_f2_fbar'], 
                              'plot_f3': param_dict[prob_name]['plot_f3'],
                              'plot_grad_f': param_dict[prob_name]['plot_grad_f'], 
                              'plot_grad_fbar': param_dict[prob_name]['plot_grad_fbar'],
                              'plot_grad_f3': param_dict[prob_name]['plot_grad_f3'],
                              'ul_stepsize_scheme': 1, 
                              'ml_stepsize_scheme': 1, 
                              'll_stepsize_scheme': 1,
                              'ml_stepsize_scheme_true_funct_armijo': 2,
                              'll_stepsize_scheme_true_funct_armijo': 2,
                              'ml_iters_true_funct_armijo': 1,
                              'll_iters_true_funct_armijo': 1,
                              'mlp_iters_max': 30,
                              'llp_iters_max': 60,                 
                              'mlp_iters_init': 1, 
                              'llp_iters_init': 1,
                              'inc_acc': True, 
                              'inc_acc_threshold_f1': 1e-2,
                              'inc_acc_threshold_f2': 1e-1,
                              'hess': 'CG-FD', 
                              'cg_fd_rtol': 1e-4,
                              'cg_fd_maxiter': 3,
                              'cg_fd_rtol_ml': 1e-4,
                              'cg_fd_maxiter_ml': 3,
                              'neumann_eta': None,
                              'neumann_hessian_q': None,
                              'neumann_eta_ml': None,
                              'neumann_hessian_q_ml': None,
                              'normalize': False, 
                              'advlearn_noise': param_dict[prob_name]['advlearn_noise'],
                              'advlearn_std_dev': param_dict[prob_name]['advlearn_std_dev'],
                              'iprint': param_dict[prob_name]['iprint'],  
                              'plot_color': '#1f77b4',
                              'plot_legend': 'TSG-N-FD',
                              'linestyle': 'dashdot',
                              'prob': prob}                   
                                
        ## TSG-AD 
        algo_name = 'TSG-AD'
        exp_param_dict[prob_name][algo_name] = {'algo': 'stoctio', 
                              'algo_full_name': algo_name, 
                              'ul_lr': 0.01, 
                              'ml_lr': 0.1, 
                              'll_lr': 0.1, 
                              'use_stopping_iter': param_dict[prob_name]['use_stopping_iter'], 
                              'max_iter': param_dict[prob_name]['max_iter'], 
                              'stopping_time': param_dict[prob_name]['stopping_time'], 
                              'true_func': param_dict[prob_name]['true_func'], 
                              'true_fbar': param_dict[prob_name]['true_fbar'],
                              'plot_f2_fbar': param_dict[prob_name]['plot_f2_fbar'], 
                              'plot_f3': param_dict[prob_name]['plot_f3'],
                              'plot_grad_f': param_dict[prob_name]['plot_grad_f'], 
                              'plot_grad_fbar': param_dict[prob_name]['plot_grad_fbar'],
                              'plot_grad_f3': param_dict[prob_name]['plot_grad_f3'],
                              'ul_stepsize_scheme': 1, 
                              'ml_stepsize_scheme': 1, 
                              'll_stepsize_scheme': 1,
                              'ml_stepsize_scheme_true_funct_armijo': 2,
                              'll_stepsize_scheme_true_funct_armijo': 2,
                              'ml_iters_true_funct_armijo': 1,
                              'll_iters_true_funct_armijo': 1,
                              'mlp_iters_max': 30,
                              'llp_iters_max': 60,                 
                              'mlp_iters_init': 1, 
                              'llp_iters_init': 1,
                              'inc_acc': True, 
                              'inc_acc_threshold_f1': 1e-2,
                              'inc_acc_threshold_f2': 1e-1,
                              'hess': 'autodiff', 
                              'cg_fd_rtol': None,
                              'cg_fd_maxiter': None,
                              'cg_fd_rtol_ml': None,
                              'cg_fd_maxiter_ml': None,
                              'neumann_eta': 0.05,
                              'neumann_hessian_q': 2,
                              'neumann_eta_ml': 0.05,
                              'neumann_hessian_q_ml': 5,
                              'normalize': False, 
                              'advlearn_noise': param_dict[prob_name]['advlearn_noise'],
                              'advlearn_std_dev': param_dict[prob_name]['advlearn_std_dev'],
                              'iprint': param_dict[prob_name]['iprint'],
                              'plot_color': '#CD5C5C',
                              'plot_legend': 'TSG-AD',
                              'linestyle': 'solid',
                              'prob': prob} 


# prob_name_list = ['Quadratic4']

 
# algo_name_list = ['TSG-H'] 
# algo_name_list = ['TSG-N-FD']
# algo_name_list = ['TSG-AD']
# algo_name_list = ['TSG-N-FD', 'TSG-AD'] 
# algo_name_list = ['TSG-N-FD', 'TSG-AD'] 

###############################################################################







######################
## Problem Quartic1 ##
######################

## Problem Quartic1
prob_name_root = "Quartic1"
prob_name = prob_name_root 
prob_dict[prob_name] = {'name_prob_to_run_val': prob_name_root}
prob = func.TrilevelProblem(name_prob_to_run=prob_dict[prob_name]['name_prob_to_run_val'])

#----------------------------------------------------------------------#
#-------------- Parameters common to all the algorithms  --------------#
#----------------------------------------------------------------------#

param_dict[prob_name] = {}

# A flag to use the total number of iterations as a stopping criterion
param_dict[prob_name]['use_stopping_iter'] = True
# Maximum number of iterations
param_dict[prob_name]['max_iter'] = 5 
# Maximum running time (in sec) used when use_stopping_iter is False
param_dict[prob_name]['stopping_time'] = 10   
# Use true function
param_dict[prob_name]['true_func'] = True  
param_dict[prob_name]['true_fbar'] = True 
# Create plots for fbar and f3
param_dict[prob_name]['plot_f2_fbar'] = True
param_dict[prob_name]['plot_f3'] = True
# Create plots for grad_f, grad_fbar, and grad_f3
param_dict[prob_name]['plot_grad_f'] = True
param_dict[prob_name]['plot_grad_fbar'] = True
param_dict[prob_name]['plot_grad_f3'] = True
# Compute test MSE with or without Gaussian noise (only for Adversarial Learning)
param_dict[prob_name]['advlearn_noise'] = None
param_dict[prob_name]['advlearn_std_dev'] = None
# Number of runs for each algorithm
param_dict[prob_name]['num_rep_value'] = 1 
# Sets the verbosity level for printing information (higher values correspond to more detailed output)
param_dict[prob_name]['iprint'] = 8

####################################################### OLD
# List of colors for the algorithms in the plots
# plot_color_list = ['#1f77b4','#2ca02c','#bcbd22','#CD5C5C','#ff7f0e']
# plot_legend_list = ['TSG-N-FD (inc. acc.)','TSG-1 (inc. acc.)','TSG-H (inc. acc.)','DARTS','StocBiO (inc. acc.)']
####################################################### OLD

#--------------------------------------------------------------------#
#-------------- Parameters specific for each algorithm --------------#
#--------------------------------------------------------------------#

exp_param_dict[prob_name] = {}

## TSG-H 
algo_name = 'TSG-H'
exp_param_dict[prob_name][algo_name] = {'algo': 'tsg', 
                     'algo_full_name': algo_name, 
                     'ul_lr': 0.3, 
                     'ml_lr': 0.2, 
                     'll_lr': 0.1, 
                     'use_stopping_iter': param_dict[prob_name]['use_stopping_iter'],
                     'max_iter': param_dict[prob_name]['max_iter'], 
                     'stopping_time': param_dict[prob_name]['stopping_time'], 
                     'true_func': param_dict[prob_name]['true_func'], 
                     'true_fbar': param_dict[prob_name]['true_fbar'],
                     'plot_f2_fbar': param_dict[prob_name]['plot_f2_fbar'], 
                     'plot_f3': param_dict[prob_name]['plot_f3'],
                     'plot_grad_f': param_dict[prob_name]['plot_grad_f'], 
                     'plot_grad_fbar': param_dict[prob_name]['plot_grad_fbar'],
                     'plot_grad_f3': param_dict[prob_name]['plot_grad_f3'],
                     'ul_stepsize_scheme': 2, 
                     'ml_stepsize_scheme': 2, 
                     'll_stepsize_scheme': 2,
                     'ml_stepsize_scheme_true_funct_armijo': 2,
                     'll_stepsize_scheme_true_funct_armijo': 2,
                     'ml_iters_true_funct_armijo': 1,
                     'll_iters_true_funct_armijo': 1,
                     'mlp_iters_max': 30,
                     'llp_iters_max': 60,                 
                     'mlp_iters_init': 1, 
                     'llp_iters_init': 1,
                     'inc_acc': True, 
                     'inc_acc_threshold_f1': 1e-2,
                     'inc_acc_threshold_f2': 1e-1,
                     'hess': True, 
                     'cg_fd_rtol': None,
                     'cg_fd_maxiter': None,
                     'cg_fd_rtol_ml': None,
                     'cg_fd_maxiter_ml': None,
                     'neumann_eta': None,
                     'neumann_hessian_q': None,
                     'neumann_eta_ml': None,
                     'neumann_hessian_q_ml': None,
                     'normalize': False, 
                     'advlearn_noise': param_dict[prob_name]['advlearn_noise'],
                     'advlearn_std_dev': param_dict[prob_name]['advlearn_std_dev'],
                     'iprint': param_dict[prob_name]['iprint'],
                     'plot_color': '#bcbd22',
                     'plot_legend': 'TSG-H',
                     'linestyle': 'dashed',
                     'prob': prob} 
    
## TSG-N-FD 
algo_name = 'TSG-N-FD'
exp_param_dict[prob_name][algo_name] = {'algo': 'tsg', 
                      'algo_full_name': algo_name, 
                      'ul_lr': 0.01, 
                      'ml_lr': 0.001, 
                      'll_lr': 0.05, 
                      'use_stopping_iter': param_dict[prob_name]['use_stopping_iter'], 
                      'max_iter': param_dict[prob_name]['max_iter'], 
                      'stopping_time': param_dict[prob_name]['stopping_time'], 
                      'true_func': param_dict[prob_name]['true_func'], 
                      'true_fbar': param_dict[prob_name]['true_fbar'],
                      'plot_f2_fbar': param_dict[prob_name]['plot_f2_fbar'], 
                      'plot_f3': param_dict[prob_name]['plot_f3'],
                      'plot_grad_f': param_dict[prob_name]['plot_grad_f'], 
                      'plot_grad_fbar': param_dict[prob_name]['plot_grad_fbar'],
                      'plot_grad_f3': param_dict[prob_name]['plot_grad_f3'],
                      'ul_stepsize_scheme': 2, 
                      'ml_stepsize_scheme': 2, 
                      'll_stepsize_scheme': 2,
                      'ml_stepsize_scheme_true_funct_armijo': 2,
                      'll_stepsize_scheme_true_funct_armijo': 2,
                      'ml_iters_true_funct_armijo': 1,
                      'll_iters_true_funct_armijo': 1,
                      'mlp_iters_max': 30,
                      'llp_iters_max': 60,                 
                      'mlp_iters_init': 1, 
                      'llp_iters_init': 1,
                      'inc_acc': True, 
                      'inc_acc_threshold_f1': 1e-2,
                      'inc_acc_threshold_f2': 1e-1,
                      'hess': 'CG-FD', 
                      'cg_fd_rtol': 1e-4,
                      'cg_fd_maxiter': 3,
                      'cg_fd_rtol_ml': 1e-4,
                      'cg_fd_maxiter_ml': 3,
                      'neumann_eta': None,
                      'neumann_hessian_q': None,
                      'neumann_eta_ml': None,
                      'neumann_hessian_q_ml': None,
                      'normalize': False, 
                      'advlearn_noise': param_dict[prob_name]['advlearn_noise'],
                      'advlearn_std_dev': param_dict[prob_name]['advlearn_std_dev'],
                      'iprint': param_dict[prob_name]['iprint'],  
                      'plot_color': '#1f77b4',
                      'plot_legend': 'TSG-N-FD',
                      'linestyle': 'dashdot',
                      'prob': prob}                       
                        
## TSG-AD 
algo_name = 'TSG-AD'
exp_param_dict[prob_name][algo_name] = {'algo': 'stoctio', 
                      'algo_full_name': algo_name, 
                      'ul_lr': 0.001, 
                      'ml_lr': 0.001, 
                      'll_lr': 0.001, 
                      'use_stopping_iter': param_dict[prob_name]['use_stopping_iter'], 
                      'max_iter': param_dict[prob_name]['max_iter'], 
                      'stopping_time': param_dict[prob_name]['stopping_time'], 
                      'true_func': param_dict[prob_name]['true_func'], 
                      'true_fbar': param_dict[prob_name]['true_fbar'],
                      'plot_f2_fbar': param_dict[prob_name]['plot_f2_fbar'], 
                      'plot_f3': param_dict[prob_name]['plot_f3'],
                      'plot_grad_f': param_dict[prob_name]['plot_grad_f'], 
                      'plot_grad_fbar': param_dict[prob_name]['plot_grad_fbar'],
                      'plot_grad_f3': param_dict[prob_name]['plot_grad_f3'],
                      'ul_stepsize_scheme': 2, 
                      'ml_stepsize_scheme': 2, 
                      'll_stepsize_scheme': 2,
                      'ml_stepsize_scheme_true_funct_armijo': 2,
                      'll_stepsize_scheme_true_funct_armijo': 2,
                      'ml_iters_true_funct_armijo': 1,
                      'll_iters_true_funct_armijo': 1,
                      'mlp_iters_max': 30,
                      'llp_iters_max': 60,                 
                      'mlp_iters_init': 1, 
                      'llp_iters_init': 1,
                      'inc_acc': True, 
                      'inc_acc_threshold_f1': 1e-2,
                      'inc_acc_threshold_f2': 1e-1,
                      'hess': 'autodiff', 
                      'cg_fd_rtol': None,
                      'cg_fd_maxiter': None,
                      'cg_fd_rtol_ml': None,
                      'cg_fd_maxiter_ml': None,
                      'neumann_eta': 0.05,
                      'neumann_hessian_q': 2,
                      'neumann_eta_ml': 2,
                      'neumann_hessian_q_ml': 2,
                      'normalize': False, 
                      'advlearn_noise': param_dict[prob_name]['advlearn_noise'],
                      'advlearn_std_dev': param_dict[prob_name]['advlearn_std_dev'],
                      'iprint': param_dict[prob_name]['iprint'],
                      'plot_color': '#CD5C5C',
                      'plot_legend': 'TSG-AD',
                      'linestyle': 'solid',
                      'prob': prob}    


# prob_name_list = ['Quartic1']

 
# algo_name_list = ['TSG-H'] 
# algo_name_list = ['TSG-N-FD']
# algo_name_list = ['TSG-AD']
# algo_name_list = ['TSG-H', 'TSG-N-FD', 'TSG-AD']                   

###############################################################################







######################
## Problem Quartic2 ##
######################

for use_stopping_iter_val, max_iter_val in [(True, 10), (False, 20000)]:
        
        if use_stopping_iter_val:
            name_aux = "iter"
        else:
            name_aux = "time"
            
        ## Problem Quartic2
        prob_name_root = "Quartic2" 
        prob_name = prob_name_root + name_aux
        prob_dict[prob_name] = {'name_prob_to_run_val': prob_name_root}
        prob = func.TrilevelProblem(name_prob_to_run=prob_dict[prob_name]['name_prob_to_run_val'])
        
        #----------------------------------------------------------------------#
        #-------------- Parameters common to all the algorithms  --------------#
        #----------------------------------------------------------------------#
        
        param_dict[prob_name] = {}
        
        # A flag to use the total number of iterations as a stopping criterion
        param_dict[prob_name]['use_stopping_iter'] = use_stopping_iter_val
        # Maximum number of iterations
        param_dict[prob_name]['max_iter'] = max_iter_val
        # Maximum running time (in sec) used when use_stopping_iter is False
        param_dict[prob_name]['stopping_time'] = 0.2   
        # Use true function
        param_dict[prob_name]['true_func'] = True  
        param_dict[prob_name]['true_fbar'] = True 
        # Create plots for fbar and f3
        param_dict[prob_name]['plot_f2_fbar'] = True
        param_dict[prob_name]['plot_f3'] = True
        # Create plots for grad_f, grad_fbar, and grad_f3
        param_dict[prob_name]['plot_grad_f'] = True
        param_dict[prob_name]['plot_grad_fbar'] = True
        param_dict[prob_name]['plot_grad_f3'] = True
        # Compute test MSE with or without Gaussian noise (only for Adversarial Learning)
        param_dict[prob_name]['advlearn_noise'] = None
        param_dict[prob_name]['advlearn_std_dev'] = None
        # Number of runs for each algorithm
        param_dict[prob_name]['num_rep_value'] = 1 
        # Sets the verbosity level for printing information (higher values correspond to more detailed output)
        param_dict[prob_name]['iprint'] = 8
        
        ####################################################### OLD
        # List of colors for the algorithms in the plots
        # plot_color_list = ['#1f77b4','#2ca02c','#bcbd22','#CD5C5C','#ff7f0e']
        # plot_legend_list = ['TSG-N-FD (inc. acc.)','TSG-1 (inc. acc.)','TSG-H (inc. acc.)','DARTS','StocBiO (inc. acc.)']
        ####################################################### OLD
        
        #--------------------------------------------------------------------#
        #-------------- Parameters specific for each algorithm --------------#
        #--------------------------------------------------------------------#
        
        exp_param_dict[prob_name] = {}
        
        ## TSG-H 
        algo_name = 'TSG-H'
        exp_param_dict[prob_name][algo_name] = {'algo': 'tsg', 
                             'algo_full_name': algo_name, 
                             'ul_lr': 0.3, 
                             'ml_lr': 0.2, 
                             'll_lr': 0.1, 
                             'use_stopping_iter': param_dict[prob_name]['use_stopping_iter'],
                             'max_iter': param_dict[prob_name]['max_iter'], 
                             'stopping_time': param_dict[prob_name]['stopping_time'], 
                             'true_func': param_dict[prob_name]['true_func'], 
                             'true_fbar': param_dict[prob_name]['true_fbar'],
                             'plot_f2_fbar': param_dict[prob_name]['plot_f2_fbar'], 
                             'plot_f3': param_dict[prob_name]['plot_f3'],
                             'plot_grad_f': param_dict[prob_name]['plot_grad_f'], 
                             'plot_grad_fbar': param_dict[prob_name]['plot_grad_fbar'],
                             'plot_grad_f3': param_dict[prob_name]['plot_grad_f3'],
                             'ul_stepsize_scheme': 1, 
                             'ml_stepsize_scheme': 1, 
                             'll_stepsize_scheme': 1,
                             'ml_stepsize_scheme_true_funct_armijo': 2,
                             'll_stepsize_scheme_true_funct_armijo': 2,
                             'ml_iters_true_funct_armijo': 1,
                             'll_iters_true_funct_armijo': 1,
                             'mlp_iters_max': 30,
                             'llp_iters_max': 60,                 
                             'mlp_iters_init': 1, 
                             'llp_iters_init': 1,
                             'inc_acc': True, 
                             'inc_acc_threshold_f1': 1e-2,
                             'inc_acc_threshold_f2': 1e-1,
                             'hess': True, 
                             'cg_fd_rtol': None,
                             'cg_fd_maxiter': None,
                             'cg_fd_rtol_ml': None,
                             'cg_fd_maxiter_ml': None,
                             'neumann_eta': None,
                             'neumann_hessian_q': None,
                             'neumann_eta_ml': None,
                             'neumann_hessian_q_ml': None,
                             'normalize': False, 
                             'advlearn_noise': param_dict[prob_name]['advlearn_noise'],
                             'advlearn_std_dev': param_dict[prob_name]['advlearn_std_dev'],
                             'iprint': param_dict[prob_name]['iprint'],
                             'plot_color': '#bcbd22',
                             'plot_legend': 'TSG-H',
                             'linestyle': 'dashed',
                             'prob': prob} 
            
        ## TSG-N-FD 
        algo_name = 'TSG-N-FD'
        exp_param_dict[prob_name][algo_name] = {'algo': 'tsg', 
                              'algo_full_name': algo_name, 
                              'ul_lr': 0.3, 
                              'ml_lr': 0.2, 
                              'll_lr': 0.0001, 
                              'use_stopping_iter': param_dict[prob_name]['use_stopping_iter'], 
                              'max_iter': param_dict[prob_name]['max_iter'], 
                              'stopping_time': param_dict[prob_name]['stopping_time'], 
                              'true_func': param_dict[prob_name]['true_func'], 
                              'true_fbar': param_dict[prob_name]['true_fbar'],
                              'plot_f2_fbar': param_dict[prob_name]['plot_f2_fbar'], 
                              'plot_f3': param_dict[prob_name]['plot_f3'],
                              'plot_grad_f': param_dict[prob_name]['plot_grad_f'], 
                              'plot_grad_fbar': param_dict[prob_name]['plot_grad_fbar'],
                              'plot_grad_f3': param_dict[prob_name]['plot_grad_f3'],
                              'ul_stepsize_scheme': 1, 
                              'ml_stepsize_scheme': 1, 
                              'll_stepsize_scheme': 1,
                              'ml_stepsize_scheme_true_funct_armijo': 2,
                              'll_stepsize_scheme_true_funct_armijo': 2,
                              'ml_iters_true_funct_armijo': 1,
                              'll_iters_true_funct_armijo': 1,
                              'mlp_iters_max': 30,
                              'llp_iters_max': 60,                 
                              'mlp_iters_init': 1, 
                              'llp_iters_init': 1,
                              'inc_acc': True, 
                              'inc_acc_threshold_f1': 1e-2,
                              'inc_acc_threshold_f2': 1e-1,
                              'hess': 'CG-FD', 
                              'cg_fd_rtol': 1e-4,
                              'cg_fd_maxiter': 3,
                              'cg_fd_rtol_ml': 1e-4,
                              'cg_fd_maxiter_ml': 3,
                              'neumann_eta': None,
                              'neumann_hessian_q': None,
                              'neumann_eta_ml': None,
                              'neumann_hessian_q_ml': None,
                              'normalize': False, 
                              'advlearn_noise': param_dict[prob_name]['advlearn_noise'],
                              'advlearn_std_dev': param_dict[prob_name]['advlearn_std_dev'],
                              'iprint': param_dict[prob_name]['iprint'],  
                              'plot_color': '#1f77b4',
                              'plot_legend': 'TSG-N-FD',
                              'linestyle': 'dashdot',
                              'prob': prob}                       
                                
        ## TSG-AD
        algo_name = 'TSG-AD'
        exp_param_dict[prob_name][algo_name] = {'algo': 'stoctio', 
                              'algo_full_name': algo_name, 
                              'ul_lr': 0.3, 
                              'ml_lr': 0.2, 
                              'll_lr': 0.0001, 
                              'use_stopping_iter': param_dict[prob_name]['use_stopping_iter'], 
                              'max_iter': param_dict[prob_name]['max_iter'], 
                              'stopping_time': param_dict[prob_name]['stopping_time'], 
                              'true_func': param_dict[prob_name]['true_func'], 
                              'true_fbar': param_dict[prob_name]['true_fbar'],
                              'plot_f2_fbar': param_dict[prob_name]['plot_f2_fbar'], 
                              'plot_f3': param_dict[prob_name]['plot_f3'],
                              'plot_grad_f': param_dict[prob_name]['plot_grad_f'], 
                              'plot_grad_fbar': param_dict[prob_name]['plot_grad_fbar'],
                              'plot_grad_f3': param_dict[prob_name]['plot_grad_f3'],
                              'ul_stepsize_scheme': 1, 
                              'ml_stepsize_scheme': 1, 
                              'll_stepsize_scheme': 1,
                              'ml_stepsize_scheme_true_funct_armijo': 2,
                              'll_stepsize_scheme_true_funct_armijo': 2,
                              'ml_iters_true_funct_armijo': 1,
                              'll_iters_true_funct_armijo': 1,
                              'mlp_iters_max': 30,
                              'llp_iters_max': 60,                 
                              'mlp_iters_init': 1, 
                              'llp_iters_init': 1,
                              'inc_acc': True, 
                              'inc_acc_threshold_f1': 1e-2,
                              'inc_acc_threshold_f2': 1e-1,
                              'hess': 'autodiff', 
                              'cg_fd_rtol': None,
                              'cg_fd_maxiter': None,
                              'cg_fd_rtol_ml': None,
                              'cg_fd_maxiter_ml': None,
                              'neumann_eta': 0.05,
                              'neumann_hessian_q': 2,
                              'neumann_eta_ml': 0.05,
                              'neumann_hessian_q_ml': 2,
                              'normalize': False, 
                              'advlearn_noise': param_dict[prob_name]['advlearn_noise'],
                              'advlearn_std_dev': param_dict[prob_name]['advlearn_std_dev'],
                              'iprint': param_dict[prob_name]['iprint'],
                              'plot_color': '#CD5C5C',
                              'plot_legend': 'TSG-AD',
                              'linestyle': 'solid',
                              'prob': prob}    


# prob_name_list = ['Quartic2']

 
# algo_name_list = ['TSG-H'] 
# algo_name_list = ['TSG-N-FD']
# algo_name_list = ['TSG-AD']
# algo_name_list = ['TSG-H', 'TSG-N-FD', 'TSG-AD']                   

###############################################################################






######################
## Problem Quartic3 ##
######################

for use_stopping_iter_val, max_iter_val in [(True, 60), (False, 2000)]:
        
        if use_stopping_iter_val:
            name_aux = "iter"
        else:
            name_aux = "time"
            
        ## Problem Quartic3
        prob_name_root = "Quartic3" 
        prob_name = prob_name_root + name_aux
        prob_dict[prob_name] = {'name_prob_to_run_val': prob_name_root}
        prob = func.TrilevelProblem(name_prob_to_run=prob_dict[prob_name]['name_prob_to_run_val'])
        
        #----------------------------------------------------------------------#
        #-------------- Parameters common to all the algorithms  --------------#
        #----------------------------------------------------------------------#
        
        param_dict[prob_name] = {}
        
        # A flag to use the total number of iterations as a stopping criterion
        param_dict[prob_name]['use_stopping_iter'] = use_stopping_iter_val
        # Maximum number of iterations
        param_dict[prob_name]['max_iter'] = max_iter_val
        # Maximum running time (in sec) used when use_stopping_iter is False
        param_dict[prob_name]['stopping_time'] = 2   
        # Use true function
        param_dict[prob_name]['true_func'] = True  
        param_dict[prob_name]['true_fbar'] = True 
        # Create plots for fbar and f3
        param_dict[prob_name]['plot_f2_fbar'] = True
        param_dict[prob_name]['plot_f3'] = True
        # Create plots for grad_f, grad_fbar, and grad_f3
        param_dict[prob_name]['plot_grad_f'] = True
        param_dict[prob_name]['plot_grad_fbar'] = True
        param_dict[prob_name]['plot_grad_f3'] = True
        # Compute test MSE with or without Gaussian noise (only for Adversarial Learning)
        param_dict[prob_name]['advlearn_noise'] = None
        param_dict[prob_name]['advlearn_std_dev'] = None
        # Number of runs for each algorithm
        param_dict[prob_name]['num_rep_value'] = 10
        # Sets the verbosity level for printing information (higher values correspond to more detailed output)
        param_dict[prob_name]['iprint'] = 8
        
        ####################################################### OLD
        # List of colors for the algorithms in the plots
        # plot_color_list = ['#1f77b4','#2ca02c','#bcbd22','#CD5C5C','#ff7f0e']
        # plot_legend_list = ['TSG-N-FD (inc. acc.)','TSG-1 (inc. acc.)','TSG-H (inc. acc.)','DARTS','StocBiO (inc. acc.)']
        ####################################################### OLD
        
        #--------------------------------------------------------------------#
        #-------------- Parameters specific for each algorithm --------------#
        #--------------------------------------------------------------------#
        
        exp_param_dict[prob_name] = {}
        
        ## TSG-H 
        algo_name = 'TSG-H'
        exp_param_dict[prob_name][algo_name] = {'algo': 'tsg', 
                             'algo_full_name': algo_name, 
                             'ul_lr': 0.3,  #0.3 
                             'ml_lr': 0.2, 
                             'll_lr': 0.1, 
                             'use_stopping_iter': param_dict[prob_name]['use_stopping_iter'],
                             'max_iter': param_dict[prob_name]['max_iter'], 
                             'stopping_time': param_dict[prob_name]['stopping_time'], 
                             'true_func': param_dict[prob_name]['true_func'], 
                             'true_fbar': param_dict[prob_name]['true_fbar'],
                             'plot_f2_fbar': param_dict[prob_name]['plot_f2_fbar'], 
                             'plot_f3': param_dict[prob_name]['plot_f3'],
                             'plot_grad_f': param_dict[prob_name]['plot_grad_f'], 
                             'plot_grad_fbar': param_dict[prob_name]['plot_grad_fbar'],
                             'plot_grad_f3': param_dict[prob_name]['plot_grad_f3'],
                             'ul_stepsize_scheme': 1, 
                             'ml_stepsize_scheme': 1, 
                             'll_stepsize_scheme': 1,
                             'ml_stepsize_scheme_true_funct_armijo': 2,
                             'll_stepsize_scheme_true_funct_armijo': 2,
                             'ml_iters_true_funct_armijo': 1,
                             'll_iters_true_funct_armijo': 1,
                             'mlp_iters_max': 30,
                             'llp_iters_max': 60,                 
                             'mlp_iters_init': 1, 
                             'llp_iters_init': 1,
                             'inc_acc': True, 
                             'inc_acc_threshold_f1': 1e-2,
                             'inc_acc_threshold_f2': 1e-1,
                             'hess': True, 
                             'cg_fd_rtol': None,
                             'cg_fd_maxiter': None,
                             'cg_fd_rtol_ml': None,
                             'cg_fd_maxiter_ml': None,
                             'neumann_eta': None,
                             'neumann_hessian_q': None,
                             'neumann_eta_ml': None,
                             'neumann_hessian_q_ml': None,
                             'normalize': False, 
                             'advlearn_noise': param_dict[prob_name]['advlearn_noise'],
                             'advlearn_std_dev': param_dict[prob_name]['advlearn_std_dev'],
                             'iprint': param_dict[prob_name]['iprint'],
                             'plot_color': '#bcbd22',
                             'plot_legend': 'TSG-H',
                             'linestyle': 'dashed',
                             'prob': prob} 
            
        ## TSG-N-FD 
        algo_name = 'TSG-N-FD'
        exp_param_dict[prob_name][algo_name] = {'algo': 'tsg', 
                              'algo_full_name': algo_name, 
                              'ul_lr': 0.01,  #0.01
                              'ml_lr': 0.01,  #0.01
                              'll_lr': 0.001,  #0.001
                              'use_stopping_iter': param_dict[prob_name]['use_stopping_iter'], 
                              'max_iter': param_dict[prob_name]['max_iter'], 
                              'stopping_time': param_dict[prob_name]['stopping_time'], 
                              'true_func': param_dict[prob_name]['true_func'], 
                              'true_fbar': param_dict[prob_name]['true_fbar'],
                              'plot_f2_fbar': param_dict[prob_name]['plot_f2_fbar'], 
                              'plot_f3': param_dict[prob_name]['plot_f3'],
                              'plot_grad_f': param_dict[prob_name]['plot_grad_f'], 
                              'plot_grad_fbar': param_dict[prob_name]['plot_grad_fbar'],
                              'plot_grad_f3': param_dict[prob_name]['plot_grad_f3'],
                              'ul_stepsize_scheme': 1, 
                              'ml_stepsize_scheme': 1, 
                              'll_stepsize_scheme': 1,
                              'ml_stepsize_scheme_true_funct_armijo': 2,
                              'll_stepsize_scheme_true_funct_armijo': 2,
                              'ml_iters_true_funct_armijo': 1,
                              'll_iters_true_funct_armijo': 1,
                              'mlp_iters_max': 30,
                              'llp_iters_max': 60,                 
                              'mlp_iters_init': 1, 
                              'llp_iters_init': 1,
                              'inc_acc': True, 
                              'inc_acc_threshold_f1': 1e-2,
                              'inc_acc_threshold_f2': 1e-1,
                              'hess': 'CG-FD', 
                              'cg_fd_rtol': 1e-4,
                              'cg_fd_maxiter': 3,
                              'cg_fd_rtol_ml': 1e-4,
                              'cg_fd_maxiter_ml': 3,
                              'neumann_eta': None,
                              'neumann_hessian_q': None,
                              'neumann_eta_ml': None,
                              'neumann_hessian_q_ml': None,
                              'normalize': False, 
                              'advlearn_noise': param_dict[prob_name]['advlearn_noise'],
                              'advlearn_std_dev': param_dict[prob_name]['advlearn_std_dev'],
                              'iprint': param_dict[prob_name]['iprint'],  
                              'plot_color': '#1f77b4',
                              'plot_legend': 'TSG-N-FD',
                              'linestyle': 'dashdot',
                              'prob': prob}                       
                                
        ## TSG-AD 
        algo_name = 'TSG-AD'
        exp_param_dict[prob_name][algo_name] = {'algo': 'stoctio', 
                              'algo_full_name': algo_name, 
                              'ul_lr': 0.3, 
                              'ml_lr': 0.2, 
                              'll_lr': 0.0001, 
                              'use_stopping_iter': param_dict[prob_name]['use_stopping_iter'], 
                              'max_iter': param_dict[prob_name]['max_iter'], 
                              'stopping_time': param_dict[prob_name]['stopping_time'], 
                              'true_func': param_dict[prob_name]['true_func'], 
                              'true_fbar': param_dict[prob_name]['true_fbar'],
                              'plot_f2_fbar': param_dict[prob_name]['plot_f2_fbar'], 
                              'plot_f3': param_dict[prob_name]['plot_f3'],
                              'plot_grad_f': param_dict[prob_name]['plot_grad_f'], 
                              'plot_grad_fbar': param_dict[prob_name]['plot_grad_fbar'],
                              'plot_grad_f3': param_dict[prob_name]['plot_grad_f3'],
                              'ul_stepsize_scheme': 1, 
                              'ml_stepsize_scheme': 1, 
                              'll_stepsize_scheme': 1,
                              'ml_stepsize_scheme_true_funct_armijo': 2,
                              'll_stepsize_scheme_true_funct_armijo': 2,
                              'ml_iters_true_funct_armijo': 1,
                              'll_iters_true_funct_armijo': 1,
                              'mlp_iters_max': 30,
                              'llp_iters_max': 60,                 
                              'mlp_iters_init': 1, 
                              'llp_iters_init': 1,
                              'inc_acc': True, 
                              'inc_acc_threshold_f1': 1e-2,
                              'inc_acc_threshold_f2': 1e-1,
                              'hess': 'autodiff', 
                              'cg_fd_rtol': None,
                              'cg_fd_maxiter': None,
                              'cg_fd_rtol_ml': None,
                              'cg_fd_maxiter_ml': None,
                              'neumann_eta': 0.05,
                              'neumann_hessian_q': 2,
                              'neumann_eta_ml': 0.05,
                              'neumann_hessian_q_ml': 2,
                              'normalize': False, 
                              'advlearn_noise': param_dict[prob_name]['advlearn_noise'],
                              'advlearn_std_dev': param_dict[prob_name]['advlearn_std_dev'],
                              'iprint': param_dict[prob_name]['iprint'],
                              'plot_color': '#CD5C5C',
                              'plot_legend': 'TSG-AD',
                              'linestyle': 'solid',
                              'prob': prob}    


# prob_name_list = ['Quartic3']

 
# algo_name_list = ['TSG-H'] 
# algo_name_list = ['TSG-N-FD']
# algo_name_list = ['TSG-AD']
# algo_name_list = ['TSG-H', 'TSG-AD'] 
# algo_name_list = ['TSG-H', 'TSG-N-FD', 'TSG-AD']                   

###############################################################################







######################
## Problem Quartic4 ##
######################

for use_stopping_iter_val, max_iter_val in [(True, 60), (False, 2000)]:
        
        if use_stopping_iter_val:
            name_aux = "iter"
        else:
            name_aux = "time"
            
        ## Problem Quartic4
        prob_name_root = "Quartic4" 
        prob_name = prob_name_root + name_aux
        prob_dict[prob_name] = {'name_prob_to_run_val': prob_name_root}
        prob = func.TrilevelProblem(name_prob_to_run=prob_dict[prob_name]['name_prob_to_run_val'])
        
        #----------------------------------------------------------------------#
        #-------------- Parameters common to all the algorithms  --------------#
        #----------------------------------------------------------------------#
        
        param_dict[prob_name] = {}
        
        # A flag to use the total number of iterations as a stopping criterion
        param_dict[prob_name]['use_stopping_iter'] = use_stopping_iter_val
        # Maximum number of iterations
        param_dict[prob_name]['max_iter'] = max_iter_val
        # Maximum running time (in sec) used when use_stopping_iter is False
        param_dict[prob_name]['stopping_time'] = 2   
        # Use true function
        param_dict[prob_name]['true_func'] = True  
        param_dict[prob_name]['true_fbar'] = True 
        # Create plots for fbar and f3
        param_dict[prob_name]['plot_f2_fbar'] = True
        param_dict[prob_name]['plot_f3'] = True
        # Create plots for grad_f, grad_fbar, and grad_f3
        param_dict[prob_name]['plot_grad_f'] = True
        param_dict[prob_name]['plot_grad_fbar'] = True
        param_dict[prob_name]['plot_grad_f3'] = True
        # Compute test MSE with or without Gaussian noise (only for Adversarial Learning)
        param_dict[prob_name]['advlearn_noise'] = None
        param_dict[prob_name]['advlearn_std_dev'] = None
        # Number of runs for each algorithm
        param_dict[prob_name]['num_rep_value'] = 10
        # Sets the verbosity level for printing information (higher values correspond to more detailed output)
        param_dict[prob_name]['iprint'] = 8
        
        ####################################################### OLD
        # List of colors for the algorithms in the plots
        # plot_color_list = ['#1f77b4','#2ca02c','#bcbd22','#CD5C5C','#ff7f0e']
        # plot_legend_list = ['TSG-N-FD (inc. acc.)','TSG-1 (inc. acc.)','TSG-H (inc. acc.)','DARTS','StocBiO (inc. acc.)']
        ####################################################### OLD
        
        #--------------------------------------------------------------------#
        #-------------- Parameters specific for each algorithm --------------#
        #--------------------------------------------------------------------#
        
        exp_param_dict[prob_name] = {}
        
        ## TSG-H 
        algo_name = 'TSG-H'
        exp_param_dict[prob_name][algo_name] = {'algo': 'tsg', 
                             'algo_full_name': algo_name, 
                             'ul_lr': 0.3,  #0.3 
                             'ml_lr': 0.2, 
                             'll_lr': 0.1, 
                             'use_stopping_iter': param_dict[prob_name]['use_stopping_iter'],
                             'max_iter': param_dict[prob_name]['max_iter'], 
                             'stopping_time': param_dict[prob_name]['stopping_time'], 
                             'true_func': param_dict[prob_name]['true_func'], 
                             'true_fbar': param_dict[prob_name]['true_fbar'],
                             'plot_f2_fbar': param_dict[prob_name]['plot_f2_fbar'], 
                             'plot_f3': param_dict[prob_name]['plot_f3'],
                             'plot_grad_f': param_dict[prob_name]['plot_grad_f'], 
                             'plot_grad_fbar': param_dict[prob_name]['plot_grad_fbar'],
                             'plot_grad_f3': param_dict[prob_name]['plot_grad_f3'],
                             'ul_stepsize_scheme': 1, 
                             'ml_stepsize_scheme': 1, 
                             'll_stepsize_scheme': 1,
                             'ml_stepsize_scheme_true_funct_armijo': 2,
                             'll_stepsize_scheme_true_funct_armijo': 2,
                             'ml_iters_true_funct_armijo': 1,
                             'll_iters_true_funct_armijo': 1,
                             'mlp_iters_max': 30,
                             'llp_iters_max': 60,                 
                             'mlp_iters_init': 1, 
                             'llp_iters_init': 1,
                             'inc_acc': True, 
                             'inc_acc_threshold_f1': 1e-2,
                             'inc_acc_threshold_f2': 1e-1,
                             'hess': True, 
                             'cg_fd_rtol': None,
                             'cg_fd_maxiter': None,
                             'cg_fd_rtol_ml': None,
                             'cg_fd_maxiter_ml': None,
                             'neumann_eta': None,
                             'neumann_hessian_q': None,
                             'neumann_eta_ml': None,
                             'neumann_hessian_q_ml': None,
                             'normalize': False, 
                             'advlearn_noise': param_dict[prob_name]['advlearn_noise'],
                             'advlearn_std_dev': param_dict[prob_name]['advlearn_std_dev'],
                             'iprint': param_dict[prob_name]['iprint'],
                             'plot_color': '#bcbd22',
                             'plot_legend': 'TSG-H',
                             'linestyle': 'dashed',
                             'prob': prob} 
            
        ## TSG-N-FD 
        algo_name = 'TSG-N-FD'
        exp_param_dict[prob_name][algo_name] = {'algo': 'tsg', 
                              'algo_full_name': algo_name, 
                              'ul_lr': 0.01, 
                              'ml_lr': 0.01, 
                              'll_lr': 0.001, 
                              'use_stopping_iter': param_dict[prob_name]['use_stopping_iter'], 
                              'max_iter': param_dict[prob_name]['max_iter'], 
                              'stopping_time': param_dict[prob_name]['stopping_time'], 
                              'true_func': param_dict[prob_name]['true_func'], 
                              'true_fbar': param_dict[prob_name]['true_fbar'],
                              'plot_f2_fbar': param_dict[prob_name]['plot_f2_fbar'], 
                              'plot_f3': param_dict[prob_name]['plot_f3'],
                              'plot_grad_f': param_dict[prob_name]['plot_grad_f'], 
                              'plot_grad_fbar': param_dict[prob_name]['plot_grad_fbar'],
                              'plot_grad_f3': param_dict[prob_name]['plot_grad_f3'],
                              'ul_stepsize_scheme': 1, 
                              'ml_stepsize_scheme': 1, 
                              'll_stepsize_scheme': 1,
                              'ml_stepsize_scheme_true_funct_armijo': 2,
                              'll_stepsize_scheme_true_funct_armijo': 2,
                              'ml_iters_true_funct_armijo': 1,
                              'll_iters_true_funct_armijo': 1,
                              'mlp_iters_max': 30,
                              'llp_iters_max': 60,                 
                              'mlp_iters_init': 1, 
                              'llp_iters_init': 1,
                              'inc_acc': True, 
                              'inc_acc_threshold_f1': 1e-2,
                              'inc_acc_threshold_f2': 1e-1,
                              'hess': 'CG-FD', 
                              'cg_fd_rtol': 1e-4,
                              'cg_fd_maxiter': 3,
                              'cg_fd_rtol_ml': 1e-4,
                              'cg_fd_maxiter_ml': 3,
                              'neumann_eta': None,
                              'neumann_hessian_q': None,
                              'neumann_eta_ml': None,
                              'neumann_hessian_q_ml': None,
                              'normalize': False, 
                              'advlearn_noise': param_dict[prob_name]['advlearn_noise'],
                              'advlearn_std_dev': param_dict[prob_name]['advlearn_std_dev'],
                              'iprint': param_dict[prob_name]['iprint'],  
                              'plot_color': '#1f77b4',
                              'plot_legend': 'TSG-N-FD',
                              'linestyle': 'dashdot',
                              'prob': prob}                       
                                
        ## TSG-AD 
        algo_name = 'TSG-AD'
        exp_param_dict[prob_name][algo_name] = {'algo': 'stoctio', 
                              'algo_full_name': algo_name, 
                              'ul_lr': 0.3, 
                              'ml_lr': 0.2, 
                              'll_lr': 0.0001, 
                              'use_stopping_iter': param_dict[prob_name]['use_stopping_iter'], 
                              'max_iter': param_dict[prob_name]['max_iter'], 
                              'stopping_time': param_dict[prob_name]['stopping_time'], 
                              'true_func': param_dict[prob_name]['true_func'], 
                              'true_fbar': param_dict[prob_name]['true_fbar'],
                              'plot_f2_fbar': param_dict[prob_name]['plot_f2_fbar'], 
                              'plot_f3': param_dict[prob_name]['plot_f3'],
                              'plot_grad_f': param_dict[prob_name]['plot_grad_f'], 
                              'plot_grad_fbar': param_dict[prob_name]['plot_grad_fbar'],
                              'plot_grad_f3': param_dict[prob_name]['plot_grad_f3'],
                              'ul_stepsize_scheme': 1, 
                              'ml_stepsize_scheme': 1, 
                              'll_stepsize_scheme': 1,
                              'ml_stepsize_scheme_true_funct_armijo': 2,
                              'll_stepsize_scheme_true_funct_armijo': 2,
                              'ml_iters_true_funct_armijo': 1,
                              'll_iters_true_funct_armijo': 1,
                              'mlp_iters_max': 30,
                              'llp_iters_max': 60,                 
                              'mlp_iters_init': 1, 
                              'llp_iters_init': 1,
                              'inc_acc': True, 
                              'inc_acc_threshold_f1': 1e-2,
                              'inc_acc_threshold_f2': 1e-1,
                              'hess': 'autodiff', 
                              'cg_fd_rtol': None,
                              'cg_fd_maxiter': None,
                              'cg_fd_rtol_ml': None,
                              'cg_fd_maxiter_ml': None,
                              'neumann_eta': 0.05,
                              'neumann_hessian_q': 2,
                              'neumann_eta_ml': 0.05,
                              'neumann_hessian_q_ml': 2,
                              'normalize': False, 
                              'advlearn_noise': param_dict[prob_name]['advlearn_noise'],
                              'advlearn_std_dev': param_dict[prob_name]['advlearn_std_dev'],
                              'iprint': param_dict[prob_name]['iprint'],
                              'plot_color': '#CD5C5C',
                              'plot_legend': 'TSG-AD',
                              'linestyle': 'solid',
                              'prob': prob}    


# prob_name_list = ['Quartic4']

 
# algo_name_list = ['TSG-H'] 
# algo_name_list = ['TSG-N-FD']
# algo_name_list = ['TSG-AD']
# algo_name_list = ['TSG-H', 'TSG-AD'] 
# algo_name_list = ['TSG-H', 'TSG-AD']                   

###############################################################################





##################################
## Problem AdversarialLearning1 ##
##################################

for advlearn_std_dev_val in [0, 5]:
    for use_stopping_iter_val, max_iter_val in [(True, 100), (False, 2000)]:
            
            if use_stopping_iter_val:
                name_aux = "iter"
            else:
                name_aux = "time"
                
            ## Problem AdversarialLearning1
            prob_name_root = "AdversarialLearning1" 
            prob_name = prob_name_root + name_aux + str(advlearn_std_dev_val)
            prob_dict[prob_name] = {'name_prob_to_run_val': prob_name_root}
            prob = func.TrilevelProblem(name_prob_to_run=prob_dict[prob_name]['name_prob_to_run_val'])
            
            #----------------------------------------------------------------------#
            #-------------- Parameters common to all the algorithms  --------------#
            #----------------------------------------------------------------------#
            
            param_dict[prob_name] = {}
            
            # A flag to use the total number of iterations as a stopping criterion
            param_dict[prob_name]['use_stopping_iter'] = use_stopping_iter_val
            # Maximum number of iterations
            param_dict[prob_name]['max_iter'] = max_iter_val
            # Maximum running time (in sec) used when use_stopping_iter is False
            param_dict[prob_name]['stopping_time'] = 8   
            # Use true function
            param_dict[prob_name]['true_func'] = False  
            param_dict[prob_name]['true_fbar'] = False 
            # Create plots for fbar and f3
            param_dict[prob_name]['plot_f2_fbar'] = False
            param_dict[prob_name]['plot_f3'] = False
            # Create plots for grad_f, grad_fbar, and grad_f3
            param_dict[prob_name]['plot_grad_f'] = False
            param_dict[prob_name]['plot_grad_fbar'] = False
            param_dict[prob_name]['plot_grad_f3'] = False
            # Compute test MSE with or without Gaussian noise
            param_dict[prob_name]['advlearn_noise'] = True 
            param_dict[prob_name]['advlearn_std_dev'] = advlearn_std_dev_val
            # Number of runs for each algorithm
            param_dict[prob_name]['num_rep_value'] = 10
            # Sets the verbosity level for printing information (higher values correspond to more detailed output)
            param_dict[prob_name]['iprint'] = 8
            
            ####################################################### OLD
            # List of colors for the algorithms in the plots
            # plot_color_list = ['#1f77b4','#2ca02c','#bcbd22','#CD5C5C','#ff7f0e']
            # plot_legend_list = ['TSG-N-FD (inc. acc.)','TSG-1 (inc. acc.)','TSG-H (inc. acc.)','DARTS','StocBiO (inc. acc.)']
            ####################################################### OLD
            
            #--------------------------------------------------------------------#
            #-------------- Parameters specific for each algorithm --------------#
            #--------------------------------------------------------------------#
            
            exp_param_dict[prob_name] = {}
                
            ## TSG-N-FD 
            algo_name = 'TSG-N-FD'
            exp_param_dict[prob_name][algo_name] = {'algo': 'tsg', 
                                  'algo_full_name': algo_name, 
                                  'ul_lr': 0.1, 
                                  'ml_lr': 0.000001, 
                                  'll_lr': 0.000001, 
                                  'use_stopping_iter': param_dict[prob_name]['use_stopping_iter'], 
                                  'max_iter': param_dict[prob_name]['max_iter'], 
                                  'stopping_time': param_dict[prob_name]['stopping_time'], 
                                  'true_func': param_dict[prob_name]['true_func'], 
                                  'true_fbar': param_dict[prob_name]['true_fbar'],
                                  'plot_f2_fbar': param_dict[prob_name]['plot_f2_fbar'], 
                                  'plot_f3': param_dict[prob_name]['plot_f3'],
                                  'plot_grad_f': param_dict[prob_name]['plot_grad_f'], 
                                  'plot_grad_fbar': param_dict[prob_name]['plot_grad_fbar'],
                                  'plot_grad_f3': param_dict[prob_name]['plot_grad_f3'],
                                  'ul_stepsize_scheme': 1, 
                                  'ml_stepsize_scheme': 1, 
                                  'll_stepsize_scheme': 1,
                                  'ml_stepsize_scheme_true_funct_armijo': 2,
                                  'll_stepsize_scheme_true_funct_armijo': 2,
                                  'ml_iters_true_funct_armijo': 1,
                                  'll_iters_true_funct_armijo': 1,
                                  'mlp_iters_max': 30,
                                  'llp_iters_max': 60,                 
                                  'mlp_iters_init': 1, 
                                  'llp_iters_init': 1,
                                  'inc_acc': True, 
                                  'inc_acc_threshold_f1': 1e-2,
                                  'inc_acc_threshold_f2': 1e-1,
                                  'hess': 'CG-FD', 
                                  'cg_fd_rtol': 1e-4,
                                  'cg_fd_maxiter': 2,
                                  'cg_fd_rtol_ml': 1e-4,
                                  'cg_fd_maxiter_ml': 2,
                                  'neumann_eta': None,
                                  'neumann_hessian_q': None,
                                  'neumann_eta_ml': None,
                                  'neumann_hessian_q_ml': None,
                                  'normalize': False, 
                                  'advlearn_noise': param_dict[prob_name]['advlearn_noise'],
                                  'advlearn_std_dev': param_dict[prob_name]['advlearn_std_dev'],
                                  'iprint': param_dict[prob_name]['iprint'],  
                                  'plot_color': '#1f77b4',
                                  'plot_legend': 'TSG-N-FD',
                                  'linestyle': 'dashdot',
                                  'prob': prob}                       
                                    
            ## TSG-AD 
            algo_name = 'TSG-AD'
            exp_param_dict[prob_name][algo_name] = {'algo': 'stoctio', 
                                  'algo_full_name': algo_name, 
                                  'ul_lr': 0.1, #0.1
                                  'ml_lr': 0.01, #0.01
                                  'll_lr': 0.1, #0.1
                                  'use_stopping_iter': param_dict[prob_name]['use_stopping_iter'], 
                                  'max_iter': param_dict[prob_name]['max_iter'], 
                                  'stopping_time': param_dict[prob_name]['stopping_time'], 
                                  'true_func': param_dict[prob_name]['true_func'], 
                                  'true_fbar': param_dict[prob_name]['true_fbar'],
                                  'plot_f2_fbar': param_dict[prob_name]['plot_f2_fbar'], 
                                  'plot_f3': param_dict[prob_name]['plot_f3'],
                                  'plot_grad_f': param_dict[prob_name]['plot_grad_f'], 
                                  'plot_grad_fbar': param_dict[prob_name]['plot_grad_fbar'],
                                  'plot_grad_f3': param_dict[prob_name]['plot_grad_f3'],
                                  'ul_stepsize_scheme': 1, 
                                  'ml_stepsize_scheme': 1, 
                                  'll_stepsize_scheme': 1,
                                  'ml_stepsize_scheme_true_funct_armijo': 2,
                                  'll_stepsize_scheme_true_funct_armijo': 2,
                                  'ml_iters_true_funct_armijo': 1,
                                  'll_iters_true_funct_armijo': 1,
                                  'mlp_iters_max': 30,
                                  'llp_iters_max': 60,                 
                                  'mlp_iters_init': 1, 
                                  'llp_iters_init': 1,
                                  'inc_acc': True, 
                                  'inc_acc_threshold_f1': 1e-2,
                                  'inc_acc_threshold_f2': 1e-1,
                                  'hess': 'autodiff', 
                                  'cg_fd_rtol': None,
                                  'cg_fd_maxiter': None,
                                  'cg_fd_rtol_ml': None,
                                  'cg_fd_maxiter_ml': None,
                                  'neumann_eta': 0.05,
                                  'neumann_hessian_q': 2,
                                  'neumann_eta_ml': 0.05,
                                  'neumann_hessian_q_ml': 2,
                                  'normalize': False, 
                                  'advlearn_noise': param_dict[prob_name]['advlearn_noise'],
                                  'advlearn_std_dev': param_dict[prob_name]['advlearn_std_dev'],
                                  'iprint': param_dict[prob_name]['iprint'],
                                  'plot_color': '#CD5C5C',
                                  'plot_legend': 'TSG-AD',
                                  'linestyle': 'solid',
                                  'prob': prob}    
            
            ## BSG-AD (without UL)
            algo_name = 'BSG-AD (without UL)'
            exp_param_dict[prob_name][algo_name] = {'algo': 'remove_ul', 
                                  'algo_full_name': algo_name, 
                                  'ul_lr': 0, #0.01
                                  'ml_lr': 0.01, #0.01
                                  'll_lr': 0.1, #0.01
                                  'use_stopping_iter': param_dict[prob_name]['use_stopping_iter'], 
                                  'max_iter': param_dict[prob_name]['max_iter'], 
                                  'stopping_time': param_dict[prob_name]['stopping_time'], 
                                  'true_func': param_dict[prob_name]['true_func'], 
                                  'true_fbar': param_dict[prob_name]['true_fbar'],
                                  'plot_f2_fbar': param_dict[prob_name]['plot_f2_fbar'], 
                                  'plot_f3': param_dict[prob_name]['plot_f3'],
                                  'plot_grad_f': param_dict[prob_name]['plot_grad_f'], 
                                  'plot_grad_fbar': param_dict[prob_name]['plot_grad_fbar'],
                                  'plot_grad_f3': param_dict[prob_name]['plot_grad_f3'],
                                  'ul_stepsize_scheme': 1, 
                                  'ml_stepsize_scheme': 1, 
                                  'll_stepsize_scheme': 1,
                                  'ml_stepsize_scheme_true_funct_armijo': 2,
                                  'll_stepsize_scheme_true_funct_armijo': 2,
                                  'ml_iters_true_funct_armijo': 1,
                                  'll_iters_true_funct_armijo': 1,
                                  'mlp_iters_max': 30,
                                  'llp_iters_max': 60,                 
                                  'mlp_iters_init': 1, 
                                  'llp_iters_init': 1,
                                  'inc_acc': True, 
                                  'inc_acc_threshold_f1': 1e-2,
                                  'inc_acc_threshold_f2': 1e-1,
                                  'hess': 'autodiff', 
                                  'cg_fd_rtol': None,
                                  'cg_fd_maxiter': None,
                                  'cg_fd_rtol_ml': None,
                                  'cg_fd_maxiter_ml': None,
                                  'neumann_eta': 0.05,
                                  'neumann_hessian_q': 2,
                                  'neumann_eta_ml': 0.05,
                                  'neumann_hessian_q_ml': 2,
                                  'normalize': False, 
                                  'advlearn_noise': param_dict[prob_name]['advlearn_noise'],
                                  'advlearn_std_dev': param_dict[prob_name]['advlearn_std_dev'],
                                  'iprint': param_dict[prob_name]['iprint'],
                                  'plot_color': '#2ca02c',
                                  'plot_legend': 'BSG-AD (without UL)',
                                  'linestyle': 'dashdot',
                                  'prob': prob}
            
            ## BSG-AD (without LL)
            algo_name = 'BSG-AD (without LL)'
            exp_param_dict[prob_name][algo_name] = {'algo': 'remove_ll', 
                                  'algo_full_name': algo_name, 
                                  'ul_lr': 0.1, #0.01
                                  'ml_lr': 0.01, #0.01
                                  'll_lr': 0, #0.01
                                  'use_stopping_iter': param_dict[prob_name]['use_stopping_iter'], 
                                  'max_iter': param_dict[prob_name]['max_iter'], 
                                  'stopping_time': param_dict[prob_name]['stopping_time'], 
                                  'true_func': param_dict[prob_name]['true_func'], 
                                  'true_fbar': param_dict[prob_name]['true_fbar'],
                                  'plot_f2_fbar': param_dict[prob_name]['plot_f2_fbar'], 
                                  'plot_f3': param_dict[prob_name]['plot_f3'],
                                  'plot_grad_f': param_dict[prob_name]['plot_grad_f'], 
                                  'plot_grad_fbar': param_dict[prob_name]['plot_grad_fbar'],
                                  'plot_grad_f3': param_dict[prob_name]['plot_grad_f3'],
                                  'ul_stepsize_scheme': 1, 
                                  'ml_stepsize_scheme': 1, 
                                  'll_stepsize_scheme': 1,
                                  'ml_stepsize_scheme_true_funct_armijo': 2,
                                  'll_stepsize_scheme_true_funct_armijo': 2,
                                  'ml_iters_true_funct_armijo': 1,
                                  'll_iters_true_funct_armijo': 1,
                                  'mlp_iters_max': 30,
                                  'llp_iters_max': 60,                 
                                  'mlp_iters_init': 1, 
                                  'llp_iters_init': 1,
                                  'inc_acc': True, 
                                  'inc_acc_threshold_f1': 1e-2,
                                  'inc_acc_threshold_f2': 1e-1,
                                  'hess': 'autodiff', 
                                  'cg_fd_rtol': None,
                                  'cg_fd_maxiter': None,
                                  'cg_fd_rtol_ml': None,
                                  'cg_fd_maxiter_ml': None,
                                  'neumann_eta': 0.05,
                                  'neumann_hessian_q': 2,
                                  'neumann_eta_ml': 0.05,
                                  'neumann_hessian_q_ml': 2,
                                  'normalize': False, 
                                  'advlearn_noise': param_dict[prob_name]['advlearn_noise'],
                                  'advlearn_std_dev': param_dict[prob_name]['advlearn_std_dev'],
                                  'iprint': param_dict[prob_name]['iprint'],
                                  'plot_color': '#ff7f0e',
                                  'plot_legend': 'BSG-AD (without LL)',
                                  'linestyle': 'dashed',
                                  'prob': prob}

# prob_name_list = ['AdversarialLearning1']

 
# algo_name_list = ['TSG-N-FD']
# algo_name_list = ['TSG-AD'] 
# algo_name_list = ['BSG-AD (without UL)'] 
# algo_name_list = ['BSG-AD (without LL)'] 
# algo_name_list = ['TSG-N-FD', 'TSG-AD
# algo_name_list = ['TSG-AD', 'BSG-AD (without UL)', 'BSG-AD (without LL)']                   

###############################################################################







##################################
## Problem AdversarialLearning2 ##
##################################

for advlearn_std_dev_val in [0, 5]:
    for use_stopping_iter_val, max_iter_val in [(True, 100), (False, 10000)]:
            
            if use_stopping_iter_val:
                name_aux = "iter"
            else:
                name_aux = "time"
                
            ## Problem AdversarialLearning2
            prob_name_root = "AdversarialLearning2" 
            prob_name = prob_name_root + name_aux + str(advlearn_std_dev_val)
            prob_dict[prob_name] = {'name_prob_to_run_val': prob_name_root}
            prob = func.TrilevelProblem(name_prob_to_run=prob_dict[prob_name]['name_prob_to_run_val'])
            
            #----------------------------------------------------------------------#
            #-------------- Parameters common to all the algorithms  --------------#
            #----------------------------------------------------------------------#
            
            param_dict[prob_name] = {}
            
            # A flag to use the total number of iterations as a stopping criterion
            param_dict[prob_name]['use_stopping_iter'] = use_stopping_iter_val
            # Maximum number of iterations
            param_dict[prob_name]['max_iter'] = max_iter_val   
            # Maximum running time (in sec) used when use_stopping_iter is False
            param_dict[prob_name]['stopping_time'] = 8   
            # Use true function
            param_dict[prob_name]['true_func'] = False  
            param_dict[prob_name]['true_fbar'] = False 
            # Create plots for fbar and f3
            param_dict[prob_name]['plot_f2_fbar'] = False
            param_dict[prob_name]['plot_f3'] = False
            # Create plots for grad_f, grad_fbar, and grad_f3
            param_dict[prob_name]['plot_grad_f'] = False
            param_dict[prob_name]['plot_grad_fbar'] = False
            param_dict[prob_name]['plot_grad_f3'] = False
            # Compute test MSE with or without Gaussian noise
            param_dict[prob_name]['advlearn_noise'] = True 
            param_dict[prob_name]['advlearn_std_dev'] = advlearn_std_dev_val
            # Number of runs for each algorithm
            param_dict[prob_name]['num_rep_value'] = 10 
            # Sets the verbosity level for printing information (higher values correspond to more detailed output)
            param_dict[prob_name]['iprint'] = 8
            
            ####################################################### OLD
            # List of colors for the algorithms in the plots
            # plot_color_list = ['#1f77b4','#2ca02c','#bcbd22','#CD5C5C','#ff7f0e']
            # plot_legend_list = ['TSG-N-FD (inc. acc.)','TSG-1 (inc. acc.)','TSG-H (inc. acc.)','DARTS','StocBiO (inc. acc.)']
            ####################################################### OLD
            
            #--------------------------------------------------------------------#
            #-------------- Parameters specific for each algorithm --------------#
            #--------------------------------------------------------------------#
            
            exp_param_dict[prob_name] = {}
                
            ## TSG-N-FD 
            algo_name = 'TSG-N-FD'
            exp_param_dict[prob_name][algo_name] = {'algo': 'tsg', 
                                  'algo_full_name': algo_name, 
                                  'ul_lr': 0.01, 
                                  'ml_lr': 0.01, 
                                  'll_lr': 0.01, 
                                  'use_stopping_iter': param_dict[prob_name]['use_stopping_iter'], 
                                  'max_iter': param_dict[prob_name]['max_iter'], 
                                  'stopping_time': param_dict[prob_name]['stopping_time'], 
                                  'true_func': param_dict[prob_name]['true_func'], 
                                  'true_fbar': param_dict[prob_name]['true_fbar'],
                                  'plot_f2_fbar': param_dict[prob_name]['plot_f2_fbar'], 
                                  'plot_f3': param_dict[prob_name]['plot_f3'],
                                  'plot_grad_f': param_dict[prob_name]['plot_grad_f'], 
                                  'plot_grad_fbar': param_dict[prob_name]['plot_grad_fbar'],
                                  'plot_grad_f3': param_dict[prob_name]['plot_grad_f3'],
                                  'ul_stepsize_scheme': 0, 
                                  'ml_stepsize_scheme': 0, 
                                  'll_stepsize_scheme': 0,
                                  'ml_stepsize_scheme_true_funct_armijo': 2,
                                  'll_stepsize_scheme_true_funct_armijo': 2,
                                  'ml_iters_true_funct_armijo': 1,
                                  'll_iters_true_funct_armijo': 1,
                                  'mlp_iters_max': 30,
                                  'llp_iters_max': 60,                 
                                  'mlp_iters_init': 1, 
                                  'llp_iters_init': 1,
                                  'inc_acc': True, 
                                  'inc_acc_threshold_f1': 1e-2,
                                  'inc_acc_threshold_f2': 1e-1,
                                  'hess': 'CG-FD', 
                                  'cg_fd_rtol': 1e-4,
                                  'cg_fd_maxiter': 10,
                                  'cg_fd_rtol_ml': 1e-4,
                                  'cg_fd_maxiter_ml': 10,
                                  'neumann_eta': None,
                                  'neumann_hessian_q': None,
                                  'neumann_eta_ml': None,
                                  'neumann_hessian_q_ml': None,
                                  'normalize': False, 
                                  'advlearn_noise': param_dict[prob_name]['advlearn_noise'],
                                  'advlearn_std_dev': param_dict[prob_name]['advlearn_std_dev'],
                                  'iprint': param_dict[prob_name]['iprint'],  
                                  'plot_color': '#1f77b4',
                                  'plot_legend': 'TSG-N-FD',
                                  'linestyle': 'dashdot',
                                  'prob': prob}                       
                                    
            ## TSG-AD 
            algo_name = 'TSG-AD'
            exp_param_dict[prob_name][algo_name] = {'algo': 'stoctio', 
                                  'algo_full_name': algo_name, 
                                  'ul_lr': 0.01, #0.009
                                  'ml_lr': 0.01, #0.01
                                  'll_lr': 0.01, #0.01
                                  'use_stopping_iter': param_dict[prob_name]['use_stopping_iter'], 
                                  'max_iter': param_dict[prob_name]['max_iter'], 
                                  'stopping_time': param_dict[prob_name]['stopping_time'], 
                                  'true_func': param_dict[prob_name]['true_func'], 
                                  'true_fbar': param_dict[prob_name]['true_fbar'],
                                  'plot_f2_fbar': param_dict[prob_name]['plot_f2_fbar'], 
                                  'plot_f3': param_dict[prob_name]['plot_f3'],
                                  'plot_grad_f': param_dict[prob_name]['plot_grad_f'], 
                                  'plot_grad_fbar': param_dict[prob_name]['plot_grad_fbar'],
                                  'plot_grad_f3': param_dict[prob_name]['plot_grad_f3'],
                                  'ul_stepsize_scheme': 1, 
                                  'ml_stepsize_scheme': 1, 
                                  'll_stepsize_scheme': 1,
                                  'ml_stepsize_scheme_true_funct_armijo': 2,
                                  'll_stepsize_scheme_true_funct_armijo': 2,
                                  'ml_iters_true_funct_armijo': 1,
                                  'll_iters_true_funct_armijo': 1,
                                  'mlp_iters_max': 30,
                                  'llp_iters_max': 60,                 
                                  'mlp_iters_init': 1, 
                                  'llp_iters_init': 1,
                                  'inc_acc': True, 
                                  'inc_acc_threshold_f1': 1e-2,
                                  'inc_acc_threshold_f2': 1e-1,
                                  'hess': 'autodiff', 
                                  'cg_fd_rtol': None,
                                  'cg_fd_maxiter': None,
                                  'cg_fd_rtol_ml': None,
                                  'cg_fd_maxiter_ml': None,
                                  'neumann_eta': 0.05,
                                  'neumann_hessian_q': 2,
                                  'neumann_eta_ml': 0.05,
                                  'neumann_hessian_q_ml': 2,
                                  'normalize': False, 
                                  'advlearn_noise': param_dict[prob_name]['advlearn_noise'],
                                  'advlearn_std_dev': param_dict[prob_name]['advlearn_std_dev'],
                                  'iprint': param_dict[prob_name]['iprint'],
                                  'plot_color': '#CD5C5C',
                                  'plot_legend': 'TSG-AD',
                                  'linestyle': 'solid',
                                  'prob': prob}    
            
            # ## BSG-AD (without UL)
            # algo_name = 'BSG-AD (without UL)'
            # exp_param_dict[prob_name][algo_name] = {'algo': 'remove_ul', 
            #                       'algo_full_name': algo_name, 
            #                       'ul_lr': 0, #0.01
            #                       'ml_lr': 0.01, #0.01
            #                       'll_lr': 0.01, #0.01
            #                       'use_stopping_iter': param_dict[prob_name]['use_stopping_iter'], 
            #                       'max_iter': param_dict[prob_name]['max_iter'], 
            #                       'stopping_time': param_dict[prob_name]['stopping_time'], 
            #                       'true_func': param_dict[prob_name]['true_func'], 
            #                       'true_fbar': param_dict[prob_name]['true_fbar'],
            #                       'plot_f2_fbar': param_dict[prob_name]['plot_f2_fbar'], 
            #                       'plot_f3': param_dict[prob_name]['plot_f3'],
            #                       'plot_grad_f': param_dict[prob_name]['plot_grad_f'], 
            #                       'plot_grad_fbar': param_dict[prob_name]['plot_grad_fbar'],
            #                       'plot_grad_f3': param_dict[prob_name]['plot_grad_f3'],
            #                       'ul_stepsize_scheme': 1, 
            #                       'ml_stepsize_scheme': 1, 
            #                       'll_stepsize_scheme': 1,
            #                       'ml_stepsize_scheme_true_funct_armijo': 2,
            #                       'll_stepsize_scheme_true_funct_armijo': 2,
            #                       'ml_iters_true_funct_armijo': 1,
            #                       'll_iters_true_funct_armijo': 1,
            #                       'mlp_iters_max': 30,
            #                       'llp_iters_max': 60,                 
            #                       'mlp_iters_init': 1, 
            #                       'llp_iters_init': 1,
            #                       'inc_acc': True, 
            #                       'inc_acc_threshold_f1': 1e-2,
            #                       'inc_acc_threshold_f2': 1e-1,
            #                       'hess': 'autodiff', 
            #                       'cg_fd_rtol': None,
            #                       'cg_fd_maxiter': None,
            #                       'cg_fd_rtol_ml': None,
            #                       'cg_fd_maxiter_ml': None,
            #                       'neumann_eta': 0.05,
            #                       'neumann_hessian_q': 2,
            #                       'neumann_eta_ml': 0.05,
            #                       'neumann_hessian_q_ml': 2,
            #                       'normalize': False, 
            #                       'iprint': param_dict[prob_name]['iprint'],
            #                       'plot_color': '#2ca02c',
            #                       'plot_legend': 'BSG-AD (without UL)',
                                    # 'linestyle': 'dashdot',
            #                       'prob': prob}
            
            # ## BSG-AD (without LL)
            # algo_name = 'BSG-AD (without ML)'
            # exp_param_dict[prob_name][algo_name] = {'algo': 'remove_ml', 
            #                       'algo_full_name': algo_name, 
            #                       'ul_lr': 0.1, #0.01
            #                       'ml_lr': 0.01, #0.01
            #                       'll_lr': 0, #0.01
            #                       'use_stopping_iter': param_dict[prob_name]['use_stopping_iter'], 
            #                       'max_iter': param_dict[prob_name]['max_iter'], 
            #                       'stopping_time': param_dict[prob_name]['stopping_time'], 
            #                       'true_func': param_dict[prob_name]['true_func'], 
            #                       'true_fbar': param_dict[prob_name]['true_fbar'],
            #                       'plot_f2_fbar': param_dict[prob_name]['plot_f2_fbar'], 
            #                       'plot_f3': param_dict[prob_name]['plot_f3'],
            #                       'plot_grad_f': param_dict[prob_name]['plot_grad_f'], 
            #                       'plot_grad_fbar': param_dict[prob_name]['plot_grad_fbar'],
            #                       'plot_grad_f3': param_dict[prob_name]['plot_grad_f3'],
            #                       'ul_stepsize_scheme': 1, 
            #                       'ml_stepsize_scheme': 1, 
            #                       'll_stepsize_scheme': 1,
            #                       'ml_stepsize_scheme_true_funct_armijo': 2,
            #                       'll_stepsize_scheme_true_funct_armijo': 2,
            #                       'ml_iters_true_funct_armijo': 1,
            #                       'll_iters_true_funct_armijo': 1,
            #                       'mlp_iters_max': 30,
            #                       'llp_iters_max': 60,                 
            #                       'mlp_iters_init': 1, 
            #                       'llp_iters_init': 1,
            #                       'inc_acc': True, 
            #                       'inc_acc_threshold_f1': 1e-2,
            #                       'inc_acc_threshold_f2': 1e-1,
            #                       'hess': 'autodiff', 
            #                       'cg_fd_rtol': None,
            #                       'cg_fd_maxiter': None,
            #                       'cg_fd_rtol_ml': None,
            #                       'cg_fd_maxiter_ml': None,
            #                       'neumann_eta': 0.05,
            #                       'neumann_hessian_q': 2,
            #                       'neumann_eta_ml': 0.05,
            #                       'neumann_hessian_q_ml': 2,
            #                       'normalize': False, 
            #                       'iprint': param_dict[prob_name]['iprint'],
            #                       'plot_color': '#ff7f0e',
            #                       'plot_legend': 'BSG-AD (without ML)',
                                    # 'linestyle': 'dashed',
            #                       'prob': prob}

# prob_name_list = ['AdversarialLearning2']

 
# algo_name_list = ['TSG-N-FD']
# algo_name_list = ['TSG-AD'] 
# algo_name_list = ['TSG-N-FD', 'TSG-AD']    
#### algo_name_list = ['BSG-AD (without UL)'] 
#### algo_name_list = ['BSG-AD (without ML)']                
#### algo_name_list = ['TSG-N-FD', 'TSG-AD', 'BSG-AD (without ML)']  

###############################################################################








##################################
## Problem AdversarialLearning3 ##
##################################

for use_stopping_iter_val, max_iter_val in [(True, 200), (False, 20000)]:
        
        if use_stopping_iter_val:
            name_aux = "iter"
        else:
            name_aux = "time"
            
        ## Problem AdversarialLearning3
        prob_name_root = "AdversarialLearning3" 
        prob_name = prob_name_root + name_aux
        prob_dict[prob_name] = {'name_prob_to_run_val': prob_name_root}
        prob = func.TrilevelProblem(name_prob_to_run=prob_dict[prob_name]['name_prob_to_run_val'])
        
        #----------------------------------------------------------------------#
        #-------------- Parameters common to all the algorithms  --------------#
        #----------------------------------------------------------------------#
        
        param_dict[prob_name] = {}
        
        # A flag to use the total number of iterations as a stopping criterion
        param_dict[prob_name]['use_stopping_iter'] = use_stopping_iter_val
        # Maximum number of iterations
        param_dict[prob_name]['max_iter'] = max_iter_val 
        # Maximum running time (in sec) used when use_stopping_iter is False
        param_dict[prob_name]['stopping_time'] = 20   
        # Use true function
        param_dict[prob_name]['true_func'] = False  
        param_dict[prob_name]['true_fbar'] = False 
        # Create plots for fbar and f3
        param_dict[prob_name]['plot_f2_fbar'] = False
        param_dict[prob_name]['plot_f3'] = False
        # Create plots for grad_f, grad_fbar, and grad_f3
        param_dict[prob_name]['plot_grad_f'] = False
        param_dict[prob_name]['plot_grad_fbar'] = False
        param_dict[prob_name]['plot_grad_f3'] = False
        # Compute test MSE with or without Gaussian noise
        param_dict[prob_name]['advlearn_noise'] = True 
        param_dict[prob_name]['advlearn_std_dev'] = 5 #0 or 5
        # Number of runs for each algorithm
        param_dict[prob_name]['num_rep_value'] = 10
        # Sets the verbosity level for printing information (higher values correspond to more detailed output)
        param_dict[prob_name]['iprint'] = 8
        
        ####################################################### OLD
        # List of colors for the algorithms in the plots
        # plot_color_list = ['#1f77b4','#2ca02c','#bcbd22','#CD5C5C','#ff7f0e']
        # plot_legend_list = ['TSG-N-FD (inc. acc.)','TSG-1 (inc. acc.)','TSG-H (inc. acc.)','DARTS','StocBiO (inc. acc.)']
        ####################################################### OLD
        
        #--------------------------------------------------------------------#
        #-------------- Parameters specific for each algorithm --------------#
        #--------------------------------------------------------------------#
        
        exp_param_dict[prob_name] = {}
            
        ## TSG-N-FD 
        algo_name = 'TSG-N-FD'
        exp_param_dict[prob_name][algo_name] = {'algo': 'tsg', 
                              'algo_full_name': algo_name, 
                              'ul_lr': 0.0001, 
                              'ml_lr': 0.0001, 
                              'll_lr': 0.0001, 
                              'use_stopping_iter': param_dict[prob_name]['use_stopping_iter'], 
                              'max_iter': param_dict[prob_name]['max_iter'], 
                              'stopping_time': param_dict[prob_name]['stopping_time'], 
                              'true_func': param_dict[prob_name]['true_func'], 
                              'true_fbar': param_dict[prob_name]['true_fbar'],
                              'plot_f2_fbar': param_dict[prob_name]['plot_f2_fbar'], 
                              'plot_f3': param_dict[prob_name]['plot_f3'],
                              'plot_grad_f': param_dict[prob_name]['plot_grad_f'], 
                              'plot_grad_fbar': param_dict[prob_name]['plot_grad_fbar'],
                              'plot_grad_f3': param_dict[prob_name]['plot_grad_f3'],
                              'ul_stepsize_scheme': 1, 
                              'ml_stepsize_scheme': 1, 
                              'll_stepsize_scheme': 1,
                              'ml_stepsize_scheme_true_funct_armijo': 2,
                              'll_stepsize_scheme_true_funct_armijo': 2,
                              'ml_iters_true_funct_armijo': 1,
                              'll_iters_true_funct_armijo': 1,
                              'mlp_iters_max': 30,
                              'llp_iters_max': 60,                 
                              'mlp_iters_init': 1, 
                              'llp_iters_init': 1,
                              'inc_acc': True, 
                              'inc_acc_threshold_f1': 1e-2,
                              'inc_acc_threshold_f2': 1e-1,
                              'hess': 'CG-FD', 
                              'cg_fd_rtol': 1e-4,
                              'cg_fd_maxiter': 2,
                              'cg_fd_rtol_ml': 1e-4,
                              'cg_fd_maxiter_ml': 2,
                              'neumann_eta': None,
                              'neumann_hessian_q': None,
                              'neumann_eta_ml': None,
                              'neumann_hessian_q_ml': None,
                              'normalize': False, 
                              'advlearn_noise': param_dict[prob_name]['advlearn_noise'],
                              'advlearn_std_dev': param_dict[prob_name]['advlearn_std_dev'],
                              'iprint': param_dict[prob_name]['iprint'],  
                              'plot_color': '#1f77b4',
                              'plot_legend': 'TSG-N-FD',
                              'linestyle': 'dashdot',
                              'prob': prob}                       
                                
        ## TSG-AD 
        algo_name = 'TSG-AD'
        exp_param_dict[prob_name][algo_name] = {'algo': 'stoctio', 
                              'algo_full_name': algo_name, 
                              'ul_lr': 0.1, #0.1
                              'ml_lr': 0.01, #0.01
                              'll_lr': 0.1, #0.1
                              'use_stopping_iter': param_dict[prob_name]['use_stopping_iter'], 
                              'max_iter': param_dict[prob_name]['max_iter'], 
                              'stopping_time': param_dict[prob_name]['stopping_time'], 
                              'true_func': param_dict[prob_name]['true_func'], 
                              'true_fbar': param_dict[prob_name]['true_fbar'],
                              'plot_f2_fbar': param_dict[prob_name]['plot_f2_fbar'], 
                              'plot_f3': param_dict[prob_name]['plot_f3'],
                              'plot_grad_f': param_dict[prob_name]['plot_grad_f'], 
                              'plot_grad_fbar': param_dict[prob_name]['plot_grad_fbar'],
                              'plot_grad_f3': param_dict[prob_name]['plot_grad_f3'],
                              'ul_stepsize_scheme': 1, 
                              'ml_stepsize_scheme': 1, 
                              'll_stepsize_scheme': 1,
                              'ml_stepsize_scheme_true_funct_armijo': 2,
                              'll_stepsize_scheme_true_funct_armijo': 2,
                              'ml_iters_true_funct_armijo': 1,
                              'll_iters_true_funct_armijo': 1,
                              'mlp_iters_max': 30,
                              'llp_iters_max': 60,                 
                              'mlp_iters_init': 1, 
                              'llp_iters_init': 1,
                              'inc_acc': True, 
                              'inc_acc_threshold_f1': 1e-2,
                              'inc_acc_threshold_f2': 1e-1,
                              'hess': 'autodiff', 
                              'cg_fd_rtol': None,
                              'cg_fd_maxiter': None,
                              'cg_fd_rtol_ml': None,
                              'cg_fd_maxiter_ml': None,
                              'neumann_eta': 0.05,
                              'neumann_hessian_q': 2,
                              'neumann_eta_ml': 0.05,
                              'neumann_hessian_q_ml': 2,
                              'normalize': False, 
                              'advlearn_noise': param_dict[prob_name]['advlearn_noise'],
                              'advlearn_std_dev': param_dict[prob_name]['advlearn_std_dev'],
                              'iprint': param_dict[prob_name]['iprint'],
                              'plot_color': '#CD5C5C',
                              'plot_legend': 'TSG-AD',
                              'linestyle': 'solid',
                              'prob': prob}    
        
        ## BSG-AD (without UL)
        algo_name = 'BSG-AD (without UL)'
        exp_param_dict[prob_name][algo_name] = {'algo': 'remove_ul', 
                              'algo_full_name': algo_name, 
                              'ul_lr': 0, #0.01
                              'ml_lr': 0.01, #0.01
                              'll_lr': 0.1, #0.1
                              'use_stopping_iter': param_dict[prob_name]['use_stopping_iter'], 
                              'max_iter': param_dict[prob_name]['max_iter'], 
                              'stopping_time': param_dict[prob_name]['stopping_time'], 
                              'true_func': param_dict[prob_name]['true_func'], 
                              'true_fbar': param_dict[prob_name]['true_fbar'],
                              'plot_f2_fbar': param_dict[prob_name]['plot_f2_fbar'], 
                              'plot_f3': param_dict[prob_name]['plot_f3'],
                              'plot_grad_f': param_dict[prob_name]['plot_grad_f'], 
                              'plot_grad_fbar': param_dict[prob_name]['plot_grad_fbar'],
                              'plot_grad_f3': param_dict[prob_name]['plot_grad_f3'],
                              'ul_stepsize_scheme': 1, 
                              'ml_stepsize_scheme': 1, 
                              'll_stepsize_scheme': 1,
                              'ml_stepsize_scheme_true_funct_armijo': 2,
                              'll_stepsize_scheme_true_funct_armijo': 2,
                              'ml_iters_true_funct_armijo': 1,
                              'll_iters_true_funct_armijo': 1,
                              'mlp_iters_max': 30,
                              'llp_iters_max': 60,                 
                              'mlp_iters_init': 1, 
                              'llp_iters_init': 1,
                              'inc_acc': True, 
                              'inc_acc_threshold_f1': 1e-2,
                              'inc_acc_threshold_f2': 1e-1,
                              'hess': 'autodiff', 
                              'cg_fd_rtol': None,
                              'cg_fd_maxiter': None,
                              'cg_fd_rtol_ml': None,
                              'cg_fd_maxiter_ml': None,
                              'neumann_eta': 0.05,
                              'neumann_hessian_q': 2,
                              'neumann_eta_ml': 0.05,
                              'neumann_hessian_q_ml': 2,
                              'normalize': False, 
                              'advlearn_noise': param_dict[prob_name]['advlearn_noise'],
                              'advlearn_std_dev': param_dict[prob_name]['advlearn_std_dev'],
                              'iprint': param_dict[prob_name]['iprint'],
                              'plot_color': '#2ca02c',
                              'plot_legend': 'BSG-AD (without UL)',
                              'linestyle': 'dashdot',
                              'prob': prob}
        
        ## BSG-AD (without LL)
        algo_name = 'BSG-AD (without LL)'
        exp_param_dict[prob_name][algo_name] = {'algo': 'remove_ll', 
                              'algo_full_name': algo_name, 
                              'ul_lr': 0.1, #0.01
                              'ml_lr': 0.01, #0.01
                              'll_lr': 0, #0.01
                              'use_stopping_iter': param_dict[prob_name]['use_stopping_iter'], 
                              'max_iter': param_dict[prob_name]['max_iter'], 
                              'stopping_time': param_dict[prob_name]['stopping_time'], 
                              'true_func': param_dict[prob_name]['true_func'], 
                              'true_fbar': param_dict[prob_name]['true_fbar'],
                              'plot_f2_fbar': param_dict[prob_name]['plot_f2_fbar'], 
                              'plot_f3': param_dict[prob_name]['plot_f3'],
                              'plot_grad_f': param_dict[prob_name]['plot_grad_f'], 
                              'plot_grad_fbar': param_dict[prob_name]['plot_grad_fbar'],
                              'plot_grad_f3': param_dict[prob_name]['plot_grad_f3'],
                              'ul_stepsize_scheme': 1, 
                              'ml_stepsize_scheme': 1, 
                              'll_stepsize_scheme': 1,
                              'ml_stepsize_scheme_true_funct_armijo': 2,
                              'll_stepsize_scheme_true_funct_armijo': 2,
                              'ml_iters_true_funct_armijo': 1,
                              'll_iters_true_funct_armijo': 1,
                              'mlp_iters_max': 30,
                              'llp_iters_max': 60,                 
                              'mlp_iters_init': 1, 
                              'llp_iters_init': 1,
                              'inc_acc': True, 
                              'inc_acc_threshold_f1': 1e-2,
                              'inc_acc_threshold_f2': 1e-1,
                              'hess': 'autodiff', 
                              'cg_fd_rtol': None,
                              'cg_fd_maxiter': None,
                              'cg_fd_rtol_ml': None,
                              'cg_fd_maxiter_ml': None,
                              'neumann_eta': 0.05,
                              'neumann_hessian_q': 2,
                              'neumann_eta_ml': 0.05,
                              'neumann_hessian_q_ml': 2,
                              'normalize': False, 
                              'advlearn_noise': param_dict[prob_name]['advlearn_noise'],
                              'advlearn_std_dev': param_dict[prob_name]['advlearn_std_dev'],
                              'iprint': param_dict[prob_name]['iprint'],
                              'plot_color': '#ff7f0e',
                              'plot_legend': 'BSG-AD (without LL)',
                              'linestyle': 'dashed',
                              'prob': prob}

# prob_name_list = ['AdversarialLearning3']

 
# algo_name_list = ['TSG-N-FD']
# algo_name_list = ['TSG-AD'] 
# algo_name_list = ['BSG-AD (without UL)'] 
# algo_name_list = ['BSG-AD (without LL)'] 
# algo_name_list = ['TSG-N-FD', 'TSG-AD
# algo_name_list = ['TSG-AD', 'BSG-AD (without UL)', 'BSG-AD (without LL)']                   

###############################################################################





##################################
## Problem AdversarialLearning5 ##
##################################

for use_stopping_iter_val, max_iter_val in [(True, 200), (False, 20000)]:
        
        if use_stopping_iter_val:
            name_aux = "iter"
        else:
            name_aux = "time"
            
        ## Problem AdversarialLearning5
        prob_name_root = "AdversarialLearning5" 
        prob_name = prob_name_root + name_aux
        prob_dict[prob_name] = {'name_prob_to_run_val': prob_name_root}
        prob = func.TrilevelProblem(name_prob_to_run=prob_dict[prob_name]['name_prob_to_run_val'])
        
        #----------------------------------------------------------------------#
        #-------------- Parameters common to all the algorithms  --------------#
        #----------------------------------------------------------------------#
        
        param_dict[prob_name] = {}
        
        # A flag to use the total number of iterations as a stopping criterion
        param_dict[prob_name]['use_stopping_iter'] = use_stopping_iter_val
        # Maximum number of iterations
        param_dict[prob_name]['max_iter'] = max_iter_val  
        # Maximum running time (in sec) used when use_stopping_iter is False
        param_dict[prob_name]['stopping_time'] = 20   
        # Use true function
        param_dict[prob_name]['true_func'] = False  
        param_dict[prob_name]['true_fbar'] = False 
        # Create plots for fbar and f3
        param_dict[prob_name]['plot_f2_fbar'] = False
        param_dict[prob_name]['plot_f3'] = False
        # Create plots for grad_f, grad_fbar, and grad_f3
        param_dict[prob_name]['plot_grad_f'] = False
        param_dict[prob_name]['plot_grad_fbar'] = False
        param_dict[prob_name]['plot_grad_f3'] = False
        # Compute test MSE with or without Gaussian noise
        param_dict[prob_name]['advlearn_noise'] = True 
        param_dict[prob_name]['advlearn_std_dev'] = 5 #0 or 5
        # Number of runs for each algorithm
        param_dict[prob_name]['num_rep_value'] = 10
        # Sets the verbosity level for printing information (higher values correspond to more detailed output)
        param_dict[prob_name]['iprint'] = 8
        
        ####################################################### OLD
        # List of colors for the algorithms in the plots
        # plot_color_list = ['#1f77b4','#2ca02c','#bcbd22','#CD5C5C','#ff7f0e']
        # plot_legend_list = ['TSG-N-FD (inc. acc.)','TSG-1 (inc. acc.)','TSG-H (inc. acc.)','DARTS','StocBiO (inc. acc.)']
        ####################################################### OLD
        
        #--------------------------------------------------------------------#
        #-------------- Parameters specific for each algorithm --------------#
        #--------------------------------------------------------------------#
        
        exp_param_dict[prob_name] = {}
            
        ## TSG-N-FD 
        algo_name = 'TSG-N-FD'
        exp_param_dict[prob_name][algo_name] = {'algo': 'tsg', 
                              'algo_full_name': algo_name, 
                              'ul_lr': 0.0000000001, 
                              'ml_lr': 0.000000001, 
                              'll_lr': 0.0000001, 
                              'use_stopping_iter': param_dict[prob_name]['use_stopping_iter'], 
                              'max_iter': param_dict[prob_name]['max_iter'], 
                              'stopping_time': param_dict[prob_name]['stopping_time'], 
                              'true_func': param_dict[prob_name]['true_func'], 
                              'true_fbar': param_dict[prob_name]['true_fbar'],
                              'plot_f2_fbar': param_dict[prob_name]['plot_f2_fbar'], 
                              'plot_f3': param_dict[prob_name]['plot_f3'],
                              'plot_grad_f': param_dict[prob_name]['plot_grad_f'], 
                              'plot_grad_fbar': param_dict[prob_name]['plot_grad_fbar'],
                              'plot_grad_f3': param_dict[prob_name]['plot_grad_f3'],
                              'ul_stepsize_scheme': 1, 
                              'ml_stepsize_scheme': 1, 
                              'll_stepsize_scheme': 1,
                              'ml_stepsize_scheme_true_funct_armijo': 2,
                              'll_stepsize_scheme_true_funct_armijo': 2,
                              'ml_iters_true_funct_armijo': 1,
                              'll_iters_true_funct_armijo': 1,
                              'mlp_iters_max': 30,
                              'llp_iters_max': 60,                 
                              'mlp_iters_init': 1, 
                              'llp_iters_init': 1,
                              'inc_acc': True, 
                              'inc_acc_threshold_f1': 1e-2,
                              'inc_acc_threshold_f2': 1e-1,
                              'hess': 'CG-FD', 
                              'cg_fd_rtol': 1e-4,
                              'cg_fd_maxiter': 2,
                              'cg_fd_rtol_ml': 1e-4,
                              'cg_fd_maxiter_ml': 2,
                              'neumann_eta': None,
                              'neumann_hessian_q': None,
                              'neumann_eta_ml': None,
                              'neumann_hessian_q_ml': None,
                              'normalize': False, 
                              'advlearn_noise': param_dict[prob_name]['advlearn_noise'],
                              'advlearn_std_dev': param_dict[prob_name]['advlearn_std_dev'],
                              'iprint': param_dict[prob_name]['iprint'],  
                              'plot_color': '#1f77b4',
                              'plot_legend': 'TSG-N-FD',
                              'linestyle': 'dashdot',
                              'prob': prob}                       
                                
        ## TSG-AD 
        algo_name = 'TSG-AD'
        exp_param_dict[prob_name][algo_name] = {'algo': 'stoctio', 
                              'algo_full_name': algo_name, 
                              'ul_lr': 0.01, #0.1
                              'ml_lr': 0.001, #0.01
                              'll_lr': 0.01, #0.1
                              'use_stopping_iter': param_dict[prob_name]['use_stopping_iter'], 
                              'max_iter': param_dict[prob_name]['max_iter'], 
                              'stopping_time': param_dict[prob_name]['stopping_time'], 
                              'true_func': param_dict[prob_name]['true_func'], 
                              'true_fbar': param_dict[prob_name]['true_fbar'],
                              'plot_f2_fbar': param_dict[prob_name]['plot_f2_fbar'], 
                              'plot_f3': param_dict[prob_name]['plot_f3'],
                              'plot_grad_f': param_dict[prob_name]['plot_grad_f'], 
                              'plot_grad_fbar': param_dict[prob_name]['plot_grad_fbar'],
                              'plot_grad_f3': param_dict[prob_name]['plot_grad_f3'],
                              'ul_stepsize_scheme': 1, 
                              'ml_stepsize_scheme': 1, 
                              'll_stepsize_scheme': 1,
                              'ml_stepsize_scheme_true_funct_armijo': 2,
                              'll_stepsize_scheme_true_funct_armijo': 2,
                              'ml_iters_true_funct_armijo': 1,
                              'll_iters_true_funct_armijo': 1,
                              'mlp_iters_max': 30,
                              'llp_iters_max': 60,                 
                              'mlp_iters_init': 1, 
                              'llp_iters_init': 1,
                              'inc_acc': True, 
                              'inc_acc_threshold_f1': 1e-2,
                              'inc_acc_threshold_f2': 1e-1,
                              'hess': 'autodiff', 
                              'cg_fd_rtol': None,
                              'cg_fd_maxiter': None,
                              'cg_fd_rtol_ml': None,
                              'cg_fd_maxiter_ml': None,
                              'neumann_eta': 0.05,
                              'neumann_hessian_q': 2,
                              'neumann_eta_ml': 0.05,
                              'neumann_hessian_q_ml': 2,
                              'normalize': False, 
                              'advlearn_noise': param_dict[prob_name]['advlearn_noise'],
                              'advlearn_std_dev': param_dict[prob_name]['advlearn_std_dev'],
                              'iprint': param_dict[prob_name]['iprint'],
                              'plot_color': '#CD5C5C',
                              'plot_legend': 'TSG-AD',
                              'linestyle': 'solid',
                              'prob': prob}    
        
        ## BSG-AD (without UL)
        algo_name = 'BSG-AD (without UL)'
        exp_param_dict[prob_name][algo_name] = {'algo': 'remove_ul', 
                              'algo_full_name': algo_name, 
                              'ul_lr': 0, #0.01
                              'ml_lr': 0.001, #0.01
                              'll_lr': 0.1, #0.1
                              'use_stopping_iter': param_dict[prob_name]['use_stopping_iter'], 
                              'max_iter': param_dict[prob_name]['max_iter'], 
                              'stopping_time': param_dict[prob_name]['stopping_time'], 
                              'true_func': param_dict[prob_name]['true_func'], 
                              'true_fbar': param_dict[prob_name]['true_fbar'],
                              'plot_f2_fbar': param_dict[prob_name]['plot_f2_fbar'], 
                              'plot_f3': param_dict[prob_name]['plot_f3'],
                              'plot_grad_f': param_dict[prob_name]['plot_grad_f'], 
                              'plot_grad_fbar': param_dict[prob_name]['plot_grad_fbar'],
                              'plot_grad_f3': param_dict[prob_name]['plot_grad_f3'],
                              'ul_stepsize_scheme': 1, 
                              'ml_stepsize_scheme': 1, 
                              'll_stepsize_scheme': 1,
                              'ml_stepsize_scheme_true_funct_armijo': 2,
                              'll_stepsize_scheme_true_funct_armijo': 2,
                              'ml_iters_true_funct_armijo': 1,
                              'll_iters_true_funct_armijo': 1,
                              'mlp_iters_max': 30,
                              'llp_iters_max': 60,                 
                              'mlp_iters_init': 1, 
                              'llp_iters_init': 1,
                              'inc_acc': True, 
                              'inc_acc_threshold_f1': 1e-2,
                              'inc_acc_threshold_f2': 1e-1,
                              'hess': 'autodiff', 
                              'cg_fd_rtol': None,
                              'cg_fd_maxiter': None,
                              'cg_fd_rtol_ml': None,
                              'cg_fd_maxiter_ml': None,
                              'neumann_eta': 0.05,
                              'neumann_hessian_q': 2,
                              'neumann_eta_ml': 0.05,
                              'neumann_hessian_q_ml': 2,
                              'normalize': False, 
                              'advlearn_noise': param_dict[prob_name]['advlearn_noise'],
                              'advlearn_std_dev': param_dict[prob_name]['advlearn_std_dev'],
                              'iprint': param_dict[prob_name]['iprint'],
                              'plot_color': '#2ca02c',
                              'plot_legend': 'BSG-AD (without UL)',
                              'linestyle': 'dashdot',
                              'prob': prob}
        
        ## BSG-AD (without LL)
        algo_name = 'BSG-AD (without LL)'
        exp_param_dict[prob_name][algo_name] = {'algo': 'remove_ll', 
                              'algo_full_name': algo_name, 
                              'ul_lr': 0.1, #0.01
                              'ml_lr': 0.001, #0.01
                              'll_lr': 0, #0.01
                              'use_stopping_iter': param_dict[prob_name]['use_stopping_iter'], 
                              'max_iter': param_dict[prob_name]['max_iter'], 
                              'stopping_time': param_dict[prob_name]['stopping_time'], 
                              'true_func': param_dict[prob_name]['true_func'], 
                              'true_fbar': param_dict[prob_name]['true_fbar'],
                              'plot_f2_fbar': param_dict[prob_name]['plot_f2_fbar'], 
                              'plot_f3': param_dict[prob_name]['plot_f3'],
                              'plot_grad_f': param_dict[prob_name]['plot_grad_f'], 
                              'plot_grad_fbar': param_dict[prob_name]['plot_grad_fbar'],
                              'plot_grad_f3': param_dict[prob_name]['plot_grad_f3'],
                              'ul_stepsize_scheme': 1, 
                              'ml_stepsize_scheme': 1, 
                              'll_stepsize_scheme': 1,
                              'ml_stepsize_scheme_true_funct_armijo': 2,
                              'll_stepsize_scheme_true_funct_armijo': 2,
                              'ml_iters_true_funct_armijo': 1,
                              'll_iters_true_funct_armijo': 1,
                              'mlp_iters_max': 30,
                              'llp_iters_max': 60,                 
                              'mlp_iters_init': 1, 
                              'llp_iters_init': 1,
                              'inc_acc': True, 
                              'inc_acc_threshold_f1': 1e-2,
                              'inc_acc_threshold_f2': 1e-1,
                              'hess': 'autodiff', 
                              'cg_fd_rtol': None,
                              'cg_fd_maxiter': None,
                              'cg_fd_rtol_ml': None,
                              'cg_fd_maxiter_ml': None,
                              'neumann_eta': 0.05,
                              'neumann_hessian_q': 2,
                              'neumann_eta_ml': 0.05,
                              'neumann_hessian_q_ml': 2,
                              'normalize': False, 
                              'advlearn_noise': param_dict[prob_name]['advlearn_noise'],
                              'advlearn_std_dev': param_dict[prob_name]['advlearn_std_dev'],
                              'iprint': param_dict[prob_name]['iprint'],
                              'plot_color': '#ff7f0e',
                              'plot_legend': 'BSG-AD (without LL)',
                              'linestyle': 'dashed',
                              'prob': prob}

# prob_name_list = ['AdversarialLearning5']

 
# algo_name_list = ['TSG-N-FD']
# algo_name_list = ['TSG-AD'] 
# algo_name_list = ['BSG-AD (without UL)'] 
# algo_name_list = ['BSG-AD (without LL)'] 
# algo_name_list = ['TSG-N-FD', 'TSG-AD
# algo_name_list = ['TSG-AD', 'BSG-AD (without UL)', 'BSG-AD (without LL)']                   

###############################################################################




# prob_name_list = ["Quadratic2iter", "Quadratic2time", "Quadratic3iter", "Quadratic3time"]
# algo_name_list = ['TSG-H','TSG-N-FD','TSG-AD']
# prob_name_list = ["Quadratic4iter", "Quadratic4time"]
# algo_name_list = ['TSG-N-FD','TSG-AD']
# prob_name_list = ["Quartic2iter", "Quartic2time", "Quartic3iter", "Quartic3time"]
# algo_name_list = ['TSG-H','TSG-N-FD','TSG-AD']
# prob_name_list = ["Quartic4iter", "Quartic4time"]
# algo_name_list = ['TSG-N-FD','TSG-AD']
# prob_name_list = ["AdversarialLearning2iter0", "AdversarialLearning2time0"]
# algo_name_list = ['TSG-N-FD','TSG-AD']
# prob_name_list = ["AdversarialLearning2iter5", "AdversarialLearning2time5"]
# algo_name_list = ['TSG-N-FD','TSG-AD']
# prob_name_list = ["AdversarialLearning1iter0", "AdversarialLearning1time0"]
# algo_name_list = ['TSG-AD', 'BSG-AD (without UL)', 'BSG-AD (without LL)']
# prob_name_list = ["AdversarialLearning1iter5", "AdversarialLearning1time5"]
# algo_name_list = ['TSG-AD', 'BSG-AD (without UL)', 'BSG-AD (without LL)']
# prob_name_list = ["AdversarialLearning3iter", "AdversarialLearning3time"]
# algo_name_list = ['TSG-AD', 'BSG-AD (without UL)', 'BSG-AD (without LL)']
# prob_name_list = ["AdversarialLearning5iter", "AdversarialLearning5time"]
# algo_name_list = ['TSG-AD', 'BSG-AD (without UL)', 'BSG-AD (without LL)']





problem_algo_dict = {
    # "Quadratic2iter": ['TSG-H', 'TSG-N-FD', 'TSG-AD'],
    # "Quadratic2time": ['TSG-H', 'TSG-N-FD', 'TSG-AD'],
    # "Quadratic3iter": ['TSG-H', 'TSG-N-FD', 'TSG-AD'],
    # "Quadratic3time": ['TSG-H', 'TSG-N-FD', 'TSG-AD'],
    # "Quadratic4iter": ['TSG-N-FD', 'TSG-AD'],
    "Quadratic4time": ['TSG-N-FD', 'TSG-AD'],
    # "Quartic2iter": ['TSG-H', 'TSG-N-FD', 'TSG-AD'],
    # "Quartic2time": ['TSG-H', 'TSG-N-FD', 'TSG-AD'],
    # "Quartic3iter": ['TSG-H', 'TSG-N-FD', 'TSG-AD'],
    # "Quartic3time": ['TSG-H', 'TSG-N-FD', 'TSG-AD'],
    # "Quartic4iter": ['TSG-H', 'TSG-AD'],
    # "Quartic4time": ['TSG-H', 'TSG-AD'],
    # "AdversarialLearning2iter0": ['TSG-N-FD', 'TSG-AD'],
    # "AdversarialLearning2time0": ['TSG-N-FD', 'TSG-AD'],
    # "AdversarialLearning2iter5": ['TSG-N-FD', 'TSG-AD'],
    # "AdversarialLearning2time5": ['TSG-N-FD', 'TSG-AD'],
    # "AdversarialLearning1iter0": ['TSG-AD', 'BSG-AD (without UL)', 'BSG-AD (without LL)'],
    # "AdversarialLearning1time0": ['TSG-AD', 'BSG-AD (without UL)', 'BSG-AD (without LL)'],
    # "AdversarialLearning1iter5": ['TSG-AD', 'BSG-AD (without UL)', 'BSG-AD (without LL)'],
    # "AdversarialLearning1time5": ['TSG-AD', 'BSG-AD (without UL)', 'BSG-AD (without LL)'],
    # "AdversarialLearning3iter": ['TSG-AD', 'BSG-AD (without UL)', 'BSG-AD (without LL)'],
    # "AdversarialLearning3time": ['TSG-AD', 'BSG-AD (without UL)', 'BSG-AD (without LL)'],
    # "AdversarialLearning5iter": ['TSG-AD', 'BSG-AD (without UL)', 'BSG-AD (without LL)'],
    # "AdversarialLearning5time": ['TSG-AD', 'BSG-AD (without UL)', 'BSG-AD (without LL)']
}



# problem_algo_dict = {
    # "Quartic2time": ['TSG-H', 'TSG-N-FD', 'TSG-AD']
    # "Quadratic2iter": ['TSG-H', 'TSG-N-FD', 'TSG-AD'],
# }





prob_name_list = list(problem_algo_dict.keys())





#----------------------------------------------------------------------#
#-------------- Run the experiments and create the plots --------------#
#----------------------------------------------------------------------#

# Create a dictionary collecting the output for each experiment
exp_out_dict = {}

for prob_name in prob_name_list:
    
    
    
    algo_name_list = problem_algo_dict[prob_name]
    


    #-------------------------------------------------#
    #-------------- Run the experiments --------------#
    #-------------------------------------------------#
    
    exp_out_dict[prob_name] = {}
    
    for algo_name in algo_name_list:
        run, values_avg, values_ci, true_func_values_avg, true_func_values_ci, times, aux_lists, accuracy_values_avg, accuracy_values_ci = run_experiment(exp_param_dict[prob_name][algo_name], num_rep_value=param_dict[prob_name]['num_rep_value'])
        exp_out_dict[prob_name][algo_name] = {'run': run, 'values_avg': values_avg, 'values_ci': values_ci, 'true_func_values_avg': true_func_values_avg, 'true_func_values_ci': true_func_values_ci, 'times': times, 'aux_lists': aux_lists, 'accuracy_values_avg': accuracy_values_avg, 'accuracy_values_ci': accuracy_values_ci}

        


    
    #--------------------------------------------------------------#
    #-------------- Create separate plot for f1 or f --------------#
    #--------------------------------------------------------------#        
        
    plt.figure()    
        
    for algo_name in algo_name_list:
        if exp_out_dict[prob_name][algo_name]['run'].use_stopping_iter:
            if exp_out_dict[prob_name][algo_name]['run'].true_func:
                val_x_axis = [i for i in range(len(exp_out_dict[prob_name][algo_name]['true_func_values_avg']))]
            else:
                val_x_axis = [i for i in range(len(exp_out_dict[prob_name][algo_name]['values_avg']))]
        else:
            val_x_axis = exp_out_dict[prob_name][algo_name]['times']
            val_x_axis = [item*10**3 for item in val_x_axis]
        if exp_out_dict[prob_name][algo_name]['run'].true_func:
            val_y_axis_avg = exp_out_dict[prob_name][algo_name]['true_func_values_avg'] 
            val_y_axis_ci = exp_out_dict[prob_name][algo_name]['true_func_values_ci'] 
        else:
            val_y_axis_avg = exp_out_dict[prob_name][algo_name]['values_avg'] 
            val_y_axis_ci = exp_out_dict[prob_name][algo_name]['values_ci']        
        string_legend = r'{0}'.format(exp_param_dict[prob_name][algo_name]['plot_legend'])
        # string_legend = r'{0} $\alpha^u = {1}$, $\alpha^\ell = {2}$'.format(exp_param_dict[prob_name][algo_name]['plot_legend'],exp_param_dict[prob_name][algo_name]['ul_lr'],exp_param_dict[prob_name][algo_name]['ll_lr'])
        sns.lineplot(x=val_x_axis, y=val_y_axis_avg, linewidth = 1, label = string_legend, color = exp_param_dict[prob_name][algo_name]['plot_color'], linestyle=exp_param_dict[prob_name][algo_name]['linestyle'])
        plt.fill_between(val_x_axis, (val_y_axis_avg-val_y_axis_ci), (val_y_axis_avg+val_y_axis_ci), alpha=.4, linewidth = 0.5, color = exp_param_dict[prob_name][algo_name]['plot_color'])   
    
    # plt.gca().set_ylim([-21000,50000]) 
    # plt.gca().set_ylim([-24,-10])
    
    # The optimal value of the trilevel problem
    if isinstance(exp_out_dict[prob_name][algo_name]['run'].func.prob, func.Quadratic) or 'Quartic1' in prob_name:
        plt.hlines(exp_out_dict[prob_name][algo_name_list[0]]['run'].func.f_opt(), 0, val_x_axis[len(val_x_axis)-1], color='black', linestyle='dotted', label=r'$f(x_*)$') 
    
    if exp_out_dict[prob_name][algo_name_list[0]]['run'].use_stopping_iter:
        plt.xlabel("UL Iterations", fontsize = 16)
    else:
        if exp_out_dict[prob_name][algo_name_list[0]]['run'].func.prob.is_machine_learning_problem:
            plt.xlabel("Time (s)", fontsize = 16)
        else:
            plt.xlabel("Time (ms)", fontsize = 16)
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ylabel(r"$f(x^i)$", fontsize = 16)
    plt.tick_params(axis='both', labelsize=11)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    # plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
    
    plt.legend(frameon=False, fontsize=16) # No borders in the legend

    if not isinstance(exp_out_dict[prob_name][algo_name]['run'].func.prob, func.AdversarialLearning):   
        string = ' UL, ML, and LL grad std devs: ' \
            + str(exp_out_dict[prob_name][algo_name_list[0]]['run'].func.prob.std_dev) \
            + ', ' + str(exp_out_dict[prob_name][algo_name_list[0]]['run'].func.prob.ml_std_dev) \
            + ', ' + str(exp_out_dict[prob_name][algo_name_list[0]]['run'].func.prob.ll_std_dev) \
            + '\n ML and LL Hess std devs: ' \
            + str(exp_out_dict[prob_name][algo_name_list[0]]['run'].func.prob.ml_hess_std_dev) \
            + ', ' + str(exp_out_dict[prob_name][algo_name_list[0]]['run'].func.prob.ll_hess_std_dev)
    
        plt.title(string, fontsize=16)
    
    
    fig = plt.gcf()
    fig.set_size_inches(7, 4.5)  
    fig.tight_layout(pad=4.5)
    
    ## Uncomment the next line to save the plot
    string = "Figures/" + prob_name + '_f.pdf'
    fig.savefig(string, dpi = 1000, format='pdf', bbox_inches='tight')
    plt.show()





    ## We create the plots below only when using iterations
    if exp_out_dict[prob_name][algo_name_list[0]]['run'].use_stopping_iter:
        
        #--------------------------------------------------------------#
        #-------------- Create plots for function values --------------#
        #--------------------------------------------------------------#
    
        plt.close('all')  # Reset all previous plots
    
        number_func_subplots = 1
                
        if exp_param_dict[prob_name][algo_name_list[0]]['plot_f2_fbar']:
                number_func_subplots += 1  
                
        if exp_param_dict[prob_name][algo_name_list[0]]['plot_f3']:
                number_func_subplots += 1                      
        
        for algo_name in algo_name_list:

            if exp_param_dict[prob_name][algo_name_list[0]]['plot_f2_fbar'] or exp_param_dict[prob_name][algo_name_list[0]]['plot_f3']: 
                
                ## Create a figure with subplots
                fig, axs = plt.subplots(number_func_subplots, 1, figsize=(8, 7.5))  
                fig.tight_layout(pad=5.0)
        
                f2_val_all_list, fbar_val_all_list, fbar_val_opt_all_list, fbar_val_opt_counter_all_list, f3_val_all_list, f3_val_opt_all_list, f3_val_opt_counter_all_list, grad_f_list, grad_fbar_all_list, grad_f3_all_list = exp_out_dict[prob_name][algo_name]['aux_lists']  
        
                subplot_counter = 0
        
                #--------------------------------------------------------------#
                #-------------- Create separate plot for f1 or f --------------#
                #--------------------------------------------------------------#        
        
                if exp_param_dict[prob_name][algo_name_list[0]]['plot_f2_fbar'] or exp_param_dict[prob_name][algo_name_list[0]]['plot_f3']:            
                        if exp_out_dict[prob_name][algo_name]['run'].true_func:
                            val_x_axis = [i for i in range(len(exp_out_dict[prob_name][algo_name]['true_func_values_avg']))]
                        else:
                            val_x_axis = [i for i in range(len(exp_out_dict[prob_name][algo_name]['values_avg']))]
                        if exp_out_dict[prob_name][algo_name]['run'].true_func:
                            val_y_axis_avg = exp_out_dict[prob_name][algo_name]['true_func_values_avg'] 
                            val_y_axis_ci = exp_out_dict[prob_name][algo_name]['true_func_values_ci'] 
                        else:
                            val_y_axis_avg = exp_out_dict[prob_name][algo_name]['values_avg'] 
                            val_y_axis_ci = exp_out_dict[prob_name][algo_name]['values_ci']   
                            
                        axs_aux = axs[subplot_counter] if number_func_subplots > 1 else axs
                        
                        string_legend = r'{0}'.format(exp_param_dict[prob_name][algo_name]['plot_legend'])
                        # string_legend = r'{0} $\alpha^u = {1}$, $\alpha^\ell = {2}$'.format(exp_param_dict[prob_name][algo_name]['plot_legend'],exp_param_dict[prob_name][algo_name]['ul_lr'],exp_param_dict[prob_name][algo_name]['ll_lr'])
                        sns.lineplot(x=val_x_axis, y=val_y_axis_avg, linewidth = 1, label = string_legend, color = exp_param_dict[prob_name][algo_name]['plot_color'], ax=axs_aux, linestyle=exp_param_dict[prob_name][algo_name]['linestyle'])
                        axs_aux.fill_between(val_x_axis, (val_y_axis_avg-val_y_axis_ci), (val_y_axis_avg+val_y_axis_ci), alpha=.4, linewidth = 0.5, color = exp_param_dict[prob_name][algo_name]['plot_color'])   
                        
                        # The optimal value of the trilevel problem
                        if isinstance(exp_out_dict[prob_name][algo_name]['run'].func.prob, func.Quadratic) or 'Quartic1' in prob_name:
                            axs[subplot_counter].hlines(exp_out_dict[prob_name][algo_name_list[0]]['run'].func.f_opt(), 0, val_x_axis[len(val_x_axis)-1], color='black', linestyle='dotted', label=r'$f(x_*)$') 
                
                        # Add axis labels and title
                        axs_aux.set_xlabel("UL Iterations" if exp_out_dict[prob_name][algo_name]['run'].use_stopping_iter else "Time (ms)", fontsize=16)
                        axs_aux.set_ylabel(r"$f(x^i)$", fontsize=16)
                        axs_aux.tick_params(axis='both', labelsize=11)
                        axs_aux.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
                        axs_aux.legend(frameon=False, fontsize=16)   
                
                        subplot_counter += 1
                
                #-------------------------------------------------------------#
                #-------------- Create the plots for f2 or fbar --------------#
                #-------------------------------------------------------------#
            
                if exp_param_dict[prob_name][algo_name_list[0]]['plot_f2_fbar'] and exp_out_dict[prob_name][algo_name_list[0]]['run'].use_stopping_iter:
                                        
                        if exp_out_dict[prob_name][algo_name]['run'].true_fbar:
                            val_x_axis = [i for i in range(len(fbar_val_all_list))]
                            val_y_axis = fbar_val_all_list
                        else:
                            val_x_axis = [i for i in range(len(f2_val_all_list))]  
                            val_y_axis = f2_val_all_list
                            
                        axs_aux = axs[subplot_counter] if number_func_subplots > 1 else axs
                        
                        string_legend = r'{0}'.format(exp_param_dict[prob_name][algo_name]['plot_legend'])
                        # string_legend = r'{0} $\alpha^u = {1}$, $\alpha^\ell = {2}$'.format(exp_param_dict[prob_name][algo_name]['plot_legend'],exp_param_dict[prob_name][algo_name]['ul_lr'],exp_param_dict[prob_name][algo_name]['ll_lr'])
                        sns.lineplot(x=val_x_axis, y=val_y_axis, linewidth = 1, label = string_legend, color = exp_param_dict[prob_name][algo_name]['plot_color'], ax=axs_aux, linestyle=exp_param_dict[prob_name][algo_name]['linestyle'])
                        
                        # The optimal value of fbar
                        if isinstance(exp_out_dict[prob_name][algo_name]['run'].func.prob, func.Quadratic) or 'Quartic1' in prob_name or 'Quartic2' in prob_name or 'Quartic3' in prob_name or 'Quartic4' in prob_name:
                            counter_mlp_iters = 0
                            # Loop through the segments and plot each piecewise constant
                            for i in range(len(fbar_val_opt_all_list)):
                                if i == len(fbar_val_opt_all_list)-1:
                                    axs_aux.hlines(fbar_val_opt_all_list[i], counter_mlp_iters+1, counter_mlp_iters+fbar_val_opt_counter_all_list[i]+1, color='black', linestyle='dotted', label=r'$\bar{f}(x^i,y(x^i))$')
                                elif i == 0:
                                    axs_aux.hlines(fbar_val_opt_all_list[i], 0, fbar_val_opt_counter_all_list[i], color='black', linestyle='dotted')   
                                    counter_mlp_iters += fbar_val_opt_counter_all_list[i]
                                else:
                                    axs_aux.hlines(fbar_val_opt_all_list[i], counter_mlp_iters+1, counter_mlp_iters+fbar_val_opt_counter_all_list[i]+1, color='black', linestyle='dotted')  
                                    counter_mlp_iters += fbar_val_opt_counter_all_list[i]+1
                                # plt.xlabel("Cumulative ML Iterations", fontsize = 13)
        
                        axs_aux.set_ylabel(r"$\bar{f}(x^i,y^{i,j})$", fontsize=16)
                        axs_aux.set_xlabel("Cumulative ML Iterations", fontsize=16) 
                        axs_aux.tick_params(axis='both', labelsize=11)
                        axs_aux.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
                        axs_aux.legend(frameon=False, fontsize=16)  # No borders in the legend ## loc='lower right' loc='center right'
                        
                        subplot_counter += 1
        
                #-----------------------------------------------------#
                #-------------- Create the plots for f3 --------------#
                #-----------------------------------------------------#
                    
                    
                if exp_param_dict[prob_name][algo_name_list[0]]['plot_f3'] and exp_out_dict[prob_name][algo_name_list[0]]['run'].use_stopping_iter:
                            
                        # plt.figure()  
                                        
                        val_x_axis = [i for i in range(len(f3_val_all_list))]
                        val_y_axis = f3_val_all_list
                        
                        axs_aux = axs[subplot_counter] if number_func_subplots > 1 else axs
                        
                        string_legend = r'{0}'.format(exp_param_dict[prob_name][algo_name]['plot_legend'])
                        # string_legend = r'{0} $\alpha^u = {1}$, $\alpha^\ell = {2}$'.format(exp_param_dict[prob_name][algo_name]['plot_legend'],exp_param_dict[prob_name][algo_name]['ul_lr'],exp_param_dict[prob_name][algo_name]['ll_lr'])
                        sns.lineplot(x=val_x_axis, y=val_y_axis, linewidth = 1, label = string_legend, color = exp_param_dict[prob_name][algo_name]['plot_color'], ax=axs_aux, linestyle=exp_param_dict[prob_name][algo_name]['linestyle'])
                        
                        # The optimal value of f3
                        if isinstance(exp_out_dict[prob_name][algo_name]['run'].func.prob, func.Quadratic) or 'Quartic1' in prob_name or 'Quartic2' in prob_name or 'Quartic3' in prob_name or 'Quartic4' in prob_name or 'AdversarialLearning2' in prob_name:
                            counter_llp_iters = 0
                            # Loop through the segments and plot each piecewise constant
                            for i in range(len(f3_val_opt_all_list)):
                                if i == len(f3_val_opt_all_list)-1:
                                    axs_aux.hlines(f3_val_opt_all_list[i], counter_llp_iters+1, counter_llp_iters+f3_val_opt_counter_all_list[i]+1, color='black', linestyle='dotted', label=r'$f_3(x^i,y^{i,j},z(x^i,y^{i,j}))$')
                                elif i == 0:
                                    axs_aux.hlines(f3_val_opt_all_list[i], 0, f3_val_opt_counter_all_list[i], color='black', linestyle='dotted')   
                                    counter_llp_iters += f3_val_opt_counter_all_list[i]
                                else:
                                    axs_aux.hlines(f3_val_opt_all_list[i], counter_llp_iters+1, counter_llp_iters+f3_val_opt_counter_all_list[i]+1, color='black', linestyle='dotted')  
                                    counter_llp_iters += f3_val_opt_counter_all_list[i]+1
                                # plt.xlabel("Cumulative LL Iterations", fontsize = 13)
        
                        axs_aux.set_ylabel(r"$f_3(x^i,y^{i,j},z^{i,j,k})$", fontsize=16)
                        axs_aux.set_xlabel("Cumulative LL Iterations", fontsize=16) 
                        axs_aux.tick_params(axis='both', labelsize=11)
                        axs_aux.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
                        axs_aux.legend(frameon=False, fontsize=16)  # No borders in the legend  ## loc='center right' 
        
                #----------------------------------------------------------#
                #----------------------------------------------------------#
                #----------------------------------------------------------#
            
                if not isinstance(exp_out_dict[prob_name][algo_name]['run'].func.prob, func.AdversarialLearning): 
                    # Add a common title and show the plot
                    string = ('UL, ML, and LL grad std devs: {0}, {1}, {2}\n'
                              'ML and LL Hess std devs: {3}, {4}').format(
                        exp_out_dict[prob_name][algo_name_list[0]]['run'].func.prob.std_dev,
                        exp_out_dict[prob_name][algo_name_list[0]]['run'].func.prob.ml_std_dev,
                        exp_out_dict[prob_name][algo_name_list[0]]['run'].func.prob.ll_std_dev,
                        exp_out_dict[prob_name][algo_name_list[0]]['run'].func.prob.ml_hess_std_dev,
                        exp_out_dict[prob_name][algo_name_list[0]]['run'].func.prob.ll_hess_std_dev
                    )
                
                    fig.suptitle(string, fontsize=16, y=1.02)

                ## Uncomment the next line to save the plot
                string = "Figures/" + prob_name + "_" + algo_name + '_func_breakdown.pdf'
                fig.savefig(string, dpi = 1000, format='pdf', bbox_inches='tight')                
                plt.show()            
            
                #----------------------------------------------------------#
                #----------------------------------------------------------#
                #----------------------------------------------------------#            
    
    
    
    
        #------------------------------------------------------------#
        #-------------- Create the plots for gradients --------------#
        #------------------------------------------------------------#
    
        plt.close('all')  # Reset all previous plots
        
        if exp_param_dict[prob_name][algo_name_list[0]]['plot_grad_f'] or exp_param_dict[prob_name][algo_name_list[0]]['plot_grad_fbar'] or exp_param_dict[prob_name][algo_name_list[0]]['plot_grad_f3']:
    
            number_grad_subplots = 0
        
            if exp_param_dict[prob_name][algo_name_list[0]]['plot_grad_f']:
                    number_grad_subplots += 1
                    
            if exp_param_dict[prob_name][algo_name_list[0]]['plot_grad_fbar']:
                    number_grad_subplots += 1  
                    
            if exp_param_dict[prob_name][algo_name_list[0]]['plot_grad_f3']:
                    number_grad_subplots += 1                      
            
            for algo_name in algo_name_list:
                
                ## Create a figure with subplots
                fig, axs = plt.subplots(number_grad_subplots, 1, figsize=(8, 7.5))  
                fig.tight_layout(pad=5.0)
        
                f2_val_all_list, fbar_val_all_list, fbar_val_opt_all_list, fbar_val_opt_counter_all_list, f3_val_all_list, f3_val_opt_all_list, f3_val_opt_counter_all_list, grad_f_list, grad_fbar_all_list, grad_f3_all_list = exp_out_dict[prob_name][algo_name]['aux_lists']  
        
                subplot_counter = 0
                
                #---------------------------------------------------------#
                #-------------- Create the plots for grad_f --------------#
                #---------------------------------------------------------#
        
                if exp_param_dict[prob_name][algo_name_list[0]]['plot_grad_f'] and exp_out_dict[prob_name][algo_name_list[0]]['run'].use_stopping_iter:
                                                        
                        val_x_axis = [i for i in range(len(grad_f_list))]
                        val_y_axis = grad_f_list
                        
                        axs_aux = axs[subplot_counter] if number_grad_subplots > 1 else axs
            
                        string_legend = r'{0}'.format(exp_param_dict[prob_name][algo_name]['plot_legend'])
                        # string_legend = r'{0} $\alpha^u = {1}$, $\alpha^\ell = {2}$'.format(exp_param_dict[prob_name][algo_name]['plot_legend'],exp_param_dict[prob_name][algo_name]['ul_lr'],exp_param_dict[prob_name][algo_name]['ll_lr'])
                        sns.lineplot(x=val_x_axis, y=val_y_axis, linewidth = 1, label = string_legend, color = exp_param_dict[prob_name][algo_name]['plot_color'], ax=axs_aux, linestyle=exp_param_dict[prob_name][algo_name]['linestyle'])
            
                        axs_aux.set_ylabel(r"$|| \nabla f(x^i) ||_2$", fontsize=16)
                        axs_aux.set_xlabel("UL Iterations", fontsize=16) 
                        axs_aux.set_ylim(-0.2, max(val_y_axis))
                        axs_aux.tick_params(axis='both', labelsize=11)
                        axs_aux.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
                        axs_aux.legend(frameon=False, fontsize=16)  # No borders in the legend
                        
                        subplot_counter += 1
                
                #------------------------------------------------------------#
                #-------------- Create the plots for grad_fbar --------------#
                #------------------------------------------------------------#
        
                if exp_param_dict[prob_name][algo_name_list[0]]['plot_grad_fbar'] and exp_out_dict[prob_name][algo_name_list[0]]['run'].use_stopping_iter:
                                                        
                        val_x_axis = [i for i in range(len(grad_fbar_all_list))]
                        val_y_axis = grad_fbar_all_list
                        
                        axs_aux = axs[subplot_counter] if number_grad_subplots > 1 else axs
            
                        string_legend = r'{0}'.format(exp_param_dict[prob_name][algo_name]['plot_legend'])
                        # string_legend = r'{0} $\alpha^u = {1}$, $\alpha^\ell = {2}$'.format(exp_param_dict[prob_name][algo_name]['plot_legend'],exp_param_dict[prob_name][algo_name]['ul_lr'],exp_param_dict[prob_name][algo_name]['ll_lr'])
                        sns.lineplot(x=val_x_axis, y=val_y_axis, linewidth = 1, label = string_legend, color = exp_param_dict[prob_name][algo_name]['plot_color'], ax=axs_aux, linestyle=exp_param_dict[prob_name][algo_name]['linestyle'])
             
                        axs_aux.set_ylabel(r"$|| \nabla \bar{f}(x^i, y^{i,j}) ||_2$", fontsize=16)
                        axs_aux.set_xlabel("Cumulative ML Iterations", fontsize=16)  
                        axs_aux.set_ylim(-0.2, max(val_y_axis))
                        axs_aux.tick_params(axis='both', labelsize=11)
                        axs_aux.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
                        axs_aux.legend(frameon=False, fontsize=16)  # No borders in the legend
                
                        subplot_counter += 1
                
                #----------------------------------------------------------#
                #-------------- Create the plots for grad_f3 --------------#
                #----------------------------------------------------------#
                    
                if exp_param_dict[prob_name][algo_name_list[0]]['plot_grad_f3'] and exp_out_dict[prob_name][algo_name_list[0]]['run'].use_stopping_iter:
                                                        
                        val_x_axis = [i for i in range(len(grad_f3_all_list))]
                        val_y_axis = grad_f3_all_list
                        
                        axs_aux = axs[subplot_counter] if number_grad_subplots > 1 else axs
                        
                        string_legend = r'{0}'.format(exp_param_dict[prob_name][algo_name]['plot_legend'])
                        # string_legend = r'{0} $\alpha^u = {1}$, $\alpha^\ell = {2}$'.format(exp_param_dict[prob_name][algo_name]['plot_legend'],exp_param_dict[prob_name][algo_name]['ul_lr'],exp_param_dict[prob_name][algo_name]['ll_lr'])
                        sns.lineplot(x=val_x_axis, y=val_y_axis, linewidth = 1, label = string_legend, color = exp_param_dict[prob_name][algo_name]['plot_color'], ax=axs_aux, linestyle=exp_param_dict[prob_name][algo_name]['linestyle'])
            
                        axs_aux.set_ylabel(r"$|| \nabla f_3(x^i, y^{i,j}, z^{i,j,k}) ||_2$", fontsize=16)
                        axs_aux.set_xlabel("Cumulative LL Iterations", fontsize=16) 
                        axs_aux.set_ylim(-1, max(val_y_axis)) 
                        axs_aux.tick_params(axis='both', labelsize=11)
                        axs_aux.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
                        axs_aux.legend(frameon=False, fontsize=16)  # No borders in the legend
                    
                #----------------------------------------------------------#
                #----------------------------------------------------------#
                #----------------------------------------------------------#
        
                if not isinstance(exp_out_dict[prob_name][algo_name]['run'].func.prob, func.AdversarialLearning):    
                    # Add a common title and show the plot
                    string = ('UL, ML, and LL grad std devs: {0}, {1}, {2}\n'
                              'ML and LL Hess std devs: {3}, {4}').format(
                        exp_out_dict[prob_name][algo_name_list[0]]['run'].func.prob.std_dev,
                        exp_out_dict[prob_name][algo_name_list[0]]['run'].func.prob.ml_std_dev,
                        exp_out_dict[prob_name][algo_name_list[0]]['run'].func.prob.ll_std_dev,
                        exp_out_dict[prob_name][algo_name_list[0]]['run'].func.prob.ml_hess_std_dev,
                        exp_out_dict[prob_name][algo_name_list[0]]['run'].func.prob.ll_hess_std_dev
                    )
            
                    fig.suptitle(string, fontsize=16, y=1.02)

                ## Uncomment the next line to save the plot
                string = "Figures/" + prob_name + "_" + algo_name + '_grad_breakdown.pdf'
                fig.savefig(string, dpi = 1000, format='pdf', bbox_inches='tight')   
                plt.show()            
            
                #----------------------------------------------------------#
                #----------------------------------------------------------#
                #----------------------------------------------------------#            






    ## We create the plots below only for machine learning problems
    if exp_out_dict[prob_name][algo_name_list[0]]['run'].func.prob.is_machine_learning_problem:
        
        #-----------------------------------------------------------------------------------------#
        #-------------- Create the plots for accuracy for machine learning problems --------------#
        #-----------------------------------------------------------------------------------------#
    
        # for prob_name in prob_name_list:
        plt.figure()  
            
        for algo_name in algo_name_list:
            if exp_out_dict[prob_name][algo_name]['run'].use_stopping_iter:
                val_x_axis = [i for i in range(len(exp_out_dict[prob_name][algo_name]['values_avg']))]
            else:
                val_x_axis = exp_out_dict[prob_name][algo_name]['times']
            val_y_axis_avg = exp_out_dict[prob_name][algo_name]['accuracy_values_avg'] 
            val_y_axis_ci = exp_out_dict[prob_name][algo_name]['accuracy_values_ci']        
            string_legend = r'{0}'.format(exp_param_dict[prob_name][algo_name]['plot_legend'])
            # string_legend = r'{0} $\alpha^u = {1}$, $\alpha^\ell = {2}$'.format(exp_param_dict[prob_name][algo_name]['plot_legend'],exp_param_dict[prob_name][algo_name]['ul_lr'],exp_param_dict[prob_name][algo_name]['ll_lr'])
            sns.lineplot(x=val_x_axis, y=val_y_axis_avg, linewidth = 1, label = string_legend, color = exp_param_dict[prob_name][algo_name]['plot_color'], linestyle=exp_param_dict[prob_name][algo_name]['linestyle'])
            # if num_rep_value > 1:
            plt.fill_between(val_x_axis, (val_y_axis_avg-val_y_axis_ci), (val_y_axis_avg+val_y_axis_ci), alpha=.4, linewidth = 0.5, color = exp_param_dict[prob_name][algo_name]['plot_color'])   

        if exp_param_dict[prob_name][algo_name]['advlearn_noise']:
            if 'AdversarialLearning1' in prob_name:
                if param_dict[prob_name]['advlearn_std_dev'] == 0:        
                    plt.gca().set_ylim([3*1e1,4.4*1e1]) 
                if param_dict[prob_name]['advlearn_std_dev'] == 5:        
                    plt.gca().set_ylim([0*1e2,5*1e2]) 
                    
            if 'AdversarialLearning2' in prob_name:
                if param_dict[prob_name]['advlearn_std_dev'] == 0:        
                    plt.gca().set_ylim([3*1e1,4.4*1e1]) 
                if param_dict[prob_name]['advlearn_std_dev'] == 5:        
                    plt.gca().set_ylim([0*1e2,5*1e2]) 
                    
        if exp_out_dict[prob_name][algo_name_list[0]]['run'].use_stopping_iter:
            plt.xlabel("UL Iterations", fontsize = 16)
        else:
            plt.xlabel("Time (s)", fontsize = 16)
            plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        plt.ylabel(r"Test MSE", fontsize = 16)
        plt.tick_params(axis='both', labelsize=11)
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        # plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
        
        plt.legend(frameon=False, fontsize=16) # No borders in the legend
        
        
        fig = plt.gcf()
        fig.set_size_inches(7, 4.5)  
        fig.tight_layout(pad=4.5)
        
        ## Uncomment the next line to save the plot
        string = "Figures/" + prob_name + '_testMSE.pdf'
        fig.savefig(string, dpi = 1000, format='pdf', bbox_inches='tight')   
        plt.show()









            