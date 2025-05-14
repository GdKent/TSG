import os
import numpy as np
import random
import time

import torch
import torch.nn as nn
import copy

from scipy.optimize import minimize_scalar 
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.sparse.linalg import cg, LinearOperator, gmres, minres
from scipy.stats import t






class TrilevelSolver:
    """
    Class used to implement trilevel stochastic algorithms for synthetic quadratic trilevel problems

    Attributes
        ul_lr (real):                         Current upper-level stepsize (default 1)
        ml_lr (real):                         Current middle-level stepsize (default 1)
        ll_lr (real):                         Current lowel-level stepsize (default 1)
        mlp_iters (int):                      Current number of middle-level iterations (default 1)
        llp_iters (int):                      Current number of lower-level iterations (default 1)
        func (obj):                           Object used to define an optimization problem to solve 
        algo (str):                           Name of the algorithm to run (default 'tsg')
        seed (int, optional):                 The seed used for the experiments (default 0)
        ul_lr_init (real, optional):          Initial upper-level stepsize (default 5)
        ml_lr_init (real, optional):          Initial middle-level stepsize (default 0.1)
        ll_lr_init (real, optional):          Initial lower-level stepsize (default 0.1)
        ul_stepsize_scheme (int, optional):   A flag to choose the upper-level stepsize scheme (default 0): 0 --> decaying; 1 --> fixed; 2 --> Trilevel Armijo backtracking line search; 3 --> Optimal stepsize 
        ml_stepsize_scheme (int, optional):   A flag to choose the middle-level stepsize scheme (default 0): 0 --> decaying; 1 --> fixed; 2 --> Bilevel Armijo backtracking line search; 3 --> Optimal stepsize 
        ll_stepsize_scheme (int, optional):   A flag to choose the lower-level stepsize scheme (default 0): 0 --> decaying; 1 --> fixed; 2 --> Armijo backtracking line search; 3 --> Optimal stepsize
        ml_stepsize_scheme_true_funct_armijo (int, optional):   A flag to choose the middle-level stepsize scheme when computing the true function for Armijo (default 0): 0 --> decaying; 1 --> fixed; 2 --> Bilevel Armijo backtracking line search; 3 --> Optimal stepsize 
        ll_stepsize_scheme_true_funct_armijo (int, optional):   A flag to choose the lower-level stepsize scheme when computing the true function for Armijo (default 0): 0 --> decaying; 1 --> fixed; 2 --> Armijo backtracking line search; 3 --> Optimal stepsize
        ml_iters_true_funct_armijo (int):     Number of middle-level iterations when using Armijo
        ll_iters_true_funct_armijo:           Number of lower-level iterations when using Armijo
        normalize (bool, optional):           A flag to normalize the direction used for the upper-level update (default False)
        hess (bool, optional):                A flag to use either the true Hessians (hess = True), rank-2 approximations (hess = False), or CG with FD (hess = 'CG-FD') (default False)
        max_iter:                             Maximum number of upper-level iterations (default 500)
        mlp_iters_max (real, optional):       Maximum number of middle-level iterations when using the increasing accuracy strategy (default 1)
        llp_iters_max (real, optional):       Maximum number of lower-level iterations when using the increasing accuracy strategy (default 1)        
        mlp_iters_init (real, optional):      Initial number of middle-level iterations (default 1)
        llp_iters_init (real, optional):      Initial number of lower-level iterations (default 1)
        ml_inc_acc (bool, optional):          A flag to use an increasing accuracy strategy for the middle-level problem; if False, the number of ml iterations is fixed (default False)
        ll_inc_acc (bool, optional):          A flag to use an increasing accuracy strategy for the lower-level problem; if False, the number of ll iterations is fixed (default False)
        true_func (bool, optional):           A flag to compute the true function (default True)
        use_stopping_iter (bool, optional):   A flag to use the total number of iterations as a stopping criterion (default True)
        stopping_time (real, optional):       Maximum running time (in sec) used when use_stopping_iter is False (default 0.5)
        iprint (int):                         Used to print info on the optimization (default 1): 0 --> no printing; 1 --> at the end of the optimization; 2 --> at each iteration 
    """    

    
    def __init__(self, func, 
                 algo, 
                 algo_full_name, 
                 ul_lr, 
                 ml_lr, 
                 ll_lr, 
                 max_iter, 
                 normalize,
                 hess, 
                 inc_acc,
                 inc_acc_threshold_f1,
                 inc_acc_threshold_f2,
                 use_stopping_iter, 
                 stopping_time,
                 ul_stepsize_scheme, 
                 ml_stepsize_scheme, 
                 ll_stepsize_scheme,
                 ml_stepsize_scheme_true_funct_armijo,
                 ll_stepsize_scheme_true_funct_armijo,
                 ml_iters_true_funct_armijo,
                 ll_iters_true_funct_armijo,
                 true_func,
                 true_fbar,
                 plot_f2_fbar,
                 plot_f3,
                 plot_grad_f,
                 plot_grad_fbar,
                 plot_grad_f3,
                 mlp_iters_max,
                 llp_iters_max,                 
                 mlp_iters_init, 
                 llp_iters_init,
                 cg_fd_rtol=None,
                 cg_fd_maxiter=None,
                 cg_fd_rtol_ml=None,
                 cg_fd_maxiter_ml=None,
                 neumann_eta=None,
                 neumann_hessian_q=None,
                 neumann_eta_ml=None,
                 neumann_hessian_q_ml=None,
                 advlearn_noise=None,
                 advlearn_std_dev=None,
                 seed=0,
                 iprint = 1):

        self.func = func
        self.algo = algo
        self.algo_full_name = algo_full_name
        self.seed = seed
        self.ul_lr_init = ul_lr
        self.ml_lr_init = ml_lr
        self.ll_lr_init = ll_lr
        self.ul_stepsize_scheme = ul_stepsize_scheme
        self.ml_stepsize_scheme = ml_stepsize_scheme
        self.ll_stepsize_scheme = ll_stepsize_scheme
        self.ml_stepsize_scheme_true_funct_armijo = ml_stepsize_scheme_true_funct_armijo
        self.ll_stepsize_scheme_true_funct_armijo = ll_stepsize_scheme_true_funct_armijo
        self.ml_iters_true_funct_armijo = ml_iters_true_funct_armijo
        self.ll_iters_true_funct_armijo = ll_iters_true_funct_armijo
        self.max_iter = max_iter
        self.normalize = normalize
        self.hess = hess
        self.cg_fd_rtol = cg_fd_rtol
        self.cg_fd_maxiter = cg_fd_maxiter
        self.cg_fd_rtol_ml = cg_fd_rtol_ml
        self.cg_fd_maxiter_ml = cg_fd_maxiter_ml
        self.neumann_eta = neumann_eta
        self.neumann_hessian_q = neumann_hessian_q
        self.neumann_eta_ml = neumann_eta_ml
        self.neumann_hessian_q_ml = neumann_hessian_q_ml
        self.mlp_iters_max = mlp_iters_max
        self.llp_iters_max = llp_iters_max
        self.mlp_iters_init = mlp_iters_init
        self.llp_iters_init = llp_iters_init
        self.inc_acc = inc_acc
        self.inc_acc_threshold_f1 = inc_acc_threshold_f1
        self.inc_acc_threshold_f2 = inc_acc_threshold_f2
        self.true_func = true_func  
        self.true_fbar = true_fbar
        self.plot_f2_fbar = plot_f2_fbar
        self.plot_f3 = plot_f3       
        self.plot_grad_f = plot_grad_f
        self.plot_grad_fbar = plot_grad_fbar 
        self.plot_grad_f3 = plot_grad_f3 
        self.use_stopping_iter = use_stopping_iter
        self.stopping_time = stopping_time
        self.advlearn_noise = advlearn_noise
        self.advlearn_std_dev = advlearn_std_dev
        self.iprint = iprint
        
        
    def set_seed(self, seed):
        """
        Sets the seed
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        os.environ['PYTHONHASHSEED'] = str(seed) 
    

    def generate_ll_vars(self):
        """
        Creates a vector of LL variables 
        """
        ll_vars = np.random.uniform(0, 0.1, (self.func.prob.z_dim, 1))
        return ll_vars  

    
    def generate_ml_vars(self):
        """
        Creates a vector of LL variables 
        """
        ml_vars = np.random.uniform(0, 0.1, (self.func.prob.y_dim, 1))
        return ml_vars 


    def update_ulp(self, ul_vars, ml_vars, ll_vars, grad_f_list): 
        """
        Updates the UL variables by taking a gradient descent step for the UL problem
        """
        if self.iprint >= 10:
            print('\nSTART UPD UL\n')
        
        # Obtain estimate of the gradient of f wrt UL variables
        grad_f_ul_vars = self.grad_f_ul_vars(ul_vars, ml_vars, ll_vars, hess=self.hess)

        # Update the variables
        if self.ul_stepsize_scheme == 2:
            alpha, ml_vars, ll_vars = self.backtracking_armijo_ul(ul_vars, ml_vars, ll_vars, grad_f_ul_vars)
            ul_vars = ul_vars - alpha * grad_f_ul_vars 
            self.ul_lr = alpha
            
        elif self.ul_stepsize_scheme == 0 or self.ul_stepsize_scheme == 1:
            ul_vars = ul_vars - self.ul_lr * grad_f_ul_vars  
            
        elif self.ul_stepsize_scheme == 3:
            
            def obj_funct_stepsize_ul(stepsize):
                aux_var = ul_vars - stepsize * grad_f_ul_vars
                ml_aux_var, ll_aux_var = self.update_mlp_true_funct_armijo(aux_var, ml_vars, ll_vars, ml_iters_true_funct=self.mlp_iters,  ll_iters_true_funct=self.llp_iters)
                # print('stepsize',stepsize,'ul_vars',ul_vars,aux_var,'ml_vars',ml_vars,ml_aux_var,'ll_vars',ll_vars,ll_aux_var,'obj_funct',self.func.f(aux_var, ml_aux_var, ll_aux_var))
                return self.func.f(aux_var, ml_aux_var, ll_aux_var)
            
            res = minimize_scalar(obj_funct_stepsize_ul, bounds=(0, 1), method='bounded')
            ul_lr_opt = res.x
            ul_vars = ul_vars - ul_lr_opt * grad_f_ul_vars
            self.ul_lr = ul_lr_opt  
            
            if self.iprint >= 10:
                print('\nOptimal ll lr: ',ul_lr_opt,'\n')            

        if self.plot_grad_f:
            grad_f_list.append(np.linalg.norm(self.grad_f_ul_vars(ul_vars, ml_vars, ll_vars, hess=True)))

        if self.iprint >= 8: 
            print('\nAFTER update_ulp')
            if self.func.prob.x_dim == 1:
                print('ul_vars',ul_vars,' ', end=' ')
            if self.true_func: 
                print('f: ',self.func.f(ul_vars, ml_vars, ll_vars),' grad_f_ul_vars: ',np.linalg.norm(self.grad_f_ul_vars(ul_vars, ml_vars, ll_vars, hess=True)))
            if self.true_func and self.func.prob.x_opt_available_analytically:
                ul_vars_opt, ml_vars_opt, ll_vars_opt = self.func.compute_args(self.func.prob.x_opt(), y0=ml_vars, z0=ll_vars)
                if self.func.prob.x_dim == 1:
                    print('ul_vars_opt',ul_vars_opt,' ', end=' ')
                print('f: ',self.func.f(ul_vars_opt, ml_vars_opt, ll_vars_opt),' grad_f_ul_vars: ',np.linalg.norm(self.grad_f_ul_vars(ul_vars_opt, ml_vars_opt, ll_vars_opt, hess=True)),'\n')

        if self.iprint >= 10:          
            print('\nEND UPD UL\n')
            
        return ul_vars, ml_vars, ll_vars


    def update_mlp(self, ul_vars, ml_vars, ll_vars, f2_val_all_list, f3_val_all_list, f3_val_opt_all_list, f3_val_opt_counter_all_list, fbar_val_all_list, fbar_val_opt_all_list, fbar_val_opt_counter_all_list, grad_f3_all_list, grad_fbar_all_list): 
        """
        Updates the ML variables by taking a gradient descent step for the ML problem
        """               
        if self.iprint >= 10:
            print('\nSTART UPD ML\n')

        if self.true_fbar == True: 
            fbar_val_all_list = self.compute_fbar_value(ul_vars, ml_vars, ll_vars, fbar_list=fbar_val_all_list)
        f2_val_all_list.append(self.func.prob.f_2(ul_vars, ml_vars, ll_vars))

        if self.plot_f2_fbar and self.true_fbar:
            ul_vars_aux, ml_vars_opt, ll_vars_opt = self.func.compute_args(ul_vars, y0=ml_vars, z0=ll_vars)
            fbar_val_opt_all_list.append(self.func.fbar(ul_vars_aux, ml_vars_opt, ll_vars_opt))
            fbar_val_opt_counter_all_list.append(self.mlp_iters)        

        pre_f2_val = self.func.prob.f_2(ul_vars, ml_vars, ll_vars)  
      
        for i in range(self.mlp_iters):
            ## When removing the LL problem, we do not need any call to update_llp
            if not self.algo == "remove_ll":
                # Update the LL variables 
                ll_vars = self.update_llp(ul_vars, ml_vars, ll_vars, f3_val_all_list, f3_val_opt_all_list, f3_val_opt_counter_all_list, grad_f3_all_list)   

            # Obtain estimate of the gradient of fbar wrt ML variables
            grad_fbar_ml_vars = self.grad_fbar_ml_vars(ul_vars, ml_vars, ll_vars, hess=self.hess) 
               
            # Update the variables
            if self.ml_stepsize_scheme == 2:
                alpha, ll_vars = self.backtracking_armijo_ml(ul_vars, ml_vars, ll_vars, grad_fbar_ml_vars)
                ml_vars = ml_vars - alpha * grad_fbar_ml_vars 
                self.ml_lr = alpha
                
            elif self.ml_stepsize_scheme == 0:
                self.ml_lr = self.ml_lr/(i+1)
                ml_vars = ml_vars - self.ml_lr * grad_fbar_ml_vars
                
            elif self.ml_stepsize_scheme == 1:
                ml_vars = ml_vars - self.ml_lr * grad_fbar_ml_vars

            elif self.ml_stepsize_scheme == 3:  
                
                def obj_funct_stepsize_ml(stepsize):
                    aux_var = ml_vars - stepsize * grad_fbar_ml_vars
                    ll_aux_var = self.update_llp_true_funct_armijo(ul_vars, aux_var, ll_vars, ll_iters_true_funct=self.mlp_iters)
                    # print('mlp_iter',i,'stepsize',stepsize,'ml_vars',ml_vars,aux_var,'ll_vars',ll_vars,ll_aux_var,'obj_funct',self.func.fbar(ul_vars, aux_var, ll_aux_var))
                    return self.func.fbar(ul_vars, aux_var, ll_aux_var)
                
                res = minimize_scalar(obj_funct_stepsize_ml, bounds=(0, 1), method='bounded')
                ml_lr_opt = res.x
                ml_vars = ml_vars - ml_lr_opt * grad_fbar_ml_vars
                self.ml_lr = ml_lr_opt
                
                if self.iprint >= 10:
                    print('\nOptimal ml lr: ',ml_lr_opt,'\n')

            if self.true_fbar == True: 
                fbar_val_all_list = self.compute_fbar_value(ul_vars, ml_vars, ll_vars, fbar_list=fbar_val_all_list)
            f2_val_all_list.append(self.func.prob.f_2(ul_vars, ml_vars, ll_vars))

            if self.plot_grad_fbar and self.true_fbar:
                grad_fbar_all_list.append(np.linalg.norm(self.grad_fbar_ml_vars(ul_vars, ml_vars, ll_vars, hess=True))) # Added hess=True on April 23, 2025
    
            if self.plot_grad_f3:
                grad_f3_all_list.append(np.linalg.norm(self.func.prob.grad_f3_ll_vars(ul_vars, ml_vars, ll_vars)))

        ## By removing one level, we obtain a bilevel problem, which no longer requires two calls to update_llp
        if not self.algo == "remove_ll" and not self.algo == "remove_ul" and not self.func.prob.remove_ll:
            # Update the LL variables 
            ll_vars = self.update_llp(ul_vars, ml_vars, ll_vars, f3_val_all_list, f3_val_opt_all_list, f3_val_opt_counter_all_list, grad_f3_all_list) 

        # if self.plot_grad_fbar:
        #     grad_fbar_all_list.append(np.linalg.norm(self.grad_fbar_ml_vars(ul_vars, ml_vars, ll_vars)))

        # if self.plot_grad_f3:
        #     grad_f3_all_list.append(np.linalg.norm(self.func.prob.grad_f3_ll_vars(ul_vars, ml_vars, ll_vars)))

        if self.iprint >= 8: 
            print('\nAFTER update_llp')
            if self.func.prob.z_dim == 1:
                print('ll_vars',ll_vars,' ', end=' ')
            print('f3: ',self.func.prob.f_3(ul_vars, ml_vars, ll_vars),' grad_f3_ll_vars: ',np.linalg.norm(self.func.prob.grad_f3_ll_vars(ul_vars, ml_vars, ll_vars)))
            if self.plot_f3:
                ul_vars_aux, ml_vars_aux, ll_vars_opt = self.func.compute_args(ul_vars, ml_vars, z0=ll_vars)
                if self.func.prob.z_dim == 1:
                    print('ll_vars_opt',ll_vars_opt,' ', end=' ')
                print('f3: ',self.func.prob.f_3(ul_vars_aux, ml_vars_aux, ll_vars_opt),' grad_f3_ll_vars: ',np.linalg.norm(self.func.prob.grad_f3_ll_vars(ul_vars_aux, ml_vars_aux, ll_vars_opt)),'\n')

            if self.func.prob.x_dim == 1 and self.func.prob.y_dim == 1 and self.func.prob.z_dim == 1:
                print('ul_vars',ul_vars,'ml_vars',ml_vars,'ll_vars',ll_vars)
                
            # print('AAA',ll_vars,np.dot(self.func.prob.Hzx,ul_vars)+np.dot(self.func.prob.Hzy,ml_vars))
            
            print('\nAFTER update_mlp')
            if self.func.prob.y_dim == 1:
                print('ml_vars',ml_vars,' ', end=' ')
            if self.plot_f2_fbar and self.true_fbar:    
                print('fbar: ',self.func.fbar(ul_vars, ml_vars, ll_vars),' grad_fbar_ml_vars: ',np.linalg.norm(self.grad_fbar_ml_vars(ul_vars, ml_vars, ll_vars, hess=True))) # Added hess=True on April 23, 2025
            if self.true_fbar:
                ul_vars_aux, ml_vars_opt, ll_vars_opt = self.func.compute_args(ul_vars, y0=ml_vars, z0=ll_vars)
                if self.func.prob.y_dim == 1:
                    print('ml_vars_opt',ml_vars_opt,' ', end=' ')
                print('fbar: ',self.func.fbar(ul_vars_aux, ml_vars_opt, ll_vars_opt),' grad_fbar_ml_vars: ',np.linalg.norm(self.grad_fbar_ml_vars(ul_vars_aux, ml_vars_opt, ll_vars_opt, hess=True)),'\n') # Added hess=True on April 23, 2025
            
        # Increasing accuracy strategy for the ML problem (if f2 does not improve, we increase the number of ll iterations)           
        if self.inc_acc == True:
         	if self.llp_iters >= self.llp_iters_max:  
          		self.llp_iters = self.llp_iters_max
         	# if self.llp_iters >= min(self.llp_iters_max,self.mlp_iters**3 * self.max_iter):  
          #    	  self.llp_iters = min(self.llp_iters_max,self.mlp_iters**3 * self.max_iter)             
         	# if self.llp_iters >= self.mlp_iters**3 * self.max_iter:
          #        self.llp_iters = self.mlp_iters**3 * self.max_iter
         	else:
         	   post_obj_val = self.func.prob.f_2(ul_vars, ml_vars, ll_vars) 
         	   obj_val_diff = abs(post_obj_val - pre_f2_val)
         	   if obj_val_diff/max(abs(pre_f2_val), np.finfo(float).eps) <= self.inc_acc_threshold_f2:     ##Before April 28 2025: if obj_val_diff/abs(pre_f2_val) <= self.inc_acc_threshold_f2:          
                    self.llp_iters += 1

        if self.iprint >= 10:
            # time.sleep(3)            
            print('\nEND UPD ML\n')    
            
        return ml_vars, ll_vars
    

    def update_llp(self, ul_vars, ml_vars, ll_vars, f3_val_all_list, f3_val_opt_all_list, f3_val_opt_counter_all_list, grad_f3_all_list): 
        """
        Updates the LL variables by taking a gradient descent step for the LL problem
        """ 
        if self.iprint >= 10:
            print('\nSTART UPD LL\n') 

        f3_val_all_list.append(self.func.prob.f_3(ul_vars, ml_vars, ll_vars))
            
        if self.plot_f3 and True:
            ul_vars_aux, ml_vars_aux, ll_vars_opt = self.func.compute_args(ul_vars, ml_vars, z0=ll_vars)
            f3_val_opt_all_list.append(self.func.prob.f_3(ul_vars_aux, ml_vars_aux, ll_vars_opt))
            f3_val_opt_counter_all_list.append(self.llp_iters)                            

        for i in range(self.llp_iters):
        # for i in range(100):            
            # Obtain gradient of the LLP wrt LL variables
            grad_llp_ll_vars = self.func.prob.grad_f3_ll_vars(ul_vars, ml_vars, ll_vars)
                
            # Update the variables
            if self.ll_stepsize_scheme == 2:
                alpha = self.backtracking_armijo_ll(ul_vars, ml_vars, ll_vars) 
                ll_vars = ll_vars - alpha * grad_llp_ll_vars
                self.ll_lr = alpha
                
            elif self.ll_stepsize_scheme == 0: 
                self.ll_lr = self.ll_lr/(i+1)
                ll_vars = ll_vars - self.ll_lr * grad_llp_ll_vars
                
            elif self.ll_stepsize_scheme == 1: 
                ll_vars = ll_vars - self.ll_lr * grad_llp_ll_vars
                
            elif self.ll_stepsize_scheme == 3:  
                
                def obj_funct_stepsize_ll(stepsize):
                    aux_var = ll_vars - stepsize * grad_llp_ll_vars
                    return self.func.prob.f_3(ul_vars, ml_vars, aux_var)
                
                res = minimize_scalar(obj_funct_stepsize_ll, bounds=(0, 1), method='bounded')
                ll_lr_opt = res.x
                ll_vars = ll_vars - ll_lr_opt * grad_llp_ll_vars
                self.ll_lr = ll_lr_opt
                                
                if self.iprint >= 10:
                    print('\nOptimal ll lr: ',ll_lr_opt,'\n')

            f3_val_all_list.append(self.func.prob.f_3(ul_vars, ml_vars, ll_vars))

            if self.plot_grad_f3:
                grad_f3_all_list.append(np.linalg.norm(self.func.prob.grad_f3_ll_vars(ul_vars, ml_vars, ll_vars)))

        if self.iprint >= 10:
            print('\nAFTER update_llp')
            if self.func.prob.z_dim == 1:
                print('ll_vars',ll_vars,' ', end=' ')
            print('f3: ',self.func.prob.f_3(ul_vars, ml_vars,ll_vars),' grad_f3_ll_vars: ',np.linalg.norm(self.func.prob.grad_f3_ll_vars(ul_vars, ml_vars, ll_vars)))
            if self.plot_f3:
                ul_vars_aux, ml_vars_aux, ll_vars_opt = self.func.compute_args(ul_vars, ml_vars, z0=ll_vars)
                if self.func.prob.z_dim == 1:
                    print('ll_vars_opt',ll_vars_opt,' ', end=' ')
                print('f3: ',self.func.prob.f_3(ul_vars_aux, ml_vars_aux, ll_vars_opt),' grad_f3_ll_vars: ',np.linalg.norm(self.func.prob.grad_f3_ll_vars(ul_vars_aux, ml_vars_aux, ll_vars_opt)),'\n')

        if self.iprint >= 10:        
            # time.sleep(3)
            print('\nEND UPD LL\n')             
        return ll_vars


    def update_mlp_true_funct_armijo(self, ul_vars, ml_vars, ll_vars, ml_iters_true_funct, ll_iters_true_funct): 
        """
        Updates the ML variables by taking a gradient descent step for the ML problem
        """               
        if self.iprint >= 10:
            print('\nSTART UPD ML True Funct\n') 
            
        ml_lr = self.ml_lr_init
        
        for i in range(ml_iters_true_funct):
            # Update the LL variables 
            ll_vars = self.update_llp_true_funct_armijo(ul_vars, ml_vars, ll_vars, ll_iters_true_funct)
            
            # Obtain estimate of the gradient of fbar wrt ML variables
            grad_fbar_ml_vars = self.grad_fbar_ml_vars(ul_vars, ml_vars, ll_vars, hess=self.hess) 
               
            # Update the variables
            if self.ml_stepsize_scheme_true_funct_armijo == 2:
                alpha, ll_vars = self.backtracking_armijo_ml(ul_vars, ml_vars, ll_vars, grad_fbar_ml_vars)
                ml_vars = ml_vars - alpha * grad_fbar_ml_vars 
                
            elif self.ml_stepsize_scheme_true_funct_armijo == 0:
                ml_lr = ml_lr/(i+1)
                ml_vars = ml_vars - ml_lr * grad_fbar_ml_vars
                
            elif self.ml_stepsize_scheme_true_funct_armijo == 1:
                ml_vars = ml_vars - self.ml_lr * grad_fbar_ml_vars

            elif self.ml_stepsize_scheme_true_funct_armijo == 3:  
                
                def obj_funct_stepsize_ml(stepsize):
                    aux_var = ml_vars - stepsize * grad_fbar_ml_vars
                    ll_aux_var = self.update_llp_true_funct_armijo(ul_vars, aux_var, ll_vars, ll_iters_true_funct)
                    return self.func.fbar(ul_vars, aux_var, ll_aux_var)
                
                res = minimize_scalar(obj_funct_stepsize_ml, bounds=(0, 1), method='bounded')
                ml_lr_opt = res.x
                ml_vars = ml_vars - ml_lr_opt * grad_fbar_ml_vars
                
                if self.iprint >= 10:
                    print('\nOptimal ml lr true funct: ',ml_lr_opt,'\n')

        # Update the LL variables 
        ll_vars = self.update_llp_true_funct_armijo(ul_vars, ml_vars, ll_vars, ll_iters_true_funct)
        # print('update_mlp_true_funct_armijo',' ul_vars',ul_vars,'ml_vars',ml_vars,'ll_vars',ll_vars)
            
        if self.iprint >= 10:
            # time.sleep(3)            
            print('\nEND UPD ML True Funct\n')    
            
        return ml_vars, ll_vars


    def update_llp_true_funct_armijo(self, ul_vars, ml_vars, ll_vars, ll_iters_true_funct): 
        """
        Updates the LL variables by taking a gradient descent step for the LL problem
        """ 
        if self.iprint >= 10:
            print('\nSTART UPD LL True Funct\n') 
            
        ll_lr = self.ll_lr_init
           
        for i in range(ll_iters_true_funct):          
            # Obtain gradient of the LLP wrt LL variables
            grad_llp_ll_vars = self.func.prob.grad_f3_ll_vars(ul_vars, ml_vars, ll_vars)
                
            # Update the variables
            if self.ll_stepsize_scheme_true_funct_armijo == 2:
                alpha = self.backtracking_armijo_ll(ul_vars, ml_vars, ll_vars) 
                ll_vars = ll_vars - alpha * grad_llp_ll_vars
                
            elif self.ll_stepsize_scheme_true_funct_armijo == 0: 
                ll_lr = ll_lr/(i+1)
                ll_vars = ll_vars - ll_lr * grad_llp_ll_vars
                
            elif self.ll_stepsize_scheme_true_funct_armijo == 1: 
                ll_vars = ll_vars - self.ll_lr * grad_llp_ll_vars
                
            elif self.ll_stepsize_scheme_true_funct_armijo == 3:  
                
                def obj_funct_stepsize_ll(stepsize):
                    aux_var = ll_vars - stepsize * grad_llp_ll_vars
                    # print('llp_iter',i,'stepsize',stepsize,'ml_vars',ml_vars,'ll_vars',ll_vars,aux_var,'obj_funct',self.func.prob.f_3(ul_vars, ml_vars, aux_var))
                    return self.func.prob.f_3(ul_vars, ml_vars, aux_var)
                
                res = minimize_scalar(obj_funct_stepsize_ll, bounds=(0, 1), method='bounded')
                ll_lr_opt = res.x
                ll_vars = ll_vars - ll_lr_opt * grad_llp_ll_vars
                
                if self.iprint >= 10:
                    print('\nOptimal ll lr true funct: ',ll_lr_opt,'\n')

        # print('update_llp_true_funct_armijo',' ul_vars',ul_vars,'ml_vars',ml_vars,'ll_vars',ll_vars)
                
        if self.iprint >= 10:        
            # time.sleep(3)
            print('\nEND UPD LL True Funct\n')
             
        return ll_vars
    

    def backtracking_armijo_ul(self, x, y, z, d, eta = 10**-4, tau = 0.5):
        if self.iprint >= 10:
            print('\nSTART Armijo UL')
        
        initial_rate = 1
        
        alpha_ul = initial_rate
        iterLS = 0 

        y_tilde, z_tilde = self.update_mlp_true_funct_armijo(x - alpha_ul * d, y, z, self.ml_iters_true_funct_armijo, self.ll_iters_true_funct_armijo) 
        # print('0y_tilde',y_tilde,'z_tilde',z_tilde)
        sub_obj = np.linalg.norm(d)**2
        
        while (self.func.prob.f_1(x - alpha_ul * d, y_tilde, z_tilde) > self.func.prob.f_1(x, y, z) - eta * alpha_ul * sub_obj): 
            # print('0f1',self.func.prob.f_1(x - alpha_ul * d, y_tilde, z_tilde),self.func.prob.f_1(x, y, z) - eta * alpha_ul * sub_obj,alpha_ul,'y_tilde: ',y_tilde,'y: ',y,'z_tilde: ',z_tilde,'z: ',z)                                   
            iterLS = iterLS + 1
            alpha_ul = alpha_ul * tau
            if alpha_ul <= 10**-8:
                alpha_ul = 0 
                y_tilde = y;
                z_tilde = z;
                break 
            
            y_tilde, z_tilde = self.update_mlp_true_funct_armijo(x - alpha_ul * d, y, z, self.ml_iters_true_funct_armijo, self.ll_iters_true_funct_armijo) 
            # print('y_tilde',y_tilde,'z_tilde',z_tilde)
            
        # print('f1',self.func.prob.f_1(x - alpha_ul * d, y_tilde, z_tilde),self.func.prob.f_1(x, y, z) - eta * alpha_ul * sub_obj,alpha_ul,'y_tilde: ',y_tilde,'y: ',y,'z_tilde: ',z_tilde,'z: ',z)            
        if self.iprint >= 10:
            print('\nArmijo alpha UL: ',alpha_ul,'\n')
    
            print('\nEND Armijo UL')    
        return alpha_ul, y_tilde, z_tilde


    def backtracking_armijo_ml(self, x, y, z, d, eta = 10**-4, tau = 0.5):
        if self.iprint >= 10:
            print('\nSTART Armijo ML')
        
        initial_rate = 1
        
        alpha_ml = initial_rate
        iterLS = 0
        z_tilde = self.update_llp_true_funct_armijo(x, y - alpha_ml * d, z, self.ll_iters_true_funct_armijo) 
        # print('z_tilde',z_tilde)
        sub_obj = np.linalg.norm(d)**2
        while (self.func.prob.f_2(x, y - alpha_ml * d, z_tilde) > self.func.prob.f_2(x, y, z) - eta * alpha_ml * sub_obj): 
            # print('f2',self.func.prob.f_2(x, y - alpha_ml * d, z_tilde),self.func.prob.f_2(x, y, z) - eta * alpha_ml * sub_obj,alpha_ml,'z_tilde:',z_tilde,'z:',z)                                   
            iterLS = iterLS + 1
            alpha_ml = alpha_ml * tau
            if alpha_ml <= 10**-8:
                alpha_ml = 0 
                z_tilde = z;
                break  
            
            z_tilde = self.update_llp_true_funct_armijo(x, y - alpha_ml * d, z, self.ll_iters_true_funct_armijo) 
            # print('z_tilde',z_tilde)
            
        # print('f2',self.func.prob.f_2(x, y - alpha_ml * d, z_tilde),self.func.prob.f_2(x, y, z) - eta * alpha_ml * sub_obj,alpha_ml,'z_tilde:',z_tilde,'z:',z)                                   
        if self.iprint >= 10:
            print('\nArmijo alpha ML: ',alpha_ml,'\n')
    
            print('\nEND Armijo ML')     
        return alpha_ml, z_tilde
    

    def backtracking_armijo_ll(self, x, y, z, eta = 10**-4, tau = 0.5):
        if self.iprint >= 10:
            print('\nSTART Armijo LL')
        
        initial_rate = 1
        
        alpha = initial_rate
        iterLS = 0
        
        grad_f_l_z = self.func.prob.grad_f3_ll_vars(x, y, z) 
        
        while (self.func.prob.f_3(x, y, z - alpha * grad_f_l_z) > self.func.prob.f_3(x,y,z) - eta * alpha * np.dot(grad_f_l_z.T, grad_f_l_z)):                                    
            iterLS = iterLS + 1
            alpha = alpha * tau
            if alpha <= 10**-8:
                alpha = 0 
                break 

        if self.iprint >= 10:
            print('\nArmijo alpha LL: ',alpha,'\n')
    
            print('\nEND Armijo LL') 
        return alpha
    
    
    def cg_fd_ll_vars(self, ul_vars, ml_vars, ll_vars, vec, rhs):
        """
        Approximates expressions of this type: B^-1c, evaluated at (ul_vars, ml_vars, ll_vars), by applying CG-FD where FD is wrt ll_vars. 
        
        Example: B = hess_f3_zz, c = grad_f2_z 
        
        hess_f3_zz c = (grad_f3_z((ul_vars, ml_vars, ll_plus) - grad_f3_z((ul_vars, ml_vars, ll_minus))/(2*eps)
        
        Args:
            vec:   grad_f3_z (reference to function definition)
            rhs:   grad_f2_z (reference to function definition)   
        """
        
        def mv(v):
            ep = 1e-1 #0.1/np.maximum(1,v_norm) #1e-1
            
            # Define z+ and z-
            ll_plus = ll_vars + ep * v.reshape(-1,1)
            ll_minus = ll_vars - ep * v.reshape(-1,1)                
            
            grad_plus = vec(ul_vars, ml_vars, ll_plus)                
            grad_minus = vec(ul_vars, ml_vars, ll_minus)
            
            return (grad_plus - grad_minus) / (2 * ep)  
        
        self.matrix = LinearOperator((self.func.prob.z_dim, self.func.prob.z_dim), matvec=mv)

        # print('\nrhs',rhs(ul_vars, ml_vars, ll_vars))        
        lambda_adj, exit_code = cg(self.matrix, rhs(ul_vars, ml_vars, ll_vars), x0=None, rtol=self.cg_fd_rtol, maxiter=self.cg_fd_maxiter) #maxiter=100
        # print('exit code', exit_code)
        return np.reshape(lambda_adj, (-1,1))


    def cg_fd_ml_vars(self, ul_vars, ml_vars, ll_vars, vec, rhs, use_minres=False):
        """
        Approximates expressions of this type: B^-1c, evaluated at (ul_vars, ml_vars, ll_vars), by applying CG-FD where FD is wrt ml_vars
        
        Example: B = hess_fbar_yy, c = grad_f1_y 
        
        hess_fbar_yy c = (grad_fbar_y((ul_vars, ml_plus, ll_vars) - grad_fbar_y((ul_vars, ml_minus, ll_vars))/(2*eps)
        
        Args:
            vec:   grad_fbar_y
            rhs:   grad_f1_y    
        """
        
        def mv(v):
            ## Finite difference approximation
            #v_norm = np.linalg.norm(v) 
            ep = 1e-1 #1e-1 April 2025 #0.1/np.maximum(1,v_norm) #1e-1
            
            # Define z+ and z-
            ml_plus = ml_vars + ep * v.reshape(-1,1)
            ml_minus = ml_vars - ep * v.reshape(-1,1)                
            
            grad_plus = vec(ul_vars, ml_plus, ll_vars)                
            grad_minus = vec(ul_vars, ml_minus, ll_vars)
            
            return (grad_plus - grad_minus) / (2 * ep)  
        
        self.matrix = LinearOperator((self.func.prob.y_dim, self.func.prob.y_dim), matvec=mv)

        if use_minres:        
            lambda_adj, exit_code = minres(self.matrix, rhs, x0=None, rtol=self.cg_fd_rtol_ml, maxiter=self.cg_fd_maxiter_ml) #maxiter=100
        else:
            lambda_adj, exit_code = cg(self.matrix, rhs, x0=None, rtol=self.cg_fd_rtol_ml, maxiter=self.cg_fd_maxiter_ml) #maxiter=100

        return np.reshape(lambda_adj, (-1,1))
    

    def fd_ll_vars(self, ul_vars, ml_vars, ll_vars, vec, v):
        """
        Approximates matrix-vector products of this type: Av, evaluated at (ul_vars, ml_vars, ll_vars), by applying FD wrt ll_vars
        
        Example: A = hess_f3_yz, v = z-dim vector
        
        hess_f3_yz v = (grad_f3_y((ul_vars, ml_vars, ll_plus) - grad_f3_y((ul_vars, ml_vars, ll_minus))/(2*eps),

                  where ll_plus = ll_vars + ep * v and ll_minus = ll_vars - ep * v                        
        
        Args:
            vec:   grad_f3_y 
        """
        ## Finite difference approximation
        #grad_norm = np.linalg.norm(lambda_adj)            
        ep = 1e-1 #0.01 / grad_norm
        
        # Define z+ and z-
        ll_plus = ll_vars + ep * v
        ll_minus = ll_vars - ep * v
        
        grad_plus = vec(ul_vars, ml_vars, ll_plus)
        grad_minus = vec(ul_vars, ml_vars, ll_minus)
    
        return (grad_plus - grad_minus) / (2 * ep)


    def fd_ml_vars(self, ul_vars, ml_vars, ll_vars, vec, v):
        """
        Approximates matrix-vector products of this type: Av, evaluated at (ul_vars, ml_vars, ll_vars), by applying FD wrt ml_vars
        
        Example: A = hess_fbar_xy, v = y-dim vector
        
        hess_fbar_xy v = (grad_fbar_x((ul_vars, ml_plus, ll_vars) - grad_fbar_x((ul_vars, ml_minus, ll_vars))/(2*eps),

                  where ml_plus = ml_vars + ep * v and ml_minus = ml_vars - ep * v                        
        
        Args:
            vec:   grad_fbar_x 
        """
        ## Finite difference approximation
        #grad_norm = np.linalg.norm(lambda_adj)            
        ep = 1e-1 #0.01 / grad_norm
        
        # Define z+ and z-
        ml_plus = ml_vars + ep * v
        ml_minus = ml_vars - ep * v
        
        grad_plus = vec(ul_vars, ml_plus, ll_vars)
        grad_minus = vec(ul_vars, ml_minus, ll_vars)
    
        return (grad_plus - grad_minus) / (2 * ep)


    def neumann_autodiff_ll_vars(self, ul_vars, ml_vars, ll_vars, vec, rhs, ml_vars_requires_grad=False):
        """
        Approximates expressions of this type: B^-1c, evaluated at (ul_vars, ml_vars, ll_vars), by applying automatic differentiation wrt ll_vars.
        
        B is a Hessian matrix of dimension z-dim times z-dim. 
        c is the z-dim rhs.
        
        Args:
            vec needs to be a torch function returning a z-dim gradient such that the Jacobian of vec is equal to B.   
        """
        eta = self.neumann_eta #0.01#0.05
        hessian_q = self.neumann_hessian_q

        ul_vars, ml_vars, ll_vars = self.func.compute_args(ul_vars, ml_vars, ll_vars)
        
        if not ml_vars_requires_grad:
            v_0 = torch.from_numpy(rhs(ul_vars, ml_vars, ll_vars)).detach()   
            
            ul_vars = torch.tensor(ul_vars, dtype=torch.float64)            
            ml_vars = torch.tensor(ml_vars, dtype=torch.float64)
            ll_vars = torch.tensor(ll_vars, dtype=torch.float64)
            ll_vars.requires_grad_()            

        else:

            ul_vars = torch.tensor(ul_vars, dtype=torch.float64)            
            ml_vars = torch.tensor(ml_vars, dtype=torch.float64)
            ll_vars = torch.tensor(ll_vars, dtype=torch.float64)
          
            v_0 = rhs(ul_vars, ml_vars, ll_vars).detach()              
            
            ll_vars.requires_grad_()
      
        if ml_vars_requires_grad:
            ul_vars.requires_grad_()
            ml_vars.requires_grad_()
        
        vec_torch = vec(ul_vars, ml_vars, ll_vars)
        
        vec_torch_aux = torch.reshape(ll_vars, [-1]) - eta * torch.reshape(vec_torch, [-1]) 
        
        # Compute B^-1c using truncated Neumann series
        z_list = []
        
        for _ in range(hessian_q):
            vec_torch_c = torch.matmul(vec_torch_aux.double(), v_0.double())
            I_minus_B_c = torch.autograd.grad(vec_torch_c, ll_vars, create_graph=True)[0]
            v_0 = torch.unsqueeze(torch.reshape(I_minus_B_c, [-1]), 1).detach()
            z_list.append(v_0) 
        invB_c = eta*v_0+torch.sum(torch.stack(z_list), dim=0)

        if not ml_vars_requires_grad:
            out = invB_c.detach().double()
        else:
            out = invB_c.detach().double(), ul_vars, ml_vars

        return out


    def neumann_autodiff_ml_vars__(self, ul_vars, ml_vars, ll_vars, vec, rhs, ml_vars_requires_grad=False):
        """
        Approximates expressions of this type: B^-1c, evaluated at (ul_vars, ml_vars, ll_vars), by applying automatic differentiation wrt ml_vars.
        
        B is a Hessian matrix of dimension y-dim times y-dim.
        c is the y-dim rhs.
        
        Args:
            vec needs to be a torch function returning a y-dim gradient such that the Jacobian of vec is equal to B.   
        """
        # ul_vars, ml_vars, ll_vars = self.func.compute_args(ul_vars, ml_vars, ll_vars)
        
        v_0 = torch.from_numpy(rhs).detach()   
        
        eta = self.neumann_eta_ml #0.01#0.05
        hessian_q = self.neumann_hessian_q_ml

        ul_vars = torch.tensor(ul_vars, dtype=torch.float64)            
        if not ml_vars_requires_grad: # if given, we do not want to break the computational graph
            ml_vars = torch.tensor(ml_vars, dtype=torch.float64)
            ml_vars.requires_grad_()
        ll_vars = torch.tensor(ll_vars, dtype=torch.float64)      
        
        vec_torch = vec
        
        vec_torch_aux = torch.reshape(ml_vars, [-1]) - eta * torch.reshape(vec_torch, [-1]) 
        
        # Compute B^-1c using truncated Neumann series
        z_list = []
        
        for _ in range(hessian_q):
            vec_torch_c = torch.matmul(vec_torch_aux.double(), v_0.double())
            I_minus_B_c = torch.autograd.grad(vec_torch_c, ml_vars, create_graph=True)[0]
            v_0 = torch.unsqueeze(torch.reshape(I_minus_B_c, [-1]), 1).detach()
            z_list.append(v_0) 
        invB_c = eta*v_0+torch.sum(torch.stack(z_list), dim=0)
        
        return invB_c.detach().double()
    

    # def neumann_autodiff_ml_vars(self, ul_vars, ml_vars, ll_vars, vec, rhs, ml_vars_requires_grad=False):
    #     """
    #     Approximates expressions of this type: B^-1c, evaluated at (ul_vars, ml_vars, ll_vars), by applying automatic differentiation wrt ml_vars.
        
    #     B is a Hessian matrix of dimension y-dim times y-dim.
    #     c is the y-dim rhs.
        
    #     Args:
    #         vec needs to be a torch function returning a y-dim gradient such that the Jacobian of vec is equal to B.   
    #     """
    #     # ul_vars, ml_vars, ll_vars = self.func.compute_args(ul_vars, ml_vars, ll_vars)         
    #     v_0 = torch.from_numpy(rhs).detach()   
        
    #     eta = self.neumann_eta_ml #0.01#0.05
    #     hessian_q = self.neumann_hessian_q_ml

    #     ul_vars = torch.tensor(ul_vars, dtype=torch.float64)            
    #     if not ml_vars_requires_grad: # if given, we do not want to break the computational graph
    #         ml_vars = torch.tensor(ml_vars, dtype=torch.float64)
    #         ml_vars.requires_grad_()
    #     ll_vars = torch.tensor(ll_vars, dtype=torch.float64)      
        
    #     vec_torch = vec(ul_vars, ml_vars, ll_vars)
        
    #     vec_torch_aux = torch.reshape(ll_vars, [-1]) - eta * torch.reshape(vec_torch, [-1]) 
        
    #     # Compute B^-1c using truncated Neumann series
    #     z_list = []
        
    #     for _ in range(hessian_q):
    #         vec_torch_c = torch.matmul(vec_torch_aux.double(), v_0.double())
    #         I_minus_B_c = torch.autograd.grad(vec_torch_c, ml_vars, create_graph=True)[0]
    #         v_0 = torch.unsqueeze(torch.reshape(I_minus_B_c, [-1]), 1).detach()
    #         z_list.append(v_0) 
    #     invB_c = eta*v_0+torch.sum(torch.stack(z_list), dim=0)
        
    #     return invB_c.detach().double()


    def autodiff_ul_vars__(self, ul_vars, ml_vars, ll_vars, vec, v, ml_vars_requires_grad=False):
        """
        Approximates matrix-vector products of this type: Av, evaluated at (ul_vars, ml_vars, ll_vars), by applying automatic differentiation wrt ul_vars.
        
        Example: A = hess_f3_xz, A = hess_fbar_xy, v = z-dim or y-dim vector
        
        Args:
            vec needs to be a torch function returning a z-dim gradient (i.e., grad_f3_z) or y-dim gradient (i.e., grad_fbar_y) such that the Jacobian of vec is equal to A.
        """
        # ul_vars, ml_vars, ll_vars = self.func.compute_args(ul_vars, ml_vars, ll_vars)

        ll_vars = torch.tensor(ll_vars, dtype=torch.float64)
             
        # ul_vars.requires_grad_()    
        if not ml_vars_requires_grad:
            ml_vars = ml_vars.detach()  #April 4, 2025 bug: ml_vars = ll_vars.detach()
        ll_vars = ll_vars.detach()

        vec_torch = vec
        
        # Compute matrix vector product: Av
        vec_torch_aux = torch.reshape(vec_torch, [-1]) 
        vec_torch_v = torch.matmul(vec_torch_aux.double(), v.detach().double())          
        Av = torch.autograd.grad(vec_torch_v, ul_vars, create_graph=True, allow_unused=True)[0]  ## Added allow_unused=True on April 23, 2025
        
        if Av is None: ## Added on April 23, 2025
            # Handle the case when gradient is not computed
            Av = torch.zeros_like(ul_vars)

        if not ml_vars_requires_grad:
            out = Av.detach().numpy()
        else:
            out = Av.detach().numpy()
        return out
    

    def autodiff_ul_vars(self, ul_vars, ml_vars, ll_vars, vec, v, ml_vars_requires_grad=False):
        """
        Approximates matrix-vector products of this type: Av, evaluated at (ul_vars, ml_vars, ll_vars), by applying automatic differentiation wrt ul_vars.
        
        Example: A = hess_f3_xz, A = hess_fbar_xy, v = z-dim or y-dim vector
        
        Args:
            vec needs to be a torch function returning a z-dim gradient (i.e., grad_f3_z) or y-dim gradient (i.e., grad_fbar_y) such that the Jacobian of vec is equal to A.
        """
        ul_vars, ml_vars, ll_vars = self.func.compute_args(ul_vars, ml_vars, ll_vars)

        ul_vars = torch.tensor(ul_vars, dtype=torch.float64)            
        ml_vars = torch.tensor(ml_vars, dtype=torch.float64)
        ll_vars = torch.tensor(ll_vars, dtype=torch.float64)
             
        ul_vars.requires_grad_()    
        if not ml_vars_requires_grad:
            ml_vars = ml_vars.detach()   #April 4, 2025 bug: ml_vars = ll_vars.detach()
        ll_vars = ll_vars.detach()

        vec_torch = vec(ul_vars, ml_vars, ll_vars)
        
        # Compute matrix vector product: Av
        vec_torch_aux = torch.reshape(vec_torch, [-1]) 
        vec_torch_v = torch.matmul(vec_torch_aux.double(), v.detach().double())

        if not vec_torch_v.requires_grad: ## Added on April 27, 2025
            Av = torch.zeros_like(ul_vars)

        else: ## Added on April 27, 2025     
            Av = torch.autograd.grad(vec_torch_v, ul_vars, create_graph=True, allow_unused=True)[0]  ## Added allow_unused=True on April 27, 2025
    
            if Av is None: ## Added on April 27, 2025
                # Handle the case when gradient is not computed
                Av = torch.zeros_like(ul_vars)

        if not ml_vars_requires_grad:
            out = Av.detach().numpy()
        else:
            out = Av.detach().numpy(), ml_vars

        return out


    def autodiff_ml_vars(self, ul_vars, ml_vars, ll_vars, vec, v, ml_vars_requires_grad=False):
        """
        Approximates matrix-vector products of this type: Av, evaluated at (ul_vars, ml_vars, ll_vars), by applying automatic differentiation wrt ml_vars.
        
        Example: A = hess_f3_yz, v = z-dim vector
        
        Args:
            vec needs to be a torch function returning a z-dim gradient (i.e., grad_f3_z) such that the Jacobian of vec is equal to A.
        """
        ul_vars, ml_vars, ll_vars = self.func.compute_args(ul_vars, ml_vars, ll_vars)

        ul_vars = torch.tensor(ul_vars, dtype=torch.float64)            
        ml_vars = torch.tensor(ml_vars, dtype=torch.float64)
        ll_vars = torch.tensor(ll_vars, dtype=torch.float64)

        if not ml_vars_requires_grad:  #April 4, 2025 bug: if not ml_vars_requires_grad:       
            ul_vars = ul_vars.detach() #April 4, 2025 bug: ul_vars = ll_vars.detach()
        else:
            ul_vars.requires_grad_()  
        ml_vars.requires_grad_()  
        ll_vars = ll_vars.detach()

        vec_torch = vec(ul_vars, ml_vars, ll_vars)
        
        # Compute matrix vector product: Av
        vec_torch_aux = torch.reshape(vec_torch, [-1]) 
        vec_torch_v = torch.matmul(vec_torch_aux.double(), v.detach().double())          

        if not vec_torch_v.requires_grad: ## Added on April 28, 2025
            Av = torch.zeros_like(ml_vars)

        else: ## Added on April 28, 2025 
            Av = torch.autograd.grad(vec_torch_v, ml_vars, create_graph=True)[0]

        if not ml_vars_requires_grad:
            out = Av.detach().numpy()
        else:
            out = Av, ul_vars, ml_vars, ll_vars

        return out
        

    def grad_fbar_ul_vars(self, ul_vars, ml_vars=None, ll_vars=None, hess='CG-FD'):
        """
        The gradient of fbar wrt the upper-level variables
    	"""
        grad_f2_x = self.func.prob.grad_f2_ul_vars
        grad_f2_z = self.func.prob.grad_f2_ll_vars

        args = self.func.compute_args(ul_vars, ml_vars, ll_vars)
        
        # Compute the ML tsg direction
        if hess == 'CG-FD':   
            # TSG-N-FD

            ## When removing the LL problem, we modify the structure of the ML adjoint gradient because f3 no longer appears in the formula
            if not self.algo == "remove_ll":            
                lambda_adj = self.cg_fd_ll_vars(*args, self.func.prob.grad_f3_ll_vars, grad_f2_z)
                second_term = self.fd_ll_vars(*args, self.func.prob.grad_f3_ul_vars, lambda_adj)
            
                tsg = grad_f2_x(*args) - second_term
                
            else:
                tsg = grad_f2_x(*args)                 

        else:
            print("grad_fbar_ul_vars is only implemented for self.hess equal to 'CG-FD'")
            
        return tsg
    

    def grad_fbar_ml_vars(self, ul_vars, ml_vars=None, ll_vars=None, hess=True):
        """
        The gradient of fbar wrt the middle-level variables
    	"""
        grad_f2_y = self.func.prob.grad_f2_ml_vars
        grad_f2_z = self.func.prob.grad_f2_ll_vars

        args = self.func.compute_args(ul_vars, ml_vars, ll_vars)
      
        # Compute the ML tsg direction
        if hess == True:
            # TSG-H

            ## When removing the LL problem, we modify the structure of the ML adjoint gradient because f3 no longer appears in the formula
            if not self.algo == "remove_ll":            
                hess_f3_yz = self.func.prob.hess_f3_ml_vars_ll_vars
                hess_f3_zz = self.func.prob.hess_f3_ll_vars_ll_vars
    
                # Inverse computed by factorization
                tsg = grad_f2_y(*args) - np.dot(hess_f3_yz(*args),np.dot(np.linalg.inv(hess_f3_zz(*args)),grad_f2_z(*args)))
                
            else:
                tsg = grad_f2_y(*args)                 

        elif hess == 'CG-FD':   
            # TSG-N-FD

            ## When removing the LL problem, we modify the structure of the ML adjoint gradient because f3 no longer appears in the formula
            if not self.algo == "remove_ll":            
                lambda_adj = self.cg_fd_ll_vars(*args, self.func.prob.grad_f3_ll_vars, grad_f2_z)
                second_term = self.fd_ll_vars(*args,self.func.prob.grad_f3_ml_vars,lambda_adj)
            
                tsg = grad_f2_y(*args) - second_term
                
            else:
                tsg = grad_f2_y(*args) 

        elif hess == 'autodiff':
            # TSG-AD

            ## When removing the LL problem, we modify the structure of the ML adjoint gradient because f3 no longer appears in the formula
            if not self.algo == "remove_ll":  
                rhs = self.neumann_autodiff_ll_vars(*args, self.func.prob.grad_f3_ll_vars_torch, grad_f2_z)
                out = self.autodiff_ml_vars(*args, self.func.prob.grad_f3_ll_vars_torch, rhs)   #April 27, 2025 bug: self.autodiff_ul_vars(*args, self.func.prob.grad_f3_ll_vars_torch, rhs) 
                
                tsg = grad_f2_y(*args) - out         

            else:                
                tsg = grad_f2_y(*args)   
                
        else:
            print('There is something wrong with self.hess')
            
        return tsg


    def grad_fbar_ml_vars_torch(self, ul_vars, ml_vars=None, ll_vars=None, outputs=False):
        """
        The gradient of fbar wrt the middle-level variables when hess is set to autodiff
    	"""
        args = self.func.compute_args(ul_vars, ml_vars, ll_vars)

        grad_f2_z_torch = self.func.prob.grad_f2_ll_vars_torch
        grad_f2_y_torch = self.func.prob.grad_f2_ml_vars_torch

        ## When removing the LL problem, we modify the structure of the ML adjoint gradient because f3 no longer appears in the formula
        if not self.algo == "remove_ll":          
            rhs, ul_vars, ml_vars = self.neumann_autodiff_ll_vars(*args, self.func.prob.grad_f3_ll_vars_torch, grad_f2_z_torch, ml_vars_requires_grad=True)
            out, ul_vars, ml_vars, ll_vars = self.autodiff_ml_vars(ul_vars, ml_vars, ll_vars, self.func.prob.grad_f3_ll_vars_torch, rhs, ml_vars_requires_grad=True)
            
            tsg = grad_f2_y_torch(ul_vars, ml_vars, ll_vars) - out         

        else:  
            ul_vars = torch.tensor(ul_vars, dtype=torch.float64)            
            ml_vars = torch.tensor(ml_vars, dtype=torch.float64)
            ll_vars = torch.tensor(ll_vars, dtype=torch.float64)

            ul_vars.requires_grad_()  
            ml_vars.requires_grad_()  
            ll_vars = ll_vars.detach()
            
            tsg = grad_f2_y_torch(ul_vars, ml_vars, ll_vars) 
            
        if not outputs:
            out = tsg
        else:
            out = tsg, ul_vars, ml_vars
            
        return out
    

    def grad_f_ul_vars(self, x, y, z, hess):
        """
        The adjoint gradient
    	"""
        grad_f1_x = self.func.prob.grad_f1_ul_vars
        grad_f1_z = self.func.prob.grad_f1_ll_vars
        grad_f1_y = self.func.prob.grad_f1_ml_vars
        grad_f2_z = self.func.prob.grad_f2_ll_vars 
        
        args = self.func.compute_args(x, y, z)
                   
        # Compute the UL tsg direction
        if hess == True:
            # TSG-H
            
            hess_f2_zz = self.func.prob.hess_f2_ll_vars_ll_vars       
            hess_f3_xz = self.func.prob.hess_f3_ul_vars_ll_vars
            hess_f3_yz = self.func.prob.hess_f3_ml_vars_ll_vars
            hess_f3_zz = self.func.prob.hess_f3_ll_vars_ll_vars
            
            trd_hess_f3_yzz = self.func.prob.trd_hess_f3_ml_vars_ll_vars_ll_vars
            trd_hess_f3_zzz = self.func.prob.trd_hess_f3_ll_vars_ll_vars_ll_vars

            hess_f2_yx = self.func.prob.hess_f2_ml_vars_ul_vars
            hess_f2_yz = self.func.prob.hess_f2_ml_vars_ll_vars
            hess_f3_zx = self.func.prob.hess_f3_ll_vars_ul_vars

            hess_f2_zx = self.func.prob.hess_f2_ll_vars_ul_vars
            
            trd_hess_f3_yzx = self.func.prob.trd_hess_f3_ml_vars_ll_vars_ul_vars
            # trd_hess_f3_yxz = self.func.prob.trd_hess_f3_ml_vars_ul_vars_ll_vars 
            trd_hess_f3_zzx = self.func.prob.trd_hess_f3_ll_vars_ll_vars_ul_vars
            # trd_hess_f3_zxz = self.func.prob.trd_hess_f3_ll_vars_ul_vars_ll_vars 

            hess_f2_yy = self.func.prob.hess_f2_ml_vars_ml_vars
            hess_f2_yz = self.func.prob.hess_f2_ml_vars_ll_vars
            hess_f3_zy = self.func.prob.hess_f3_ll_vars_ml_vars

            hess_f2_zy = self.func.prob.hess_f2_ll_vars_ml_vars
            
            trd_hess_f3_yzy = self.func.prob.trd_hess_f3_ml_vars_ll_vars_ml_vars
            # trd_hess_f3_yyz = self.func.prob.trd_hess_f3_ml_vars_ml_vars_ll_vars 
            trd_hess_f3_zzy = self.func.prob.trd_hess_f3_ll_vars_ll_vars_ml_vars
            # trd_hess_f3_zyz = self.func.prob.trd_hess_f3_ll_vars_ml_vars_ll_vars 
            
            def hess_fbar_ml_vars_ul_vars(x, y, z):
                 """
                 The Hessian of fbar wrt the middle and upper level variables
             	 """
                 def grad_z_x(x, y, z):  
                     args = x, y, z  
                     return -np.dot(np.linalg.inv(hess_f3_zz(*args)),hess_f3_zx(*args)).T # Note that here there is a transpose
                 
                 def par_der_ul_vars(x, y, z):
                     args = x, y, z  

                     aux = np.dot(np.linalg.inv(hess_f3_zz(*args)),grad_f2_z(*args))
                     
                     aux_1 = np.squeeze(np.einsum('nmt,tr->nmr',trd_hess_f3_yzx(*args),aux)) + np.squeeze(np.einsum('nmt,tr->nmr',np.einsum('tmh,tn->nmh',trd_hess_f3_yzz(*args),grad_z_x(*args).T),aux))
                     aux_2 = np.squeeze(np.einsum('nmt,tr->nmr',np.einsum('ht,ntl->nhl',hess_f3_yz(*args),np.einsum('ht,ntl->nhl',np.linalg.inv(hess_f3_zz(*args)),trd_hess_f3_zzx(*args) + np.einsum('tmh,tn->nmh',trd_hess_f3_zzz(*args),grad_z_x(*args).T))),aux))                 
                     aux_3 = np.dot(hess_f3_yz(*args),np.dot(np.linalg.inv(hess_f3_zz(*args)),hess_f2_zx(*args) + np.dot(hess_f2_zz(*args),grad_z_x(*args).T)))
                     
                     out = - (aux_1 - aux_2).T - aux_3

                     return out
        
                 par_der_x = par_der_ul_vars

                 args = x, y, z                  
                 out = hess_f2_yx(*args) + np.dot(hess_f2_yz(*args),grad_z_x(*args).T) + par_der_x(*args)
                 return out
             
            def hess_fbar_ml_vars_ml_vars(x, y, z):
                 """
                 The Hessian of fbar wrt the middle level variables
             	 """   
                 def grad_z_y(x, y, z):  
                     args = x, y, z 
                     return -np.dot(np.linalg.inv(hess_f3_zz(*args)),hess_f3_zy(*args)).T # Note that here there is a transpose
                 
                 def par_der_ml_vars(x, y, z):
                     args = x, y, z  

                     aux = np.dot(np.linalg.inv(hess_f3_zz(*args)),grad_f2_z(*args))
                     
                     aux_1 = np.squeeze(np.einsum('nmt,tr->nmr',trd_hess_f3_yzy(*args),aux)) + np.squeeze(np.einsum('nmt,tr->nmr',np.einsum('tmh,tn->nmh',trd_hess_f3_yzz(*args),grad_z_y(*args).T),aux))
                     aux_2 = np.squeeze(np.einsum('nmt,tr->nmr',np.einsum('ht,ntl->nhl',hess_f3_yz(*args),np.einsum('ht,ntl->nhl',np.linalg.inv(hess_f3_zz(*args)),trd_hess_f3_zzy(*args) + np.einsum('tmh,tn->nmh',trd_hess_f3_zzz(*args),grad_z_y(*args).T))),aux))                 
                     aux_3 = np.dot(hess_f3_yz(*args),np.dot(np.linalg.inv(hess_f3_zz(*args)),hess_f2_zy(*args) + np.dot(hess_f2_zz(*args),grad_z_y(*args).T)))
                     
                     out = - (aux_1 - aux_2).T - aux_3

                     return out
        
                 par_der_y = par_der_ml_vars

                 args = x, y, z                 
                 out = hess_f2_yy(*args) + np.dot(hess_f2_yz(*args),grad_z_y(*args).T) + par_der_y(*args)
                 return out
             
            hess_fbar_yx = hess_fbar_ml_vars_ul_vars 
            hess_fbar_yy = hess_fbar_ml_vars_ml_vars

            ## When removing the LL problem, we modify the structure of the UL adjoint gradient because f3 no longer appears in the formula
            if not self.algo == "remove_ll":             
                tsg = grad_f1_x(*args) - np.dot(hess_f3_xz(*args),np.dot(np.linalg.inv(hess_f3_zz(*args)),grad_f1_z(*args))) - \
                    np.dot(hess_fbar_yx(*args).T,np.dot(np.linalg.inv(hess_fbar_yy(*args)),grad_f1_y(*args) - np.dot(hess_f3_yz(*args),np.dot(np.linalg.inv(hess_f3_zz(*args)),grad_f1_z(*args)))))

            else:
                tsg = grad_f1_x(*args) - \
                    np.dot(hess_fbar_yx(*args).T,np.dot(np.linalg.inv(hess_fbar_yy(*args)),grad_f1_y(*args)))
                
            # lambda_adj, exit_code = cg(hess_f3_zz,grad_f2_z, x0=None, tol=1e-4, maxiter=3)
            # lambda_adj = np.reshape(lambda_adj, (-1,1))
            
            # tsg = grad_f2_y - np.dot(hess_f3_yz,lambda_adj)
                       
        elif hess == 'CG-FD':   
            # TSG-N-FD
            
            grad_f3_x = self.func.prob.grad_f3_ul_vars
            grad_f3_y = self.func.prob.grad_f3_ml_vars
            grad_f3_z = self.func.prob.grad_f3_ll_vars

            # For convenience, we create an auxiliary grad_fbar_ul_vars_aux that is specific for hess='CG-FD'
            def grad_fbar_ul_vars_aux(ul_vars, ml_vars, ll_vars):
                return self.grad_fbar_ul_vars(ul_vars, ml_vars, ll_vars, hess='CG-FD')

            # For convenience, we create an auxiliary grad_fbar_ml_vars_aux that is specific for hess='CG-FD' ## Added on April 23, 2025
            def grad_fbar_ml_vars_aux(ul_vars, ml_vars, ll_vars):
                return self.grad_fbar_ml_vars(ul_vars, ml_vars, ll_vars, hess='CG-FD')
                
            grad_fbar_x = grad_fbar_ul_vars_aux
            grad_fbar_y = grad_fbar_ml_vars_aux  ## Fixed on April 23, 2025 (it was self.grad_fbar_ml_vars before (hess is always True, which is incorrect))

            ## When removing the LL problem, we modify the structure of the UL adjoint gradient because f3 no longer appears in the formula
            if not self.algo == "remove_ll":
                lambda_adj_left_term = self.cg_fd_ll_vars(*args, grad_f3_z, grad_f1_z)
                left_term = grad_f1_x(*args) - self.fd_ll_vars(*args, grad_f3_x, lambda_adj_left_term)
                
                lambda_adj_rhs = self.cg_fd_ll_vars(*args, grad_f3_z, grad_f1_z)
                rhs = grad_f1_y(*args) - self.fd_ll_vars(*args, grad_f3_y, lambda_adj_rhs)          
    
                lambda_adj_fbar = self.cg_fd_ml_vars(*args, grad_fbar_y, rhs, use_minres=True) # We use minres instead of cg just to avoid ReentrancyError (=Recursion error). Both solvers are for symmetric problems (minres allows singular matrix and is less efficient)
                right_term = self.fd_ml_vars(*args,grad_fbar_x,lambda_adj_fbar)
                
                tsg = left_term - right_term

            else:
                left_term = grad_f1_x(*args) 
                rhs = grad_f1_y(*args)          
    
                lambda_adj_fbar = self.cg_fd_ml_vars(*args, grad_fbar_y, rhs, use_minres=True) # We use minres instead of cg just to avoid ReentrancyError (=Recursion error). Both solvers are for symmetric problems (minres allows singular matrix and is less efficient)
                right_term = self.fd_ml_vars(*args,grad_fbar_x,lambda_adj_fbar)
                
                tsg = left_term - right_term                        

        elif hess == 'autodiff':
            # TSG-AD

            ## When removing the LL problem, we modify the structure of the UL adjoint gradient because f3 no longer appears in the formula
            if not self.algo == "remove_ll":
                rhs_aux_1 = self.neumann_autodiff_ll_vars(*args, self.func.prob.grad_f3_ll_vars_torch, grad_f1_z)
                left_term = grad_f1_x(*args) - self.autodiff_ul_vars(*args, self.func.prob.grad_f3_ll_vars_torch, rhs_aux_1)
            
                rhs_aux_2 = self.neumann_autodiff_ll_vars(*args, self.func.prob.grad_f3_ll_vars_torch, grad_f1_z)
                rhs = grad_f1_y(*args) - self.autodiff_ml_vars(*args, self.func.prob.grad_f3_ll_vars_torch, rhs_aux_2)
    
                grad_fbar_y_torch, ul_vars, ml_vars = self.grad_fbar_ml_vars_torch(*args, outputs=True)
    
                # rhs_aux_3 = self.neumann_autodiff_ml_vars(args[0], ml_vars, args[2], self.grad_fbar_ml_vars_torch, rhs, ml_vars_requires_grad=True)             
                rhs_aux_3 = self.neumann_autodiff_ml_vars__(ul_vars, ml_vars, args[2], grad_fbar_y_torch, rhs, ml_vars_requires_grad=True)             
                # right_term = self.autodiff_ul_vars(args[0], ml_vars, args[2], self.grad_fbar_ml_vars_torch, rhs_aux_3, ml_vars_requires_grad=True)
                right_term = self.autodiff_ul_vars__(ul_vars, ml_vars, args[2], grad_fbar_y_torch, rhs_aux_3, ml_vars_requires_grad=True)
    
                tsg = left_term - right_term            

            else:
                left_term = grad_f1_x(*args) 
                rhs = grad_f1_y(*args)
    
                grad_fbar_y_torch, ul_vars, ml_vars = self.grad_fbar_ml_vars_torch(*args, outputs=True)
    
                rhs_aux_3 = self.neumann_autodiff_ml_vars__(ul_vars, ml_vars, args[2], grad_fbar_y_torch, rhs, ml_vars_requires_grad=True)             
                right_term = self.autodiff_ul_vars__(ul_vars, ml_vars, args[2], grad_fbar_y_torch, rhs_aux_3, ml_vars_requires_grad=True)
    
                tsg = left_term - right_term                 
            
        return tsg
    
    
    def compute_f_value(self, ul_vars, ml_vars, ll_vars, f_list): 
        """
        Computes the true function value
        """         
        f_list.append(self.func.f(ul_vars, y=None, z=None, y0=ml_vars, z0=ll_vars)) 
        return f_list


    def compute_fbar_value(self, ul_vars, ml_vars, ll_vars, fbar_list): 
        """
        Computes the true function value
        """ 
        fbar_list.append(self.func.fbar(ul_vars, ml_vars, z=None, z0=ll_vars)) 
        return fbar_list
    
    
    def main_algorithm(self): 
        """
        Main body of a bilevel stochastic algorithm
        """
        print('\nSTART Algorithm\n')
        
        # Initialize lists
        f1_val_list = [] # At each iteration
        f2_val_list = [] # At each iteration
        f3_val_list = [] # At each iteration
        f_list = [] # At each iteration
        fbar_list = [] # At each iteration
        time_list = [] # At each iteration
        
        f2_val_all_list = [] # After each update of the ml variables
        f3_val_all_list = [] # Current values of f3; After each update of the ll variables
        f3_val_opt_all_list = [] # Optimal f3 for current x and y; Before update of the ll variables
        f3_val_opt_counter_all_list = [] # Before update of the ll variables
        fbar_val_all_list = [] # Current values of fbar; After each update of the ml variables
        fbar_val_opt_all_list = [] # Optimal fbar for current x; Before update of the ml variables
        fbar_val_opt_counter_all_list = [] # Before update of the ml variables

        grad_f_list = [] # Current values of grad_f; At each iteration    
        grad_fbar_all_list = [] # Current values of grad_fbar for current x; After each update of the ml variables
        grad_f3_all_list = [] # Current values of grad_f3 for current x and y; After each update of the ll variables

        accuracy_list = [] # Only for machine learning problems
        
        # Initialize the variables
        ul_vars = self.func.prob.ul_vars
        ml_vars = self.func.prob.ml_vars
        ll_vars = self.func.prob.ll_vars

        # ul_vars = np.random.uniform(0, 1, (self.func.prob.x_dim, 1))*self.func.prob.ul_vars_init_constant
        # ml_vars = np.random.uniform(0, 1, (self.func.prob.y_dim, 1))*self.func.prob.ml_vars_init_constant
        # ll_vars = np.random.uniform(0, 1, (self.func.prob.z_dim, 1))*self.func.prob.ll_vars_init_constant
        
        # ul_vars = np.array([[-0.4]])
        # ml_vars = np.array([[-0.2]])
        # ll_vars = np.array([[-0.6]])

        if self.func.prob.is_machine_learning_problem:
            noise_val=self.advlearn_noise 
            std_dev_val=self.advlearn_std_dev   
            accuracy_metric = self.func.prob.mse_func    #mse_func #r_squared
                
            ## If minibatch approach, generate next minibatch
            if not self.func.prob.batch_dataset:
                self.func.prob.next_minibatch()
            
            if self.algo == "remove_ul":
                ## In our paper, we only need to remove the UL variables for the adversarial formulation
                ## We don't need to set the UL variable (regularization parameter) to zero because we want to keep the regularization term
                self.func.prob.remove_ul = True

            if self.algo == "remove_ml":
                if not self.func.prob.sato_formulation_min_max_min:
                    raise AssertionError("Removing the ML problem only makes sense for Sato formulation, which is not the formulation currently used.")
                ## We first switch to non-Sato formulation. Then, we remove the LL problem.
                ml_vars, ll_vars = ll_vars, ml_vars
                self.func.prob.y_dim, self.func.prob.z_dim = self.func.prob.z_dim, self.func.prob.y_dim
                self.func.prob.sato_formulation_min_max_min = False
                self.func.prob.remove_ll = True

            if self.algo == "remove_ll" or self.func.prob.remove_ll:
                if self.func.prob.sato_formulation_min_max_min:
                    raise AssertionError("Removing the LL problem only makes sense for non-Sato formulation, which is not the formulation currently used.")
                ## In our paper, we only need to remove the LL variables for the adversarial formulation when using the non-Sato formulation
                ## We need to set the LL variables (perturbation) to zero because they do not appear in the resulting bilevel formulation
                ll_vars = np.zeros_like(ll_vars)
                self.func.prob.remove_ll = True
                
            accuracy_list.append(accuracy_metric(ml_vars, ll_vars, noise=noise_val, std_dev=std_dev_val))

        # Compute the true function (only for plotting purposes)                
        if self.true_func == True: 
            f_list = self.compute_f_value(ul_vars, ml_vars, ll_vars, f_list=f_list) 
        if self.true_fbar == True:
            fbar_list = self.compute_fbar_value(ul_vars, ml_vars, ll_vars, fbar_list=fbar_list)
        
        f1_val_list.append(self.func.prob.f_1(ul_vars, ml_vars, ll_vars))
        f2_val_list.append(self.func.prob.f_2(ul_vars, ml_vars, ll_vars))
        f3_val_list.append(self.func.prob.f_3(ul_vars, ml_vars, ll_vars))
        f2_val_all_list.append(self.func.prob.f_2(ul_vars, ml_vars, ll_vars))
        f3_val_all_list.append(self.func.prob.f_3(ul_vars, ml_vars, ll_vars))
            
        # print('\n x0',ul_vars[0],'x1',ul_vars[1],'y0',ml_vars[0],'y1',ml_vars[1],'z0',ll_vars[0],'z1',ll_vars[1])
        
        self.ul_lr = self.ul_lr_init
        self.ml_lr = self.ml_lr_init
        self.ll_lr = self.ll_lr_init
        
        self.mlp_iters = self.mlp_iters_init
        self.llp_iters = self.llp_iters_init

        cur_time = time.time()
        
        end_time =  time.time() - cur_time #- time_true_func_eval_cumul
        time_list.append(end_time)        
        
        j = 1            
        for it in range(self.max_iter):

            # self.llp_iters = self.mlp_iters**3 * self.max_iter 
           
            # Check if we stop the algorithm based on time
            if not(self.use_stopping_iter) and (time.time() - cur_time >= self.stopping_time): 
                break

            if self.func.prob.is_machine_learning_problem:
                ## If minibatch approach, generate next minibatch
                if not self.func.prob.batch_dataset and it >= 2:
                    self.func.prob.next_minibatch()
                    
                accuracy_list.append(accuracy_metric(ml_vars, ll_vars, noise=noise_val, std_dev=std_dev_val))
                
            pre_f1_val = self.func.prob.f_1(ul_vars, ml_vars, ll_vars) 

            # print('\nPre ML upd: x0',ul_vars[0],'y0',ml_vars[0],'z0',ll_vars[0])
            # print('\nx0',ul_vars[0],'x1',ul_vars[1],'y0',ml_vars[0],'y1',ml_vars[1],'z0',ll_vars[0],'z1',ll_vars[1])

            # Update the ML variables(and update the LL variables for each ML update)
            ml_vars, ll_vars = self.update_mlp(ul_vars, ml_vars, ll_vars, f2_val_all_list, f3_val_all_list, f3_val_opt_all_list, f3_val_opt_counter_all_list, fbar_val_all_list, fbar_val_opt_all_list, fbar_val_opt_counter_all_list, grad_f3_all_list, grad_fbar_all_list) 
 
            # print('\nPost ML upd: x0',ul_vars[0],'y1',ml_vars[0],'z1',ll_vars[0])
            # print('\nx0',ul_vars[0],'x1',ul_vars[1],'y0',ml_vars[0],'y1',ml_vars[1],'z0',ll_vars[0],'z1',ll_vars[1])

            ## When removing the LL, we keep update_ulp and we modify the structure of the adjoint gradients of the UL and ML problems            
            if not self.algo == "remove_ul" and not self.func.prob.remove_ll:
                # Update the UL variables
                ul_vars, ml_vars, ll_vars  = self.update_ulp(ul_vars, ml_vars, ll_vars, grad_f_list) 
               
            f1_val_list.append(self.func.prob.f_1(ul_vars, ml_vars, ll_vars))            
            f2_val_list.append(self.func.prob.f_2(ul_vars, ml_vars, ll_vars))
            f3_val_list.append(self.func.prob.f_3(ul_vars, ml_vars, ll_vars))
            j += 1  

            # print('\nPost UL upd: x1',ul_vars[0],'y1',ml_vars[0],'z1',ll_vars[0],'\n')
            # print('\nx0',ul_vars[0],'x1',ul_vars[1],'y0',ml_vars[0],'y1',ml_vars[1],'z0',ll_vars[0],'z1',ll_vars[1],'\n')
            
            end_time =  time.time() - cur_time #- time_true_func_eval_cumul
            time_list.append(end_time) 


            time_true_func_eval_start_time = time.time()
            # Compute the true function (only for plotting purposes)                
            if self.true_func == True:                 
                f_list = self.compute_f_value(ul_vars, ml_vars, ll_vars, f_list=f_list) 
            if self.true_fbar == True:
                fbar_list = self.compute_fbar_value(ul_vars, ml_vars, ll_vars, fbar_list=fbar_list)
            time_true_func_eval = time.time() - time_true_func_eval_start_time  


            # Increasing accuracy strategy for the UL problem (if f1 does not improve, we increase the number of ml iterations)           
            if self.inc_acc == True:
             	if self.mlp_iters >= self.mlp_iters_max:
              		self.mlp_iters = self.mlp_iters_max
             	else:
                    post_obj_val = self.func.prob.f_1(ul_vars, ml_vars, ll_vars) 
                    obj_val_diff = abs(post_obj_val - pre_f1_val)
                    if obj_val_diff/max(abs(pre_f1_val), np.finfo(float).eps) <= self.inc_acc_threshold_f1:     ## Before April 28 2025: if obj_val_diff/abs(pre_f1_val) <= self.inc_acc_threshold_f1:         
                        self.mlp_iters += 1 
 

            if self.iprint >= 2:
                if self.true_func and self.true_fbar:
                    print("Algorithm: ",self.algo_full_name,' iter: ',it,' f_1: ',f"{f1_val_list[len(f1_val_list)-1]:.4g}",' f_2: ',f"{f2_val_list[len(f2_val_list)-1]:.4g}",' f_3: ',f"{f3_val_list[len(f3_val_list)-1]:.4g}",' f: ',f"{f_list[len(f_list)-1]:.4g}",' fbar: ',f"{fbar_list[len(fbar_list)-1]:.4g}",' #ML iters: ',self.mlp_iters,' #LL iters: ',self.llp_iters,' ul_lr: ',f"{self.ul_lr:.4g}",' ml_lr: ',f"{self.ml_lr:.4g}",' ll_lr: ',f"{self.ll_lr:.4g}")    
                elif self.true_func:
                    print("Algorithm: ",self.algo_full_name,' iter: ',it,' f_1: ',f"{f1_val_list[len(f1_val_list)-1]:.4g}",' f_2: ',f"{f2_val_list[len(f2_val_list)-1]:.4g}",' f_3: ',f"{f3_val_list[len(f3_val_list)-1]:.4g}",' f: ',f"{f_list[len(f_list)-1]:.4g}",' #ML iters: ',self.mlp_iters,' #LL iters: ',self.llp_iters,' ul_lr: ',f"{self.ul_lr:.4g}",' ml_lr: ',f"{self.ml_lr:.4g}",' ll_lr: ',f"{self.ll_lr:.4g}")    
                else:
                    print("Algorithm: ",self.algo_full_name,' iter: ',it,' f_1: ',f"{f1_val_list[len(f1_val_list)-1]:.4g}",' f_2: ',f"{f2_val_list[len(f2_val_list)-1]:.4g}",' f_3: ',f"{f3_val_list[len(f3_val_list)-1]:.4g}",' #ML iters: ',self.mlp_iters,' #LL iters: ',self.llp_iters,' ul_lr: ',self.ul_lr,' ml_lr: ',self.ml_lr,' ll_lr: ',self.ll_lr)   


            # Update the UL learning rate 
            if self.ul_stepsize_scheme == 0:
                self.ul_lr = self.ul_lr_init/j 
            elif self.ul_stepsize_scheme == 1:
                self.ul_lr = self.ul_lr_init
                
            # Re-initialize the ML and LL learning rates (it only matters for the decaying stepsize case, i.e., if self.ml_stepsize_scheme == 0 or self.ll_stepsize_scheme == 0)
            self.ml_lr = self.ml_lr_init                
            self.ll_lr = self.ll_lr_init
            
            
        if self.iprint >= 1:
            if self.true_func and self.true_fbar:
                print("Algorithm: ",self.algo_full_name,' f_1: ',f1_val_list[len(f1_val_list)-1],' f_2: ',f2_val_list[len(f2_val_list)-1],' f_3: ',f3_val_list[len(f3_val_list)-1],' f: ',f_list[len(f_list)-1],' fbar: ',fbar_list[len(fbar_list)-1],' time: ',time.time() - cur_time,' #ML iters: ',self.mlp_iters,' #LL iters: ',self.llp_iters,' ul_lr: ',self.ul_lr,' ml_lr: ',self.ml_lr,' ll_lr: ',self.ll_lr)    
            elif self.true_func:
                print("Algorithm: ",self.algo_full_name,' f_1: ',f1_val_list[len(f1_val_list)-1],' f_2: ',f2_val_list[len(f2_val_list)-1],' f_3: ',f3_val_list[len(f3_val_list)-1],' f: ',f_list[len(f_list)-1],' time: ',time.time() - cur_time,' #ML iters: ',self.mlp_iters,' #LL iters: ',self.llp_iters,' ul_lr: ',self.ul_lr,' ml_lr: ',self.ml_lr,' ll_lr: ',self.ll_lr)    
            else:
                print("Algorithm: ",self.algo_full_name,' f_1: ',f1_val_list[len(f1_val_list)-1],' f_2: ',f2_val_list[len(f2_val_list)-1],' f_3: ',f3_val_list[len(f3_val_list)-1],' time: ',time.time() - cur_time,' #ML iters: ',self.mlp_iters,' #LL iters: ',self.llp_iters,' ul_lr: ',self.ul_lr,' ml_lr: ',self.ml_lr,' ll_lr: ',self.ll_lr)    


        ## This is to set self.func.prob.sato_formulation_min_max_min back to True and avoid the assertion error when number of replications is > 1  
        if self.func.prob.is_machine_learning_problem:
            if self.algo == "remove_ml":
                self.func.prob.sato_formulation_min_max_min = True

        return [f1_val_list, f_list, time_list, f2_val_all_list, fbar_val_all_list, fbar_val_opt_all_list, fbar_val_opt_counter_all_list, f3_val_all_list, f3_val_opt_all_list, f3_val_opt_counter_all_list, grad_f_list, grad_fbar_all_list, grad_f3_all_list, accuracy_list] 


    # def main_algorithm_machine_learning(self): 
    #     """
    #     Main body of a bilevel stochastic algorithm
    #     """
    #     print('\nSTART Algorithm\n')
        
    #     # Initialize lists
    #     f1_val_list = [] # At each iteration
    #     f2_val_list = [] # At each iteration
    #     f3_val_list = [] # At each iteration
    #     f_list = [] # At each iteration
    #     fbar_list = [] # At each iteration
    #     time_list = [] # At each iteration
        
    #     f2_val_all_list = [] # After each update of the ml variables
    #     f3_val_all_list = [] # Current values of f3; After each update of the ll variables
    #     f3_val_opt_all_list = [] # Optimal f3 for current x and y; Before update of the ll variables
    #     f3_val_opt_counter_all_list = [] # Before update of the ll variables
    #     fbar_val_all_list = [] # Current values of fbar; After each update of the ml variables
    #     fbar_val_opt_all_list = [] # Optimal fbar for current x; Before update of the ml variables
    #     fbar_val_opt_counter_all_list = [] # Before update of the ml variables

    #     grad_f_list = [] # Current values of grad_f; At each iteration    
    #     grad_fbar_all_list = [] # Current values of grad_fbar for current x; After each update of the ml variables
    #     grad_f3_all_list = [] # Current values of grad_f3 for current x and y; After each update of the ll variables

        
    #     # Initialize the variables
    #     ul_vars = self.func.prob.ul_vars
    #     ml_vars = self.func.prob.ml_vars
    #     ll_vars = self.func.prob.ll_vars

        
    #     # Compute the true function (only for plotting purposes)                
    #     if self.true_func == True: 
    #         f_list = self.compute_f_value(ul_vars, ml_vars, ll_vars, f_list=f_list) 
    #     if self.true_fbar == True:
    #         fbar_list = self.compute_fbar_value(ul_vars, ml_vars, ll_vars, fbar_list=fbar_list)
        
    #     f1_val_list.append(self.func.prob.f_1(ul_vars, ml_vars, ll_vars))
    #     f2_val_list.append(self.func.prob.f_2(ul_vars, ml_vars, ll_vars))
    #     f3_val_list.append(self.func.prob.f_3(ul_vars, ml_vars, ll_vars))
    #     f2_val_all_list.append(self.func.prob.f_2(ul_vars, ml_vars, ll_vars))
    #     f3_val_all_list.append(self.func.prob.f_3(ul_vars, ml_vars, ll_vars))
            
    #     # print('\n x0',ul_vars[0],'x1',ul_vars[1],'y0',ml_vars[0],'y1',ml_vars[1],'z0',ll_vars[0],'z1',ll_vars[1])
        
    #     self.ul_lr = self.ul_lr_init
    #     self.ml_lr = self.ml_lr_init
    #     self.ll_lr = self.ll_lr_init
        
    #     self.mlp_iters = self.mlp_iters_init
    #     self.llp_iters = self.llp_iters_init

    #     cur_time = time.time()
        
    #     end_time =  time.time() - cur_time #- time_true_func_eval_cumul
    #     time_list.append(end_time)        
        
    #     j = 1            
    #     for it in range(self.max_iter):
    #         # self.llp_iters = self.mlp_iters**3 * self.max_iter 
            
    #         # Check if we stop the algorithm based on time
    #         if not(self.use_stopping_iter) and (time.time() - cur_time >= self.stopping_time): 
    #             break
            
    #         pre_f1_val = self.func.prob.f_1(ul_vars, ml_vars, ll_vars) 

    #         # print('\nPre ML upd: x0',ul_vars[0],'y0',ml_vars[0],'z0',ll_vars[0])
    #         # print('\nx0',ul_vars[0],'x1',ul_vars[1],'y0',ml_vars[0],'y1',ml_vars[1],'z0',ll_vars[0],'z1',ll_vars[1])

    #         # Update the ML variables(and update the LL variables for each ML update)
    #         ml_vars, ll_vars = self.update_mlp(ul_vars, ml_vars, ll_vars, f2_val_all_list, f3_val_all_list, f3_val_opt_all_list, f3_val_opt_counter_all_list, fbar_val_all_list, fbar_val_opt_all_list, fbar_val_opt_counter_all_list, grad_f3_all_list, grad_fbar_all_list) 

    #         # print('\nPost ML upd: x0',ul_vars[0],'y1',ml_vars[0],'z1',ll_vars[0])
    #         # print('\nx0',ul_vars[0],'x1',ul_vars[1],'y0',ml_vars[0],'y1',ml_vars[1],'z0',ll_vars[0],'z1',ll_vars[1])
            
    #         # Update the UL variables
    #         ul_vars, ml_vars, ll_vars  = self.update_ulp(ul_vars, ml_vars, ll_vars, grad_f_list) 
    #         f1_val_list.append(self.func.prob.f_1(ul_vars, ml_vars, ll_vars))            
    #         f2_val_list.append(self.func.prob.f_2(ul_vars, ml_vars, ll_vars))
    #         f3_val_list.append(self.func.prob.f_3(ul_vars, ml_vars, ll_vars))
    #         j += 1  

    #         # print('\nPost UL upd: x1',ul_vars[0],'y1',ml_vars[0],'z1',ll_vars[0],'\n')
    #         # print('\nx0',ul_vars[0],'x1',ul_vars[1],'y0',ml_vars[0],'y1',ml_vars[1],'z0',ll_vars[0],'z1',ll_vars[1],'\n')
            
    #         end_time =  time.time() - cur_time #- time_true_func_eval_cumul
    #         time_list.append(end_time) 


    #         time_true_func_eval_start_time = time.time()
    #         # Compute the true function (only for plotting purposes)                
    #         if self.true_func == True:                 
    #             f_list = self.compute_f_value(ul_vars, ml_vars, ll_vars, f_list=f_list) 
    #         if self.true_fbar == True:
    #             fbar_list = self.compute_fbar_value(ul_vars, ml_vars, ll_vars, fbar_list=fbar_list)
    #         time_true_func_eval = time.time() - time_true_func_eval_start_time  


    #         # Increasing accuracy strategy for the UL problem (if f1 does not improve, we increase the number of ml iterations)           
    #         if self.inc_acc == True:
    #          	if self.mlp_iters >= self.mlp_iters_max:
    #           		self.mlp_iters = self.mlp_iters_max
    #          	else:
    #                 post_obj_val = self.func.prob.f_1(ul_vars, ml_vars, ll_vars) 
    #                 obj_val_diff = abs(post_obj_val - pre_f1_val)
    #                 if obj_val_diff/abs(pre_f1_val) <= self.inc_acc_threshold_f1:              
    #                     self.mlp_iters += 1 
 

    #         if self.iprint >= 2:
    #             if self.true_func and self.true_fbar:
    #                 print("Algorithm: ",self.algo_full_name,' iter: ',it,' f_1: ',f"{f1_val_list[len(f1_val_list)-1]:.4g}",' f_2: ',f"{f2_val_list[len(f2_val_list)-1]:.4g}",' f_3: ',f"{f3_val_list[len(f3_val_list)-1]:.4g}",' f: ',f"{f_list[len(f_list)-1]:.4g}",' fbar: ',f"{fbar_list[len(fbar_list)-1]:.4g}",' #ML iters: ',self.mlp_iters,' #LL iters: ',self.llp_iters,' ul_lr: ',f"{self.ul_lr:.4g}",' ml_lr: ',f"{self.ml_lr:.4g}",' ll_lr: ',f"{self.ll_lr:.4g}")    
    #             elif self.true_func:
    #                 print("Algorithm: ",self.algo_full_name,' iter: ',it,' f_1: ',f"{f1_val_list[len(f1_val_list)-1]:.4g}",' f_2: ',f"{f2_val_list[len(f2_val_list)-1]:.4g}",' f_3: ',f"{f3_val_list[len(f3_val_list)-1]:.4g}",' f: ',f"{f_list[len(f_list)-1]:.4g}",' #ML iters: ',self.mlp_iters,' #LL iters: ',self.llp_iters,' ul_lr: ',f"{self.ul_lr:.4g}",' ml_lr: ',f"{self.ml_lr:.4g}",' ll_lr: ',f"{self.ll_lr:.4g}")    
    #             else:
    #                 print("Algorithm: ",self.algo_full_name,' iter: ',it,' f_1: ',f"{f1_val_list[len(f1_val_list)-1]:.4g}",' f_2: ',f"{f2_val_list[len(f2_val_list)-1]:.4g}",' f_3: ',f"{f3_val_list[len(f3_val_list)-1]:.4g}",' #ML iters: ',self.mlp_iters,' #LL iters: ',self.llp_iters,' ul_lr: ',self.ul_lr,' ml_lr: ',self.ml_lr,' ll_lr: ',self.ll_lr)   


    #         # Update the UL learning rate 
    #         if self.ul_stepsize_scheme == 0:
    #             self.ul_lr = self.ul_lr_init/j 
    #         elif self.ul_stepsize_scheme == 1:
    #             self.ul_lr = self.ul_lr_init
                
    #         # Re-initialize the ML and LL learning rates (it only matters for the decaying stepsize case, i.e., if self.ml_stepsize_scheme == 0 or self.ll_stepsize_scheme == 0)
    #         self.ml_lr = self.ml_lr_init                
    #         self.ll_lr = self.ll_lr_init
            
            
    #     if self.iprint >= 1:
    #         if self.true_func and self.true_fbar:
    #             print("Algorithm: ",self.algo_full_name,' f_1: ',f1_val_list[len(f1_val_list)-1],' f_2: ',f2_val_list[len(f2_val_list)-1],' f_3: ',f3_val_list[len(f3_val_list)-1],' f: ',f_list[len(f_list)-1],' fbar: ',fbar_list[len(fbar_list)-1],' time: ',time.time() - cur_time,' #ML iters: ',self.mlp_iters,' #LL iters: ',self.llp_iters,' ul_lr: ',self.ul_lr,' ml_lr: ',self.ml_lr,' ll_lr: ',self.ll_lr)    
    #         elif self.true_func:
    #             print("Algorithm: ",self.algo_full_name,' f_1: ',f1_val_list[len(f1_val_list)-1],' f_2: ',f2_val_list[len(f2_val_list)-1],' f_3: ',f3_val_list[len(f3_val_list)-1],' f: ',f_list[len(f_list)-1],' time: ',time.time() - cur_time,' #ML iters: ',self.mlp_iters,' #LL iters: ',self.llp_iters,' ul_lr: ',self.ul_lr,' ml_lr: ',self.ml_lr,' ll_lr: ',self.ll_lr)    
    #         else:
    #             print("Algorithm: ",self.algo_full_name,' f_1: ',f1_val_list[len(f1_val_list)-1],' f_2: ',f2_val_list[len(f2_val_list)-1],' f_3: ',f3_val_list[len(f3_val_list)-1],' time: ',time.time() - cur_time,' #ML iters: ',self.mlp_iters,' #LL iters: ',self.llp_iters,' ul_lr: ',self.ul_lr,' ml_lr: ',self.ml_lr,' ll_lr: ',self.ll_lr)    

    #     return [f1_val_list, f_list, time_list, f2_val_all_list, fbar_val_all_list, fbar_val_opt_all_list, fbar_val_opt_counter_all_list, f3_val_all_list, f3_val_opt_all_list, f3_val_opt_counter_all_list, grad_f_list, grad_fbar_all_list, grad_f3_all_list] 
    
    
    def piecewise_func(self, x, boundaries, func_val):
        """
        Computes the value of a piecewise constant function defined by boundaries and func_val at x
        """
        for i in range(len(boundaries)):
            if x <= boundaries[i]:
              return func_val[i]
        return func_val[len(boundaries)-1]
    
    
    def main_algorithm_avg_ci(self, num_rep=1):
        """
        Returns arrays with averages and 95% confidence interval half-widths for function values or true function values at each iteration obtained over multiple runs
        """
        self.set_seed(self.seed) 
        
        # Solve the problem for the first time
        sol = self.main_algorithm()
        
        values = sol[0]
        true_func_values = sol[1]
        times = sol[2]
        accuracy_values = sol[13]
        
        values_rep = np.zeros((len(values),num_rep))
        values_rep[:,0] = np.asarray(values) 
        accuracy_values_rep = np.zeros((len(accuracy_values),num_rep)) 
        accuracy_values_rep[:,0] = np.asarray(accuracy_values)
        
        if self.true_func:
            true_func_values_rep = np.zeros((len(true_func_values),num_rep))
            true_func_values_rep[:,0] = np.asarray(true_func_values)
        
        # Solve the problem num_rep-1 times
        for i in range(num_rep-1):
          self.set_seed(self.seed+1+i) 
          sol = self.main_algorithm()
          if self.use_stopping_iter:
              values_rep[:,i+1] = np.asarray(sol[0])
              if self.func.prob.is_machine_learning_problem:
                  accuracy_values_rep[:,i+1] = np.asarray(sol[13])
              if self.true_func:
                  true_func_values_rep[:,i+1] = np.asarray(sol[1])
          else:
              values_rep[:,i+1] = list(map(lambda x: self.piecewise_func(x,sol[2],sol[0]),times))
              if self.func.prob.is_machine_learning_problem:
                  accuracy_values_rep[:,i+1] = list(map(lambda x: self.piecewise_func(x,sol[2],sol[13]),times))
              if self.true_func:
                  true_func_values_rep[:,i+1] = list(map(lambda x: self.piecewise_func(x,sol[2],sol[1]),times))
        
        values_avg = np.mean(values_rep, axis=1)
        values_ci = t.ppf(0.975,num_rep-1)*(np.sqrt(np.var(values_rep, ddof=1, axis=1))/np.sqrt(num_rep))
        
        if self.func.prob.is_machine_learning_problem:
            accuracy_values_avg = np.mean(accuracy_values_rep, axis=1) 
            accuracy_values_ci = t.ppf(0.975,num_rep-1)*(np.sqrt(np.var(accuracy_values_rep, ddof=1, axis=1))/np.sqrt(num_rep))
        else: 
            accuracy_values_avg = []
            accuracy_values_ci = []
        
        if self.true_func:
            true_func_values_avg = np.mean(true_func_values_rep, axis=1)
            true_func_values_ci = t.ppf(0.975,num_rep-1)*(np.sqrt(np.var(true_func_values_rep, ddof=1, axis=1))/np.sqrt(num_rep))
        else:
            true_func_values_avg = []
            true_func_values_ci = []
        
        # Only for the first replication    
        aux_lists = np.asarray(sol[3]), np.asarray(sol[4]), np.asarray(sol[5]), np.asarray(sol[6]), np.asarray(sol[7]), np.asarray(sol[8]), np.asarray(sol[9]), np.asarray(sol[10]), np.asarray(sol[11]), np.asarray(sol[12])
            
        return values_avg, values_ci, true_func_values_avg, true_func_values_ci, times, aux_lists, accuracy_values_avg, accuracy_values_ci


















