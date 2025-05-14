import os
import numpy as np
import random
import math
import pandas as pd

from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_url
from torchvision.transforms import ToTensor
from torch.utils.data.sampler import SubsetRandomSampler

from sklearn.datasets import make_spd_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing

import tarfile
import time



def set_seed(seed):
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


class TrilevelProblem:
    """
    Class used to define synthetic quadratic trilevel problems.

    Attributes
        x_dim:                        Dimension of the upper-level problem
        y_dim:                        Dimension of the middle-level problem 
        z_dim:                        Dimension of the lower-level problem
        std_dev:                      Standard deviation of the upper-level stochastic gradient estimates 
        ml_std_dev:                   Standard deviation of the middle-level stochastic gradient estimates
        ll_std_dev:                   Standard deviation of the lower-level stochastic gradient estimates
        ml_hess_std_dev:              Standard deviation of the middle-level stochastic Hessian estimates
        ll_hess_std_dev:              Standard deviation of the lower-level stochastic Hessian estimates
        prob:                         Class representing the trilevel problem we aim to solve 
        name_prob_to_run (str):       A string representing the name of the problem we aim to solve
        seed (int, optional):         The seed used for the experiments (default 42)
    """
    
    def __init__(self, name_prob_to_run, seed=42):
        
        self.seed = seed
        
        set_seed(self.seed)

        if name_prob_to_run == "Quadratic1":
            ## Deterministic case, unidimensional
            self.prob = Quadratic(x_dim=1, y_dim=1, z_dim=1, std_dev=0, ml_std_dev=0, ll_std_dev=0, ml_hess_std_dev=0, ll_hess_std_dev=0, seed=self.seed, configuration=1)
            self.name_prob_to_run = "Quadratic1"

        if name_prob_to_run == "Quadratic2":
            ## Deterministic case, multidimensional
            self.prob = Quadratic(x_dim=50, y_dim=50, z_dim=50, std_dev=0, ml_std_dev=0, ll_std_dev=0, ml_hess_std_dev=0, ll_hess_std_dev=0, seed=self.seed, configuration=2)
            self.name_prob_to_run = "Quadratic2"

        if name_prob_to_run == "Quadratic3":
            ## Stochastic case, multidimensional, low noise
            self.prob = Quadratic(x_dim=50, y_dim=50, z_dim=50, std_dev=1, ml_std_dev=1, ll_std_dev=0.2, ml_hess_std_dev=0.1, ll_hess_std_dev=0, seed=self.seed, configuration=3)
            self.name_prob_to_run = "Quadratic3"

        if name_prob_to_run == "Quadratic4":
            ## Stochastic case, multidimensional, high noise
            # self.prob = Quadratic(x_dim=50, y_dim=50, z_dim=50, std_dev=1, ml_std_dev=1, ll_std_dev=0.5, ml_hess_std_dev=0.1, ll_hess_std_dev=0.1, seed=self.seed, configuration=3)
            self.prob = Quadratic(x_dim=50, y_dim=50, z_dim=50, std_dev=1, ml_std_dev=1, ll_std_dev=1.75, ml_hess_std_dev=0.1, ll_hess_std_dev=0.1, seed=self.seed, configuration=3)
            self.name_prob_to_run = "Quadratic4"
            
            
            
        elif name_prob_to_run == "Quartic1":
            ## Deterministic case, unidimensional
            self.prob = Quartic(x_dim=1, y_dim=1, z_dim=1, std_dev=0, ml_std_dev=0, ll_std_dev=0, ml_hess_std_dev=0, ll_hess_std_dev=0, seed=self.seed, configuration=1)
            self.name_prob_to_run = "Quartic1"

        elif name_prob_to_run == "Quartic2":
            ## Deterministic case, multidimensional
            self.prob = Quartic(x_dim=5, y_dim=5, z_dim=1, std_dev=0, ml_std_dev=0, ll_std_dev=0, ml_hess_std_dev=0, ll_hess_std_dev=0, seed=self.seed, configuration=2)
            self.name_prob_to_run = "Quartic2"

        elif name_prob_to_run == "Quartic3":
            ## Stochastic case, multidimensional, low noise
            self.prob = Quartic(x_dim=5, y_dim=5, z_dim=1, std_dev=0.01, ml_std_dev=0.01, ll_std_dev=0.000001, ml_hess_std_dev=0.1, ll_hess_std_dev=0, seed=self.seed, configuration=3)
            self.name_prob_to_run = "Quartic3"

        elif name_prob_to_run == "Quartic4":
            ## Stochastic case, multidimensional, high noise
            self.prob = Quartic(x_dim=5, y_dim=5, z_dim=1, std_dev=0.02, ml_std_dev=0.02, ll_std_dev=0.000002, ml_hess_std_dev=0.1, ll_hess_std_dev=0, seed=self.seed, configuration=3)
            self.name_prob_to_run = "Quartic4"
            
            

        elif name_prob_to_run == "AdversarialLearning1":
            ## Red wine, non-Sato formulation   
            self.prob = AdversarialLearning(seed=self.seed, configuration=1)
            self.name_prob_to_run = "AdversarialLearning1"

        elif name_prob_to_run == "AdversarialLearning2":
            ## Red wine, Sato formulation
            self.prob = AdversarialLearning(seed=self.seed, configuration=2)
            self.name_prob_to_run = "AdversarialLearning2"
            
            

        elif name_prob_to_run == "AdversarialLearning3":
            ## White wine, non-Sato formulation
            self.prob = AdversarialLearning(seed=self.seed, configuration=3)
            self.name_prob_to_run = "AdversarialLearning3"
            
            

        elif name_prob_to_run == "AdversarialLearning5":
            ## California, non-Sato formulation
            self.prob = AdversarialLearning(seed=self.seed, configuration=5)
            self.name_prob_to_run = "AdversarialLearning5"
            
            

    def compute_args(self, x, y=None, z=None, y0=None, z0=None):
        """
        Compute the arguments for exact and inexact functions.
        When the input is (x, y) or (x, y, z0=z), it will return z(x,y) using z0 as a starting point (if z0 is not provided, z0 will be randomly generated).
        When the input is (x), (x, y0=y), (x, z0=z), (x, y0=y, z0=z), it will return y(x) using y0 as a starting point; z0 will be used as a starting point to find z(x,y); (if y0 and z0 are not provided, they will be randomly generated).   
        """
        if z is None:
            
            if z0 is None:
                z = np.random.uniform(0, 1, (self.prob.z_dim, 1))*0.1
            else:
                z = z0
                
            if y is None:
                if y0 is None:
                    y = np.random.uniform(0, 1, (self.prob.y_dim, 1))*0.1
                else:
                    y = y0
                
                if self.prob.y_opt_available_analytically:
                    y = self.prob.y_opt(x)
                else:
                    y = self.y_opt(x, y, z)

            if self.prob.z_opt_available_analytically:
                z = self.prob.z_opt(x, y)
            else:                
                z = self.z_opt(x, y, z)
            
        return x, y, z


    def f(self, x, y=None, z=None, y0=None, z0=None):
        """
        The true objective function of the trilevel problem;
        y0 is used as a starting point to find y(x); z0 is used as a starting point to find z(x,y).
       	"""
        args = self.compute_args(x, y, z, y0=y0, z0=z0)
        return self.prob.f_1(*args)


    def fbar(self, x, y, z=None, z0=None):
        """
        The true objective function of the middle-level problem;
        z0 is used as a starting point to find z(x,y).
       	"""
        args = self.compute_args(x, y, z, z0=z0)
        return self.prob.f_2(*args)


    def z_opt(self, x, y, z):
        """
        The approximate optimal solution of the lower-level problem as a function of x and y (i.e., z(x,y)); 
        z is the current value of the ll vars and it is used as a starting point.
    	"""
        result = minimize(lambda z_var: self.prob.f_3(x, y, z_var.reshape(-1,1)), x0=z.flatten(), method='BFGS', options={'disp': False})
        
        return result.x.reshape(-1,1)


    def y_opt(self, x, y, z):
        """
        The approximate optimal solution of the middle-level problem as a function of x; 
        y and z are the current values of the ml and ll vars, respectively; 
        y is used as a starting point to find y(x); z is used as a starting point to find z(x,y).
        """
        result = minimize(lambda y_var: self.fbar(x, y_var.reshape(-1,1), None, z0=z), x0=y.flatten(), method='BFGS', options={'disp': False})
        
        # print('yopt1:',result.x.reshape(-1,1))
        # out = 2*np.linalg.inv((1 + np.matmul(self.prob.hess_f3_ml_vars_ll_vars(x, y, z),np.linalg.inv(self.prob.hess_f3_ll_vars_ll_vars(x, y, z)))))*x
        # print('yopt2:',out)
        # out = np.matmul(np.matmul(np.linalg.inv(self.prob.Hyy),self.prob.Hyx),x)
        # print('yopt3:',out)
        
        return result.x.reshape(-1,1)


    def f_opt(self):
        """
        The optimal value of the trilevel problem
    	  """
        return self.f(self.prob.x_opt())
    
    
class Quadratic:
    """
    Class used to define the following synthetic quadratic trilevel problem:
        min_{x} f_1(x,y,z) =  hx'x + hy'y + hz'z + 0.5 x'Hxx x + x'Hxy y + x'Hxz z
            s.t. y = argmin_{y} f_2(x,y,z) = 0.5 y'Hyy y - y'Hyx x - y'Hyz z
                s.t. z = argmin_{z} f_3(x,y,z) = 0.5 z'Hzz z - z'Hzx x - z'Hzy y
    """
    
    
    def __init__(self, x_dim, y_dim, z_dim, std_dev, ml_std_dev, ll_std_dev, ml_hess_std_dev, ll_hess_std_dev, seed=42, configuration=1):
        
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.std_dev = std_dev
        self.ml_std_dev = ml_std_dev
        self.ll_std_dev = ll_std_dev
        self.ml_hess_std_dev = ml_hess_std_dev
        self.ll_hess_std_dev = ll_hess_std_dev
        self.seed = seed
        
        set_seed(self.seed)

        self.z_opt_available_analytically = True
        self.y_opt_available_analytically = True
        self.x_opt_available_analytically = True
        self.is_machine_learning_problem = False

        self.remove_ul = False # If True, obtain a bilevel problem by removing the UL problem (only works correctly for adverarial learning)
        self.remove_ml = False # If True, obtain a bilevel problem by removing the LL problem (only works correctly for adverarial learning)
        self.remove_ll = False # If True, obtain a bilevel problem by removing the LL problem (only works correctly for adverarial learning)
            
        if configuration == 1:        
            self.hx = np.random.uniform(0,10,(self.x_dim,1)) 
            self.hy = np.random.uniform(0,10,(self.y_dim,1)) 
            self.hz = np.random.uniform(0,10,(self.z_dim,1)) 
    
            self.Hxx = np.eye(self.x_dim,self.x_dim)  #make_spd_matrix(self.x_dim, random_state=self.seed) 
            self.Hyy = 4*np.eye(self.y_dim,self.y_dim)  #make_spd_matrix(self.y_dim, random_state=self.seed+1)         
            self.Hzz = np.eye(self.z_dim,self.z_dim)  #make_spd_matrix(self.z_dim, random_state=self.seed+2) 
            self.Hxy = np.eye(self.x_dim,self.y_dim) 
            self.Hyx = self.Hxy.T
            self.Hxz = np.eye(self.x_dim,self.z_dim) 
            self.Hzx = self.Hxz.T
            self.Hyz = np.eye(self.y_dim,self.z_dim) 
            self.Hzy = self.Hyz.T
            
            self.ul_vars_init_constant = 20
            self.ml_vars_init_constant = 20
            self.ll_vars_init_constant = 20

            self.ul_vars = np.random.uniform(0, 1, (self.x_dim, 1))*self.ul_vars_init_constant
            self.ml_vars = np.random.uniform(0, 1, (self.y_dim, 1))*self.ml_vars_init_constant
            self.ll_vars = np.random.uniform(0, 1, (self.z_dim, 1))*self.ll_vars_init_constant
            
        elif configuration == 2:        
            self.hx = np.random.uniform(0,10,(self.x_dim,1)) 
            self.hy = np.random.uniform(0,10,(self.y_dim,1)) 
            self.hz = np.random.uniform(0,10,(self.z_dim,1)) 
    
            self.Hxx = np.eye(self.x_dim,self.x_dim) #make_spd_matrix(self.x_dim, random_state=self.seed+0)
            self.Hyy = 4*np.eye(self.y_dim,self.y_dim)         
            self.Hzz = np.eye(self.z_dim,self.z_dim) #make_spd_matrix(self.z_dim, random_state=self.seed+2)
            self.Hxy = np.eye(self.x_dim,self.y_dim) 
            self.Hyx = self.Hxy.T
            self.Hxz = np.eye(self.x_dim,self.z_dim) 
            self.Hzx = self.Hxz.T
            self.Hyz = np.eye(self.y_dim,self.z_dim) 
            self.Hzy = self.Hyz.T
            
            self.ul_vars_init_constant = 20
            self.ml_vars_init_constant = 20
            self.ll_vars_init_constant = 20

            self.ul_vars = np.random.uniform(0, 1, (self.x_dim, 1))*self.ul_vars_init_constant
            self.ml_vars = np.random.uniform(0, 1, (self.y_dim, 1))*self.ml_vars_init_constant
            self.ll_vars = np.random.uniform(0, 1, (self.z_dim, 1))*self.ll_vars_init_constant
            
        elif configuration == 3:        
            self.hx = np.random.uniform(0,10,(self.x_dim,1)) 
            self.hy = np.random.uniform(0,10,(self.y_dim,1)) 
            self.hz = np.random.uniform(0,10,(self.z_dim,1)) 
    
            self.Hxx = np.eye(self.x_dim,self.x_dim) #make_spd_matrix(self.x_dim, random_state=self.seed+0)
            self.Hyy = 4*np.eye(self.y_dim,self.y_dim) #make_spd_matrix(self.y_dim, random_state=self.seed+1)         
            self.Hzz = np.eye(self.z_dim,self.z_dim) #make_spd_matrix(self.z_dim, random_state=self.seed+2)
            self.Hxy = np.eye(self.x_dim,self.y_dim) 
            self.Hyx = self.Hxy.T
            self.Hxz = np.eye(self.x_dim,self.z_dim) 
            self.Hzx = self.Hxz.T
            self.Hyz = np.eye(self.y_dim,self.z_dim) 
            self.Hzy = self.Hyz.T
            
            self.ul_vars_init_constant = 20
            self.ml_vars_init_constant = 20
            self.ll_vars_init_constant = 20

            self.ul_vars = np.random.uniform(0, 1, (self.x_dim, 1))*self.ul_vars_init_constant
            self.ml_vars = np.random.uniform(0, 1, (self.y_dim, 1))*self.ml_vars_init_constant
            self.ll_vars = np.random.uniform(0, 1, (self.z_dim, 1))*self.ll_vars_init_constant
            
        
    def z_opt(self, x, y):
        """
        The optimal solution of the lower-level problem as a function of x and y
    	"""
        z_opt = np.dot(np.linalg.inv(self.Hzz),np.dot(self.Hzx,x) + np.dot(self.Hzy,y))
        
        # print('\nz_opt',z_opt,' grad_f3_ll_vars: ',self.grad_f3_ll_vars(x, y, z_opt).T,'\n')
        
        return z_opt


    def y_opt(self, x):
        """
        The optimal solution of the middle-level problem as a function of x
    	"""
        inv_Hzz = np.linalg.inv(self.Hzz)
        aux_1 = self.Hyy - 2*np.dot(self.Hyz,np.dot(inv_Hzz,self.Hzy))
        aux_2 = self.Hyx + np.dot(self.Hyz,np.dot(inv_Hzz,self.Hzx))
        y_opt = np.dot(np.linalg.inv(aux_1),np.dot(aux_2,x))
        
        # print('\ny_opt',y_opt,' grad_f2_ll_vars: ',self.grad_f2_ml_vars(x, y_opt, self.z_opt(x, y_opt)).T,'\n')
        
        return y_opt
        
    
    def x_opt(self):
        """
        The optimal value of the trilevel problem
    	    """
        inv_Hzz = np.linalg.inv(self.Hzz)
        
        aux_1 = self.Hyy - 2*np.dot(self.Hyz,np.dot(inv_Hzz,self.Hzy))
        aux_2 = self.Hyx + np.dot(self.Hyz,np.dot(inv_Hzz,self.Hzx))
        
        aux_12 = np.dot(np.linalg.inv(aux_1),aux_2)
        
        aux_3 = np.dot(inv_Hzz, self.Hzx + np.dot(self.Hzy,aux_12))
        aux_4 = self.Hyx + np.dot(self.Hyz,np.dot(inv_Hzz,self.Hzx))
        
        aux_E = self.Hxx + np.dot(self.Hxy,aux_12) + np.dot(self.Hxz,aux_3) +\
            np.dot(self.Hxz,np.dot(inv_Hzz,self.Hzx)) + np.dot(aux_12.T,aux_4)
        
        aux_D = self.hx + np.dot(self.Hxz,np.dot(inv_Hzz,self.hz)) + np.dot(aux_12.T,self.hy + np.dot(self.Hyz,np.dot(inv_Hzz,self.hz)))

        x_opt = - np.dot(np.linalg.inv(aux_E),aux_D)
        
        # print('\nx_opt',x_opt,' grad_f1_ul_vars: ',self.grad_f1_ul_vars(x_opt, self.y_opt(x_opt), self.z_opt(x_opt, self.y_opt(x_opt))).T,'\n')
        
        return x_opt   
    
    
    def f_1(self, x, y, z):
        """
        The upper-level objective function
    	"""
        out = np.dot(self.hx.T,x) + np.dot(self.hy.T,y) + np.dot(self.hz.T,z) +\
            0.5*np.dot(x.T,np.dot(self.Hxx,x)) + np.dot(x.T,np.dot(self.Hxy,y)) + np.dot(x.T,np.dot(self.Hxz,z))
        return np.squeeze(out)
    

    def f_2(self, x, y, z):
        """
        The middle-level objective function
    	"""
        out = 0.5*np.dot(y.T,np.dot(self.Hyy,y)) - np.dot(y.T,np.dot(self.Hyx,x)) - np.dot(y.T,np.dot(self.Hyz,z)) 
        return np.squeeze(out)


    def f_3(self, x, y, z):
        """
        The lower-level objective function
    	"""
        out = 0.5*np.dot(z.T,np.dot(self.Hzz,z)) - np.dot(z.T,np.dot(self.Hzx,x)) - np.dot(z.T,np.dot(self.Hzy,y))
        return np.squeeze(out)
    

    def grad_f1_ul_vars(self, x, y, z):
        """
        The gradient of the upper-level objective function wrt the upper-level variables
    	"""
        out = self.hx  + np.dot(self.Hxx,x) + np.dot(self.Hxy,y) + np.dot(self.Hxz,z)
        out = out + self.std_dev*np.random.randn(out.shape[0],out.shape[1])
        return out


    def grad_f1_ml_vars(self, x, y, z):
        """
        The gradient of the upper-level objective function wrt the middle-level variables
    	"""
        out = self.hy  + np.dot(self.Hyx,x) 
        out = out + self.std_dev*np.random.randn(out.shape[0],out.shape[1])
        return out


    def grad_f1_ll_vars(self, x, y, z):
        """
        The gradient of the upper-level objective function wrt the lower-level variables
    	"""
        out = self.hz  + np.dot(self.Hzx,x) 
        out = out + self.std_dev*np.random.randn(out.shape[0],out.shape[1])
        return out


    def grad_f2_ul_vars(self, x, y, z):
        """
        The gradient of the middle-level objective function wrt the upper-level variables
    	"""
        out = - np.dot(self.Hxy,y)
        out = out + self.ml_std_dev*np.random.randn(out.shape[0],out.shape[1])
        return out
    

    def grad_f2_ml_vars(self, x, y, z):
        """
        The gradient of the middle-level objective function wrt the middle-level variables
    	"""
        out = np.dot(self.Hyy,y) - np.dot(self.Hyx,x) - np.dot(self.Hyz,z)
        out = out + self.ml_std_dev*np.random.randn(out.shape[0],out.shape[1])
        return out


    def grad_f2_ml_vars_torch(self, x, y, z):
        """
        The gradient of the middle-level objective function wrt the middle-level variables
    	"""
        out = torch.matmul(torch.tensor(self.Hyy, dtype=torch.float64),y) - torch.matmul(torch.tensor(self.Hyx, dtype=torch.float64),x) - torch.matmul(torch.tensor(self.Hyz, dtype=torch.float64),z)
        out = out + self.ml_std_dev*torch.randn(out.shape[0],out.shape[1])
        return out
    

    def grad_f2_ll_vars(self, x, y, z):
        """
        The gradient of the middle-level objective function wrt the lower-level variables
    	"""
        out = -np.dot(self.Hzy,y)
        out = out + self.ml_std_dev*np.random.randn(out.shape[0],out.shape[1])
        return out


    def grad_f2_ll_vars_torch(self, x, y, z):
        """
        The gradient of the middle-level objective function wrt the lower-level variables
    	"""
        out = torch.matmul(torch.tensor(self.Hzy, dtype=torch.float64),y)   
        out = out + self.ml_std_dev*torch.randn(out.shape[0],out.shape[1])
        return out
    

    def grad_f3_ul_vars(self, x, y, z):
        """
        The gradient of the lower-level objective function wrt the uppwe-level variables
    	"""
        out = -np.dot(self.Hxz,z)
        out = out + self.ll_std_dev*np.random.randn(out.shape[0],out.shape[1])
        return out
    
    
    def grad_f3_ml_vars(self, x, y, z):
        """
        The gradient of the lower-level objective function wrt the middle-level variables
    	"""
        out = -np.dot(self.Hyz,z)
        out = out + self.ll_std_dev*np.random.randn(out.shape[0],out.shape[1])
        return out
    

    def grad_f3_ll_vars(self, x, y, z):
        """
        The gradient of the lower-level objective function wrt the lower-level variables
    	"""
        out = np.dot(self.Hzz,z) - np.dot(self.Hzx,x) - np.dot(self.Hzy,y)
        out = out + self.ll_std_dev*np.random.randn(out.shape[0],out.shape[1])
        return out


    def grad_f3_ll_vars_torch(self, x, y, z):
        """
        The gradient of the lower-level objective function wrt the lower-level variables
     	"""
        out = torch.matmul(torch.tensor(self.Hzz, dtype=torch.float64),z) - torch.matmul(torch.tensor(self.Hzx, dtype=torch.float64),x) - torch.matmul(torch.tensor(self.Hzy, dtype=torch.float64),y) 
        out = out + self.ll_std_dev*torch.randn(out.shape[0],out.shape[1])
        return out        


    def hess_f2_ml_vars_ul_vars(self, x, y, z):
        """
        The Hessian of the middle-level objective function wrt the lower-level variables
    	"""
        out = -self.Hyx
        out = out + self.ml_hess_std_dev*np.random.randn(out.shape[0],out.shape[1])
        return out


    def hess_f2_ml_vars_ml_vars(self, x, y, z):
        """
        The Hessian of the middle-level objective function wrt the lower-level variables
    	"""
        out = self.Hyy
        out = out + self.ml_hess_std_dev*np.random.randn(out.shape[0],out.shape[1])
        return out
    

    def hess_f2_ml_vars_ll_vars(self, x, y, z):
        """
        The Hessian of the middle-level objective function wrt the lower-level variables
    	"""
        out = -self.Hyz
        out = out + self.ml_hess_std_dev*np.random.randn(out.shape[0],out.shape[1])
        return out


    def hess_f2_ll_vars_ul_vars(self, x, y, z):
        """
        The Hessian of the middle-level objective function wrt the lower-level variables
    	"""
        out = np.zeros((self.z_dim,self.x_dim))
        out = out + self.ml_hess_std_dev*np.random.randn(out.shape[0],out.shape[1])
        return out


    def hess_f2_ll_vars_ml_vars(self, x, y, z):
        """
        The Hessian of the middle-level objective function wrt the lower-level variables
    	"""
        out = -self.Hzy
        out = out + self.ml_hess_std_dev*np.random.randn(out.shape[0],out.shape[1])
        return out
    

    def hess_f2_ll_vars_ll_vars(self, x, y, z):
        """
        The Hessian of the middle-level objective function wrt the lower-level variables
    	"""
        out = np.zeros((self.z_dim,self.z_dim))
        out = out + self.ml_hess_std_dev*np.random.randn(out.shape[0],out.shape[1])
        return out
    

    def hess_f3_ul_vars_ll_vars(self, x, y, z):
        """
        The Hessian of the lower-level objective function wrt the lower-level variables
    	"""
        out = -self.Hxz
        out = out + self.ll_hess_std_dev*np.random.randn(out.shape[0],out.shape[1])
        return out
    

    def hess_f3_ml_vars_ll_vars(self, x, y, z):
        """
        The Hessian of the lower-level objective function wrt the middle and lower level variables
    	"""
        out = -self.Hyz
        out = out + self.ll_hess_std_dev*np.random.randn(out.shape[0],out.shape[1])
        return out


    def hess_f3_ll_vars_ul_vars(self, x, y, z):
        """
        The Hessian of the lower-level objective function wrt the lower-level variables
    	"""
        out = self.hess_f3_ul_vars_ll_vars(x, y, z).T
        out = out + self.ll_hess_std_dev*np.random.randn(out.shape[0],out.shape[1])
        return out


    def hess_f3_ll_vars_ml_vars(self, x, y, z):
        """
        The Hessian of the lower-level objective function wrt the lower-level variables
    	"""
        out = self.hess_f3_ml_vars_ll_vars(x, y, z).T
        out = out + self.ll_hess_std_dev*np.random.randn(out.shape[0],out.shape[1])
        return out
    

    def hess_f3_ll_vars_ll_vars(self, x, y, z):
        """
        The Hessian of the lower-level objective function wrt the lower-level variables
    	"""
        out = self.Hzz
        out = out + self.ll_hess_std_dev*np.random.randn(out.shape[0],out.shape[1])
        return out


    def trd_hess_f3_ml_vars_ll_vars_ll_vars(self, x, y, z): 
        out = np.zeros((self.z_dim,self.y_dim,self.z_dim))
        out = out + self.ll_hess_std_dev*np.random.randn(*out.shape)
        return out        


    def trd_hess_f3_ll_vars_ll_vars_ll_vars(self, x, y, z): 
        out = np.zeros((self.z_dim,self.z_dim,self.z_dim))
        out = out + self.ll_hess_std_dev*np.random.randn(*out.shape)
        return out          


    def trd_hess_f3_ml_vars_ll_vars_ul_vars(self, x, y, z): 
        out = np.zeros((self.x_dim,self.y_dim,self.z_dim))
        out = out + self.ll_hess_std_dev*np.random.randn(*out.shape)
        return out    


    def trd_hess_f3_ml_vars_ul_vars_ll_vars(self, x, y, z): 
        return np.transpose(self.trd_hess_f3_ml_vars_ll_vars_ul_vars(x, y, z), (2, 1, 0))
    

    def trd_hess_f3_ll_vars_ll_vars_ul_vars(self, x, y, z): 
        out = np.zeros((self.x_dim,self.z_dim,self.z_dim))
        out = out + self.ll_hess_std_dev*np.random.randn(*out.shape)
        return out    


    def trd_hess_f3_ll_vars_ul_vars_ll_vars(self, x, y, z): 
        return np.transpose(self.trd_hess_f3_ll_vars_ll_vars_ul_vars(x, y, z), (2, 1, 0)) 
    

    def trd_hess_f3_ml_vars_ll_vars_ml_vars(self, x, y, z): 
        out = np.zeros((self.y_dim,self.y_dim,self.z_dim))
        out = out + self.ll_hess_std_dev*np.random.randn(*out.shape)
        return out    


    def trd_hess_f3_ml_vars_ml_vars_ll_vars(self, x, y, z): 
        return np.transpose(self.trd_hess_f3_ml_vars_ll_vars_ml_vars(x, y, z), (2, 1, 0))    
    

    def trd_hess_f3_ll_vars_ll_vars_ml_vars(self, x, y, z): 
        out = np.zeros((self.y_dim,self.z_dim,self.z_dim))
        out = out + self.ll_hess_std_dev*np.random.randn(*out.shape)
        return out 


    def trd_hess_f3_ll_vars_ml_vars_ll_vars(self, x, y, z): 
        return np.transpose(self.trd_hess_f3_ll_vars_ll_vars_ml_vars(x, y, z), (2, 1, 0))               


class Quartic:
    """
    Class used to define the following synthetic quadratic trilevel problem:
        min_{x} f_1(x,y,z) =  hx'x + hy'y + hz'z + 0.5 x'Hxx x + x'Hxy y + x'Hxz z
            s.t. y = argmin_{y} f_2(x,y,z) = 0.5 y'Hyy y - y'Hyx x - y'Hyz z
                s.t. z = argmin_{z} f_3(x,y,z) = 0.5 ||z'Hzz z - z'Hzx x - z'Hzy y||^2

    Expanding f_3, we get that f_3 is given by the sum of these terms: a11 + 2*a12 + 2*a13 + a22 + 2*a23 + a33, where
    a11 = (z' Hzz z)^2, a12 = -(z' Hzz z)(z' Hzx x), a13 = -(z' Hzz z)(z' Hzy y),
    a21 = a12,          a22 = (z' Hzx x)^2,          a23 = (z' Hzx x)(z' Hzy y),                                                          
    a31 = a13,          a32 = a23,                   a33 = (z' Hzy y)^2    
    """    
    

    def __init__(self, x_dim, y_dim, z_dim, std_dev, ml_std_dev, ll_std_dev, ml_hess_std_dev, ll_hess_std_dev, seed=42, configuration=1):
        
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.std_dev = std_dev
        self.ml_std_dev = ml_std_dev
        self.ll_std_dev = ll_std_dev
        self.ml_hess_std_dev = ml_hess_std_dev
        self.ll_hess_std_dev = ll_hess_std_dev
        self.seed = seed
        
        set_seed(self.seed)
        self.is_machine_learning_problem = False

        self.remove_ul = False # If True, obtain a bilevel problem by removing the UL problem (only works correctly for adverarial learning)
        self.remove_ml = False # If True, obtain a bilevel problem by removing the LL problem (only works correctly for adverarial learning)
        self.remove_ll = False # If True, obtain a bilevel problem by removing the LL problem (only works correctly for adverarial learning)
        
        if configuration == 1:
            self.z_opt_available_analytically = True
            self.y_opt_available_analytically = True
            self.x_opt_available_analytically = True

            self.hx = np.array([1]) 
            self.hy = np.array([1]) 
            self.hz = np.array([1])
    
            self.Hxx = np.array([[1]]) 
            self.Hyy = np.array([[3]])         
            self.Hzz = np.array([[1]])
            self.Hxy = np.array([[1]]) 
            self.Hyx = np.array([[1]])
            self.Hxz = np.array([[1]]) 
            self.Hzx = np.array([[1]])
            self.Hyz = np.array([[1]]) 
            self.Hzy = np.array([[1]])

            # self.ul_vars_init_constant = -1
            # self.ml_vars_init_constant = -1
            # self.ll_vars_init_constant = -1

            # self.ul_vars = np.random.uniform(0, 1, (self.x_dim, 1))*self.ul_vars_init_constant
            # self.ml_vars = np.random.uniform(0, 1, (self.y_dim, 1))*self.ml_vars_init_constant
            # self.ll_vars = np.random.uniform(0, 1, (self.z_dim, 1))*self.ll_vars_init_constant

            self.ul_vars = np.array([[-0.4]])    
            self.ml_vars = np.array([[-0.2]])
            self.ll_vars = np.array([[-0.6]])
            
        elif configuration == 2:        
            self.z_opt_available_analytically = True
            self.y_opt_available_analytically = False
            self.x_opt_available_analytically = False

            self.hx = np.random.uniform(0,0.1,(self.x_dim,1)) 
            self.hy = np.random.uniform(0,0.1,(self.y_dim,1)) 
            self.hz = np.random.uniform(0,0.1,(self.z_dim,1))
    
            self.Hxx = np.eye(self.x_dim,self.x_dim) #make_spd_matrix(self.x_dim, random_state=self.seed)*0.1 
            self.Hyy = 4*np.eye(self.y_dim,self.y_dim) #make_spd_matrix(self.y_dim, random_state=self.seed+1)*0.1         
            self.Hzz = np.eye(self.z_dim,self.z_dim) #make_spd_matrix(self.z_dim, random_state=self.seed+2)*0.1 
            self.Hxy = np.eye(self.x_dim,self.y_dim) #np.random.rand(x_dim,y_dim)*0.1 #np.eye(self.x_dim,self.y_dim) 
            self.Hyx = self.Hxy.T
            self.Hxz = np.eye(self.x_dim,self.z_dim) #np.random.rand(x_dim,z_dim)*0.1 #np.eye(self.x_dim,self.z_dim) 
            self.Hzx = self.Hxz.T
            self.Hyz = np.eye(self.y_dim,self.z_dim) #np.random.rand(y_dim,z_dim)*0.1 #np.eye(self.y_dim,self.z_dim) 
            self.Hzy = self.Hyz.T

            self.ul_vars_init_constant = -0.40 #-0.1
            self.ml_vars_init_constant = -0.20 #-0.1
            self.ll_vars_init_constant = -0.60 #-0.1

            self.ul_vars = np.random.uniform(0, 1, (self.x_dim, 1))*self.ul_vars_init_constant
            self.ml_vars = np.random.uniform(0, 1, (self.y_dim, 1))*self.ml_vars_init_constant
            self.ll_vars = np.random.uniform(0, 1, (self.z_dim, 1))*self.ll_vars_init_constant
 
            # self.ul_vars = np.array([[-0.4]])*np.ones((self.x_dim, 1))    
            # self.ml_vars = np.array([[-0.2]])*np.ones((self.y_dim, 1)) 
            # self.ll_vars = np.array([[-0.6]])**np.ones((self.z_dim, 1)) 
            
        elif configuration == 3:        
            self.z_opt_available_analytically = True
            self.y_opt_available_analytically = False
            self.x_opt_available_analytically = False

            self.hx = np.random.uniform(0,0.1,(self.x_dim,1)) 
            self.hy = np.random.uniform(0,0.1,(self.y_dim,1)) 
            self.hz = np.random.uniform(0,0.1,(self.z_dim,1))
    
            self.Hxx = np.eye(self.x_dim,self.x_dim) #make_spd_matrix(self.x_dim, random_state=self.seed)*0.1 
            self.Hyy = 4*np.eye(self.y_dim,self.y_dim) #make_spd_matrix(self.y_dim, random_state=self.seed+1)*0.1         
            self.Hzz = np.eye(self.z_dim,self.z_dim) #make_spd_matrix(self.z_dim, random_state=self.seed+2)*0.1 
            self.Hxy = np.eye(self.x_dim,self.y_dim) #np.random.rand(x_dim,y_dim)*0.1 #np.eye(self.x_dim,self.y_dim) 
            self.Hyx = self.Hxy.T
            self.Hxz = np.eye(self.x_dim,self.z_dim) #np.random.rand(x_dim,z_dim)*0.1 #np.eye(self.x_dim,self.z_dim) 
            self.Hzx = self.Hxz.T
            self.Hyz = np.eye(self.y_dim,self.z_dim) #np.random.rand(y_dim,z_dim)*0.1 #np.eye(self.y_dim,self.z_dim) 
            self.Hzy = self.Hyz.T

            self.ul_vars_init_constant = -0.40 #-0.1
            self.ml_vars_init_constant = -0.20 #-0.1
            self.ll_vars_init_constant = -0.60 #-0.1

            self.ul_vars = np.random.uniform(0, 1, (self.x_dim, 1))*self.ul_vars_init_constant
            self.ml_vars = np.random.uniform(0, 1, (self.y_dim, 1))*self.ml_vars_init_constant
            self.ll_vars = np.random.uniform(0, 1, (self.z_dim, 1))*self.ll_vars_init_constant
            

    def z_opt(self, x, y):
        """
        The approximate optimal solution of the lower-level problem as a function of x and y; z is the current value of the ll vars
        	"""
        return np.dot(self.Hzx,x) + np.dot(self.Hzy,y)


    def y_opt(self, x):
        """
        The approximate optimal solution of the middle-level problem as a function of x; y and z are the current values of the ml and ll vars, respectively
        	"""
        B = self.Hyy - 2*np.matmul(self.Hzy,self.Hyz)
        invB = np.linalg.inv(B)
        C = self.Hyx + np.matmul(self.Hzx,self.Hyz)
        y_opt = np.matmul(np.matmul(invB,C),x)
        return y_opt
        
    
    def x_opt(self):
        """
        The optimal value of the trilevel problem
    	    """
        B = self.Hyy - 2*np.matmul(self.Hzy,self.Hyz)
        invB = np.linalg.inv(B)
        C = self.Hyx + np.matmul(self.Hzx,self.Hyz)
        
        invBC = np.matmul(invB,C)
        
        aux_1 = self.Hxx + np.matmul(self.Hxy,invBC) + np.matmul(self.Hxz,self.Hzx) + np.matmul(self.Hxz,np.matmul(self.Hzy,invBC)) + np.matmul(self.Hxz,self.Hzx) + np.matmul(invBC.T,self.Hyx) + np.matmul(invBC.T,np.matmul(self.Hyz,self.Hzx))
        aux_2 = self.hx + np.matmul(self.Hxz,self.hz) + np.matmul(invBC.T,self.hy) + np.matmul(invBC.T,np.matmul(self.Hyz,self.hz))
        
        x_opt = -np.matmul(np.linalg.inv(aux_1),aux_2)
        
        x_opt = x_opt.reshape(1,-1)
                
        return x_opt   
    
    
    def f_1(self, x, y, z):
        """
        The upper-level objective function
    	  """
        out = np.dot(self.hx.T,x) + np.dot(self.hy.T,y) + np.dot(self.hz.T,z) +\
            0.5*np.dot(x.T,np.dot(self.Hxx,x)) + np.dot(x.T,np.dot(self.Hxy,y)) + np.dot(x.T,np.dot(self.Hxz,z))
        return np.squeeze(out)
    

    def f_2(self, x, y, z):
        """
        The middle-level objective function
      	"""
        out = 0.5*np.dot(y.T,np.dot(self.Hyy,y)) - np.dot(y.T,np.dot(self.Hyx,x)) - np.dot(y.T,np.dot(self.Hyz,z)) 
        return np.squeeze(out)


    def f_3(self, x, y, z):
        """
        The lower-level objective function
    	  """
        out = 0.5*np.linalg.norm(np.dot(z.T,np.dot(self.Hzz,z)) - np.dot(z.T,np.dot(self.Hzx,x)) - np.dot(z.T,np.dot(self.Hzy,y)))**2
        return np.squeeze(out)
    

    def grad_f1_ul_vars(self, x, y, z):
        """
        The gradient of the upper-level objective function wrt the upper-level variables
    	  """
        out = self.hx  + np.dot(self.Hxx,x) + np.dot(self.Hxy,y) + np.dot(self.Hxz,z)
        out = out + self.std_dev*np.random.randn(out.shape[0],out.shape[1])
        return out


    def grad_f1_ml_vars(self, x, y, z):
        """
        The gradient of the upper-level objective function wrt the middle-level variables
    	  """
        out = self.hy  + np.dot(self.Hyx,x) 
        out = out + self.std_dev*np.random.randn(out.shape[0],out.shape[1])
        return out


    def grad_f1_ll_vars(self, x, y, z):
        """
        The gradient of the upper-level objective function wrt the lower-level variables
      	"""
        out = self.hz  + np.dot(self.Hzx,x) 
        out = out + self.std_dev*np.random.randn(out.shape[0],out.shape[1])
        return out


    def grad_f2_ul_vars(self, x, y, z):
        """
        The gradient of the middle-level objective function wrt the upper-level variables
       	"""
        out = - np.dot(self.Hxy,y)
        out = out + self.ml_std_dev*np.random.randn(out.shape[0],out.shape[1])
        return out
    

    def grad_f2_ml_vars(self, x, y, z):
        """
        The gradient of the middle-level objective function wrt the middle-level variables
      	"""
        out = np.dot(self.Hyy,y) - np.dot(self.Hyx,x) - np.dot(self.Hyz,z)
        out = out + self.ml_std_dev*np.random.randn(out.shape[0],out.shape[1])
        return out


    def grad_f2_ml_vars_torch(self, x, y, z):
        """
        The gradient of the middle-level objective function wrt the middle-level variables
      	"""
        out = torch.matmul(torch.tensor(self.Hyy, dtype=torch.float64),y) - torch.matmul(torch.tensor(self.Hyx, dtype=torch.float64),x) - torch.matmul(torch.tensor(self.Hyz, dtype=torch.float64),z)
        out = out + self.ml_std_dev*torch.randn(out.shape[0],out.shape[1])
        return out
    

    def grad_f2_ll_vars(self, x, y, z):
        """
        The gradient of the middle-level objective function wrt the lower-level variables
      	"""
        out = -np.dot(self.Hzy,y)
        out = out + self.ml_std_dev*np.random.randn(out.shape[0],out.shape[1])
        return out


    def grad_f2_ll_vars_torch(self, x, y, z):
        """
        The gradient of the middle-level objective function wrt the lower-level variables
      	"""
        out = torch.matmul(torch.tensor(self.Hzy, dtype=torch.float64),y)   
        out = out + self.ml_std_dev*torch.randn(out.shape[0],out.shape[1])
        return out
    

    def grad_f3_ul_vars(self, x, y, z):
        """
        The gradient of the lower-level objective function wrt the upper-level variables
      	"""
        a11 = np.zeros((self.x_dim,1))
        a12 = -np.dot(z.T,np.dot(self.Hzz,z))*np.dot(self.Hxz,z)
        a13 = np.zeros((self.x_dim,1))
        a22 = 2*np.dot(z.T,np.dot(self.Hzx,x))*np.dot(self.Hxz,z)
        a23 = np.dot(z.T,np.dot(self.Hzy,y))*np.dot(self.Hxz,z)
        a33 = np.zeros((self.x_dim,1))
        out = 0.5*(a11 + 2*a12 + 2*a13 + a22 + 2*a23 + a33)
        out = out + self.ll_std_dev*np.random.randn(out.shape[0],out.shape[1])
        
        # print('out grad_f3_ul_vars 1',out)       
        # out = 0.5*(2*(- z**3*self.Hzx) + 2*z**2*self.Hzx*x*self.Hzx + 2*(z**2* self.Hzy*y*self.Hzx))
        # print('out grad_f3_ul_vars 2',out)
        
        return out
    
    
    def grad_f3_ml_vars(self, x, y, z):
        """
        The gradient of the lower-level objective function wrt the middle-level variables
    	"""
        a11 = np.zeros((self.y_dim,1))
        a12 = np.zeros((self.y_dim,1))
        a13 = -np.dot(z.T,np.dot(self.Hzz,z))*np.dot(self.Hyz,z)
        a22 = np.zeros((self.y_dim,1))
        a23 = np.dot(z.T,np.dot(self.Hzx,x))*np.dot(self.Hyz,z)
        a33 = 2*np.dot(z.T,np.dot(self.Hzy,y))*np.dot(self.Hyz,z)
        out = 0.5*(a11 + 2*a12 + 2*a13 + a22 + 2*a23 + a33)
        out = out + self.ll_std_dev*np.random.randn(out.shape[0],out.shape[1])
        
        # print('out grad_f3_ml_vars 1',out)
        # out = 0.5*(2*(- z**3 * self.Hzy) + 2*(z**2 * self.Hzx * x * self.Hzy) + 2*z**2 * self.Hzy * y * self.Hzy)      
        # print('out grad_f3_ml_vars 2',out)
        
        return out
    

    def grad_f3_ll_vars(self, x, y, z):
        """
        The gradient of the lower-level objective function wrt the lower-level variables
    	"""
        a11 = 4*np.dot(z.T,np.dot(self.Hzz,z))*np.dot(self.Hzz,z)
        a12 = -(np.dot(z.T,np.dot(self.Hzz,z))*np.dot(self.Hzx,x) + 2*np.dot(z.T,np.dot(self.Hzx,x))*np.dot(self.Hzz,z))
        a13 = -(np.dot(z.T,np.dot(self.Hzz,z))*np.dot(self.Hzy,y) + 2*np.dot(z.T,np.dot(self.Hzy,y))*np.dot(self.Hzz,z))
        a22 = 2*np.dot(z.T,np.dot(self.Hzx,x))*np.dot(self.Hzx,x)
        a23 = (np.dot(z.T,np.dot(self.Hzx,x))*np.dot(self.Hzy,y) + np.dot(z.T,np.dot(self.Hzy,y))*np.dot(self.Hzx,x))
        a33 = 2*np.dot(z.T,np.dot(self.Hzy,y))*np.dot(self.Hzy,y)
        out = 0.5*(a11 + 2*a12 + 2*a13 + a22 + 2*a23 + a33)
        out = out + self.ll_std_dev*np.random.randn(out.shape[0],out.shape[1])
        
        # print('out grad_f3_ll_vars 1',out) 
        # out = 0.5*(4*z**3 + 2*(- 3*z**2*self.Hzx*x) + 2*( - 3*z**2*self.Hzy*y) + 2*z*(self.Hzx*x)**2  + 2*( 2*z*(self.Hzx*x*self.Hzy*y))  + 2*z*(self.Hzy*y)**2)
        # print('out grad_f3_ll_vars 2',out)
        
        return out


    def grad_f3_ll_vars_torch(self, x, y, z):
        """
        The gradient of the lower-level objective function wrt the lower-level variables
     	""" 
        a11 = 4*torch.matmul(z.T,torch.matmul(torch.tensor(self.Hzz,dtype=torch.float64),z))*torch.matmul(torch.tensor(self.Hzz,dtype=torch.float64),z)
        a12 = -(torch.matmul(z.T,torch.matmul(torch.tensor(self.Hzz,dtype=torch.float64),z))*torch.matmul(torch.tensor(self.Hzx,dtype=torch.float64),x) + 2*torch.matmul(z.T,torch.matmul(torch.tensor(self.Hzx,dtype=torch.float64),x))*torch.matmul(torch.tensor(self.Hzz,dtype=torch.float64),z))
        a13 = -(torch.matmul(z.T,torch.matmul(torch.tensor(self.Hzz,dtype=torch.float64),z))*torch.matmul(torch.tensor(self.Hzy,dtype=torch.float64),y) + 2*torch.matmul(z.T,torch.matmul(torch.tensor(self.Hzy,dtype=torch.float64),y))*torch.matmul(torch.tensor(self.Hzz,dtype=torch.float64),z))
        a22 = 2*torch.matmul(z.T,torch.matmul(torch.tensor(self.Hzx,dtype=torch.float64),x))*torch.matmul(torch.tensor(self.Hzx,dtype=torch.float64),x)
        a23 = (torch.matmul(z.T,torch.matmul(torch.tensor(self.Hzx,dtype=torch.float64),x))*torch.matmul(torch.tensor(self.Hzy,dtype=torch.float64),y) + torch.matmul(z.T,torch.matmul(torch.tensor(self.Hzy,dtype=torch.float64),y))*torch.matmul(torch.tensor(self.Hzx,dtype=torch.float64),x))
        a33 = 2*torch.matmul(z.T,torch.matmul(torch.tensor(self.Hzy,dtype=torch.float64),y))*torch.matmul(torch.tensor(self.Hzy,dtype=torch.float64),y)
        out = 0.5*(a11 + 2*a12 + 2*a13 + a22 + 2*a23 + a33)
        out = out + self.ll_std_dev*torch.randn(out.shape[0],out.shape[1])
        return out        


    def hess_f2_ml_vars_ul_vars(self, x, y, z):
        """
        The Hessian of the middle-level objective function wrt the lower-level variables
    	"""
        out = -self.Hyx
        out = out + self.ml_hess_std_dev*np.random.randn(out.shape[0],out.shape[1])
        return out


    def hess_f2_ml_vars_ml_vars(self, x, y, z):
        """
        The Hessian of the middle-level objective function wrt the lower-level variables
    	"""
        out = self.Hyy
        out = out + self.ml_hess_std_dev*np.random.randn(out.shape[0],out.shape[1])
        return out
    

    def hess_f2_ml_vars_ll_vars(self, x, y, z):
        """
        The Hessian of the middle-level objective function wrt the lower-level variables
    	"""
        out = -self.Hyz
        out = out + self.ml_hess_std_dev*np.random.randn(out.shape[0],out.shape[1])
        return out


    def hess_f2_ll_vars_ul_vars(self, x, y, z):
        """
        The Hessian of the middle-level objective function wrt the lower-level variables
    	"""
        out = np.zeros((self.z_dim,self.x_dim))
        out = out + self.ml_hess_std_dev*np.random.randn(out.shape[0],out.shape[1])
        return out


    def hess_f2_ll_vars_ml_vars(self, x, y, z):
        """
        The Hessian of the middle-level objective function wrt the lower-level variables
    	"""
        out = -self.Hzy
        out = out + self.ml_hess_std_dev*np.random.randn(out.shape[0],out.shape[1])
        return out
    

    def hess_f2_ll_vars_ll_vars(self, x, y, z):
        """
        The Hessian of the middle-level objective function wrt the lower-level variables
    	"""
        out = np.zeros((self.z_dim,self.z_dim))
        out = out + self.ml_hess_std_dev*np.random.randn(out.shape[0],out.shape[1])
        return out
    

    def hess_f3_ul_vars_ll_vars(self, x, y, z):
        """
        The Hessian of the lower-level objective function wrt the lower-level variables
    	"""
        a11 = np.zeros((self.x_dim,self.z_dim))
        a12 = -(np.dot(z.T,np.dot(self.Hzz,z))*self.Hxz + 2*np.matmul(np.dot(self.Hxz,z),np.dot(z.T,self.Hzz)))
        a13 = np.zeros((self.x_dim,self.z_dim))
        a22 = 2*(np.dot(z.T,np.dot(self.Hzx,x))*self.Hxz + np.matmul(np.dot(self.Hxz,z),np.dot(x.T,self.Hxz)))
        a23 = np.dot(z.T,np.dot(self.Hzy,y))*self.Hxz + np.matmul(np.dot(self.Hxz,z),np.dot(y.T,self.Hyz))
        a33 = np.zeros((self.x_dim,self.z_dim))
        out = 0.5*(a11 + 2*a12 + 2*a13 + a22 + 2*a23 + a33)
        out = out + self.ll_hess_std_dev*np.random.randn(out.shape[0],out.shape[1])
        
        # print('out hess_f3_ul_vars_ll_vars 1',out)      
        # out = 0.5*(2*(- 3 * z**2 * self.Hzx) + 4*z*self.Hzx**2 * x + 2*( 2*z*self.Hzy*y*self.Hzx))
        # print('out hess_f3_ul_vars_ll_vars 2',out)
        
        return out
    

    def hess_f3_ml_vars_ll_vars(self, x, y, z):
        """
        The Hessian of the lower-level objective function wrt the middle and lower level variables
    	"""
        a11 = np.zeros((self.y_dim,self.z_dim))
        a12 = np.zeros((self.y_dim,self.z_dim))
        a13 = -(np.dot(z.T,np.dot(self.Hzz,z))*self.Hyz + 2*np.matmul(np.dot(self.Hyz,z),np.dot(z.T,self.Hzz)))
        a22 = np.zeros((self.y_dim,self.z_dim))
        a23 = np.dot(z.T,np.dot(self.Hzx,x))*self.Hyz + np.matmul(np.dot(self.Hyz,z),np.dot(x.T,self.Hxz))
        a33 = 2*(np.dot(z.T,np.dot(self.Hzy,y))*self.Hyz + np.matmul(np.dot(self.Hyz,z),np.dot(y.T,self.Hyz)))
        out = 0.5*(a11 + 2*a12 + 2*a13 + a22 + 2*a23 + a33)
        out = out + self.ll_hess_std_dev*np.random.randn(out.shape[0],out.shape[1])
        
        # print('out hess_f3_ml_vars_ll_vars 1',out)
        # out = 0.5*(2*(- 3*z**2*self.Hzy) + 2*(2*z*self.Hzx * x * self.Hzy) + 4*z*self.Hzy**2 * y)
        # print('out hess_f3_ml_vars_ll_vars 2',out)
        
        return out


    def hess_f3_ll_vars_ul_vars(self, x, y, z):
        """
        The Hessian of the lower-level objective function wrt the lower-level variables
    	"""
        a11 = np.zeros((self.z_dim,self.x_dim))
        a12 = -(np.dot(z.T,np.dot(self.Hzz,z))*self.Hzx + 2*np.matmul(np.dot(self.Hzz,z),np.dot(z.T,self.Hzx)))
        a13 = np.zeros((self.z_dim,self.x_dim))
        a22 = 2*(np.dot(z.T,np.dot(self.Hzx,x))*self.Hzx + np.matmul(np.dot(self.Hzx,x),np.dot(z.T,self.Hzx)))
        a23 = np.dot(z.T,np.dot(self.Hzy,y))*self.Hzx + np.matmul(np.dot(self.Hzy,y),np.dot(z.T,self.Hzx))
        a33 = np.zeros((self.z_dim,self.x_dim))
        out = 0.5*(a11 + 2*a12 + 2*a13 + a22 + 2*a23 + a33)
        out = out + self.ll_hess_std_dev*np.random.randn(out.shape[0],out.shape[1])
        
        # print('out hess_f3_ll_vars_ul_vars 1',out)
        # out = 0.5*(2*(- 3*z**2*self.Hzx) + 4*z*self.Hzx**2 * x + 2*(2*z*self.Hzy*y*self.Hzx)) 
        # print('out hess_f3_ll_vars_ul_vars 2',out)
        
        return out


    def hess_f3_ll_vars_ml_vars(self, x, y, z):
        """
        The Hessian of the lower-level objective function wrt the lower-level variables
    	"""
        a11 = np.zeros((self.z_dim,self.y_dim))
        a12 = np.zeros((self.z_dim,self.y_dim))
        a13 = -(np.dot(z.T,np.dot(self.Hzz,z))*self.Hzy + 2*np.matmul(np.dot(self.Hzz,z),np.dot(z.T,self.Hzy)))
        a22 = np.zeros((self.z_dim,self.y_dim))
        a23 = np.dot(z.T,np.dot(self.Hzx,x))*self.Hzy + np.matmul(np.dot(self.Hzx,x),np.dot(z.T,self.Hzy))
        a33 = 2*(np.dot(z.T,np.dot(self.Hzy,y))*self.Hzy + np.matmul(np.dot(self.Hzy,y),np.dot(z.T,self.Hzy)))
        out = 0.5*(a11 + 2*a12 + 2*a13 + a22 + 2*a23 + a33)
        out = out + self.ll_hess_std_dev*np.random.randn(out.shape[0],out.shape[1])
        
        # print('out hess_f3_ll_vars_ml_vars 1',out)
        # out = 0.5*(2*(- 3*z**2*self.Hzy) + 2*(2*z*self.Hzx*x*self.Hzy) + 4*z*self.Hzy*y*self.Hzy)  
        # print('out hess_f3_ll_vars_ml_vars 2',out)
        
        return out
    

    def hess_f3_ll_vars_ll_vars(self, x, y, z):
        """
        The Hessian of the lower-level objective function wrt the lower-level variables
    	"""
        a11 = 4*(np.dot(z.T,np.dot(self.Hzz,z))*self.Hzz + 2*np.matmul(np.dot(self.Hzz,z),np.dot(z.T,self.Hzz)))
        a12 = -(2*np.matmul(np.dot(self.Hzx,x),np.dot(z.T,self.Hzz)) + 2*(np.dot(z.T,np.dot(self.Hzx,x))*self.Hzz + np.matmul(np.dot(self.Hzz,z),np.dot(x.T,self.Hxz))))
        a13 = -(2*np.matmul(np.dot(self.Hzy,y),np.dot(z.T,self.Hzz)) + 2*(np.dot(z.T,np.dot(self.Hzy,y))*self.Hzz + np.matmul(np.dot(self.Hzz,z),np.dot(y.T,self.Hyz))))
        a22 = 2*np.matmul(np.dot(self.Hzx,x),np.dot(x.T,self.Hxz))
        a23 = (np.matmul(np.dot(self.Hzx,x),np.dot(y.T,self.Hyz)) + np.matmul(np.dot(self.Hzy,y),np.dot(x.T,self.Hxz)))
        a33 = 2*np.matmul(np.dot(self.Hzy,y),np.dot(y.T,self.Hyz))
        out = 0.5*(a11 + 2*a12 + 2*a13 + a22 + 2*a23 + a33)
        out = out + self.ll_hess_std_dev*np.random.randn(out.shape[0],out.shape[1])
        
        # print('out hess_f3_ll_vars_ll_vars 1',out)      
        # out = 0.5*(4*3*z**2 + 2*(- 6*z*self.Hzx*x) + 2*(- 6*z*self.Hzy*y) + 2*(self.Hzx*x)**2  + 2*(2*(self.Hzx*x*self.Hzy*y)) + 2*(self.Hzy*y)**2)
        # print('out hess_f3_ll_vars_ll_vars 2',out)
        
        return out


    def compute_grad_aTvvTb(self,a,b,v):
        """
        Given column vectors a, b, and v, compute the gradient wrt v of a'*v*v'*b: (ab' + ba')v (same size as v)
        """
        return a*np.dot(b.T,v) + b*np.dot(a.T,v) 
    
    
    def compute_grad_aTuvTb(self,a,b,u):
        """
        Given column vectors a, b, and u, compute the gradient wrt v of a'*u*v'*b: ba'u (same size as b)
        """
        return b*np.dot(a.T,u) 
    
    
    def compute_grad_aTvuTb(self,a,b,u):
        """
        Given column vectors a, b, and u, compute the gradient wrt v of a'*v*u'*b: ab'u (same size as a)
        """
        return a*np.dot(b.T,u)
    
    
    def compute_trd_hess_aux_1(self,func_grad_type,H1,H2,vec):
        """
        Takes derivatives wrt v of expressions of these types: 
            H1'*v*v'*H2 (or H1'*v*v'*H1) --> func_grad_type = compute_grad_aTvvTb   (vec is v)
            H1'*u*v'*H2                  --> func_grad_type = compute_grad_aTuvTb   (vec is u)
            H1'*v*u'*H2                  --> func_grad_type = compute_grad_aTvuTb   (vec is u)
            
        The output is a 3D tensor.
        """
 
        if func_grad_type.__name__ == 'compute_grad_aTvvTb':
            out = np.zeros((vec.shape[0], H1.shape[0], H2.shape[1])) #3rd dimension, n rows, n columns of the result
        elif func_grad_type.__name__ == 'compute_grad_aTuvTb':
            out = np.zeros((H2.shape[0], H1.shape[0], H2.shape[1])) #3rd dimension, n rows, n columns of the result        
        elif func_grad_type.__name__ == 'compute_grad_aTvuTb':
            out = np.zeros((H1.shape[1], H1.shape[0], H2.shape[1])) #3rd dimension, n rows, n columns of the result
        
        #h varies across the columns of H2
        for h in range(H2.shape[1]):
            #i varies across the rows of H1
            for i in range(H1.shape[0]):
                    out[:, i, h] = func_grad_type(H1[i,:].reshape(-1,1),H2[:,h].reshape(-1,1),vec).squeeze()
                    
        return out
    

    def compute_trd_hess_aux_2(self,vec,H):
        """
        Given a column vector vec and a matrix H, returns a 3D tensor out such that:
            out[0,:,:] = vec[0]*H
            out[1,:,:] = vec[1]*H
            out[2,:,:] = vec[2]*H
        """
        out = vec[:, np.newaxis, :] * H[np.newaxis, :, :]
        return out


    def trd_hess_f3_ml_vars_ll_vars_ll_vars(self, x, y, z): 
        a11 = np.zeros((self.z_dim,self.y_dim,self.z_dim))
        a12 = np.zeros((self.z_dim,self.y_dim,self.z_dim))
        a13 = -(2*self.compute_trd_hess_aux_2(np.dot(self.Hzz,z),self.Hyz) + 2*self.compute_trd_hess_aux_1(self.compute_grad_aTvvTb,self.Hyz,self.Hzz,z))
        a22 = np.zeros((self.z_dim,self.y_dim,self.z_dim))
        a23 = self.compute_trd_hess_aux_2(np.dot(self.Hzx,x),self.Hyz) + self.compute_trd_hess_aux_1(self.compute_grad_aTvuTb,self.Hyz,self.Hxz,x)
        a33 = 2*self.compute_trd_hess_aux_2(np.dot(self.Hzy,y),self.Hyz) + self.compute_trd_hess_aux_1(self.compute_grad_aTvuTb,self.Hyz,self.Hyz,y)
        out = 0.5*(a11 + 2*a12 + 2*a13 + a22 + 2*a23 + a33)
        # out = out + self.ll_hess_std_dev*np.random.randn(*out.shape)
        
        # print('out trd_hess_f3_ml_vars_ll_vars_ll_vars 1',out)       
        # out = 0.5*(2*(- 6*z*self.Hzy) + 2*(2*self.Hzx * x * self.Hzy) + 4*self.Hzy*y*self.Hzy)
        # print('out trd_hess_f3_ml_vars_ll_vars_ll_vars 2',out)
        
        return out        


    def trd_hess_f3_ll_vars_ll_vars_ll_vars(self, x, y, z): 
        a11 = 4*(2*self.compute_trd_hess_aux_2(np.dot(self.Hzz,z),self.Hzz) + 2*self.compute_trd_hess_aux_1(self.compute_grad_aTvvTb,self.Hzz,self.Hzz,z))
        a12 = -(2*self.compute_trd_hess_aux_1(self.compute_grad_aTuvTb,self.Hzx,self.Hzz,x) + 2*(self.compute_trd_hess_aux_2(np.dot(self.Hzx,x),self.Hzz) + self.compute_trd_hess_aux_1(self.compute_grad_aTvuTb,self.Hzz,self.Hxz,x)))
        a13 = -(2*self.compute_trd_hess_aux_1(self.compute_grad_aTuvTb,self.Hzy,self.Hzz,y) + 2*(self.compute_trd_hess_aux_2(np.dot(self.Hzy,y),self.Hzz) + self.compute_trd_hess_aux_1(self.compute_grad_aTvuTb,self.Hzz,self.Hyz,y)))
        a22 = np.zeros((self.z_dim,self.z_dim,self.z_dim))
        a23 = np.zeros((self.z_dim,self.z_dim,self.z_dim))
        a33 = np.zeros((self.z_dim,self.z_dim,self.z_dim))
        out = 0.5*(a11 + 2*a12 + 2*a13 + a22 + 2*a23 + a33)
        # out = out + self.ll_hess_std_dev*np.random.randn(*out.shape)
        
        # print('out trd_hess_f3_ll_vars_ll_vars_ll_vars 1',out)     
        # out = 0.5*(4*3*2*z +2*(- 6*self.Hzx*x) + 2*(- 6*self.Hzy*y))
        # print('out trd_hess_f3_ll_vars_ll_vars_ll_vars 2',out)
        
        return out          


    def trd_hess_f3_ml_vars_ll_vars_ul_vars(self, x, y, z): 
        a11 = np.zeros((self.x_dim,self.y_dim,self.z_dim))
        a12 = np.zeros((self.x_dim,self.y_dim,self.z_dim))
        a13 = np.zeros((self.x_dim,self.y_dim,self.z_dim))
        a22 = np.zeros((self.x_dim,self.y_dim,self.z_dim))
        a23 = self.compute_trd_hess_aux_2(np.dot(self.Hxz,z),self.Hyz) + self.compute_trd_hess_aux_1(self.compute_grad_aTuvTb,self.Hyz,self.Hxz,z)
        a33 = np.zeros((self.x_dim,self.y_dim,self.z_dim))
        out = 0.5*(a11 + 2*a12 + 2*a13 + a22 + 2*a23 + a33)
        # out = out + self.ll_hess_std_dev*np.random.randn(*out.shape)
        
        # print('out trd_hess_f3_ml_vars_ll_vars_ul_vars 1',out)        
        # out = 0.5*(2*(2*z*self.Hzx*self.Hzy))
        # print('out trd_hess_f3_ml_vars_ll_vars_ul_vars 2',out)
        
        return out    


    def trd_hess_f3_ml_vars_ul_vars_ll_vars(self, x, y, z): 
        return np.transpose(self.trd_hess_f3_ml_vars_ll_vars_ul_vars(x, y, z), (2, 1, 0))
    

    def trd_hess_f3_ll_vars_ll_vars_ul_vars(self, x, y, z): 
        a11 = np.zeros((self.x_dim,self.z_dim,self.z_dim))
        a12 = -(2*self.compute_trd_hess_aux_1(self.compute_grad_aTvuTb,self.Hzx,self.Hzz,z) + 2*(self.compute_trd_hess_aux_2(np.dot(self.Hxz,z),self.Hzz) + self.compute_trd_hess_aux_1(self.compute_grad_aTuvTb,self.Hzz,self.Hxz,z)))
        a13 = np.zeros((self.x_dim,self.z_dim,self.z_dim))
        a22 = 2*self.compute_trd_hess_aux_1(self.compute_grad_aTvvTb,self.Hzx,self.Hxz,x)
        a23 = self.compute_trd_hess_aux_1(self.compute_grad_aTvuTb,self.Hzx,self.Hyz,y) + self.compute_trd_hess_aux_1(self.compute_grad_aTuvTb,self.Hzy,self.Hxz,y)
        a33 = np.zeros((self.x_dim,self.z_dim,self.z_dim))
        out = 0.5*(a11 + 2*a12 + 2*a13 + a22 + 2*a23 + a33)
        # out = out + self.ll_hess_std_dev*np.random.randn(*out.shape)
        
        # print('out trd_hess_f3_ll_vars_ll_vars_ul_vars 1',out,a12,a22,a23)    
        # out = 0.5*(2*(- 6 * z*self.Hzx) + 4*self.Hzx*self.Hzx*x + 2*(2*self.Hzy*y*self.Hzx))
        # print('out trd_hess_f3_ll_vars_ll_vars_ul_vars 2',out,- 6 * z*self.Hzx,4*self.Hzx*self.Hzx*x,2*self.Hzy*y*self.Hzx)
        
        return out    


    def trd_hess_f3_ll_vars_ul_vars_ll_vars(self, x, y, z): 
        return np.transpose(self.trd_hess_f3_ll_vars_ll_vars_ul_vars(x, y, z), (2, 1, 0))  
    

    def trd_hess_f3_ml_vars_ll_vars_ml_vars(self, x, y, z): 
        a11 = np.zeros((self.y_dim,self.y_dim,self.z_dim))
        a12 = np.zeros((self.y_dim,self.y_dim,self.z_dim))
        a13 = np.zeros((self.y_dim,self.y_dim,self.z_dim))
        a22 = np.zeros((self.y_dim,self.y_dim,self.z_dim))
        a23 = np.zeros((self.y_dim,self.y_dim,self.z_dim))
        a33 = 2*(self.compute_trd_hess_aux_2(np.dot(self.Hyz,z),self.Hyz) + self.compute_trd_hess_aux_1(self.compute_grad_aTuvTb,self.Hyz,self.Hyz,z))
        out = 0.5*(a11 + 2*a12 + 2*a13 + a22 + 2*a23 + a33)
        # out = out + self.ll_hess_std_dev*np.random.randn(*out.shape)
        
        # print('out trd_hess_f3_ml_vars_ll_vars_ml_vars 1',out,a33)  
        # out = 0.5*(4*z*self.Hzy*self.Hzy)
        # print('out trd_hess_f3_ml_vars_ll_vars_ml_vars 2',out,4*z*self.Hzy*self.Hzy)
        
        return out    


    def trd_hess_f3_ml_vars_ml_vars_ll_vars(self, x, y, z): 
        return np.transpose(self.trd_hess_f3_ml_vars_ll_vars_ml_vars(x, y, z), (2, 1, 0)) 
    

    def trd_hess_f3_ll_vars_ll_vars_ml_vars(self, x, y, z): 
        a11 = np.zeros((self.y_dim,self.z_dim,self.z_dim))
        a12 = np.zeros((self.y_dim,self.z_dim,self.z_dim))
        a13 = -(2*self.compute_trd_hess_aux_1(self.compute_grad_aTvuTb,self.Hzy,self.Hzz,z) + 2*(self.compute_trd_hess_aux_2(np.dot(self.Hyz,z),self.Hzz) + self.compute_trd_hess_aux_1(self.compute_grad_aTuvTb,self.Hzz,self.Hyz,z)))
        a22 = np.zeros((self.y_dim,self.z_dim,self.z_dim))
        a23 = self.compute_trd_hess_aux_1(self.compute_grad_aTuvTb,self.Hzx,self.Hyz,x) + self.compute_trd_hess_aux_1(self.compute_grad_aTvuTb,self.Hzy,self.Hxz,x)
        a33 = 2*self.compute_trd_hess_aux_1(self.compute_grad_aTvvTb,self.Hzy,self.Hyz,y)
        out = 0.5*(a11 + 2*a12 + 2*a13 + a22 + 2*a23 + a33)
        # out = out + self.ll_hess_std_dev*np.random.randn(*out.shape)
        
        # print('out trd_hess_f3_ll_vars_ll_vars_ml_vars 1',out)
        # out = 0.5*(2*(- 6 * z*self.Hzy) + 2*(2*self.Hzx*x*self.Hzy) + 4*self.Hzy*y*self.Hzy)
        # print('out trd_hess_f3_ll_vars_ll_vars_ml_vars 2',out)
        
        return out


    def trd_hess_f3_ll_vars_ml_vars_ll_vars(self, x, y, z): 
        return np.transpose(self.trd_hess_f3_ll_vars_ll_vars_ml_vars(x, y, z), (2, 1, 0))


class AdversarialLearning:
    """
    Class used to define a trilevel adversarial learning problem.   
    """    

    class IndexedTensorDataset(Dataset):
        def __init__(self, X, y):
            self.X = X
            self.y = y
    
        def __len__(self):
            return len(self.X)
    
        def __getitem__(self, idx):
            return self.X[idx], self.y[idx], idx  # include index
    

    def __init__(self, seed=42, configuration=1):
        
        self.seed = seed
        
        set_seed(self.seed)
        
        self.configuration = configuration
        self.is_machine_learning_problem = True

        self.z_opt_available_analytically = False
        self.y_opt_available_analytically = False
        self.x_opt_available_analytically = False
        
        self.remove_ul = False # If True, obtain a bilevel problem by removing the UL problem (only works correctly for adverarial learning)
        self.remove_ml = False # If True, obtain a bilevel problem by removing the LL problem (only works correctly for adverarial learning)
        self.remove_ll = False # If True, obtain a bilevel problem by removing the LL problem (only works correctly for adverarial learning)
            
        if configuration == 1 or configuration == 2:

            if configuration == 1:
                ## If self.batch_dataset is True, use the full dataset (or a subset of it if self.dataset_subset is True) and then apply a batch algorithm; 
                ## If self.batch_dataset is False, use a minibatch approach on the full dataset (or a subset of it if self.dataset_subset is True)
                self.batch_dataset = False
                self.dataset_subset = True
                self.train_subset_size = 40
                self.valid_subset_size = 100
                
                self.minibatch_size = 64
            
            if configuration == 2: 
                ## If self.batch_dataset is True, use the full dataset (or a subset of it if self.dataset_subset is True) and then apply a batch algorithm; 
                ## If self.batch_dataset is False, use a minibatch approach on the full dataset (or a subset of it if self.dataset_subset is True)
                self.batch_dataset = False
                self.dataset_subset = True
                self.train_subset_size = 40
                self.valid_subset_size = 100
                
                self.minibatch_size = 64
                        
            # (Red and white) wine quality dataset: https://www.sciencedirect.com/science/article/pii/S0167923609001377
            # Replace these file paths with the correct paths to your CSV files
            red_wine_path = "wine+quality/winequality-red.csv"
            
            # Load the datasets
            red_wine = pd.read_csv(red_wine_path, delimiter=";")   # Assuming the delimiter is `;`

            # Define a split ratio
            train_ratio = 0.7; val_ratio = 0.15; test_ratio = 0.15
            
            # For the red wine dataset
            red_train, red_temp = train_test_split(red_wine, test_size=(1 - train_ratio), random_state=42)
            red_val, red_test = train_test_split(red_temp, test_size=test_ratio / (val_ratio + test_ratio), random_state=42)

            # Save splits to instance variables
            self.train_data = red_train
            self.val_data = red_val
            self.test_data = red_test
            
            # Print the sizes of each split
            print(f"\nRed Wine Dataset: Training set: {len(red_train)}, Validation set: {len(red_val)}, Test set: {len(red_test)}\n")             
                

        if configuration == 3 or configuration == 4:

            if configuration == 3:
                ## If self.batch_dataset is True, use the full dataset (or a subset of it if self.dataset_subset is True) and then apply a batch algorithm; 
                ## If self.batch_dataset is False, use a minibatch approach on the full dataset (or a subset of it if self.dataset_subset is True)
                self.batch_dataset = False
                self.dataset_subset = True
                self.train_subset_size = 40
                self.valid_subset_size = 100
                
                self.minibatch_size = 64
                
            if configuration == 4: 
                ## If self.batch_dataset is True, use the full dataset (or a subset of it if self.dataset_subset is True) and then apply a batch algorithm; 
                ## If self.batch_dataset is False, use a minibatch approach on the full dataset (or a subset of it if self.dataset_subset is True)
                self.batch_dataset = False
                self.dataset_subset = True
                self.train_subset_size = 40
                self.valid_subset_size = 100
                
                self.minibatch_size = 64
                
            # (Red and white) wine quality dataset: https://www.sciencedirect.com/science/article/pii/S0167923609001377
            # Replace these file paths with the correct paths to your CSV files
            white_wine_path = "wine+quality/winequality-white.csv"
            
            # Load the datasets
            white_wine = pd.read_csv(white_wine_path, delimiter=";")

            # Define a split ratio
            train_ratio = 0.7; val_ratio = 0.15; test_ratio = 0.15
            
            # For the white wine dataset
            white_train, white_temp = train_test_split(white_wine, test_size=(1 - train_ratio), random_state=42)
            white_val, white_test = train_test_split(white_temp, test_size=test_ratio / (val_ratio + test_ratio), random_state=42)

            # Save splits to instance variables
            self.train_data = white_train
            self.val_data = white_val
            self.test_data = white_test
            
            # Print the sizes of each split        
            print(f"\nWhite Wine Dataset: Training set: {len(white_train)}, Validation set: {len(white_val)}, Test set: {len(white_test)}\n")             


        if configuration == 5 or configuration == 6:

            if configuration == 5:
                ## If self.batch_dataset is True, use the full dataset (or a subset of it if self.dataset_subset is True) and then apply a batch algorithm; 
                ## If self.batch_dataset is False, use a minibatch approach on the full dataset (or a subset of it if self.dataset_subset is True)
                self.batch_dataset = False
                self.dataset_subset = True
                self.train_subset_size = 40
                self.valid_subset_size = 100
                
                self.minibatch_size = 64 #64
                
            if configuration == 6: 
                ## If self.batch_dataset is True, use the full dataset (or a subset of it if self.dataset_subset is True) and then apply a batch algorithm; 
                ## If self.batch_dataset is False, use a minibatch approach on the full dataset (or a subset of it if self.dataset_subset is True)
                self.batch_dataset = False
                self.dataset_subset = True
                self.train_subset_size = 40
                self.valid_subset_size = 100
                
                self.minibatch_size = 64
                
            # Load the California housing dataset
            housing = fetch_california_housing(as_frame=True)  # directly returns a DataFrame

            # Convert to a single DataFrame with features + target
            housing_df = housing.frame  # includes both data and target in one DataFrame

            # Define a split ratio
            train_ratio = 0.7; val_ratio = 0.15; test_ratio = 0.15
            
            # Split into train, val, and test)
            housing_train, housing_temp = train_test_split(housing_df, test_size=(1 - train_ratio), random_state=42)
            housing_val, housing_test = train_test_split(housing_temp, test_size=test_ratio / (val_ratio + test_ratio), random_state=42)

            # Save splits to instance variables
            self.train_data = housing_train
            self.val_data = housing_val
            self.test_data = housing_test
            
            # Print the sizes of each split        
            print(f"\nCalifornia Housing Dataset: Training set: {len(housing_train)}, Validation set: {len(housing_val)}, Test set: {len(housing_test)}\n")


        self.X_train, self.y_train, self.X_valid, self.y_valid, self.X_test, self.y_test, self.train_loader = self.extract_data()
        
        train_size = self.X_train.shape[0] if self.batch_dataset else self.minibatch_size

        self.a = self.X_valid.shape[0]
        self.b = train_size

        if configuration == 1:
            ## We use our formulation: min, min, max
            self.sato_formulation_min_max_min = False ## We use our formulation: min, min, max
            self.c = 0.1 #0.1

            self.ul_vars_init_constant = -2 #-1
            self.ml_vars_init_constant = -2 #-1
            self.ll_vars_init_constant = -2 #-1
            
            self.x_dim = 1
            self.y_dim = self.X_train.shape[1]
            self.z_dim = self.X_train.shape[0]*self.X_train.shape[1]   #train_size*self.X_train.shape[1]
            
            self.ul_vars = np.random.uniform(0, 1, (self.x_dim, 1))*self.ul_vars_init_constant
            self.ml_vars = np.random.uniform(0, 1, (self.y_dim, 1))*self.ml_vars_init_constant
            self.ll_vars = np.random.uniform(0, 1, (self.z_dim, 1))*self.ll_vars_init_constant
        
        if configuration == 2:
            ## We use Sato's formulation: min, max, min
            self.sato_formulation_min_max_min = True
            self.c = 0.1 #0.1
            
            self.ul_vars_init_constant = -2 #-1 
            self.ml_vars_init_constant = -2 #-0.5
            self.ll_vars_init_constant = -2 #-0.5
    
            self.x_dim = 1
            self.y_dim = self.X_train.shape[0]*self.X_train.shape[1]   #train_size*self.X_train.shape[1]
            self.z_dim = self.X_train.shape[1]
    
            self.ul_vars = np.random.uniform(0, 1, (self.x_dim, 1))*self.ul_vars_init_constant
            self.ll_vars = np.random.uniform(0, 1, (self.z_dim, 1))*self.ll_vars_init_constant
            self.ml_vars = np.random.uniform(0, 1, (self.y_dim, 1))*self.ml_vars_init_constant
            
        if configuration == 3:
            ## We use our formulation: min, min, max
            self.sato_formulation_min_max_min = False 
            self.c = 0.1 #0.1

            self.ul_vars_init_constant = -2
            self.ml_vars_init_constant = -2 
            self.ll_vars_init_constant = -2 
    
            self.x_dim = 1
            self.y_dim = self.X_train.shape[1]
            self.z_dim = self.X_train.shape[0]*self.X_train.shape[1]   #train_size*self.X_train.shape[1]
    
            self.ul_vars = np.random.uniform(0, 1, (self.x_dim, 1))*self.ul_vars_init_constant
            self.ml_vars = np.random.uniform(0, 1, (self.y_dim, 1))*self.ml_vars_init_constant
            self.ll_vars = np.random.uniform(0, 1, (self.z_dim, 1))*self.ll_vars_init_constant
            
        if configuration == 4:
            ## We use Sato's formulation: min, max, min
            self.sato_formulation_min_max_min = True 
            self.c = 0.1 #0.1

            self.ul_vars_init_constant = -2 
            self.ml_vars_init_constant = -2 
            self.ll_vars_init_constant = -2 
    
            self.x_dim = 1
            self.y_dim = self.X_train.shape[0]*self.X_train.shape[1]   #train_size*self.X_train.shape[1]
            self.z_dim = self.X_train.shape[1]
    
            self.ul_vars = np.random.uniform(0, 1, (self.x_dim, 1))*self.ul_vars_init_constant
            self.ll_vars = np.random.uniform(0, 1, (self.z_dim, 1))*self.ll_vars_init_constant
            self.ml_vars = np.random.uniform(0, 1, (self.y_dim, 1))*self.ml_vars_init_constant

        if configuration == 5:
            ## We use our formulation: min, min, max
            self.sato_formulation_min_max_min = False 
            self.c = 0.1 #0.1

            self.ul_vars_init_constant = -2
            self.ml_vars_init_constant = -2 
            self.ll_vars_init_constant = -2 
    
            self.x_dim = 1
            self.y_dim = self.X_train.shape[1]
            self.z_dim = self.X_train.shape[0]*self.X_train.shape[1]   #train_size*self.X_train.shape[1]
    
            self.ul_vars = np.random.uniform(0, 1, (self.x_dim, 1))*self.ul_vars_init_constant
            self.ml_vars = np.random.uniform(0, 1, (self.y_dim, 1))*self.ml_vars_init_constant
            self.ll_vars = np.random.uniform(0, 1, (self.z_dim, 1))*self.ll_vars_init_constant
            
        if configuration == 6:
            ## We use Sato's formulation: min, max, min
            self.sato_formulation_min_max_min = True 
            self.c = 0.1 #0.1

            self.ul_vars_init_constant = -2 
            self.ml_vars_init_constant = -2 
            self.ll_vars_init_constant = -2 
    
            self.x_dim = 1
            self.y_dim = self.X_train.shape[0]*self.X_train.shape[1]   #train_size*self.X_train.shape[1]
            self.z_dim = self.X_train.shape[1]
    
            self.ul_vars = np.random.uniform(0, 1, (self.x_dim, 1))*self.ul_vars_init_constant
            self.ll_vars = np.random.uniform(0, 1, (self.z_dim, 1))*self.ll_vars_init_constant
            self.ml_vars = np.random.uniform(0, 1, (self.y_dim, 1))*self.ml_vars_init_constant
            

    def extract_data(self):
        """
        Extract features (X) and target values (y) from the training, validation, and test data.
        """
        # For the training data
        X_train = self.train_data.iloc[:, :-1].values  # All columns except the last one (features)
        y_train = self.train_data.iloc[:, -1].values   # Last column (target)
    
        # For the validation data
        X_valid = self.val_data.iloc[:, :-1].values    # All columns except the last one (features)
        y_valid = self.val_data.iloc[:, -1].values     # Last column (target)
    
        # For the test data (if needed)
        X_test = self.test_data.iloc[:, :-1].values    # All columns except the last one (features)
        y_test = self.test_data.iloc[:, -1].values     # Last column (target)
    
        # Standardize the features using the training data statistics
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_valid = scaler.transform(X_valid)
        X_test = scaler.transform(X_test)
        
        self.X_train_full = X_train
        self.y_train_full = y_train
        self.X_valid_full = X_valid
        self.y_valid_full = y_valid
        
        train_loader = None
        
        if self.batch_dataset:
            if self.dataset_subset:
                # Randomly select 40 samples for training
                train_indices = np.random.choice(len(X_train), size=self.train_subset_size, replace=False)
                X_train = X_train[train_indices]
                y_train = y_train[train_indices]
            
                # Randomly select 100 samples for validation
                valid_indices = np.random.choice(len(X_valid), size=self.valid_subset_size, replace=False)
                X_valid = X_valid[valid_indices]
                y_valid = y_valid[valid_indices]        

                self.X_train_full = X_train
                self.y_train_full = y_train
                self.X_valid_full = X_valid
                self.y_valid_full = y_valid

        else:
            def worker_init_fn(worker_id):
                # Each worker gets a unique but reproducible seed
                seed = torch.initial_seed() % 2**32
                np.random.seed(seed)
                random.seed(seed)
                set_seed(self.seed)

            generator = torch.Generator()
            generator.manual_seed(self.seed)

            # Convert to PyTorch tensors
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
          
            # Create dataset and loader
            train_dataset = self.IndexedTensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=self.minibatch_size, shuffle=True, drop_last=True, worker_init_fn=worker_init_fn, generator=generator)
        
            self.train_loader_iterator = iter(train_loader)
        
        return X_train, y_train, X_valid, y_valid, X_test, y_test, train_loader            


    def mse_func(self, y, z, noise=False, std_dev=0): 
        """
        MSE calculation function based on theta (which is y in the non-Sato formulaton and is z in the Sato formulation)
        """
        theta = z if self.sato_formulation_min_max_min else y
        
        y_true = self.y_test.reshape(-1, 1)
        
        mse_values = []

        for _ in range(100):
            if noise:
                P = np.random.normal(loc=0.0, scale=std_dev, size=self.X_test.shape)
                
            else:
                P = np.zeros_like(self.X_test)
        
            y_pred = np.dot(self.X_test + P, theta).reshape(-1, 1)
            residual =  y_true - y_pred
            mse = (1 / residual.shape[0]) * np.dot(residual.T, residual)
            
            mse_values.append(np.squeeze(mse))
    
        out = np.mean(mse_values)

        return out


    def r_squared(self, y, z, noise=False, std_dev=0):
        """
        R^2 calculation function based on theta (which is y in the non-Sato formulation and is z in the Sato formulation).
        """
        theta = z if self.sato_formulation_min_max_min else y
        
        y_true = self.y_test.reshape(-1, 1)
        total_variance = np.sum((y_true - np.mean(y_true))**2)
        
        r2_values = []

        for _ in range(100):    
            if noise:
                P = np.random.normal(loc=0.0, scale=std_dev, size=self.X_test.shape)
                
            else:
                P = np.zeros_like(self.X_test)
        
            y_pred = np.dot(self.X_test + P, theta).reshape(-1, 1)
            residual =  y_true - y_pred
            mse = np.dot(residual.T, residual)
            
            
            r2 = 1 - (mse / total_variance)
            r2_values.append(np.squeeze(r2))
    
        out = np.mean(r2_values)

        return out
        

    def next_minibatch(self):
        try:
            X_batch, y_batch, idx_batch = next(self.train_loader_iterator)
        except StopIteration:
            # Reinitialize iterator if exhausted
            self.train_loader_iterator = iter(self.train_loader)
            X_batch, y_batch, idx_batch = next(self.train_loader_iterator)
        self.X_train = X_batch.numpy()
        self.y_train = y_batch.numpy()
        self.minibatch_indices = idx_batch.numpy()


    def smoothed_ell1_norm(self, x, mu=0.25):
        smoothed = np.zeros_like(x)
    
        # Region: x > mu
        mask1 = x > mu
        smoothed[mask1] = x[mask1]
    
        # Region: x < -mu
        mask2 = x < -mu
        smoothed[mask2] = -x[mask2]
    
        # Region: -mu <= x <= mu
        mask3 = ~mask1 & ~mask2
        x_mid = x[mask3]
        smoothed[mask3] = (
            - (x_mid ** 4) / (8 * mu ** 3)
            + (3 * x_mid ** 2) / (4 * mu)
            + (3 * mu) / 8
        )
    
        return np.sum(smoothed)


    def smoothed_ell1_gradient(self, x, mu=0.25, flag_pytorch=False):

        if flag_pytorch:
            grad = torch.zeros_like(x)
            
        else:                
            grad = np.zeros_like(x)
    
        # Region: x > mu
        mask1 = x > mu
        grad[mask1] = 1
    
        # Region: x < -mu
        mask2 = x < -mu
        grad[mask2] = -1
    
        # Region: -mu <= x <= mu
        mask3 = ~mask1 & ~mask2
        x_mid = x[mask3]
        grad[mask3] = - (x_mid ** 3) / (2 * mu ** 3) + (3 * x_mid) / (2 * mu)
    
        return grad
     

    def f_1_general(self, lam, theta, delta): 
        residual = self.y_valid.reshape(-1,1) - np.dot(self.X_valid, theta)
        out = (1 / self.a) * np.dot(residual.T, residual)
        return np.squeeze(out)


    def g(self, lam, theta, delta):
        if self.batch_dataset:
            P = delta.reshape(self.X_train.shape[0], self.X_train.shape[1])
            residual = self.y_train.reshape(-1,1) - np.dot(self.X_train + P, theta)
            out = (1 / self.b) * np.dot(residual.T, residual)            
            
        else:
            P_aux = delta.reshape(-1, self.X_train.shape[1])  ## P = delta.reshape(self.X_train.shape[0], self.X_train.shape[1])
            P = P_aux[self.minibatch_indices]
            residual = self.y_train.reshape(-1,1) - np.dot(self.X_train + P, theta)
            out = (1 / self.b) * np.dot(residual.T, residual)
        return np.squeeze(out)


    def phi_delta(self, lam, theta, delta):  
        if self.batch_dataset:
            d = theta.shape[0]
            P = delta.reshape(self.X_train.shape[0], self.X_train.shape[1])
            regularization = (self.c / (self.b * d)) * np.linalg.norm(P, 'fro')**2
            
        else:
            d = theta.shape[0]
            P_aux = delta.reshape(-1, self.X_train.shape[1])  ## P = delta.reshape(self.X_train.shape[0], self.X_train.shape[1])
            P = P_aux[self.minibatch_indices]
            regularization = (self.c / (self.b * d)) * np.linalg.norm(P, 'fro')**2   
        return np.squeeze(regularization)


    def phi_theta(self, lam, theta, delta):
        d = theta.shape[0]
        regularization = np.exp(lam) * self.smoothed_ell1_norm(theta) / d
        return np.squeeze(regularization)        


    def f_1_general_gradient_lam(self, lam, theta, delta):  
        return np.zeros_like(lam)


    def f_1_general_gradient_theta(self, lam, theta, delta):
        residual = self.y_valid.reshape(-1,1) - self.X_valid @ theta
        out = (-2 / self.a) * self.X_valid.T @ residual
        return out


    # def f_1_general_gradient_theta_torch(self, lam, theta, delta):
    #     residual = torch.from_numpy(self.y_valid).float().reshape(-1, 1) - self.X_valid @ theta
    #     out = (-2 / self.a) * torch.from_numpy(self.X_valid.T).float() @ residual
    #     return out
  

    def f_1_general_gradient_delta(self, lam, theta, delta):  
        return np.zeros_like(delta)


    # def f_1_general_gradient_delta_torch(self, lam, theta, delta):  
    #     return torch.zeros_like(delta)
    

    def g_gradient_lam(self, lam, theta, delta):
        return np.zeros_like(lam)
    

    def g_gradient_theta(self, lam, theta, delta):
        if self.batch_dataset:
            P = delta.reshape(self.X_train.shape[0], self.X_train.shape[1])
            out = (-2 / self.b) * (self.X_train + P).T @ (self.y_train.reshape(-1,1) - (self.X_train + P) @ theta) 
            
        else:
            P_aux = delta.reshape(-1, self.X_train.shape[1])  ## P = delta.reshape(self.X_train.shape[0], self.X_train.shape[1])
            P = P_aux[self.minibatch_indices]
            out = (-2 / self.b) * (self.X_train + P).T @ (self.y_train.reshape(-1,1) - (self.X_train + P) @ theta) 
        return out


    def g_gradient_theta_torch(self, lam, theta, delta):
        if self.batch_dataset:
            P = delta.view(self.X_train.shape)
            A = torch.tensor(self.X_train, dtype=P.dtype, device=P.device) + P
            residual = torch.tensor(self.y_train, dtype=A.dtype, device=A.device).reshape(-1,1) - A @ theta  
            out = (-2 / self.b) * A.T @ residual
            
        else: 
            P_aux = delta.view(-1, self.X_train.shape[1])  ## P = delta.view(self.X_train.shape)
            P = P_aux[self.minibatch_indices]
            A = torch.tensor(self.X_train, dtype=P.dtype, device=P.device) + P
            residual = torch.tensor(self.y_train, dtype=A.dtype, device=A.device).reshape(-1,1) - A @ theta  
            out = (-2 / self.b) * A.T @ residual
        return out
    

    def g_gradient_delta(self, lam, theta, delta):
        if self.batch_dataset:
            P = delta.reshape(self.X_train.shape[0], self.X_train.shape[1])
            grad_P = (-2 / self.b) * np.outer(self.y_train.reshape(-1,1) - (self.X_train + P) @ theta, theta)  
            out = grad_P.reshape(-1,1)
        
        else:
            P_aux = delta.reshape(-1, self.X_train.shape[1])  ## P = delta.reshape(self.X_train.shape[0], self.X_train.shape[1])
            P = P_aux[self.minibatch_indices]
            grad_P = (-2 / self.b) * np.outer(self.y_train.reshape(-1,1) - (self.X_train + P) @ theta, theta)  
            # out = grad_P.reshape(-1,1)
            
            # Initialize full-sized gradient with zeros
            grad_full = np.zeros_like(P_aux)
            # Place minibatch gradients in correct rows
            grad_full[self.minibatch_indices] = grad_P
            # Flatten to match the shape of delta
            out = grad_full.reshape(-1, 1)
            
        return out


    def g_gradient_delta_torch(self, lam, theta, delta):
        if self.batch_dataset:
            P = delta.view(self.X_train.shape)
            A = torch.tensor(self.X_train, dtype=P.dtype, device=P.device) + P
            residual = torch.tensor(self.y_train, dtype=A.dtype, device=A.device).reshape(-1,1).reshape(-1,1) - A @ theta
            grad_P = (-2 / self.b) * torch.outer(residual.view(-1), theta.view(-1)) 
            out = grad_P.reshape(-1, 1)
        
        else:
            P_aux = delta.view(-1, self.X_train.shape[1])  ## P = delta.view(self.X_train.shape)
            P = P_aux[self.minibatch_indices]
            A = torch.tensor(self.X_train, dtype=P.dtype, device=P.device) + P
            residual = torch.tensor(self.y_train, dtype=A.dtype, device=A.device).reshape(-1,1).reshape(-1,1) - A @ theta
            grad_P = (-2 / self.b) * torch.outer(residual.view(-1), theta.view(-1)) 
            # out = grad_P.reshape(-1, 1)
    
            # Initialize full-sized gradient with zeros
            grad_full = torch.zeros_like(P_aux)
            # Place minibatch gradients in correct rows
            grad_full[self.minibatch_indices] = grad_P
            # Flatten to match the shape of delta
            out = grad_full.view(-1, 1)
        return out
    

    def phi_delta_gradient_lam(self, lam, theta, delta):  
        return np.zeros_like(lam)


    def phi_delta_gradient_delta(self, lam, theta, delta):  
        if self.batch_dataset:
            d = theta.shape[0]
            P = delta.reshape(self.X_train.shape[0], self.X_train.shape[1])
            out = (2 * self.c / (self.b * d)) * P
            out = out.reshape(-1,1)
            
        else:
            d = theta.shape[0]
            P_aux = delta.reshape(-1, self.X_train.shape[1])  ## P = delta.reshape(self.X_train.shape[0], self.X_train.shape[1])
            P = P_aux[self.minibatch_indices]
            out = (2 * self.c / (self.b * d)) * P
            # out.reshape(-1,1)
    
            # Initialize full-sized gradient with zeros
            grad_full = np.zeros_like(P_aux)
            # Place minibatch gradients in correct rows
            grad_full[self.minibatch_indices] = out
            # Flatten to match the shape of delta
            out = grad_full.reshape(-1, 1)
            
        return out


    def phi_delta_gradient_delta_torch(self, lam, theta, delta):  
        if self.batch_dataset:
            d = theta.shape[0]
            P = delta.view(self.X_train.shape)
            out = (2 * self.c / (self.b * d)) * P
            out = out.reshape(-1,1)
            
        else:
            d = theta.shape[0]
            P_aux = delta.view(-1, self.X_train.shape[1])  ## P = delta.view(self.X_train.shape)
            P = P_aux[self.minibatch_indices]
            out = (2 * self.c / (self.b * d)) * P
            # out.reshape(-1,1)
            
            # Initialize full-sized gradient with zeros
            grad_full = torch.zeros_like(P_aux)
            # Place minibatch gradients in correct rows
            grad_full[self.minibatch_indices] = out
            # Flatten to match the shape of delta
            out = grad_full.view(-1, 1)    
        
        return out
    

    def phi_delta_gradient_theta(self, lam, theta, delta):  
        return np.zeros_like(theta)


    def phi_delta_gradient_theta_torch(self, lam, theta, delta):  
        return torch.zeros_like(theta)
    

    def phi_theta_gradient_lam(self, lam, theta, delta):
        d = theta.shape[0]
        out = np.exp(lam) * self.smoothed_ell1_norm(theta) / d
        return out


    def phi_theta_gradient_delta(self, lam, theta, delta):
        return np.zeros_like(delta)


    def phi_theta_gradient_delta_torch(self, lam, theta, delta):  
        return torch.zeros_like(delta)
    

    def phi_theta_gradient_theta(self, lam, theta, delta):  
        d = theta.shape[0]
        out = (np.exp(lam) / d) * self.smoothed_ell1_gradient(theta)
        return out


    def phi_theta_gradient_theta_torch(self, lam, theta, delta):  
        d = theta.shape[0]
        out = (torch.exp(lam) / d) * self.smoothed_ell1_gradient(theta, flag_pytorch=True)
        return out         
    
    
    def f_1(self, x, y, z):
        """
        The upper-level objective function
      	""" 
        if self.remove_ul:
            ## Compute f_1 using f_2
            if not self.sato_formulation_min_max_min:
                out = self.g(x, y, z) + self.phi_theta(x, y, z)
                
            if self.sato_formulation_min_max_min:
                out = - (self.g(x, z, y) - self.phi_delta(x, z, y))
          
        else:
            if not self.sato_formulation_min_max_min:
                out = self.f_1_general(x, y, z)
    
            if self.sato_formulation_min_max_min:
                out = self.f_1_general(x, z, y)
            
        return out
    

    def f_2(self, x, y, z):
        """
        The middle-level objective function
     	"""
        if not self.sato_formulation_min_max_min:
            out = self.g(x, y, z) + self.phi_theta(x, y, z)
            
        if self.sato_formulation_min_max_min:
            out = - (self.g(x, z, y) - self.phi_delta(x, z, y))
            
        return out


    def f_3(self, x, y, z):
        """
        The lower-level objective function
     	"""
        if self.remove_ll:
            out = 0
            
        else:
            if not self.sato_formulation_min_max_min:
                out = - (self.g(x, y, z) - self.phi_delta(x, y, z)) 
    
            if self.sato_formulation_min_max_min:
                out = self.g(x, z, y) + self.phi_theta(x, z, y)  
        
        return out
    

    def grad_f1_ul_vars(self, x, y, z):
        """
        The gradient of the upper-level objective function wrt the upper-level variables
    	  """
        if self.remove_ul:
            out = np.zeros_like(x)

        else:
            if not self.sato_formulation_min_max_min:
                out = self.f_1_general_gradient_lam(x, y, z)
    
            if self.sato_formulation_min_max_min:
                out = self.f_1_general_gradient_lam(x, z, y)
            
        return out


    def grad_f1_ml_vars(self, x, y, z):    
        """
        The gradient of the upper-level objective function wrt the middle-level variables
    	  """
        if not self.sato_formulation_min_max_min:
            if self.remove_ul:
                out = np.zeros_like(y)  
            else:
                out = self.f_1_general_gradient_theta(x, y, z)

        if self.sato_formulation_min_max_min:
            if self.remove_ul:
                out = np.zeros_like(z) 
            else:
                out = self.f_1_general_gradient_delta(x, z, y)
                
        return out


    def grad_f1_ml_vars_torch(self, x, y, z):    
        """
        The gradient of the upper-level objective function wrt the middle-level variables
    	  """
        if not self.sato_formulation_min_max_min:
            if self.remove_ul:
                out = torch.zeros_like(y)  
            else:
                out = self.f_1_general_gradient_theta_torch(x, y, z)

        if self.sato_formulation_min_max_min:
            if self.remove_ul:
                out = torch.zeros_like(z) 
            else:
                out = self.f_1_general_gradient_delta_torch(x, z, y)
                
        return out
    

    def grad_f1_ll_vars(self, x, y, z):    
        """
        The gradient of the upper-level objective function wrt the lower-level variables
      	"""
        if not self.sato_formulation_min_max_min:
            if self.remove_ul:
                out = np.zeros_like(z)
            else:
                out = self.f_1_general_gradient_delta(x, y, z)
                
        if self.sato_formulation_min_max_min:
            if self.remove_ul:
                out = np.zeros_like(y)
            else:
                out = self.f_1_general_gradient_theta(x, z, y)           
                
        return out


    def grad_f2_ul_vars(self, x, y, z):     
        """
        The gradient of the middle-level objective function wrt the upper-level variables
       	""" 
        if not self.sato_formulation_min_max_min:
            out = self.g_gradient_lam(x, y, z) + self.phi_theta_gradient_lam(x, y, z) 
                
        if self.sato_formulation_min_max_min:
            out = -(self.g_gradient_lam(x, z, y) - self.phi_delta_gradient_lam(x, z, y))
            
        return out
    

    def grad_f2_ml_vars(self, x, y, z):
        """
        The gradient of the middle-level objective function wrt the middle-level variables
      	"""
        if not self.sato_formulation_min_max_min:
            out = self.g_gradient_theta(x, y, z) + self.phi_theta_gradient_theta(x, y, z)
                
        if self.sato_formulation_min_max_min:
            out = -(self.g_gradient_delta(x, z, y) - self.phi_delta_gradient_delta(x, z, y))
                
        return out


    def grad_f2_ml_vars_torch(self, x, y, z):
        """
        The gradient of the middle-level objective function wrt the middle-level variables
      	"""
        if not self.sato_formulation_min_max_min:
            out = self.g_gradient_theta_torch(x, y, z) + self.phi_theta_gradient_theta_torch(x, y, z)
                
        if self.sato_formulation_min_max_min:
            out = -(self.g_gradient_delta_torch(x, z, y) - self.phi_delta_gradient_delta_torch(x, z, y))

        return out
    

    def grad_f2_ll_vars(self, x, y, z):
        """
        The gradient of the middle-level objective function wrt the lower-level variables
      	"""
        if not self.sato_formulation_min_max_min:
            out = self.g_gradient_delta(x, y, z) + self.phi_theta_gradient_delta(x, y, z)
                
        if self.sato_formulation_min_max_min:
            out = -(self.g_gradient_theta(x, z, y) - self.phi_delta_gradient_theta(x, z, y)) 
                
        return out


    def grad_f2_ll_vars_torch(self, x, y, z):
        """
        The gradient of the middle-level objective function wrt the lower-level variables
      	"""
        if not self.sato_formulation_min_max_min:
            out = self.g_gradient_delta_torch(x, y, z) + self.phi_theta_gradient_delta_torch(x, y, z)
                
        if self.sato_formulation_min_max_min:
            out = -(self.g_gradient_theta_torch(x, z, y) - self.phi_delta_gradient_theta_torch(x, z, y)) 
                
        return out
    

    def grad_f3_ul_vars(self, x, y, z):
        """
        The gradient of the lower-level objective function wrt the upper-level variables
      	"""
        if self.remove_ll:
            out = np.zeros_like(x)
            
        else:
            if not self.sato_formulation_min_max_min:
                out = -(self.g_gradient_lam(x, y, z) - self.phi_delta_gradient_lam(x, y, z))
    
            if self.sato_formulation_min_max_min:
                out = self.g_gradient_lam(x, z, y) + self.phi_theta_gradient_lam(x, z, y)         
        
        return out
    
    
    def grad_f3_ml_vars(self, x, y, z):
        """
        The gradient of the lower-level objective function wrt the middle-level variables
    	"""
        if not self.sato_formulation_min_max_min:
            if self.remove_ll:
                out = np.zeros_like(y)
            else:
                out = -(self.g_gradient_theta(x, y, z) - self.phi_delta_gradient_theta(x, y, z))
                
        if self.sato_formulation_min_max_min:
            if self.remove_ll:
                out = np.zeros_like(z)
            else:
                out = self.g_gradient_delta(x, z, y) + self.phi_theta_gradient_delta(x, z, y)         
                
        return out
    

    def grad_f3_ll_vars(self, x, y, z):
        """
        The gradient of the lower-level objective function wrt the lower-level variables
    	"""
        if not self.sato_formulation_min_max_min:
            if self.remove_ll:
                out = np.zeros_like(z)
            else:
                out = -(self.g_gradient_delta(x, y, z) - self.phi_delta_gradient_delta(x, y, z))
                
        if self.sato_formulation_min_max_min:
            if self.remove_ll:
                out = np.zeros_like(y)
            else:
                out = self.g_gradient_theta(x, z, y) + self.phi_theta_gradient_theta(x, z, y)                           
                
        return out


    def grad_f3_ll_vars_torch(self, x, y, z):
        """
        The gradient of the lower-level objective function wrt the lower-level variables
     	""" 
        if not self.sato_formulation_min_max_min:
            if self.remove_ll:
                out = torch.zeros_like(z)
            else:
                out = -(self.g_gradient_delta_torch(x, y, z) - self.phi_delta_gradient_delta_torch(x, y, z))
                
        if self.sato_formulation_min_max_min:
            if self.remove_ll:
                out = torch.zeros_like(y)
            else:
                out = self.g_gradient_theta_torch(x, z, y) + self.phi_theta_gradient_theta_torch(x, z, y)
                
        return out     
    
    

















