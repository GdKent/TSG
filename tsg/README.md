					  tsg (Version 0.1, May 2025)
					  ===========================
		  
						   Read me
						 ----------


tsg is the name of the code associated with the manuscript titled “A stochastic gradient method for trilevel optimization”.


tsg is freely available for research, educational or commercial use, under a GNU lesser general public license.


This file describes the following topics:

1. System requirements
2. Code structure
3. How to reproduce the results in the manuscript
4. The tsg team 


## 1. System requirements

The code is written in Python 3.

To run the code, first install the required dependencies using the command: pip install -r requirements.txt.

We recommend using Python 3.10.15. System requirements are available at 
https://www.python.org/downloads/release/python-31015

## 2. Code structure

The code includes the following files:

  - **trilevel_solver.py**: 
  
  Contains one class: 
	
 TrilevelSolver (used to implement gradient-based trilevel or bilevel optimization algorithms).

  - **functions.py**:	
  
  Contains several classes:
  
  TrilevelProblem (used to define the trilevel problem we aim to solve)
	
Quadratic (trilevel problem with quadratic objective functions at all levels)

Quartic (trilevel problem with quartic lower-level objective function)

AdversarialLearning (trilevel problem for adversarial hyperparameter tuning)

  - **driver_trilevel.py**:
         
  To run the numerical experiments and obtain the figures in the manuscript, or to conduct custom experiments.

In the driver_trilevel.py file, the dictionary exp_param_dict contains several parameters that can be used to define 
a numerical experiment. The values of such parameters can be set by modifying the values associated with the
corresponding keys.

  - **requirements.txt**:

  Contains all the Python packages and their specific versions needed to run the code.

  - **README.md**:    
  
  The current file.


## 3. How to reproduce the results in the manuscript 

To reproduce the results in the manuscript, simply run driver_trilevel.py as is. All figures, including those in the manuscript and technical appendices, will be saved as PDF files in the current working directory.


## 4. Citation 

In case you'd like to cite our work, please refer to the following paper: T. Giovannelli, G. D. Kent, and L. N. Vicente, A stochastic gradient method for trilevel optimization, arXiv preprint (2025).
     

## 5. The tsg team 

   - Tommaso Giovannelli (University of Cincinnati)
   - Griffin Dean Kent (Lehigh University)
   - Luis Nunes Vicente (Lehigh University)



