from scipy import optimize
from scipy.sparse.linalg.eigen.arpack.arpack import ArpackNoConvergence
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.optimize import minimize
np.random.seed(1) 
# Creating the distribution
#Visualizing the distribution

# def ackley(x):
#     A = 10
#     sum = x[1] * 2 - x[0] * 3 + 1
#     return sum 

# x = np.random.rand((2))
# y = np.zeros((2))
# optimize = minimize(ackley, x, bounds = ((0,1), (0,1)))
# print(optimize)
# print(ackley(optimize.x))
# print(ackley(y))

y = 3

x = 5 if y > 3 else 4 
print(x) 