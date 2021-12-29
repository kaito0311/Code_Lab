import numpy as np
import tensorflow as tf
from utils import * 
import matplotlib.pyplot as plt 

task = ackley_function(3)
a = np.array([0.5,0.5,0.5,0.5])
print(task.calculate_fitness(a))