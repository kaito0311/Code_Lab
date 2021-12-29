import numpy as np
from numpy.core.defchararray import split
from numpy.lib.npyio import load

a = np.array([[1, 2], [3, 4]])
b = np.array([3, 4])


def load_file(path_file):
    input = np.loadtxt(path_file)
    return input

# print(load_file("Gecco/Tasks/benchmark_1/matrix_1"))


from utils_mf import * 

shift = np.ones((50))
matrix_rotate = np.eye(50)

sphere = Sphere(matrix_rotate, shift)
print(sphere.decode(np.zeros(50)))
sum = sphere.calculate_fitness(np.zeros((50)))
print(sum)
print(sphere.bias)
