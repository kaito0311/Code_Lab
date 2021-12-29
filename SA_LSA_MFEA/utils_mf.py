import numpy as np
from numpy.core.defchararray import array, find
from numpy.random import choice
import random
from scipy.stats import multivariate_normal
from scipy.stats import norm

from utils_sa_lsa import index_convert_matrix_to_1D

class function: 
    def __init__(self,  matrix, bias,dimension = 50, lower = -50, upper = 50):
        self.dimension= dimension 
        self.matrix = matrix
        self.bias = bias 
        self.lower = lower 
        self.upper = upper 
        self.name = self.__class__.__name__ + " _dimension: " + str(dimension)
    

    
    def decode(self, array_value):
        array_value = array_value[: self.dimension]
        x = array_value
        x = x * (self.upper - self.lower) + self.lower
        # x = (np.dot(np.linalg.inv(self.matrix), (x - self.bias).reshape((self.dimension, 1)))); 
        x = (np.dot((self.matrix), (x - self.bias).reshape((self.dimension, 1)))); 
        return x

class Rosenbrock(function):
    def calculate_fitness(self, array_value):
        x = self.decode(array_value) 
        sum = 0

        l = 100*np.sum((np.delete(x, 0, 0) - np.delete(x, -1, 0 )**2) ** 2)
        r = np.sum((np.delete(x, -1, 0) - 1) ** 2)
        sum = l + r
        return sum 

class Ackley(function):
    def calculate_fitness(self, array_value): 
        x = self.decode(array_value)
        sum1, sum2 = 0,0
        # for i in range(self.dimension):
        #     sum1 += x[i] ** 2
        #     sum2 += np.cos(2 * np.pi * x[i]) 
        sum1 = np.sum(x**2) 
        sum2 = np.sum(np.cos(2 * np.pi * x))
        
        avg = sum1/self.dimension 
        avg2 = sum2/self.dimension 
    
        return -20 * np.exp(-0.2*np.sqrt(avg)) - np.exp(avg2) + 20 + np.exp(1)

class Girewank(function):
    def calculate_fitness(self, array_value):
        x = self.decode(array_value) 
        sum1 = 0
        sum2 = 1 
        sum1 = np.sum(x ** 2)

        # for i in range(self.dimension):
        #     sum2 *= np.cos(x[i]/np.sqrt(i+1))
        sum2 = np.prod(np.cos(x/np.sqrt(np.arange(self.dimension) + 1)))
        
        return 1 + 1/4000 * sum1 - sum2


class Rastrigin(function): 
    def calculate_fitness(self, array_value): 
        x = self.decode(array_value)
        sum = 10 * self.dimension

        # for i in range(self.dimension): 
        #     sum += (x[i] ** 2 - 10 * np.cos(2 * np.pi * x[i])) 
        sum += np.sum(x**2)
        sum += np.sum(-10 * np.cos(2 * np.pi * x))
        
        return sum; 


class Schewefel(function): 
    def calculate_fitness(self, array_value): 
        x = self.decode(array_value) 
        sum = 0 
        # for i in range(self.dimension): 
        #     sum += x[i] * np.sin(np.sqrt(np.abs(x[i])))
        sum = np.sum(x * np.sin(np.sqrt(np.abs(x))))
        return 418.9829 * self.dimension - sum 
        

class Sphere(function):
    def calculate_fitness(self, array_value): 
        x = self.decode(array_value) 
        sum = 0; 
        x = x**2
        sum = np.sum(x)
        sum = float(sum) 
        return sum; 

class Weierstrass(function):
    kmax = 21
    a = 0.5 
    b = 3 
    vt1 = np.power(np.zeros(kmax) + a, np.arange(kmax))
    vt2 = np.power(np.zeros(kmax) + b, np.arange(kmax))
    
    def calculate_fitness(self, array_value): 
        x = self.decode(array_value) 
        sum = 0 

       
        # for i in range(self.dimension): 
        #     for k in range(kmax): 
        #         sum += np.power(a, k) *  np.cos(2 * np.pi * np.power(b, k) * (x[i] + 0.5))


        # for i in range(self.dimension):
        #     for k in range(kmax): 
        #         sum += vt1[k] * np.cos(2 * np.pi * vt2[k] * (x[i] + 0.5))
        
        # for k in range(kmax):
        #     sum -= self.dimension * vt1[k] * np.cos(2 * np.pi * vt2[k] * 0.5) 
        # return sum
        left = 0
        for i in range(self.dimension):
            left += np.sum(self.a ** np.arange(self.kmax) * \
                np.cos(2*np.pi * self.b ** np.arange(self.kmax) * (x[i]  + 0.5)))
            
        right = self.dimension * np.sum(self.a ** np.arange(self.kmax) * \
            np.cos(2 * np.pi * self.b ** np.arange(self.kmax) * 0.5)
        )
        return left - right


# class rosenbrock_function:
#     def __init__(self, dimension, delta=0, lower=-500, upper=500):
#         self.dimension = dimension
#         self.delta = delta
#         self.lower = lower
#         self.upper = upper
#         self.name = "rosenbrock " + f"{delta} "

#     def decode(self, array_value):
#         array_value = array_value[: self.dimension]
#         return array_value

#     def calculate_fitness(self, array_value):
#         x = self.decode(array_value)
#         x = x * (self.upper - self.lower) + self.lower
#         x = x - self.delta
#         sum = 0
#         for i in range(self.dimension - 1):
#             sum += 100.0 * (x[i] * x[i] - x[i + 1]) * (x[i] * x[i] - x[i + 1]) + 1.0 * (x[i] - 1) * (x[i] - 1)
#         return float(sum)


# class schwefel_function:
#     def __init__(self, dimension, delta=0, lower=-500, upper=500):
#         self.dimension = dimension
#         self.delta = delta
#         self.lower = lower
#         self.upper = upper
#         self.name = "schwefel " + f"{delta} "

#     def decode(self, array_value):
#         array_value = array_value[: self.dimension]
#         return array_value

#     def calculate_fitness(self, array_value):
#         x = self.decode(array_value)
#         x = x * (self.upper - self.lower) + self.lower
#         x = x - self.delta
#         sum = 418.9829 * self.dimension - np.sum(x * np.sin(np.sqrt(np.abs(x))))
#         return float(sum)


# class griewank_function:
#     def __init__(self, dimension, delta=0, lower=-100, upper=100):
#         self.dimension = dimension
#         self.delta = delta
#         self.lower = lower
#         self.upper = upper
#         self.name = "griewank " + f"{delta} "

#     def decode(self, array_value):
#         array_value = array_value[: self.dimension]
#         return array_value

#     def calculate_fitness(self, array_value):
#         x = self.decode(array_value)
#         x = x * (self.upper - self.lower) + self.lower
#         x = x - self.delta
#         i = np.arange(self.dimension) + 1
#         sum = np.sum(x * x / 4000) - np.prod(np.cos(x / np.sqrt(i))) + 1
#         return float(sum)


# class sphere_function:
#     def __init__(self, dimension, delta=0, lower=-100, upper=100):
#         self.dimension = dimension
#         self.delta = delta
#         self.lower = lower
#         self.upper = upper
#         self.name = "sphere " + f"{delta} "

#     def decode(self, array_value):
#         array_value = array_value[: self.dimension]
#         return array_value

#     def calculate_fitness(self, array_value):
#         x = self.decode(array_value) 
#         x = x * (self.upper - self.lower) + self.lower
#         x = x - self.delta
#         sum = np.sum(x * x, keepdims=True)
#         return float(sum)


# class rastrigin_function:
#     def __init__(self, dimension, A=10, delta=0, lower=-50, upper=50):
#         self.dimension = dimension
#         self.A = A
#         self.delta = delta
#         self.lower = lower
#         self.upper = upper
#         self.name = "rastrigin " + f"{delta} "

#     def decode(self, array_value):
#         array_value = array_value[: self.dimension]
#         return np.array(array_value)

#     def calculate_fitness(self, array_value):
#         x = self.decode(array_value)
#         x = x * (self.upper - self.lower) + self.lower
#         x = x - self.delta
#         sum = (
#             self.A * self.dimension
#             + np.sum(x * x)
#             - self.A * np.sum(np.cos(2 * np.pi * np.cos(x)))
#         )
#         return float(sum)


# class ackley_function:
#     def __init__(
#         self, dimension, delta=0, lower=-50, upper=50, a=20, b=0.2, c=2 * np.pi
#     ):
#         self.dimesion = dimension
#         self.delta = delta
#         self.a = a
#         self.b = b
#         self.c = c
#         self.lower = lower
#         self.upper = upper
#         self.name = "ackley " + f"{delta} "

#     def decode(self, array_value):
#         array_value = array_value[: self.dimesion]
#         return np.array(array_value)

#     def calculate_fitness(self, array_value):
#         x = self.decode(array_value)
#         x = x * (self.upper - self.lower) + self.lower
#         x = x - self.delta
#         result = (
#             -self.a * np.exp(-self.b * np.sqrt(1.0 / self.dimesion * np.sum(x * x)))
#             - np.exp(1.0 / self.dimesion * np.sum(np.cos(self.c * x)))
#             + self.a
#             + np.exp(1)
#         )
#         return result


# class weitr_function:
#     def __init__(self, dimension, delta=0, lower=-0.5, upper=0.5, a=0.5, b=3, k=20):
#         self.dimesion = dimension
#         self.delta = delta
#         self.a = a
#         self.b = b
#         self.k = k
#         self.lower = lower
#         self.upper = upper

#     def decode(self, array_value):
#         array_value = array_value[: self.dimesion]
#         return np.array(array_value)

#     def calculate_fitness(self, array_value):
#         x = self.decode(array_value)
#         x = x * (self.upper - self.lower) + self.lower
#         x = x - self.delta
#         c = 0
#         for k in range(self.k):
#             x += np.power(self.a, k) * np.cos(
#                 2 * np.pi * np.power(self.b, k) * (x + 0.5)
#             )
#             c += np.power(self.a, k) * np.cos(2 * np.pi * np.power(self.b, k) * (0.5))

#         result = np.sum(x) - self.dimesion * c * 1.0
#         return result


def create_population(number_population, dimension, lower, upper):
    population = np.random.uniform(
        low=lower, high=upper, size=(number_population, dimension)
    )
    return population


def generate_population(population, tasks, number_population):
    """
    Gán cho các cá thể trong quần thể với skill factor ngẫu nhiên.\n
    Tính toán factorial cost tương ứng với skill factor đã được gán
    """
    number_task = len(tasks)
    skill_factor = np.zeros(number_population, dtype=int)
    factorial_cost = np.zeros(number_population, dtype=float)
    number_task = len(tasks)
    for individual in range(len(population)):
        best_task = individual % number_task
        skill_factor[individual] = best_task
        factorial_cost[individual] = tasks[best_task].calculate_fitness(
            population[individual]
        )

    return skill_factor, factorial_cost


def compute_scalar_fitness(factorial_cost, skill_factor):
    number_task = np.max(skill_factor) + 1
    number_population = len(factorial_cost)
    temp = [[] for i in range(number_task)]
    index = [[] for i in range(number_task)]
    scalar_fitness = np.zeros_like(skill_factor, dtype=float)
    for ind in range(number_population):
        task = skill_factor[ind]
        temp[task].append(factorial_cost[ind])
        index[task].append(ind)

    for task in range(number_task):
        index_sorted = np.argsort(np.array(temp[task]))
        for order in range(len(index_sorted)):
            scalar_fitness[index[task][index_sorted[order]]] = 1.0 / float(1.0 + order)

    return scalar_fitness


def compute_beta_for_sbx(u, nc=2):
    if u < 0.5:
        return np.power(2.0 * u, 1.0 / float(nc + 1))
    else:
        return np.power(1.0 / (2.0 * (1 - u)), 1.0 / (nc + 1))


def sbx_crossover(p1, p2):
    sbxdi = 2
    D = p1.shape[0]
    cf = np.empty([D])
    u = np.random.rand(D)        

    cf[u <= 0.5] = np.power((2 * u[u <= 0.5]), (1 / (sbxdi + 1)))
    cf[u > 0.5] = np.power((2 * (1 - u[u > 0.5])), (-1 / (sbxdi + 1)))

    c1 = 0.5 * ((1 + cf) * p1 + (1 - cf) * p2)
    c2 = 0.5 * ((1 + cf) * p2 + (1 - cf) * p1)

    c1 = np.clip(c1, 0, 1)
    c2 = np.clip(c2, 0, 1)

    return c1, c2

# def sbx_crossover(parent1, parent2):
#     # rand = random.random()
#     # beta = compute_beta_for_sbx(rand)

#     # child1 = 0.5 * ((1.0 + beta) * parent1 + (1.0 - beta) * parent2)
#     # child2 = 0.5 * ((1.0 - beta) * parent1 + (1.0 + beta) * parent2)

#     rand = np.random.rand(len(parent1))
#     beta = np.zeros_like(rand) 
#     # for i in range(len(rand)):
#     #     beta[i] = compute_beta_for_sbx(rand[i])
#     nc = 2 
#     beta = np.where(rand <= 0.5, np.power(2.0 * rand, 1.0 / float(nc + 1)), np.power(1.0 / (2.0 * (1 - rand)), 1.0 / (nc + 1)))

#     child1 = 0.5 * ((1.0 + beta) * parent1 + (1.0 - beta) * parent2)
#     child2 = 0.5 * ((1.0 - beta) * parent1 + (1.0 + beta) * parent2)

#     child1, child2 = np.clip(child1, 0, 1), np.clip(child2, 0, 1)
#     return child1, child2


# def poly_mutation(parent, nm=5):
#     # u = random.random()

#     # r = 1 - np.power(2.0 * (1.0 - u), 1.0 / (nm + 1.0))
#     # l = np.power(2.0 * u, 1.0 / (nm + 1.0)) - 1
#     # # p = 1.0 / float(len(parent))
#     # child = np.zeros_like(parent)
#     # if u <= 0.5:
#     #     child = parent + l * parent
#     # else:
#     #     child = parent + r * (1 - parent)


#     child = np.copy(parent)
#     u = np.random.rand(len(parent))
#     r = 1 - np.power(2.0 * (1.0 - u), 1.0 / (nm + 1.0))
#     l = np.power(2.0 * u, 1.0 / (nm + 1.0)) - 1
#     rand = np.random.rand(len(parent))
#     a = np.where(u <= 0.5, parent *(1+l) , parent + r * (1-parent))
#     child = np.where(rand < 1.0/len(parent), a, parent)
    
#     return child
def poly_mutation(p, pmdi=5):
    # u = random.random()

    # r = 1 - np.power(2.0 * (1.0 - u), 1.0 / (nm + 1.0))
    # l = np.power(2.0 * u, 1.0 / (nm + 1.0)) - 1
    # # p = 1.0 / float(len(parent))
    # child = np.zeros_like(parent)
    # if u <= 0.5:
    #     child = parent + l * parent
    # else:
    #     child = parent + r * (1 - parent)


    mp = float(1. / p.shape[0])
    u = np.random.uniform(size=[p.shape[0]])
    r = np.random.uniform(size=[p.shape[0]])
    tmp = np.copy(p)
    for i in range(p.shape[0]):
        if r[i] < mp:
            if u[i] < 0.5:
                delta = (2*u[i]) ** (1/(1+pmdi)) - 1
                tmp[i] = p[i] + delta * p[i]
            else:
                delta = 1 - (2 * (1 - u[i])) ** (1/(1+pmdi))
                tmp[i] = p[i] + delta * (1 - p[i])
    tmp = np.clip(tmp, 0, 1)

    return tmp 


def find_individual_same_skill(skill_factor, individual):
    a = np.array(np.where(skill_factor == skill_factor[individual]))
    result = np.random.choice(a.flatten())

    return int(result)


def evaluate_child(childs, tasks, skill_factor_child):
    number_child = len(skill_factor_child)
    factorial_cost_child = np.zeros((number_child), dtype=float)
    for index in range(number_child):
        factorial_cost_child[index] = tasks[
            skill_factor_child[index]
        ].calculate_fitness(childs[index])

    return factorial_cost_child


def optimize_result(population, skill_factor, factorial_cost, tasks):
    class result:
        def __init__(self, cost=100000, task=-1):
            self.cost = cost
            self.task = task

    results = [result(task=i) for i in range(np.max(skill_factor) + 1)]

    for i in range(len(population)):
        if results[skill_factor[i]].cost > factorial_cost[i]:
            results[skill_factor[i]].cost = factorial_cost[i]
    # for result in results:
    #     print("tasks: {} | cost: {} ".format(result.task, result.cost))

    return results
