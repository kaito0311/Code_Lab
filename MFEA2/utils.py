import numpy as np
from numpy.random import choice
import random
from scipy.stats import multivariate_normal
from scipy.stats import norm



class sphere_function:
    def __init__(self, dimension, delta =0, lower = 0, upper = 50):
        self.dimension = dimension
        self.delta = delta 
        self.lower = lower 
        self.upper = upper 

    def decode(self, array_value):
        array_value = array_value[:self.dimension]
        return array_value

    def calculate_fitness(self, array_value):
        x = self.decode(array_value)
        x = x * (self.upper - self.lower) + self.lower 
        x = x - self.delta
        sum = np.sum(x * x, keepdims=True)
        return float(sum)

class rastrigin_function:
    def __init__(self, dimension, A=10, delta = 0, lower = 0, upper = 50):
        self.dimension = dimension
        self.A = A
        self.delta = delta 
        self.lower = lower 
        self.upper = upper 

    def decode(self, array_value):
        array_value = array_value[:self.dimension]
        return np.array(array_value)
    def calculate_fitness(self, array_value):
        x = self.decode(array_value)
        x = x * (self.upper - self.lower) + self.lower 
        x = x - self.delta
        sum = self.A * self.dimension + \
            np.sum(x * x) - self.A * np.sum(np.cos(2 * np.pi * np.cos(x)))
        return float(sum)

class ackley_function:
    def __init__(self, dimension, delta = 0,lower = 0, upper = 50,  a = 20, b = 0.2, c = 2 * np.pi):
        self.dimesion = dimension 
        self.delta = delta 
        self.a = a 
        self.b = b 
        self.c = c 
        self.lower = lower 
        self.upper = upper 
    def decode(self, array_value):
        array_value = array_value[:self.dimesion] 
        return np.array(array_value) 
    def calculate_fitness(self, array_value):
        x = self.decode(array_value) 
        x = x * (self.upper - self.lower) + self.lower 
        x = x - self.delta
        result = -self.a * np.exp(-self.b * np.sqrt(1.0 / self.dimesion * np.sum(x * x))) - np.exp(1.0 / self.dimesion * np.sum(np.cos(self.c * x))) + self.a + np.exp(1) 
        return result


def create_population(number_subpopulation, number_tasks, dimension, lower, upper):
    population = np.random.uniform(
        low=lower, high=upper, size=(number_subpopulation * number_tasks, dimension))
    return population


def generate_population(population, tasks, number_population):
    '''
    Gán cho các cá thể trong quần thể với skill factor ngẫu nhiên.\n
    Tính toán factorial cost tương ứng với skill factor đã được gán
    '''
    number_task = len(tasks)
    skill_factor = np.zeros(number_population, dtype=int)
    factorial_cost = np.zeros(number_population, dtype=float)
    number_task = len(tasks)
    for individual in range(len(population)):
        best_task = individual % number_task
        skill_factor[individual] = best_task
        factorial_cost[individual] = (
            tasks[best_task].calculate_fitness(population[individual]))

    return skill_factor, factorial_cost


def compute_scalar_fitness(population, skill_factor):
    number_task = np.max(skill_factor) + 1
    number_population = len(population)
    temp = [[] for i in range(number_task)]
    index = [[] for i in range(number_task)]
    scalar_fitness = np.zeros_like(skill_factor, dtype=float)
    for ind in range(number_population):
        task = skill_factor[ind]
        temp[task].append(population[ind])
        index[task].append(ind)

    for task in range(number_task):
        index_sorted = np.argsort(np.array(temp[task]))
        for order in range(len(index_sorted)):
            scalar_fitness[index[task][index_sorted[order]]
                           ] = 1.0 / float(1.0 + order)

    return scalar_fitness






def choose_parent(number_subpopulation, scalar_fitness, number_child, skill_factor):
    """
    Chon cha me o 1/2 phía trên
    Arguments: 
        number_subpopulation: so luong ca the o moi quan the con. Shape(N, 1) 
        scalar_fitness: scalar_fitness cho quan the cha me. Shape(number_subpopulation * number_task, 1) 
        number_child: so luong child da tao o moi task. shape(number_task,1)
    Returns:
        2 ca the cha me 
    """

    top_parent = np.where(scalar_fitness >= 2.0 / number_subpopulation)[0]

    index_parent_not_enough_child = np.where(number_child[skill_factor[top_parent]] < number_subpopulation)[0] 
    parent = np.random.choice(top_parent[index_parent_not_enough_child], size= (2))
    
    return parent


def compute_beta_for_sbx(u, nc=5):
    if u < 0.5:
        return np.power(2.0 * u, 1.0 / float(nc + 1))
    else:
        return np.power(1.0 / (2.0 * (1 - u)), 1.0/(nc + 1))


def sbx_crossover(parent1, parent2):
    rand = random.random()
    beta = compute_beta_for_sbx(rand)

    child1 = 0.5 * ((1.0 + beta) * parent1 + (1.0 - beta) * parent2)
    child2 = 0.5 * ((1.0 - beta) * parent1 + (1.0 + beta) * parent2)
    return child1, child2


def poly_mutation(parent, nm=40.0):
    u = random.random()

    r = 1 - np.power(2.0 * (1.0 - u), 1.0/(nm + 1.0))
    l = np.power(2.0 * u, 1.0 / (nm + 1.0)) - 1
    # p = 1.0 / float(len(parent))
    child = np.zeros_like(parent)
    if u <= 0.5:
        child = parent  + l * parent
    else:
        child = parent + r * (1 - parent)
    return child


def create_child(parent0, parent1, skill_factor_parent0, skill_factor_parent1, rmp):
    skill_factor_child = np.zeros((2), dtype=int)
    if skill_factor_parent1 == skill_factor_parent0:
        skill_factor_child[0] = skill_factor_parent0
        skill_factor_child[1] = skill_factor_parent1
        child1, child2 = sbx_crossover(parent0, parent1)
    elif random.random() < rmp:
        child1, child2 = sbx_crossover(parent0, parent1)
        if random.random() < 0.5:
            skill_factor_child[0] = skill_factor_parent0
            skill_factor_child[1] = skill_factor_parent1
        else:
            skill_factor_child[1] = skill_factor_parent0
            skill_factor_child[0] = skill_factor_parent1
    else:

        child1 = poly_mutation(parent0)
        skill_factor_child[0] = skill_factor_parent0
        child2 = poly_mutation(parent1)
        skill_factor_child[1] = skill_factor_parent1

    return (child1, child2, skill_factor_child[0], skill_factor_child[1])

def create_child_v2(parent1, parent2, skill_factor, population, rmp):
    skill_factor_child = np.zeros((2), dtype= int)
    child1 = child2 = None 
    if(skill_factor[parent1] == skill_factor[parent2]):
        skill_factor_child[0] = skill_factor[parent1]
        skill_factor_child[1] = skill_factor[parent2]
        child1, child2 = sbx_crossover(population[parent1], population[parent2])
    elif random.random() < rmp:
        child1, child2 = sbx_crossover(population[parent1], population[parent2]) 
        if random.random() < 0.5: 
            skill_factor_child[0] = skill_factor[parent1]
            skill_factor_child[1] = skill_factor[parent2]
        else : 
            skill_factor_child[1] = skill_factor[parent1] 
            skill_factor_child[0] = skill_factor[parent2]
    else: 
        p2 = find_individual_same_skill(skill_factor, parent1)
        child1, child2 = sbx_crossover(population[parent1], population[p2])
        skill_factor_child[1] = skill_factor_child[0] = skill_factor[p2] 

    child1 = poly_mutation(child1)
    child2 = poly_mutation(child2) 
    return child1, child2, skill_factor_child[0], skill_factor_child[1] 

        
def find_individual_same_skill( skill_factor, individual):
    a = np.array(np.where(skill_factor == skill_factor[individual]))
    # print(a.shape)
    result = np.random.choice(a.flatten())

    return int(result)
 

def evaluate_child(childs, tasks, skill_factor_child):
    number_child = len(skill_factor_child)
    factorial_cost_child = np.zeros((number_child), dtype=float)
    for index in range(number_child):
        factorial_cost_child[index] = tasks[skill_factor_child[index]].calculate_fitness(
            childs[index])

    return factorial_cost_child


def update(population, number_population, skill_factor, scalar_fitness, factorial_cost):
    temp = np.argpartition(-scalar_fitness, number_population)
    result_index = temp[:number_population]
    delete_index = []
    for i in range(len(population)):
        if i not in result_index:
            delete_index.append(i)
    # delete_index = np.array(delete_index, dtype= int)

    population = np.delete(population, delete_index, axis=0)
    factorial_cost = np.delete(factorial_cost, delete_index, axis=0)
    skill_factor = np.delete(skill_factor, delete_index, axis=0)
    scalar_fitness = np.delete(scalar_fitness, delete_index, axis=0)

    assert (len(population) == number_population)

    return population, skill_factor, scalar_fitness, factorial_cost

def optimize_result(population, skill_factor, factorial_cost, tasks):
    class result: 
        def __init__(self, cost = 100000, task = -1):
            self.cost = cost 
            self.task = task 
    
    results = [result(task=i) for i in range(np.max(skill_factor) + 1)]

    for i in range(len(population)):
        if results[skill_factor[i]].cost > factorial_cost[i] :
            results[skill_factor[i]].cost = factorial_cost[i] 
    # for result in results: 
    #     print("tasks: {} | cost: {} ".format(result.task, result.cost))
    
    return results