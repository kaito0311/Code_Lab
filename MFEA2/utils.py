import numpy as np
from numpy.random import choice
import random
from scipy.stats import multivariate_normal
from scipy.stats import norm



class sphere_function:
    def __init__(self, dimension, delta =0):
        self.dimension = dimension
        self.delta = delta 

    def decode(self, array_value):
        array_value = array_value[:self.dimension]
        return array_value

    def calculate_fitness(self, array_value):
        x = self.decode(array_value)
        # print(array_value.shape)
        # print(array_va)
        x = x - self.delta
        sum = np.sum(x * x, keepdims=True)
        return float(sum)


class rastrigin_function:
    def __init__(self, dimension, A=10):
        self.dimension = dimension
        self.A = A

    def decode(self, array_value):
        array_value = array_value[:self.dimension]
        return np.array(array_value)
    def calculate_fitness(self, array_value):
        x = self.decode(array_value)
        sum = self.A * self.dimension + \
            np.sum(x * x) - self.A * np.sum(np.cos(2 * np.pi * np.cos(x)))
        return float(sum)


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


def compute_gauss_density(mean, cov, x, task, skill_factor):
    """
    mean: shape (1, dimension) 
    cov : shape (dimension, dimension) 
    x: shape (number """
    # print(x.shape)
    number_population, _ = x.shape 
    # mean = np.reshape(mean, (1, dimension))
    xsuat = 0.0
    f = np.zeros((x.shape[0])) 
    for i in range(number_population):
        # print(x[i].T) 
        f[i] = multivariate_normal.pdf(x[i].T, mean, cov)
        if(skill_factor[i] == task):
            xsuat += f[i]
    
    return f


def learn_probabilistic_model(population, skill_factor):
    '''
    Tạo ra mô hình xác suất với phân phối chuẩn dựa trên quần thể\n

    Returns:
    mô hình xác suất: shape (number_task, number_population)
    '''

    number_task = np.max(skill_factor) + 1
    number_population, dimension = population.shape
    task_index = np.zeros(number_task, dtype=int)

    u = np.zeros((number_task, dimension), dtype=float)
    cov = np.zeros((number_task, dimension, dimension), dtype=float)

    # Tach quan the cha me thanh cac quan the con ứng với mỗi task
    subpopulation = np.zeros(
        (number_task, int(number_population/number_task), dimension), dtype=float)
    for individual in range(number_population):
        task = skill_factor[individual]
        subpopulation[task][task_index[task]] = np.copy(population[individual])
        task_index[task] += 1

    prob_model = np.zeros((number_task, number_population), dtype=float)

    # Tính trung bình (kỳ vọng) và độ lệch chuẩn cho mỗi task ứng với mỗi chiều
    for task in range(number_task):
        u[task] = np.mean(subpopulation[task], axis=0, keepdims=True)
        cov[task] = np.cov(subpopulation[task].T)
        # Tạo ra mô hình xac suat
        prob_model[task] = compute_gauss_density_v2(
            u[task], cov[task], population, task, skill_factor)
    print(prob_model)
    return prob_model

def compute_gauss_density_v2(u, std, population): 

    
    number_population, dimensions = population.shape 
    pro_model = np.ones((number_population), dtype = float) 
    # for individual in range(number_population): 
    for dimension in range(dimensions) : 
        pro_model *= norm.pdf(population[:,dimension], loc = u[dimension], scale = std[dimension]) 
    return pro_model


def learn_probabilistic_model_v2(population, skill_factor):
    number_task = np.max(skill_factor) + 1; 
    number_population, dimension = population.shape 
    task_index = np.zeros(number_task, dtype = int) 

    u = np.zeros((number_task, dimension), dtype = float) 

    std = np.zeros((number_task, dimension), dtype = float) 

    subpopulation = np.zeros( shape = (number_task, int(number_population/number_task), dimension), dtype= float) 

    for individual in range(number_population): 
        task = skill_factor[individual] 
        subpopulation[task][task_index[task]] = np.copy(population[individual])
        task_index[task] +=1 
    
    pro_model = np.zeros((number_task, number_population), dtype = float) 

    for task in range(number_task): 
        u[task] = np.mean(subpopulation[task], axis = 0, keepdims= True) 
        std[task] = np.std(subpopulation[task], axis =0, keepdims = True) 
        pro_model[task] = compute_gauss_density_v2(u[task], std[task], population) 

    return pro_model

def convert_1D_to_matrix(variable, number_task, diagonal_value=0):
    rows = cols = number_task
    rmp = np.zeros((number_task, number_task), dtype=float)
    index = 0
    for row in range(rows):
        rmp[row][row] = diagonal_value
        for col in range(row + 1, cols):
            rmp[row][col] = rmp[col][row] = variable[index]
            index += 1
    return rmp


def convert_matrix_to_1D(rmp, number_task):
    index = 0
    variable = np.zeros(int(number_task * (number_task - 1) / 2), dtype=float)
    for row in range(number_task):
        for col in range(row+1, number_task):
            variable[index] = rmp[row][col]
    return variable


def optimize_rmp(variable, g, population, scalar_fitness, skill_factor):
    '''
    rmp: shape( numbertask , number_task) 
    g: (probabilisticc model) shape( number_task, numberpopulation) 
    population: (numberpopulation, dimension) 
    // select ontop parent 
    '''
    '''
    chon ra K * N/2 cá thể từ population với scalar fitness >= 2/N
    thiết lập biểu thức
    '''
    number_population, dimension = population.shape
    number_task = len(g)
    gc = np.ones((number_task, number_population), dtype=float)

    rmp = convert_1D_to_matrix(variable, number_task)
    rmp = np.reshape(rmp, (-1, number_task))
    g.reshape((number_task, number_population))
    count = 0
    # print(rmp)
    for individual in range(len(population)):
        if scalar_fitness[individual] >= 2.0 / float(number_population/number_task):
            k = skill_factor[individual]
            gc[k][individual] *= (1 - 0.5/number_task * (np.sum(rmp[k]))) * g[k][individual] + \
                (0.5 / number_task) * \
                (np.sum(np.multiply(rmp[k], g.T[individual])))

    # print(count)
    variable = convert_matrix_to_1D(rmp, number_task)
    result = np.sum(np.sum(np.log(gc)))
    # print("variable: " , variable)
    
    # print(result) 
    # result = -result;
    return -result


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


def compute_beta_for_sbx(u, nc=10):
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


def poly_mutation(parent, nm=20.0):
    u = random.random()
    rand = random.random()

    r = 1 - np.power(2.0 * (1.0 - u), 1.0/(nm + 1.0))
    l = np.power(2.0 * u, 1.0 / (nm + 1.0)) - 1
    # p = 1.0 / float(len(parent))
    child = np.zeros_like(parent)
    if u > 0.5:
        child = parent * (1.0 - r) + r
    else:
        child = parent * (1 + l)
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
        child1 = poly_mutation(child1)
        child2 = poly_mutation(child2) 
        skill_factor_child[1] = skill_factor_child[0] = skill_factor[p2] 
    
    return child1, child2, skill_factor_child[0], skill_factor_child[1] 

        
def find_individual_same_skill( skill_factor, individual):
    a = np.array(np.where(skill_factor == skill_factor[individual]))
    # print(a.shape)
    result = np.random.choice(a.flatten())

    return int(result)


def different_evolution(xj_parent0, xj_parent1, F = 0.8):
    return xj_parent0 + F * (xj_parent0 - xj_parent1) 

def intra_task_crossover(parent, population, skill_factor, tasks):
    parent1 = parent
    while parent1 == parent and skill_factor[parent] != skill_factor[parent1]: 
        parent1 = choice(np.arange(len(population)), size = 1)

    _, dimensions = population.shape 
    child = np.copy(population[parent])
    for dimension in range(dimensions):
        child[dimension] =  different_evolution(parent[dimension], parent1[dimension])
        if tasks[skill_factor[parent]].calculate_fitness(child) < tasks[skill_factor[parent]].calculate_fitness(population[parent]):
            population[parent][dimension] = child[dimension] 
        else: 
            child[dimension] = population[parent][dimension] 
    return child



def inter_task_crossover(parent0, parent1, population, skill_factor):
    child1, child2 = sbx_crossover(population[parent0], population[parent1])
    return child1, child2 
     

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
