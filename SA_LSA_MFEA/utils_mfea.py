
import numpy as np
from numpy.random import choice
import random
from scipy.stats import multivariate_normal
from scipy.stats import norm
from utils_mf import * 


def create_population(number_subpopulation, number_tasks, dimension, lower, upper):
    population = np.random.uniform(
        low=lower, high=upper, size=(number_subpopulation * number_tasks, dimension))
    return population

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

def create_child_v2(parent1, parent2, skill_factor, population, rmp):
    skill_factor_child = np.zeros((2), dtype= int)
    child1 = child2 = None 
    if(skill_factor[parent1] == skill_factor[parent2]):
        skill_factor_child[0] = skill_factor[parent1]
        skill_factor_child[1] = skill_factor[parent2]
        child1, child2 = sbx_crossover(population[parent1], population[parent2])
    elif random.random() < rmp:
        child1, child2 = sbx_crossover(population[parent1], population[parent2]) 

        skill_factor_child[0], skill_factor_child[1] =\
            np.random.choice([skill_factor[parent1], skill_factor[parent2]], replace= True, size=(2,))
    else: 
        p2 = find_individual_same_skill(skill_factor, parent1)
        child1, _ = sbx_crossover(population[parent1], population[p2])
        skill_factor_child[0] = skill_factor[p2] 

        p3 = find_individual_same_skill(skill_factor, parent2)
        _, child2 = sbx_crossover(population[parent2], population[p3])
        skill_factor_child[1] = skill_factor[p3] 

    child1 = poly_mutation(child1)
    child2 = poly_mutation(child2) 
    return child1, child2, skill_factor_child[0], skill_factor_child[1] 

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

    top_parent = np.where(scalar_fitness >= 0.0 / number_subpopulation)[0]

    index_parent_not_enough_child = np.where(number_child[skill_factor[top_parent]] < number_subpopulation)[0] 
    parent = np.random.choice(top_parent[index_parent_not_enough_child], size= (2))
    
    return parent

def learn_probabilistic_model(population, skill_factor, scalar_fitness):
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
        if(scalar_fitness[individual] >= 2.0 / ( number_population / number_task)):
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


def learn_probabilistic_model_v2(population, skill_factor, scalar_fitness):
    number_task = np.max(skill_factor) + 1; 
    number_population, dimension = population.shape 
    task_index = np.zeros(number_task, dtype = int) 

    u = np.zeros((number_task, dimension), dtype = float) 
    std = np.zeros((number_task, dimension), dtype = float) 

    subpopulation = np.zeros( shape = (number_task, int(number_population/(number_task*2)), dimension), dtype= float) 

    for individual in range(number_population): 
        if scalar_fitness[individual] >= 2.0 / (number_population * 1.0 / number_task):
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

    variable = convert_matrix_to_1D(rmp, number_task)
    result = np.sum(np.sum(np.log(gc)))

    return -result
