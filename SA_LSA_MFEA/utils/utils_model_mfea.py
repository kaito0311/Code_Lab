from typing import Tuple
import numpy as np 


def create_population(number_population, dimension, lower, upper) -> Tuple[np.array]:
    '''
    Arguments: 
        number_population: Number individuals in population 
        dimension : the number of genes in each individual
        lower: lower bound of gene 
        upper: upper bound of gene 
    Returns:
        population: the list of individual
    '''
    population = np.random.uniform(
        low=lower, high=upper, size=(number_population, dimension)
    )
    return population


def generate_population(population, tasks, number_population)->Tuple[np.array, np.array]:
    """
    Gán cho các cá thể trong quần thể với skill factor ngẫu nhiên.\n
    Tính toán factorial cost tương ứng với skill factor đã được gán

    Arguments: 
        population: np.array, list of individual 
        tasks: list of tasks 
        number_population: the number of individual 
    Returns: 
        skill_factors: skill factor corresponds with each individual 
        factorial_costs: factorial_cost of each individual in its task
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
    """
    Compute scalar fitness for individual in its task 

    Arguments: 
        factorial_cost: np.array(size population,) factorial cost of each individual 
        skill_factor: np.array(size population, ) skill_factor of each individual 

    Returns:
        Scalar fitness: np.array(size population, ) 1/rank of individual of its task. 
    """
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



def find_individual_same_skill(skill_factor, individual):
    """
    Returns: 
        index of other individual has skill as same as that of parameter individual
    """
    a = np.array(np.where(skill_factor == skill_factor[individual]))
    
    result= individual
    while result == individual:
        result = np.random.choice(a.flatten())

    return int(result)


def evaluate_child(childs, tasks, skill_factor_child):
    '''
    Returns: 
       factorial cost: compute factorial cost for each individual in list childs on its tasks
    '''
    number_child = len(skill_factor_child)
    factorial_cost_child = np.zeros((number_child), dtype=float)
    for index in range(number_child):
        factorial_cost_child[index] = tasks[
            skill_factor_child[index]
        ].calculate_fitness(childs[index])

    return factorial_cost_child


def optimize_result(population, skill_factor, factorial_cost, tasks):
    class result:
        def __init__(self, cost=1e10, task=-1):
            self.cost = cost
            self.task = task

    results = [result(task=i) for i in range(np.max(skill_factor) + 1)]

    for i in range(len(population)):
        if results[skill_factor[i]].cost > factorial_cost[i]:
            results[skill_factor[i]].cost = factorial_cost[i]
    # for result in results:
    #     print("tasks: {} | cost: {} ".format(result.task, result.cost))

    return results