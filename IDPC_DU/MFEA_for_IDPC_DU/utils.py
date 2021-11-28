import numpy as np
from idpc_du import *


def compute_max_out_edges(tasks, NODES):
    NUMBER_TASK = len(tasks)
    max_out_edges = np.zeros((NODES), dtype=int)

    for i in range(NUMBER_TASK):
        for node in range(tasks[i].number_nodes):
            a = []
            for domain in range(tasks[i].number_domains):
                a = a + np.where(tasks[i].E[domain][node] != 0)[0].tolist()
            max_out_edges[node] = max(len(set(a)), max_out_edges[node])
    return max_out_edges


def pmx_crossover(parent1, parent2):
    length = len(parent1)
    point1 = int(np.random.randint(1, len(parent1) - 4, size=1))

    # test
    point2 = point1 + 3

    child1 = np.zeros_like(parent1) - 1
    child2 = np.zeros_like(parent2) - 1

    child1[point1: point2] = parent2[point1:point2]
    child2[point1: point2] = parent1[point1: point2]

    for i in range(length):
        if child1[i] == -1:
            if parent1[i] not in child1:
                child1[i] = parent1[i]
            else:
                temp = parent1[i]
                while(temp in child1):
                    index = np.where(child1 == temp)[0][0]
                    temp = parent1[index]

                child1[i] = temp
        if child2[i] == -1:
            if parent2[i] not in child2:
                child2[i] = parent2[i]
            else:
                temp = parent2[i]
                while(temp in child2):
                    index = np.where(child2 == temp)[0][0]
                    temp = parent2[index]

                child2[i] = temp
    return child1, child2


def two_point_crossover(parent1, parent2):
    point1 = int(np.random.randint(1, len(parent1) - 4, size=1))

    # test

    point2 = point1 + 3

    child1 = np.zeros_like(parent1) - 1
    child2 = np.zeros_like(parent2) - 1

    child1[point1: point2] = parent2[point1:point2]
    child2[point1: point2] = parent1[point1: point2]

    child1[:point1] = parent1[:point1]
    child1[point2:] = parent1[point2:]

    child2[:point1] = parent2[:point1]
    child2[point2:] = parent2[point2:]

    return child1, child2


def mutation_for_idpcdu(individual, S):
    point1, point2 = np.random.choice(len(individual.node_priority), size=(2))
    temp = individual.node_priority[point1]
    individual.node_priority[point1] = individual.node_priority[point2]
    individual.node_priority[point2] = temp

    point1 = np.random.choice(len(individual.node_priority), size=1)
    individual.edge_index[point1] = np.random.randint(
        low=0, high=S[point1] - 1, dtype=int)

    return individual


def create_population(number_subpopulation, number_tasks, number_nodes, S):
    population = [individual(number_nodes, S, 1)
                  for i in range(number_subpopulation * number_tasks)]
    return population


def generate_population(population, tasks):
    number_tasks = len(tasks)
    number_population = len(population)

    skill_factor = np.zeros(number_population, dtype=int)
    factorial_cost = np.zeros((number_population), dtype=float)
    path = [] 
    for individual in range(len(population)):
        task = individual % number_tasks
        skill_factor[individual] = task
        p, factorial_cost[individual] = tasks[task].grow_path_alogrithms(
            population[individual])
        path.append(p)

    return skill_factor, factorial_cost, path


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
            scalar_fitness[index[task][index_sorted[order]]
                           ] = 1.0 / float(1.0 + order)

    return scalar_fitness


def assormative(population, skill_factor, S, rmp=0.3, pm = 0.05):
    number_population = len(population)
    child = [individual(len(population[0].edge_index), S, 0)
             for i in range(number_population)]
    number_child = 0
    skill_factor_child = np.zeros_like(skill_factor, dtype=int)
    while number_child < number_population:
        parent1, parent2 = np.random.choice(number_population, size=(2))
        if skill_factor[parent1] == skill_factor[parent2] or np.random.rand() < rmp:
            child[number_child].node_priority, child[number_child+1].node_priority = pmx_crossover(population[parent1].node_priority, population[parent2].node_priority)
            child[number_child].edge_index, child[number_child+1].edge_index = two_point_crossover(population[parent1].edge_index, population[parent2].edge_index)

            if(np.random.rand() < pm):
                child[number_child]= mutation_for_idpcdu(child[number_child], S) 
                child[number_child+1]= mutation_for_idpcdu(child[number_child+1], S) 

            if np.random.rand() < 0.5:
                skill_factor_child[number_child] = skill_factor[parent1]
                skill_factor_child[number_child+1] = skill_factor[parent2]
            else:
                skill_factor_child[number_child] = skill_factor[parent2]
                skill_factor_child[number_child+1] = skill_factor[parent1]
        else:
            child[number_child] = mutation_for_idpcdu(population[parent1], S)
            child[number_child +1] = (mutation_for_idpcdu(population[parent2], S))
            skill_factor_child[number_child] = skill_factor[parent1]
            skill_factor_child[number_child + 1] = skill_factor[parent2]

        number_child += 2

    return child, skill_factor_child

def evaluate_child(childs, tasks, skill_factor_child):
    number_child = len(skill_factor_child) 
    factorial_cost_child = np.zeros((number_child), dtype = float) 
    path = []
    for index in range(number_child):
        p, factorial_cost_child[index]= tasks[skill_factor_child[index]].grow_path_alogrithms(childs[index])
        path.append(p)

    return factorial_cost_child, path

def update(population, number_population, skill_factor, scalar_fitness, factorial_cost, path):
    temp = np.argpartition(-scalar_fitness, number_population)
    result_index = temp[:number_population]
    delete_index = []  
    for i in range(len(population)):
        if i not in result_index.tolist():
            delete_index.append(i)

    population = np.delete(population, delete_index)
    factorial_cost = np.delete(factorial_cost, delete_index)
    skill_factor = np.delete(skill_factor, delete_index)
    scalar_fitness = np.delete(scalar_fitness, delete_index)
    path = np.delete(np.array(path, dtype= object), delete_index)

    assert(len(population) == number_population)

    return population, skill_factor, scalar_fitness, factorial_cost, path


def optimize_result(population, skill_factor, factorial_cost, tasks):
    class result: 
        def __init__(self, cost = 100000, task = -1):
            self.cost = cost 
            self.task = task 
    
    results = [result(task=i) for i in range(np.max(skill_factor) + 1)]

    for i in range(len(population)):
        if results[skill_factor[i]].cost > factorial_cost[i] :
            results[skill_factor[i]].cost = factorial_cost[i] 
    for result in results: 
        print("tasks: {} | cost: {} ".format(tasks[result.task].path, result.cost))
