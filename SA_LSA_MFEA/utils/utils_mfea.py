
import numpy as np
from numpy.random import choice
import random
from .utils_model_mfea import find_individual_same_skill 
from .operators import * 


def update_constant_subpop(population, number_population, skill_factor, scalar_fitness, factorial_cost):
    """
    Subpopulations has same size on every generation
    """
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




def create_child_mfea(parent1, parent2, skill_factor, population, rmp):
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
        child1 = population[parent1]
        skill_factor_child[0] = skill_factor[parent1]
        child2 = population[parent2]
        skill_factor_child[1] = skill_factor[parent2]
        child1 = poly_mutation(child1)
        child2 = poly_mutation(child2) 

    return child1, child2, skill_factor_child[0], skill_factor_child[1] 
def create_child_mfea1(parent1, parent2, skill_factor, population, rmp):

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
    Chon cha me o 1/2 ph??a tr??n
    Arguments: 
        number_subpopulation: so luong ca the o moi quan the con. Shape(N, 1) 
        scalar_fitness: scalar_fitness cho quan the cha me. Shape(number_subpopulation * number_task, 1) 
        number_child: so luong child da tao o moi task. shape(number_task,1)
    Returns:
        2 ca the cha me 
    """

    top_parent = np.where(scalar_fitness >0)[0]

    index_parent_not_enough_child = np.where(number_child[skill_factor[top_parent]] < number_subpopulation)[0] 
    parent = np.random.choice(top_parent[index_parent_not_enough_child], size= (2))
    
    return parent

