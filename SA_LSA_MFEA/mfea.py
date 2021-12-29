import numpy as np
import matplotlib.pyplot as plt
from utils_mf import *
from scipy.optimize import minimize
from tqdm import trange 
from utils_mfea import * 
from config import * 

np.random.seed(0)

def mfea2(tasks, version = 1):

    MFEA_VERSION = version
    population = create_population(
        NUMBER_SUBPOPULATION, NUMBER_TASKS, DIMENSIONS, LOWER_BOUND, UPPER_BOUND)

    assert (len(tasks) == NUMBER_TASKS)

    # Initial Population
    skill_factor, factorial_cost = generate_population(
        population, tasks, NUMBER_SUBPOPULATION * NUMBER_TASKS)

    scalar_fitness = compute_scalar_fitness(factorial_cost, skill_factor)

    count = 0  # count the number loop

    # element of rmp matrix will be changed
    variable = np.random.rand(int(NUMBER_TASKS * (NUMBER_TASKS - 1) / 2))
    RMP = convert_1D_to_matrix(variable, NUMBER_TASKS)
    # mfea2
    history_rmp = []
    history_cost = []
    iterator = trange(LOOP)
    # iterator = range(LOOP)

    for count in iterator:

        if MFEA_VERSION == 1:
            RMP = np.zeros((NUMBER_TASKS, NUMBER_TASKS), dtype=float)
            RMP = RMP + 0.3

        child = np.zeros_like(population, dtype=float)
        skill_factor_child = np.zeros_like(skill_factor)
        factorial_cost_child = np.zeros_like(factorial_cost)
        count_child = 0
        number_child = np.zeros((NUMBER_TASKS), dtype=int)

        while count_child < NUMBER_SUBPOPULATION*NUMBER_TASKS:

            parent0, parent1 = choose_parent(
                NUMBER_SUBPOPULATION, scalar_fitness, number_child, skill_factor)
            task0, task1 = skill_factor[parent0], skill_factor[parent1]
            child[count_child], child[count_child+1], skill_factor_child[count_child], skill_factor_child[count_child+1] = \
                create_child_v2(parent0,parent1, skill_factor, population,  RMP[skill_factor[parent0]][skill_factor[parent1]])
            # child[count_child], child[count_child+1], skill_factor_child[count_child], skill_factor_child[count_child+1] = create_child(
                # population[parent0], population[parent1], skill_factor[parent0], skill_factor[parent1],  RMP[skill_factor[parent0]][skill_factor[parent1]])

            number_child[skill_factor_child[0]] += 1
            number_child[skill_factor_child[1]] += 1
            count_child += 2
        
        factorial_cost_child = evaluate_child(child, tasks, skill_factor_child)
        population = np.concatenate((population, child))
        skill_factor = np.concatenate((skill_factor, skill_factor_child))
        factorial_cost = np.concatenate((factorial_cost, factorial_cost_child))
        scalar_fitness = compute_scalar_fitness(factorial_cost, skill_factor)

        # population = child 
        # skill_factor = skill_factor_child
        # factorial_cost = factorial_cost_child
        # scalar_fitness=  compute_scalar_fitness(factorial_cost, skill_factor)

        population, skill_factor, scalar_fitness, factorial_cost = update(population, NUMBER_POPULATON, skill_factor,
                                                                          scalar_fitness, factorial_cost)
        count += 1
        results = optimize_result(population, skill_factor, factorial_cost, tasks)
        history_cost.append(results)


        iterator.set_description("loop: {} / {} ||".format(count, LOOP) )
        # print(pro_model_for_parent)
        # print(results[0].cost, "  ", results[1].cost, "  ", variable)
    if MFEA_VERSION == 2:  
        optimize_result(population, skill_factor, factorial_cost, tasks)
        _, bieudo = plt.subplots(1,variable.shape[0], figsize= (5,5))
        for i in range(variable.shape[0]):
            bieudo[i].plot(np.arange(LOOP), [x[i] for x in history_rmp])
        # plt.show() 

    for i in range(NUMBER_TASKS):
        print(results[i].cost)
    return history_cost
