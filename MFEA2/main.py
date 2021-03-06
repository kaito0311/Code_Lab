import numpy as np
import matplotlib.pyplot as plt
from utils import *
from scipy.optimize import minimize
from tqdm import trange 
from mfea2 import * 

LOWER_BOUND = 0
UPPER_BOUND = 1
NUMBER_SUBPOPULATION = 50
NUMBER_TASK = 3
DIMENSIONS = 30
MFEA_VERSION = 2
NUMBER_CHILD = NUMBER_SUBPOPULATION * NUMBER_TASK
NUMBER_POPULATON = NUMBER_SUBPOPULATION * NUMBER_TASK
LOOP = 500
np.random.seed(1)

def mfea2(version = 1):

    MFEA_VERSION = version
    population = create_population(
        NUMBER_SUBPOPULATION, NUMBER_TASK, DIMENSIONS, LOWER_BOUND, UPPER_BOUND)

    # Number task
    tasks = [rastrigin_function(dimension=20, delta = 15),ackley_function(dimension=15, delta= 14), ackley_function(dimension= 15, delta= 10)]
    assert (len(tasks) == NUMBER_TASK)

    # Initial Population
    skill_factor, factorial_cost = generate_population(
        population, tasks, NUMBER_SUBPOPULATION * NUMBER_TASK)

    scalar_fitness = compute_scalar_fitness(factorial_cost, skill_factor)

    count = 0  # count the number loop

    # element of rmp matrix will be changed
    variable = np.random.rand(int(NUMBER_TASK * (NUMBER_TASK - 1) / 2))
    RMP = convert_1D_to_matrix(variable, NUMBER_TASK)
    # print(RMP)
    # Bounds for rmp
    x = (0.0, 1.0)
    bounds = []
    for i in range(int(NUMBER_TASK * (NUMBER_TASK - 1) / 2)):
        bounds.append(x)
    bounds = tuple(bounds)

    # mfea2
    history_rmp = []
    history_cost = []
    iterator = trange(LOOP)
    # iterator = range(LOOP)

    for count in iterator:

        if MFEA_VERSION == 1:
            RMP = np.zeros((NUMBER_TASK, NUMBER_TASK), dtype=float)
            RMP = RMP + 0.3
        else:
            opt_loop = 5 
            value_opt = 100000
            variable_opt = None 
            for count_loop in range(opt_loop):

                variable = np.random.rand(int(NUMBER_TASK * (NUMBER_TASK - 1) / 2))
                # variable = np.zeros((int(NUMBER_TASK * (NUMBER_TASK - 1) / 2)))
                pro_model_for_parent = learn_probabilistic_model_v2(
                    population, skill_factor, scalar_fitness)
                optimize = minimize(optimize_rmp, variable, args=(
                    pro_model_for_parent, population, scalar_fitness, skill_factor,), bounds=bounds)
                variable = optimize.x
                if value_opt > optimize_rmp(variable, pro_model_for_parent, population, scalar_fitness, skill_factor):
                    variable_opt = variable 
                    value_opt = optimize_rmp(variable, pro_model_for_parent, population, scalar_fitness, skill_factor)
            
            variable = variable_opt
            RMP = convert_1D_to_matrix(optimize.x, NUMBER_TASK, 1)

            history_rmp.append(variable)

        child = np.zeros_like(population, dtype=float)
        skill_factor_child = np.zeros_like(skill_factor)
        factorial_cost_child = np.zeros_like(factorial_cost)
        count_child = 0
        number_child = np.zeros((NUMBER_TASK), dtype=int)

        while count_child < NUMBER_SUBPOPULATION*NUMBER_TASK:

            parent0, parent1 = choose_parent(
                NUMBER_SUBPOPULATION, scalar_fitness, number_child, skill_factor)
            task0, task1 = skill_factor[parent0], skill_factor[parent1]
            child[count_child], child[count_child+1], skill_factor_child[count_child], skill_factor_child[count_child+1] = \
                create_child_v2(parent0,parent1, skill_factor, population,  RMP[skill_factor[parent0]][skill_factor[parent1]])
            # child[count_child], child[count_child+1], skill_factor_child[count_child], skill_factor_child[count_child+1] = create_child(
                # population[parent0], population[parent1], skill_factor[parent0], skill_factor[parent1],  RMP[skill_factor[parent0]][skill_factor[parent1]])

            number_child[task0] += 1
            number_child[task1] += 1
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


        iterator.set_description("loop: {} / {} || tasks : {} || cost {}\
                                ".format(count, LOOP, {task.task for task in results}, {result.cost for result in results}) )
        # print(pro_model_for_parent)
        # print(results[0].cost, "  ", results[1].cost, "  ", variable)
    if MFEA_VERSION == 2:  
        optimize_result(population, skill_factor, factorial_cost, tasks)
        _, bieudo = plt.subplots(1,variable.shape[0], figsize= (5,5))
        for i in range(variable.shape[0]):
            bieudo[i].plot(np.arange(LOOP), [x[i] for x in history_rmp])
        # plt.show() 

    for i in range(NUMBER_TASK):
        print(results[i].cost)
    return history_cost

def visual_result(history):
    _, bieudo = plt.subplots(1, NUMBER_TASK)
    color = ['r', 'b', 'y']
    name = ['mfea', 'mfea2', '....']

    for task in range(NUMBER_TASK):
        for version in range(len(history)):
            bieudo[task].plot(np.arange(LOOP), [x[task].cost for x in history[version]], color[version], label = name[version])
    plt.legend(loc = 'best')
    plt.show()
for i in range(1):
    history = [] 
    versions = [1,2]
    for version in versions:
        print("version: ", version)
        history.append(mfea2(version=version))
    visual_result(history)
    
