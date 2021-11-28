import numpy as np
import matplotlib.pyplot as plt
from utils import *
from scipy.optimize import minimize
from tqdm import trange 

LOWER_BOUND = 0
UPPER_BOUND = 1
NUMBER_SUBPOPULATION = 100
NUMBER_TASK = 4
DIMENSIONS = 30
MFEA_VERSION = 2
NUMBER_CHILD = NUMBER_SUBPOPULATION * NUMBER_TASK
NUMBER_POPULATON = NUMBER_SUBPOPULATION * NUMBER_TASK
LOOP = 500

np.random.seed(1)


def mfea2():

    population = create_population(
        NUMBER_SUBPOPULATION, NUMBER_TASK, DIMENSIONS, LOWER_BOUND, UPPER_BOUND)

    # Number task
    tasks = [sphere_function(dimension=25, delta= 0.3),sphere_function(dimension=20, delta= 0.3),sphere_function(dimension=30, delta= 0.1),rastrigin_function(dimension=30, delta = 0.3)]
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
    history = []
    iterator = trange(LOOP)

    for count in iterator:

        if MFEA_VERSION == 1:
            RMP = np.zeros((NUMBER_TASK, NUMBER_TASK), dtype=float)
            RMP = RMP + 0.3
        else:
            pro_model_for_parent = learn_probabilistic_model_v2(
                population, skill_factor)
            optimize = minimize(optimize_rmp, variable, args=(
                pro_model_for_parent, population, scalar_fitness, skill_factor,), method="L-BFGS-B", bounds=bounds)
            variable = optimize.x
            RMP = convert_1D_to_matrix(optimize.x, NUMBER_TASK, 1)

            history.append(variable)

        child = np.zeros_like(population, dtype=float)
        skill_factor_child = np.zeros_like(skill_factor)
        factorial_cost_child = np.zeros_like(factorial_cost)
        count_child = 0
        number_child = np.zeros((NUMBER_TASK), dtype=int)

        while count_child < NUMBER_SUBPOPULATION*NUMBER_TASK:

            parent0, parent1 = choose_parent(
                NUMBER_SUBPOPULATION, scalar_fitness, number_child, skill_factor)
            task0, task1 = skill_factor[parent0], skill_factor[parent1]
            child[count_child], child[count_child+1], skill_factor_child[count_child], skill_factor_child[count_child+1] = create_child(
                population[parent0], population[parent1], skill_factor[parent0], skill_factor[parent1],  RMP[skill_factor[parent0]][skill_factor[parent1]])

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
        dem = 0 
        for i in variable: 
            dem += 1 
        if (dem < NUMBER_TASK):
            dem = 0
        results = optimize_result(population, skill_factor, factorial_cost, tasks)
        iterator.set_description("loop: {} / {} || tasks : {} || cost {}".format(count, LOOP, {task.task for task in results}, {result.cost for result in results}) )
        # print(pro_model_for_parent)

    if MFEA_VERSION == 2:  
        optimize_result(population, skill_factor, factorial_cost, tasks)
        _, bieudo = plt.subplots(1,variable.shape[0], figsize= (5,5))
        for i in range(variable.shape[0]):
            bieudo[i].plot(np.arange(LOOP), [x[i] for x in history])
        
        plt.show()


for i in range(1):
    mfea2() 
