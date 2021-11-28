
import numpy as np

from utils import *
from scipy.optimize import minimize
LOWER_BOUND = 0
UPPER_BOUND = 1
NUMBER_SUBPOPULATION = 50
NUMBER_TASK = 2
DIMENSIONS = 30
MFEA_VERSION = 1

NUMBER_CHILD = NUMBER_SUBPOPULATION * NUMBER_TASK
NUMBER_POPULATON = NUMBER_SUBPOPULATION * NUMBER_TASK
LOOP = 100
np.random.seed(1)

population = create_population(
    NUMBER_SUBPOPULATION, NUMBER_TASK, DIMENSIONS, LOWER_BOUND, UPPER_BOUND)

tasks = [sphere_function(dimension=10), rastrigin_function(
    dimension=10)]
assert (len(tasks) == NUMBER_TASK)

skill_factor, factorial_cost = generate_population(
    population, tasks, NUMBER_SUBPOPULATION * NUMBER_TASK)
scalar_fitness = compute_scalar_fitness(factorial_cost, skill_factor)
count = 0
variable = np.random.rand(int(NUMBER_TASK * (NUMBER_TASK - 1) / 2))
print(convert_1D_to_matrix(variable, NUMBER_TASK))
x = (0.0, 1.0)
bounds = []
for i in range(int(NUMBER_TASK * (NUMBER_TASK - 1) / 2)):
    bounds.append(x)
bounds = tuple(bounds)
RMP = convert_1D_to_matrix(variable, NUMBER_TASK)

while count < LOOP:

    if MFEA_VERSION == 1:
        RMP = np.zeros((NUMBER_TASK, NUMBER_TASK), dtype=float)
        RMP = RMP + 0.0
    else:

        pro_model_for_parent = learn_probabilistic_model_v2(
            population, skill_factor)
        optimize = minimize(optimize_rmp, variable, args=(
            pro_model_for_parent, population, scalar_fitness, skill_factor,), bounds=bounds)
        variable = optimize.x
        RMP = convert_1D_to_matrix(optimize.x, NUMBER_TASK, 1)

    child = np.zeros_like(population, dtype=float)
    skill_factor_child = np.zeros_like(skill_factor)
    factorial_cost_child = np.zeros_like(factorial_cost)

    number_child = 0
    while number_child < NUMBER_CHILD:
        parent0, parent1 = choose_parent(NUMBER_POPULATON)
        # child[number_child], child[number_child+1], skill_factor_child[number_child], skill_factor_child[number_child+1] = create_child(
        #     parent0,parent1, skill_factor, population, RMP[skill_factor[parent0]][skill_factor[parent1]])
        child[number_child], child[number_child+1], skill_factor_child[number_child], skill_factor_child[number_child+1] = create_child(
            population[parent0], population[parent1], skill_factor[parent0], skill_factor[parent1],  RMP[skill_factor[parent0]][skill_factor[parent1]])
        number_child += 2

    factorial_cost_child = evaluate_child(child, tasks, skill_factor_child)

    population = np.concatenate((population, child))
    skill_factor = np.concatenate((skill_factor, skill_factor_child))
    factorial_cost = np.concatenate((factorial_cost, factorial_cost_child))
    scalar_fitness = compute_scalar_fitness(factorial_cost, skill_factor)

    population, skill_factor, scalar_fitness, factorial_cost = update(population, NUMBER_POPULATON, skill_factor,
                                                                      scalar_fitness, factorial_cost)
    count += 1
    # print(count)
    # print(RMP)
    # print(pro_model_for_parent)


print(factorial_cost)
print(skill_factor)
