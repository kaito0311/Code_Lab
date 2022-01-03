import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from tqdm import trange
from utils.utils_model_mfea import create_population, generate_population, compute_scalar_fitness, evaluate_child, optimize_result
from utils.utils_mfea import choose_parent, create_child_mfea1, update_constant_subpop, create_child_mfea
from config import *

np.random.seed(0)


def mfea(tasks):

    population = create_population(
        NUMBER_SUBPOPULATION * NUMBER_TASKS, DIMENSIONS, LOWER_BOUND, UPPER_BOUND
    )

    assert len(tasks) == NUMBER_TASKS

    # Initial Population
    skill_factor, factorial_cost = generate_population(
        population, tasks, NUMBER_SUBPOPULATION * NUMBER_TASKS
    )

    scalar_fitness = compute_scalar_fitness(factorial_cost, skill_factor)

    count = 0  # count the number loop

    # element of rmp matrix will be changed
    RMP = np.zeros((NUMBER_TASKS, NUMBER_TASKS), dtype=float)
    RMP = RMP + rmp_mfea
    # mfea2
    history_cost = []
    iterator = trange(LOOP)
    # iterator = range(LOOP)

    for count in iterator:
        child = np.zeros_like(population, dtype=float)
        skill_factor_child = np.zeros_like(skill_factor)
        factorial_cost_child = np.zeros_like(factorial_cost)
        count_child = 0
        number_child = np.zeros((NUMBER_TASKS), dtype=int)

        while count_child < NUMBER_SUBPOPULATION * NUMBER_TASKS:

            parent0, parent1 = choose_parent(
                NUMBER_SUBPOPULATION, scalar_fitness, number_child, skill_factor
            )

            (
                child[count_child],
                child[count_child + 1],
                skill_factor_child[count_child],
                skill_factor_child[count_child + 1],
            ) = create_child_mfea1(
                parent0,
                parent1,
                skill_factor,
                population,
                RMP[skill_factor[parent0]][skill_factor[parent1]],
            )

            number_child[skill_factor_child[0]] += 1
            number_child[skill_factor_child[1]] += 1
            count_child += 2


        
        factorial_cost_child = evaluate_child(child, tasks, skill_factor_child)
        population = np.concatenate((population, child))
        skill_factor = np.concatenate((skill_factor, skill_factor_child))
        factorial_cost = np.concatenate((factorial_cost, factorial_cost_child))
        scalar_fitness = compute_scalar_fitness(factorial_cost, skill_factor)

        population, skill_factor, scalar_fitness, factorial_cost = update_constant_subpop(
            population, NUMBER_POPULATON, skill_factor, scalar_fitness, factorial_cost)

        count += 1
        results = optimize_result(population, skill_factor, factorial_cost, tasks)
        history_cost.append(results)

        # iterator.set_description("loop: {} / {} : {} ||".format(count, LOOP))
        iterator.set_description(f"loop: {count} / {LOOP} : {[results[i].cost for i in range(NUMBER_TASKS)]} ||")

    for i in range(NUMBER_TASKS):
        print(results[i].cost)
    return history_cost
