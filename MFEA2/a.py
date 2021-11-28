import numpy as np
from scipy.optimize import optimize
LOWER_BOUND = -50
UPPER_BOUND = 50
NUMBER_SUBPOPULATION = 2
NUMBER_TASK = 2
DIMENSIONS = 5
np.random.seed(1)


class sphere_function:
    def __init__(self, dimension):
        self.dimension = dimension

    def decode(self, array_value):
        array_value = array_value[:self.dimension]
        return array_value

    def calculate_fitness(self, array_value):
        x = self.decode(array_value)
        # print(array_value.shape)
        # print(array_va)
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


population = np.zeros(
    (NUMBER_SUBPOPULATION * NUMBER_TASK, DIMENSIONS), dtype=float)
skill_factor = np.zeros((1, NUMBER_SUBPOPULATION * NUMBER_TASK), dtype=int)
RMP = np.zeros((NUMBER_TASK, NUMBER_TASK), dtype=float)
factorial_cost = np.zeros((1, NUMBER_SUBPOPULATION * NUMBER_TASK), dtype=int)

tasks = [sphere_function(dimension=2), rastrigin_function(dimension=2)]


def evaluate_population(population, skill_factor, factorial_cost, tasks):
    for individual in range(len(population)):
        best_task = individual % NUMBER_TASK
        skill_factor[0][individual] = best_task
        factorial_cost[0][individual] = (
            tasks[best_task].calculate_fitness(population[individual]))


def create_population(population):
    population = np.random.uniform(
        low=LOWER_BOUND, high=UPPER_BOUND, size=population.shape)
    return population


def compute_scalarfitness(factorial_cost, skill_factor):

    temp= np.zeros(factorial_cost.shape, dtype = float)
    index = np.zeros((NUMBER_TASK, 1), dtype=int)

    for i in range(len(factorial_cost)):

        temp[0][int(skill_factor[0][i] * NUMBER_SUBPOPULATION +
                 index[skill_factor[0][i]])] = factorial_cost[0][i]
        index[skill_factor[0][i]] += 1

    temp = np.array(temp)
    ranks = np.empty_like(temp)

    for task in range(NUMBER_TASK):
        temp2 = temp[task * NUMBER_SUBPOPULATION: (task+1) * NUMBER_SUBPOPULATION].argsort()
        ranks[temp2 + task * NUMBER_SUBPOPULATION] = np.arange(NUMBER_SUBPOPULATION)

    real_ranks = np.zeros(temp.shape, dtype=float)
    index = np.zeros((NUMBER_TASK, 1), dtype=int)

    for i in range(len(factorial_cost)):
        real_ranks[0][i] = 1.0/(ranks[skill_factor[0][i] *
                             NUMBER_SUBPOPULATION + index[skill_factor[0][i]]] + 1)
        index[skill_factor[0][i]] += 1
    print(real_ranks)


population = create_population(population) 
print(population)
# evaluate_population(population, skill_factor, factorial_cost, tasks)
# compute_scalarfitness(factorial_cost, skill_factor)
