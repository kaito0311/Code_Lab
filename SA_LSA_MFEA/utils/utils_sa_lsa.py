import numpy as np
from numpy.core.fromnumeric import size

from utils.operators import * 
from utils.utils_model_mfea import * 


def index_convert_matrix_to_1D(task1, task2, number_task):

    if task1 < task2:
        index = number_task * (task1) - task1 * (task1 + 1) / 2 + (task2 - task1)
    elif task1 > task2:
        index = number_task * (task2) - task2 * (task2 + 1) / 2 + (task1 - task2)
    else:
        return 1
    return abs(int(index)) - 1


class Memory:
    def __init__(self, H=5, sigma=0.1):
        self.H = H
        self.index = 0
        self.sigma = sigma
        self.M = np.zeros((H), dtype=float) + 0.5

    def random_Gauss(self):
        mean = np.random.choice(self.M)
        # rmp_sampled = np.random.normal(loc=mean, scale=self.varience)
        # mu + sigma * Math.sqrt(-2.0 * Math.log(rand.nextDouble())) * Math.sin(2.0 * Math.PI * rand.nextDouble());
        rmp_sampled = 0
        while rmp_sampled <= 0:
            rmp_sampled = mean + self.sigma * np.sqrt(-2.0 * np.log(np.random.uniform())) * np.sin(2.0 * np.pi * np.random.uniform())

        if rmp_sampled > 1:
            return 1
        return rmp_sampled

    def update_M(self, value):
        self.M[self.index] = value
        self.index = (self.index + 1) % self.H


def Linear_population_size_reduction(
    evaluations, current_size_population, maxEvaluations, number_tasks, maxSize, minSize
):
    for task in range(number_tasks):
        new_size = (minSize[task] - maxSize[task]) / maxEvaluations[task] * evaluations[
            task
        ] + maxSize[task]
        if new_size < current_size_population[task]:
            current_size_population[task] = new_size

    return current_size_population


def choose_parent(number_subpopulation, scalar_fitness, number_child, skill_factor) -> np.ndarray(2):
    """
    Arguments:
        number_subpopulation: so luong ca the o moi quan the con. Shape(N, 1)
        scalar_fitness: scalar_fitness cho quan the cha me. Shape(number_subpopulation * number_task, 1)
        number_child: so luong child da tao o moi task. shape(number_task,1)
    Returns:
        2 ca the cha me
    """

    top_parent = np.where(scalar_fitness > 0)[0]
    a = np.arange(int(len(skill_factor)/2))
    np.random.shuffle(a)
    parent0 = int(np.random.choice(a, size = (1)))
    # index_parent_not_enough_child = np.where(
    #     number_child[skill_factor[top_parent]]
    #     < number_subpopulation[skill_factor[top_parent]]
    # )[0]
    # if len(index_parent_not_enough_child) == 0:
    #     return None
    # parent = np.random.choice(top_parent[index_parent_not_enough_child], size=(2))
    parent = int(np.random.choice(np.arange(len(skill_factor)), size= (1)))
    return [parent0, parent]

def choose_parent(list_population):
    parent0 = int(np.random.choice(list_population[:int(len(list_population)/2)], size= (1)))
    parent1 = int(np.random.choice(list_population, size= (1)))
    return[parent0, parent1] 

def compute_beta_for_sbx(u, nc=2):
    if u < 0.5:
        return np.power(2.0 * u, 1.0 / float(nc + 1))
    else:
        return np.power(1.0 / (2.0 * (1 - u)), 1.0 / (nc + 1))



def create_child_lsa_sa(population, parent, skill_factor, rmp, S, xichma, tasks):

    """
    Arguments:
        population: an array has shape (number_population, dimensions) that contains individuals
        parent: an array has shape (2,) that contains two indexs of parent
        skill_factor: an array has shape (nummber_population, )
        rmp: for each tasks in parent
        S: set of success rmp in time t
        xichma : percentage improvement of each rmp in S
        tasks: array has shape (number_tasks,) that contains the tasks
    Returns:
        Two child\n
        skill_factor_for_child\n
        S\n
        xichma
    """
    # print(parent[0])
    # print(population)
    
    skill_factor_child = np.zeros((2), dtype=int)
    childs = np.zeros((2, population.shape[1]), dtype=float)
    factorial_cost_child = np.zeros((2), dtype=float)

    if skill_factor[parent[0]] == skill_factor[parent[1]]:
        childs[0], childs[1] = sbx_crossover( population[parent[0]], population[parent[1]], swap= True)

        childs[0] = poly_mutation(childs[0])
        childs[1] = poly_mutation(childs[1])

        skill_factor_child[0] = skill_factor_child[1] = skill_factor[parent[0]]

        factorial_cost_child[0] = tasks[skill_factor_child[0]].calculate_fitness(childs[0])
        factorial_cost_child[1] = tasks[skill_factor_child[1]].calculate_fitness(childs[1])
    else:
        r = np.random.uniform()

        if r < rmp:
            childs[0], childs[1] = sbx_crossover(population[parent[0]], population[parent[1]],swap=False)
            
            skill_factor_child[0] = int(np.random.choice(skill_factor[parent]))
            skill_factor_child[1] = int(np.random.choice(skill_factor[parent]))

        else:
            # parent0 = find_individual_same_skill(skill_factor, parent[0])
            # child1, child2 = sbx_crossover(population[parent[0]], population[parent0])
            # if tasks[skill_factor[parent[0]]].calculate_fitness(child1) < tasks[skill_factor[parent[0]]].calculate_fitness(child2) :
            #     childs[0] = np.copy(child1)
            # else :
            #     childs[0] = np.copy(child2)

            # parent1 = find_individual_same_skill(skill_factor, parent[1])
            # child1, child2 = sbx_crossover(population[parent[1]], population[parent1])
            # if tasks[skill_factor[parent[1]]].calculate_fitness(child1) < tasks[skill_factor[parent[1]]].calculate_fitness(child2) :
            #     childs[1] = np.copy(child1) 
            # else :
            #     childs[1] = np.copy(child2) 
            
            # skill_factor_child[0] = skill_factor[parent[0]]
            # skill_factor_child[1] = skill_factor[parent[1]]
            #==============================================
            parent0 = find_individual_same_skill(skill_factor, parent[0])
            childs[0], _ = sbx_crossover(population[parent[0]], population[parent0])

            parent1 = find_individual_same_skill(skill_factor, parent[1])
            childs[1], _ = sbx_crossover(population[parent1], population[parent[1]])

            skill_factor_child[0] = skill_factor[parent[0]]
            skill_factor_child[1] = skill_factor[parent[1]]

            #================================ 
            # p0 = find_individual_same_skill(skill_factor, parent[0])
            # childs[0], childs[1] = sbx_crossover(population[parent[0]], population[p0], swap= True) 
            # skill_factor_child[0] = skill_factor_child[1] = skill_factor[parent[0]] 
            

        childs[0] = poly_mutation(childs[0])
        childs[1] = poly_mutation(childs[1])

        factorial_cost_child[0] = tasks[skill_factor_child[0]].calculate_fitness(childs[0])
        factorial_cost_child[1] = tasks[skill_factor_child[1]].calculate_fitness(childs[1])

        delta = calculate_improvement_percentage(population, parent, childs, tasks, skill_factor, skill_factor_child)

        if delta > 0 :
            index = index_convert_matrix_to_1D(skill_factor[parent[0]], skill_factor[parent[1]], len(tasks))
            S[index].append(float(rmp))
            xichma[index].append(float(delta))

    return childs, skill_factor_child, factorial_cost_child, S, xichma


def calculate_improvement_percentage( population, parent, childs, tasks, skill_factor, skill_factor_child ):
    delta = 0

    def calculate(p, o):
        task = skill_factor[p]
        fitness_pa = tasks[task].calculate_fitness(population[p])
        fitness_oa = tasks[task].calculate_fitness(childs[o])

        return (fitness_pa - fitness_oa + 1e-10) / (fitness_pa + 1e-10)

    if skill_factor_child[0] == skill_factor[parent[0]]:
        delta = np.max([delta, calculate(parent[0], 0)])
    else:
        delta = np.max([delta, calculate(parent[1], 0)])

    if skill_factor_child[1] == skill_factor[parent[0]]:
        delta = np.max([delta, calculate(parent[0], 1)])
    else:
        delta = np.max([delta, calculate(parent[1], 1)])

    return delta


def Update_Success_History_Memory(M, S, xichma, number_tasks):

    for i in range(number_tasks):
        j = i + 1
        while j < number_tasks:
            index = index_convert_matrix_to_1D(i, j, number_tasks)
            if len(S[index]) != 0:
                M[index].update_M(np.sum(np.array(xichma[index]) * np.array(S[index])**2) / (1e-50 + np.sum(np.array(xichma[index]) * np.array(S[index]))))
            j += 1
    return M


def update(population, number_population, skill_factor, scalar_fitness, factorial_cost):
    """
    number population : an array has shape (number_tasks, ). Present the number of individual will maintain to next generation 

    Returns: 
        population: list of ind
        skill_factor : list of skill factor 
        scalar_fitness : list of scalar_fitness 
        factorical_cost : list of factorial cost
    """
    delete_index = []

    for ind in range(len(population)):
        if scalar_fitness[ind] < 1.0 / number_population[skill_factor[ind]]:
            delete_index.append(ind)

    # temp = np.argpartition(-scalar_fitness, number_population)
    # result_index = temp[:number_population]
    # for i in range(len(population)):
    #     if i not in result_index:
    #         delete_index.append(i)
    # delete_index = np.array(delete_index, dtype= int)

    population = np.delete(population, delete_index, axis=0)
    factorial_cost = np.delete(factorial_cost, delete_index, axis=0)
    skill_factor = np.delete(skill_factor, delete_index, axis=0)
    scalar_fitness = np.delete(scalar_fitness, delete_index, axis=0)

    assert len(population) == np.sum(number_population)

    return population, skill_factor, scalar_fitness, factorial_cost
