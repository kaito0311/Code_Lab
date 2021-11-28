import numpy as np
from idpc_du import *
from utils import *
from tqdm import trange
from alert import *
np.random.seed(1)
tasks = [graph("data/set1/idpc_10x20x2713.idpc"), graph("data/set1/idpc_25x12x4817.idpc")]

NUMBER_TASK = len(tasks)
DOMAINS = max(task.number_domains for task in tasks)
NODES = max(task.number_nodes for task in tasks)
POPULATION = 50
LOOP = 500
REPEAT = 5
def mfea_for_idpc_du():
    S = compute_max_out_edges(tasks, NODES)
    population = create_population(POPULATION, NUMBER_TASK, NODES, S)
    skill_factor, factorial_cost, path = generate_population(population, tasks)
    count = 0
    iterator = trange(LOOP) 
    for count in iterator:
        count += 1 
        
        child, skill_factor_child= assormative(population, skill_factor, S, 0.5)

        factorial_cost_child, path_child = evaluate_child(child, tasks, skill_factor_child)

        # concat
        population= np.concatenate((population, child))
        skill_factor = np.concatenate((skill_factor, skill_factor_child))
        factorial_cost = np.concatenate((factorial_cost, factorial_cost_child))
        scalar_fitness = compute_scalar_fitness(factorial_cost, skill_factor)
        path =np.concatenate(np.array((path, path_child), dtype = object), axis = 0)

        population, skill_factor, scalar_fitness, factorial_cost, path = update(population, POPULATION * NUMBER_TASK, skill_factor, scalar_fitness, factorial_cost, path)

        # print("[+] mfea - %d/ %d " % (count, LOOP))
        iterator.set_description("loop: {} / {} ".format(count, LOOP))
    
    optimize_result(population, skill_factor, factorial_cost, tasks)




def main():
    for i in range(REPEAT):
        print(i,  "/" , REPEAT)
        mfea_for_idpc_du() 

    annoucemnt("okey done")
if __name__ == '__main__':
    main()
