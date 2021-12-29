import matplotlib.pyplot as plt
import numpy as np
from utils_mf import *
from utils_sa_lsa import *
from config import * 
from tqdm import trange
np.random.seed(0)




def lsa_mfea(tasks, lsa = True):

    history_memories = [Memory(H) for i in range(NUMBER_RMPS)]

    # Kich thuoc quan the 
    initial_size_population = np.zeros((NUMBER_TASKS), dtype=int) + 100
    current_size_population = np.copy(initial_size_population)
    min_size_population = np.zeros((NUMBER_TASKS), dtype=int) + 50 

    
    evaluations = np.zeros((NUMBER_TASKS), dtype=int)
    maxEvals = np.zeros_like(evaluations, dtype=int) + MAXEVALS / NUMBER_TASKS


    # Khoi tao quan the
    population = create_population(
        np.sum(initial_size_population), DIMENSIONS, LOWER_BOUND, UPPER_BOUND)

    
    skill_factor = np.zeros((np.sum(initial_size_population)), dtype=int)
    factorial_cost = np.zeros((np.sum(initial_size_population)), dtype=float)
    skill_factor, factorial_cost = generate_population(
        population, tasks, np.sum(initial_size_population))
    scalar_fitness = compute_scalar_fitness(factorial_cost, skill_factor)

    
    # For visual 
    dem = 0
    history_cost = []
    history_rmp = [[] for i in range(NUMBER_RMPS)]
    

    iterator = trange(LOOP) 


    # while np.sum(evaluations) < MAXEVALS:
    for dem in iterator:
        if np.sum(evaluations) >= MAXEVALS: 
            break 
        S = [[] for i in range(NUMBER_RMPS)]
        xichma = [[] for i in range(NUMBER_RMPS)]

        childs = []
        skill_factor_childs = []
        factorial_cost_childs = []

        generate_rmps = [[] for i in range(NUMBER_RMPS)]
        count_rmp = np.zeros(NUMBER_RMPS)

        count = 0
        while count < np.sum(current_size_population) and np.sum(evaluations) < MAXEVALS:

            parent = choose_parent(maxEvals, scalar_fitness, evaluations, skill_factor)
            if parent is None:
                break

            index = index_convert_matrix_to_1D(skill_factor[parent[0]], skill_factor[parent[1]], len(tasks))
            generate_rmp = history_memories[index].random_Gauss()

            generate_rmps[index].append(generate_rmp)
            count_rmp[index] += 1 
            
            child, skf_child, \
            fac_cost_child, S, xichma = create_child_v3(
                                        population, parent, skill_factor, \
                                        generate_rmp, S, \
                                        xichma, tasks)

            # for i in range(2):
            #     childs.append(child[i])
            #     if len(np.array(childs).shape) == 1:
            #         print("hmmm")
            #     skill_factor_childs.append(skf_child[i])
            #     factorial_cost_childs.append(fac_cost_child[i])
            evaluations[skf_child[0]] += 1 
            evaluations[skf_child[1]] += 1 
            if len(childs) == 0 : 
                childs = child
                skill_factor_childs = skf_child
                factorial_cost_childs = fac_cost_child
            else: 
                childs = np.concatenate([childs, child], axis= 0)
                skill_factor_childs = np.concatenate([skill_factor_childs, skf_child])
                factorial_cost_childs = np.concatenate([factorial_cost_childs, fac_cost_child])

            count += 2

        

        for i in range(NUMBER_RMPS):
            history_rmp[i].append(np.sum(generate_rmps[i])/ count_rmp[i])
        # For visual
        # print(history_memories[1].M)
        history_memories = Update_Success_History_Memory(
                            history_memories, S, xichma, NUMBER_TASKS)

        
        # Linear_population_size_reduction()
        if lsa is True: 
            current_size_population = Linear_population_size_reduction(
                                        evaluations, current_size_population, \
                                        maxEvals, NUMBER_TASKS, initial_size_population, \
                                        min_size_population)


        # Combine population parent with childs
        population = np.concatenate([population, np.array(childs)])
        skill_factor = np.concatenate([skill_factor, np.array(skill_factor_childs)])
        factorial_cost = np.concatenate([factorial_cost, np.array(factorial_cost_childs)])
        scalar_fitness = compute_scalar_fitness(factorial_cost, skill_factor)

        # Check 
        assert len(population) == len(skill_factor)
        assert len(population) == len(factorial_cost)

        population, skill_factor, scalar_fitness, factorial_cost = update(population, current_size_population, \
                                                                    skill_factor, scalar_fitness, factorial_cost)

        
        results = optimize_result(population, skill_factor, factorial_cost, tasks)
        history_cost.append(results)
                                    
        assert len(population) == np.sum(current_size_population)

        if dem > LOOP :
            break 
        dem += 1
    
    
    for i in range(NUMBER_TASKS):
        print(results[i].cost)
    return history_cost, history_rmp

