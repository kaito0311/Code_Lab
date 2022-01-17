from pathlib import PurePath
from turtle import shape
import matplotlib.pyplot as plt
import numpy as np
import sys
from numpy.core.defchararray import find
from numpy.core.fromnumeric import choose, take
from SaMTPSO.memory import Memory_SaMTPSO
from utils.utils_sa_lsa import *
from config import *
from tqdm import trange

np.random.seed(1)


def skill_factor_best_task(pop, tasks):
    population = np.copy(pop)
    maxtrix_cost = np.array(
        [np.apply_along_axis(t.calculate_fitness, 1, population) for t in tasks]
    ).T
    matrix_rank_pop = np.argsort(np.argsort(maxtrix_cost, axis=0), axis=0)

    N = len(population) / len(tasks)
    count_inds = np.array([0] * len(tasks))
    skill_factor_arr = np.zeros(
        int(
            (N * len(tasks)),
        ),
        dtype=np.int,
    )
    condition = False

    while not condition:
        idx_task = np.random.choice(np.where(count_inds < N)[0])

        idx_ind = np.argsort(matrix_rank_pop[:, idx_task])[0]

        skill_factor_arr[idx_ind] = idx_task

        matrix_rank_pop[idx_ind] = len(pop) + 1
        count_inds[idx_task] += 1

        condition = np.all(count_inds == N)

    return skill_factor_arr


def cal_factor_cost(population, tasks, skill_factor):
    factorial_cost = np.zeros_like(skill_factor, dtype=float)
    for i in range(len(population)):
        factorial_cost[i] = tasks[skill_factor[i]].calculate_fitness(population[i])

    return factorial_cost


def lsa_SaMTPSO(tasks, lsa=True):

    initial_size_population = np.zeros((NUMBER_TASKS), dtype=int) + 100
    current_size_population = np.copy(initial_size_population)
    min_size_population = np.zeros((NUMBER_TASKS), dtype=int) + 50

    evaluations = np.zeros((NUMBER_TASKS), dtype=int)
    maxEvals = np.zeros_like(evaluations, dtype=int) + int(MAXEVALS / NUMBER_TASKS)

    skill_factor = np.zeros((np.sum(initial_size_population)), dtype=int)
    factorial_cost = np.zeros((np.sum(initial_size_population)), dtype=float)

    population = create_population(
        np.sum(initial_size_population), DIMENSIONS, LOWER_BOUND, UPPER_BOUND
    )

    skill_factor = skill_factor_best_task(population, tasks)

    factorial_cost = cal_factor_cost(population, tasks, skill_factor)

    scalar_fitness = compute_scalar_fitness(factorial_cost, skill_factor)
    
    # Khởi tạo có phần giống với cái xác suất của nó . 
    p_matrix= np.ones(shape= (len(tasks), len(tasks)), dtype= float) / len(tasks)
    # Khởi tạo một mảng các memory cho các task 
    memory_task = [Memory_SaMTPSO(len(tasks)) for i in range(len(tasks))]


    history_cost = [] 
    history_p_matrix = []
    history_p_matrix.append(p_matrix)
    while np.sum(evaluations) <= MAXEVALS:

        childs = []
        skill_factor_childs = []
        factorial_cost_childs = []
        # XEM CON SINH RA Ở VỊ TRÍ THỨ BAO NHIÊU ĐỂ TÍ XÓA
        index_child_each_tasks = []

        # TASK ĐƯỢC CHỌN LÀM PARTNER LÀ TASK NÀO TƯƠNG ỨNG VỚI MỖI CON 
        task_partner = [] 

        list_population = np.arange(len(population))
        np.random.shuffle(list_population) 
        index= len(population) 
        number_child_each_task = np.zeros(shape=(len(tasks)), dtype = int)
        while len(childs) < np.sum(current_size_population):

            for task in range(len(tasks)):
                while(number_child_each_task[task] < current_size_population[task]):
                    # random de chon task nao de giao phoi 
                    task2 = None 
                    index_pa= int(np.random.choice(np.where(skill_factor == task)[0], size= 1))
                    index_pb = index_pa 
                    if memory_task[task].isFocus == False: 
                        task2 = np.random.choice(np.arange(len(tasks)), p= p_matrix[task]) 
                        while index_pa == index_pb:
                            index_pb = int(np.random.choice(np.where(skill_factor == task2)[0], size= 1))
                    else: 
                        task2 = task 
                        while index_pa == index_pb:
                            # index_pb = int(np.random.choice(np.where(skill_factor == task2)[0], size= 1))
                            index_pb = int(np.random.choice(np.array(list((set(np.where(scalar_fitness >= 0.75)[0]) & set(np.where(skill_factor == task2)[0])))), size= (1)))
                    
                    # CHỌN CHA MẸ # TRONG BÀI BÁO LÀ CHỌN CON KHỎE NHẤT. 
                    # CROSSOVER 
                    if task == task2: 
                        oa, ob = sbx_crossover(population[index_pa], population[index_pb],swap=True)
                    else: 
                        oa, ob = sbx_crossover(population[index_pa], population[index_pb],swap=False)
                        
                    skf_oa  = skf_ob= task 
                    fcost_oa = tasks[skf_oa].calculate_fitness(oa)
                    fcost_ob= tasks[skf_ob].calculate_fitness(ob) 



                    skill_factor_childs.append(skf_oa) 
                    skill_factor_childs.append(skf_ob) 

                    factorial_cost_childs.append(fcost_oa) 
                    factorial_cost_childs.append(fcost_ob) 

                    childs.append(oa) 
                    childs.append(ob) 


                    
                    index_child_each_tasks.append(index)
                    index_child_each_tasks.append(index + 1) 
                    task_partner.append(task2) 
                    task_partner.append(task2) 

                    number_child_each_task[task] += 2 
                    evaluations[task] += 2
                    index += 2 
        
    

        #NOTE
        if lsa is True:
            current_size_population = Linear_population_size_reduction(
                evaluations,
                current_size_population,
                maxEvals,
                NUMBER_TASKS,
                initial_size_population,
                min_size_population,
            )
        # for task in range(len(tasks)):
        #     index_child_each_tasks[task] += len(population) 
        population = np.concatenate([population, np.array(childs)])
        skill_factor = np.concatenate([skill_factor, np.array(skill_factor_childs)])
        factorial_cost = np.concatenate(
            [factorial_cost, np.array(factorial_cost_childs)]
        )
        scalar_fitness = compute_scalar_fitness(factorial_cost, skill_factor)

        # Cập nhật index của bọn child cho mỗi tác vụ 
        # Check
        assert len(population) == len(skill_factor)
        assert len(population) == len(factorial_cost)

        #
        delete_index = []
        choose_index = [] 
        for ind in range(len(population)):
            #FIXME: THIẾU ĐI CHẤT LƯỢNG KHI THÊM.  
            if(scalar_fitness[ind]) < 1.0 / current_size_population[skill_factor[ind]]:
                delete_index.append(ind) 
                if ind >= index_child_each_tasks[0]:
                    task1 = skill_factor[ind] 
                    task2 = task_partner[ind - index_child_each_tasks[0]]
                    memory_task[task1].update_history_memory(task2, success=False)
            else: 
                choose_index.append(ind)
                if ind >= index_child_each_tasks[0]:
                    task1 = skill_factor[ind]
                    task2 = task_partner[ind - index_child_each_tasks[0]]
                    memory_task[task1].update_history_memory(task2, success=True) 
        # print("HMMM")
        # Tính toán lại ma trận prob 
        for task in range(len(tasks)):
            p = np.copy(memory_task[task].compute_prob())
            assert p_matrix[task].shape == p.shape
            p_matrix[task] = p_matrix[task] * 0.9 + p * 0.1
        
        history_p_matrix.append(np.copy(p_matrix))


        #: UPDATE QUẦN THỂ
        population = population[choose_index]
        scalar_fitness = scalar_fitness[choose_index]
        skill_factor = skill_factor[choose_index]
        factorial_cost = factorial_cost[choose_index]

        assert len(population) == np.sum(current_size_population)

        if int(evaluations[0] / 100) > len(history_cost):
            
            results = optimize_result(population, skill_factor, factorial_cost, tasks)
            history_cost.append(results)

            sys.stdout.write("\r")
            sys.stdout.write(
                "Epoch {}, [%-20s] %3d%% ,pop_size: {},focus: {},  func_val: {}".format(
                    int(evaluations[0] / 100) + 1,
                    len(population),
                    [memory_task[i].isFocus for i in range(len(tasks))],
                    [results[i].cost for i in range(NUMBER_TASKS)],
                    # [p_matrix[6][i] for i in range(NUMBER_TASKS)],
                )
                % (
                    "=" * np.int((np.sum(evaluations) + 1) // (MAXEVALS // 20)) + ">",
                    (np.sum(evaluations) + 1) * 100 // MAXEVALS,
                )
            )
            sys.stdout.flush()
    print("")
    for i in range(NUMBER_TASKS):
        print(tasks[i].name, ": ", results[i].cost)
    return history_cost, history_p_matrix 

