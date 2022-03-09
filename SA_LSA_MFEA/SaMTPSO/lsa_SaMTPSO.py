from ast import operator
from math import factorial
from pathlib import PurePath
from re import T
from turtle import shape
import matplotlib.pyplot as plt
import numpy as np
import sys
from numpy.core.defchararray import find
from numpy.core.fromnumeric import choose, take
from SaMTPSO.learn_phase import DE, Gauss, learn_phase_task
from SaMTPSO.memory import Memory_SaMTPSO, gauss_mutation
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
    mutation = [gauss_mutation(np.zeros(shape= (50,)) + 0.1, population, i, skill_factor, scalar_fitness, factorial_cost) for i in range(len(tasks))]
    # gauss_mutation(population, population, population, population, population, population)
    history_cost = [] 
    history_p_matrix = []
    history_p_matrix.append(p_matrix)

    search_operator_task = [learn_phase_task() for i in range(len(tasks))]
    de = [DE() for i in range(len(tasks))]

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
        delta = [] 
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
                            # index_pb = int(np.random.choice(np.array(list((set(np.where(scalar_fitness >= 1/(0.5 * current_size_population[task2]))[0]) & set(np.where(skill_factor == task2)[0])))), size= (1)))
                            index_pb = int(np.random.choice(np.where(skill_factor == task2)[0], size= 1))
                    else: 
                        task2 = task 
                        while index_pa == index_pb:
                            # index_pb = int(np.random.choice(np.where(skill_factor == task2)[0], size= 1))
                            index_pb = int(np.random.choice(np.array(list((set(np.where(scalar_fitness >= 1/(0.75 * current_size_population[task]))[0]) & set(np.where(skill_factor == task2)[0])))), size= (1)))
                    
                    # CHỌN CHA MẸ # TRONG BÀI BÁO LÀ CHỌN CON KHỎE NHẤT. 
                    # CROSSOVER 
                    skf_oa = skf_ob= task
                    if task == task2: 
                        oa, ob = sbx_crossover(population[index_pa], population[index_pb],swap=True)
                    else: 
                        
                        oa, ob = sbx_crossover(population[index_pa], population[index_pb],swap=False)
                        # ob = mutation[skf_ob].mutation(population[index_pa])
                    
                    # oa = mutation[skf_oa].mutation(oa)
                    # ob = mutation[skf_ob].mutation(ob)
                    
                    
                    #ANCHOR 
                    
                    fcost_oa = tasks[skf_oa].calculate_fitness(oa)
                    fcost_ob= tasks[skf_ob].calculate_fitness(ob) 

                    # tinh phan tram cai thien 
                    delta_oa = (factorial_cost[index_pa] - fcost_oa) / (factorial_cost[index_pa] + 1e-10)
                    delta_ob = (factorial_cost[index_pa] - fcost_ob) / (factorial_cost[index_pa] + 1e-10) 
                    if task2 == task: 
                        delta_oa = max(delta_oa, (factorial_cost[index_pb] - fcost_oa) / (factorial_cost[index_pb] + 1e-10)) 
                        delta_ob = max(delta_ob, (factorial_cost[index_pb] - fcost_ob)/ (factorial_cost[index_pb] + 1e-10))

                    # if delta_oa > 0 or delta_ob > 0: 
                    #     if delta_oa >= delta_ob : 
                    #         # swap 
                    #         population[index_pa], oa = oa, population[index_pa]
                    #         fcost_oa, factorial_cost[index_pa] = factorial_cost[index_pa], fcost_oa
                    #     if delta_ob > delta_oa: 
                    #         population[index_pa], ob = ob, population[index_pa]
                    #         fcost_ob, factorial_cost[index_pa] = factorial_cost[index_pa], fcost_ob
                    # if delta_oa > 0 or delta_ob > 0: 
                    #     if delta_oa >= delta_ob : 
                    #         # swap 
                    #         population[index_pa], oa = oa, population[index_pa]
                    #         fcost_oa, factorial_cost[index_pa] = factorial_cost[index_pa], fcost_oa
                    #     if delta_ob > delta_oa: 
                    #         population[index_pa], ob = ob, population[index_pa]
                    #         fcost_ob, factorial_cost[index_pa] = factorial_cost[index_pa], fcost_ob
                    # if delta_oa > 0: 

                    delta.append(delta_oa)
                    delta.append(delta_ob) 

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

            if(scalar_fitness[ind]) < 1.0 / current_size_population[skill_factor[ind]]:
                delete_index.append(ind) 
                if ind >= index_child_each_tasks[0]:
                    task1 = skill_factor[ind] 
                    task2 = task_partner[ind - index_child_each_tasks[0]]

                    # if delta[ind-index_child_each_tasks[0]] > 0: # Không được giữ lại nhưng lại tốt hơn cha hoặc mẹ -> vẫn được tính là thành công  
                    #     memory_task[task1].update_history_memory(task2, delta[ind-index_child_each_tasks[0]]/2, success=True)

                    # else: # Không được giữ lại và không tốt hơn cha mẹ -> đảo dấu để tăng mẫu số
                    #     delta[ind-index_child_each_tasks[0]] =  -delta[ind-index_child_each_tasks[0]]
                    #     memory_task[task1].update_history_memory(task2, delta[ind-index_child_each_tasks[0]], success=False)

                    #ANCHOR: QUAY VỀ BẢN GỐC 
                    delta[ind-index_child_each_tasks[0]] = 1 
                    memory_task[task1].update_history_memory(task2, delta[ind-index_child_each_tasks[0]], success=False)
            else: 
                choose_index.append(ind)
                if ind >= index_child_each_tasks[0]:
                    task1 = skill_factor[ind]
                    task2 = task_partner[ind - index_child_each_tasks[0]]
                    # if delta[ind-index_child_each_tasks[0]] < 0: # Nếu mà chỉ được giữ lại thì để nguyên 
                    #     delta[ind-index_child_each_tasks[0]] = -delta[ind-index_child_each_tasks[0]]
                    # if delta[ind-index_child_each_tasks[0]] > 0: # Nếu được giữ lại mà còn tốt hơn cha hoặc mẹ -> tăng đôi trọng só
                    #     delta[ind-index_child_each_tasks[0]] *= 2 
                    
                    #ANCHOR: QUAY VỀ BẢN GỐC :)
                    delta[ind-index_child_each_tasks[0]] = 1 
                    memory_task[task1].update_history_memory(task2, delta[ind-index_child_each_tasks[0]], success=True)
                    
        # print("HMMM")
        # Tính toán lại ma trận prob 
        for task in range(len(tasks)):
            p = np.copy(memory_task[task].compute_prob())
            assert p_matrix[task].shape == p.shape
            p_matrix[task] = p_matrix[task] * 0.9 + p * 0.1
            # p_matrix[task] =p 
        
        history_p_matrix.append(np.copy(p_matrix))


        #: UPDATE QUẦN THỂ
        population = population[choose_index]
        scalar_fitness = scalar_fitness[choose_index]
        skill_factor = skill_factor[choose_index]
        factorial_cost = factorial_cost[choose_index]

        assert len(population) == np.sum(current_size_population)


        # #ANCHOR: Thêm SHADE 
        # index_population_tasks = [[] for i in range(len(tasks))]
        # for ind in range(len(population)): 
        #     index_population_tasks[skill_factor[ind]].append(ind) 
        
        # for subpop in range(len(tasks)):
        #     for ind in index_population_tasks[subpop]:
        #         pbest= int(np.random.choice(np.array(list((set(np.where(scalar_fitness >= 1/(0.1 * current_size_population[skill_factor[ind]]))[0]) & set(np.where(skill_factor == skill_factor[ind])[0])))), size= (1)))
        #         pr1 = int(np.random.choice(np.where(skill_factor == skill_factor[ind])[0], size= 1))
        #         pr2 = int(np.random.choice(np.where(skill_factor == skill_factor[ind])[0], size= 1))

        #         new_ind = de[skill_factor[ind]].DE_cross(population[ind], population[pbest], population[pr1], population[pr2])

        #         delta_fcost=  factorial_cost[ind] - tasks[skill_factor[ind]].calculate_fitness(new_ind) 
        #         evaluations[skill_factor[ind]] += 1
        #         if delta_fcost > 0: 
        #             de[skill_factor[ind]].update(delta_fcost) 
        #             population[ind] = new_ind 
        #             factorial_cost[ind]= factorial_cost[ind] - delta_fcost
                 
        #ANCHOR: BAI A THANG 
        # for ind in range(len(population)):
        #     # xac dinh xem no thuoc vao De hay gauss 
        #     if np.random.uniform() < 0.3: 
        #         # thuoc ve operator 1 
        #         if search_operator_task[skill_factor[ind]].operator1.name == "gauss": 
        #             new_ind = search_operator_task[skill_factor[ind]].operator1.gauss_mutation(ind) 
        #         else : 
        #             pbest= int(np.random.choice(np.array(list((set(np.where(scalar_fitness >= 1/(0.1 * current_size_population[skill_factor[ind]]))[0]) & set(np.where(skill_factor == skill_factor[ind])[0])))), size= (1)))
        #             pr1 = int(np.random.choice(np.where(skill_factor == skill_factor[ind])[0], size= 1))
        #             pr2 = int(np.random.choice(np.where(skill_factor == skill_factor[ind])[0], size= 1))
        #             new_ind= search_operator_task[skill_factor[ind]].operator1.DE_cross(ind, pbest, pr1, pr2)
        #         delta_fcost= tasks[skill_factor[ind]].calculate_fitness(new_ind) - factorial_cost[ind]

        #         search_operator_task[skill_factor[ind]].operator1.update(delta_fcost)
        #     else : 
        #         # thuoc ve operator 2 : 
        #         if search_operator_task[skill_factor[ind]].operator2.name == "gauss":
        #             new_ind= search_operator_task[skill_factor[ind]].operator2.gauss_mutation(ind) 
        #         else: 
        #             pbest= int(np.random.choice(np.array(list((set(np.where(scalar_fitness >= 1/(0.1 * current_size_population[skill_factor[ind]]))[0]) & set(np.where(skill_factor == skill_factor[ind])[0])))), size= (1)))
        #             pr1 = int(np.random.choice(np.where(skill_factor == skill_factor[ind])[0], size= 1))
        #             pr2 = int(np.random.choice(np.where(skill_factor == skill_factor[ind])[0], size= 1))
        #             new_ind= search_operator_task[skill_factor[ind]].operator2.DE_cross(ind, pbest, pr1, pr2)
        #         delta_fcost= tasks[skill_factor[ind]].calculate_fitness(new_ind) - factorial_cost[ind]

        #         search_operator_task[skill_factor[ind]].operator2.update(delta_fcost)
            
        #     # xem co thay the khong 
        #     # neu tot hon thi thay the 
        #     # khong thi phu thuoc vao xac suat 
        #     fmax = np.max(factorial_cost[np.where(skill_factor== skill_factor[ind])[0]]) 
        #     fmin = np.min(factorial_cost[np.where(skill_factor == skill_factor[ind])[0]])

        #     # tinh do da dang cua quan the 
        #     index = 

                
            # fmax = np.max(factorial_cost[index_population_tasks[subpop]])
            # fmin = np.min(factorial_cost[index_population_tasks[subpop]])
            # best = np.where(scalar_fitness[index_population_tasks[subpop]] == 1.0)[0]
            # tinh do da dang cua quan the 

            # for ind in index_population_tasks[subpop]:
            # # xac dinh xem no thuoc vao De hay gauss 
            #     if np.random.uniform() < 0.3: 
            #         # thuoc ve operator 1 
            #         if search_operator_task[skill_factor[ind]].operator1.name == "gauss": 
            #             new_ind = search_operator_task[skill_factor[ind]].operator1.gauss_mutation(ind) 
            #         else : 
            #             pbest= int(np.random.choice(np.array(list((set(np.where(scalar_fitness >= 1/(0.1 * current_size_population[skill_factor[ind]]))[0]) & set(np.where(skill_factor == skill_factor[ind])[0])))), size= (1)))
            #             pr1 = int(np.random.choice(np.where(skill_factor == skill_factor[ind])[0], size= 1))
            #             pr2 = int(np.random.choice(np.where(skill_factor == skill_factor[ind])[0], size= 1))
            #             new_ind= search_operator_task[skill_factor[ind]].operator1.DE_cross(ind, pbest, pr1, pr2)
            #         delta_fcost= tasks[skill_factor[ind]].calculate_fitness(new_ind) - factorial_cost[ind]

            #         search_operator_task[skill_factor[ind]].operator1.update(delta_fcost)
            #     else : 
            #         # thuoc ve operator 2 : 
            #         if search_operator_task[skill_factor[ind]].operator2.name == "gauss":
            #             new_ind= search_operator_task[skill_factor[ind]].operator2.gauss_mutation(ind) 
            #         else: 
            #             pbest= int(np.random.choice(np.array(list((set(np.where(scalar_fitness >= 1/(0.1 * current_size_population[skill_factor[ind]]))[0]) & set(np.where(skill_factor == skill_factor[ind])[0])))), size= (1)))
            #             pr1 = int(np.random.choice(np.where(skill_factor == skill_factor[ind])[0], size= 1))
            #             pr2 = int(np.random.choice(np.where(skill_factor == skill_factor[ind])[0], size= 1))
            #             new_ind= search_operator_task[skill_factor[ind]].operator2.DE_cross(ind, pbest, pr1, pr2)
            #         delta_fcost= tasks[skill_factor[ind]].calculate_fitness(new_ind) - factorial_cost[ind]

            #         search_operator_task[skill_factor[ind]].operator2.update(delta_fcost)
                
            #     # xem co thay the khong 
            #     # neu tot hon thi thay the 
            #     # khong thi phu thuoc vao xac suat 

            #     # tinh do da dang cua quan the 
            #     index = 
                
                


                        
            
        #ANCHOR: DE :))
        # for d in de: 
        #     d.reset() 

        



        if int(evaluations[0] / 100) > len(history_cost):
            
            results = optimize_result(population, skill_factor, factorial_cost, tasks)
            history_cost.append(results)
            #ANCHOR:
            end = len(history_cost) -1 
            # for i in range(len(tasks)):
            #     # mutation[i].update_scale(history_cost[end][i].ind, history_cost[end -1][i].ind, history_cost[end][i].cost)
        
            #     mutation[i].update_scale_base_population(population, i, skill_factor, scalar_fitness, factorial_cost)


            sys.stdout.write("\r")
            sys.stdout.write(
                "Epoch {}, [%-20s] %3d%% ,pop_size: {},Focus: {},p: {},  func_val: {}".format(
                    int(evaluations[0] / 100) + 1,
                    len(population),
                    [memory_task[i].isFocus for i in range(len(tasks))],
                    [mutation[i].scale[0] for i in range(len(tasks))],
                    # [memory_task[9].success_history[i][memory_task[9].next_position_update-1] for i in range(len(tasks))],
                    # [mutation[i].jam for i in range(NUMBER_TASKS)],
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

