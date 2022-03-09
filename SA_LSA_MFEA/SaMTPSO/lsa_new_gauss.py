
from dataclasses import replace
from re import sub
import matplotlib.pyplot as plt
import numpy as np
import sys
from SaMTPSO.learn_phase import DE, Gauss, learn_phase_task
from SaMTPSO.memory import Memory_SaMTPSO, gauss_mutation, gauss_mutation_jam
from utils.utils_gauss_adapt import add_coefficient_gauss, gauss_base_population, gauss_mutation_self_adap
from utils.utils_sa_lsa import *
from config import *
from tqdm import trange

# np.random.seed(1)



def skill_factor_best_task(pop, tasks,):
    
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




def lsa_SaMTPSO_DE_SBX_new_gauss(tasks, lsa=True, seed = 1, rate= 0.5, max_popu = 100, min_popu= 20, ti_le_giu_lai = 0.9):

    np.random.seed(seed)

    initial_size_population = np.zeros((NUMBER_TASKS), dtype=int) + max_popu
    current_size_population = np.copy(initial_size_population)
    min_size_population = np.zeros((NUMBER_TASKS), dtype=int) + min_popu

    evaluations = np.zeros((NUMBER_TASKS), dtype=int)
    maxEvals = np.zeros_like(evaluations, dtype=int) + int(MAXEVALS / NUMBER_TASKS)

    skill_factor = np.zeros((np.sum(initial_size_population)), dtype=int)
    factorial_cost = np.zeros((np.sum(initial_size_population)), dtype=float)
    population = create_population(
        np.sum(initial_size_population), DIMENSIONS, LOWER_BOUND, UPPER_BOUND
    )

    
    #NOTE: Dimension += 1 do thêm một chiều của chỉ số gauss 
    population = add_coefficient_gauss(population) 

    skill_factor = skill_factor_best_task(population, tasks)

    factorial_cost = cal_factor_cost(population, tasks, skill_factor)

    scalar_fitness = compute_scalar_fitness(factorial_cost, skill_factor)


    
    # Khởi tạo có phần giống với cái xác suất của nó . 
    p_matrix= np.ones(shape= (len(tasks), len(tasks)), dtype= float) / len(tasks)
    # Khởi tạo một mảng các memory cho các task 
    memory_task = [Memory_SaMTPSO(len(tasks)) for i in range(len(tasks))]
    # gauss_mutation(population, population, population, population, population, population)
    history_cost = [] 
    history_p_matrix = []
    history_p_matrix.append(p_matrix)

   
    
    de = [DE() for i in range(len(tasks))]
    block = np.zeros(shape= (len(tasks)), dtype= int)
    ti_le_DE_gauss=np.zeros(shape= (len(tasks)), dtype= float) + 0.5 
    sbx = newSBX(len(tasks), nc = 2, gamma= 0.9, alpha= 6)
    sbx.get_dim_uss(DIMENSIONS+1)
    ti_le_dung_de = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]) - 0.2
    #NOTE: new gauss jam: init
    gauss_mu_jam  = [gauss_mutation_jam() for i in range(len(tasks))] 

    while np.sum(evaluations) <= MAXEVALS:

        dc_sinh_ra_de = [] 

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
      

        for task in range(len(tasks)):
            # if block[task] != 0: 
            #     continue 
            # if len(history_cost) > 2 and  history_cost[len(history_cost) - 1][task].cost == 0.0: 
            #     continue 
            while(number_child_each_task[task] < current_size_population[task]):
                # random de chon task nao de giao phoi 
                task2 = None 
                index_pa= int(np.random.choice(np.where(skill_factor == task)[0], size= 1))
                
                # index_pa = int(np.random.choice(np.array(list((set(np.where(scalar_fitness >= 1/(0.5 * current_size_population[task]))[0]) & set(np.where(skill_factor == task)[0])))), size= (1)))
                index_pb = index_pa 
                task2 = np.random.choice(np.arange(len(tasks)), p= p_matrix[task])
                # while index_pa == index_pb:  
                #     index_pb = int(np.random.choice(np.where(skill_factor == task2)[0], size = 1))
                while index_pb == index_pa : 
                    index_for_pb= np.random.choice(np.where(skill_factor == task2)[0], size= (2,), replace= False)
                    if factorial_cost[index_for_pb[0]] < factorial_cost[index_for_pb[1]]:
                        index_pb = int(index_for_pb[0]) 
                    else: 
                        index_pb = int(index_for_pb[1]) 

                # if memory_task[task].isFocus == False: 
                # if True:
                #     task2 = np.random.choice(np.arange(len(tasks)), p= p_matrix[task]) 
                #     while index_pa == index_pb:
                #         index_pb = int(np.random.choice(np.array(list((set(np.where(scalar_fitness >= 1/(1.0 * current_size_population[task2]))[0]) & set(np.where(skill_factor == task2)[0])))), size= (1)))
                #         # index_pb = int(np.random.choice(np.where(skill_factor == task2)[0], size= 1))
                # else: 
                #     task2 = task 
                #     while index_pa == index_pb:
                #         # index_pb = int(np.random.choice(np.where(skill_factor == task2)[0], size= 1))
                #         index_pb = int(np.random.choice(np.array(list((set(np.where(scalar_fitness >= 1/(0.25 * current_size_population[task]))[0]) & set(np.where(skill_factor == task2)[0])))), size= (1)))
                
                # CHỌN CHA MẸ # TRONG BÀI BÁO LÀ CHỌN CON KHỎE NHẤT. 
                # CROSSOVER 
                skf_oa = skf_ob= task
                # oa, ob = sbx(population[index_pa], population[index_pb], (task, task2))
                if task == task2: 
                    oa, ob = sbx_crossover(population[index_pa], population[index_pb], swap= True)
                else: 
                    oa, ob = sbx_crossover(population[index_pa], population[index_pb], swap= False)

                # oa = gauss_base_population(population[np.where(skill_factor == task)[0]], oa) 
                # ob = gauss_base_population(population[np.where(skill_factor == task)[0]], ob) 
                # #NOTE: new gauss jam: mutation 
                # if task == task2: 
                #     oa, _ = gauss_mu_jam[skf_oa].mutation(oa, population[np.where(skill_factor == skf_oa)[0]], skf_oa)
                #     ob, _ = gauss_mu_jam[skf_ob].mutation(ob, population[np.where(skill_factor == skf_ob)[0]], skf_ob)
                    

            
                fcost_oa = tasks[skf_oa].calculate_fitness(oa)
                fcost_ob = tasks[skf_ob].calculate_fitness(ob) 

                # tinh phan tram cai thien 
                delta_oa = (factorial_cost[index_pa] - fcost_oa) / (factorial_cost[index_pa] + 1e-10)
                delta_ob = (factorial_cost[index_pa] - fcost_ob) / (factorial_cost[index_pa] + 1e-10) 

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
        delete_index = [[] for i in range(len(tasks))]
        choose_index = [] 
        index_child_success = [] 
        index_child_fail = [] 
        delta_improvment = [] 
        delta_decrease = [] 
        nb_choosed_each_tasks = np.zeros(shape= (len(tasks),), dtype= int)
        for ind in range(len(population)):
      
            if(scalar_fitness[ind]) < 1.0 / (current_size_population[skill_factor[ind]] * ti_le_giu_lai) :
                delete_index[skill_factor[ind]].append(ind) 
            else: 
                choose_index.append(ind)
                nb_choosed_each_tasks[skill_factor[ind]] += 1 
               
            if ind >= index_child_each_tasks[0] : 
                task1 = skill_factor[ind]
                task2 = task_partner[ind - index_child_each_tasks[0]]

                if scalar_fitness[ind] < 1.0 / current_size_population[skill_factor[ind]]: 
                    memory_task[task1].update_history_memory(task2, 1, success= False)
                else: 
                    index_child_success.append(ind- index_child_each_tasks[0])
                    memory_task[task1].update_history_memory(task2, 1, success= True) 


        if ti_le_giu_lai < 1.0 : 
            for i in range(len(tasks)):
            # while nb_choosed_each_tasks[i] < current_size_population[i]:
                    # chọn random từ đống kia 1 vài con 
                choose_index = np.concatenate([np.array(choose_index),np.random.choice(delete_index[i], size= int(current_size_population[i] - nb_choosed_each_tasks[i]), replace = False)])
                    
        # Tính toán lại ma trận prob 
        for task in range(len(tasks)):
            p = np.copy(memory_task[task].compute_prob())
            assert p_matrix[task].shape == p.shape
            p_matrix[task] = p_matrix[task] * 0.8 + p * 0.2
            # p_matrix[task] = p 




        #: UPDATE QUẦN THỂ
        np.random.shuffle(choose_index)
        population = population[choose_index]
        scalar_fitness = scalar_fitness[choose_index]
        skill_factor = skill_factor[choose_index]
        factorial_cost = factorial_cost[choose_index]

        assert len(population) == np.sum(current_size_population)

        #NOTE: UPDATE SBX CODE KIEN 
        # sbx.update(index_child_success)
        # sbx.update_success_fail(index_child_success, delta_improvment, index_child_fail, delta_decrease, c=2)

        index_population_tasks = [[] for i in range(len(tasks))]
        for ind in range(len(population)): 
            index_population_tasks[skill_factor[ind]].append(ind) 
        

      
        

        # xsuat_lay = 0
        # # ANCHOR: learn phase DE
        # for subpop in range(len(tasks)):
        #     # if gauss_mu_jam[subpop].is_jam:
        #     #     continue 
        #     count_mutation = 0; 
        #     count_de = 0; 
        #     danh_gia_DE = 0 
        #     danh_gia_Gauss= 0 
            
        #     # for ind in np.random.shuffle(index_population_tasks[subpop])[:int(len(index_child_each_tasks[subpop]))]:
        #     max_f = np.max(factorial_cost[np.array(list((set(np.where(scalar_fitness >= 1/(ti_le_giu_lai * current_size_population[subpop]))[0]) & set(np.where(skill_factor == subpop)[0]))))])
        #     min_f = np.min(factorial_cost[np.array(list((set(np.where(scalar_fitness >= 1/(ti_le_giu_lai * current_size_population[subpop]))[0]) & set(np.where(skill_factor == subpop)[0]))))])
        #     # min_f = np.min(factorial_cost[index_population_tasks[subpop]])
        #     np.random.shuffle(index_population_tasks[subpop])
       
        #     for ind in index_population_tasks[subpop][:int(len(index_population_tasks[subpop])/2)]:
        #         # if np.random.uniform() <ti_le_DE_gauss[subpop] and gauss_mu_jam[subpop].is_jam is False:
        #         if np.random.uniform() < 1.1:

        #             pbest = pr1 = pr2 = -1 
                   
        #             while pbest == pr1 or pr1 == pr2: 
        #                 pbest= int(np.random.choice(np.array(list((set(np.where(scalar_fitness >= 1/(0.1 * current_size_population[skill_factor[ind]]))[0]) & set(np.where(skill_factor == skill_factor[ind])[0])))), size= (1)))
        #                 pr1= int(np.random.choice(np.array(list((set(np.where(scalar_fitness >= 1/(ti_le_giu_lai * current_size_population[skill_factor[ind]]))[0]) & set(np.where(skill_factor == skill_factor[ind])[0])))), size= (1)))
        #                 pr2= int(np.random.choice(np.array(list((set(np.where(scalar_fitness >= 1/(ti_le_giu_lai * current_size_population[skill_factor[ind]]))[0]) & set(np.where(skill_factor == skill_factor[ind])[0])))), size= (1)))
              
        #             # pr1 = int(np.random.choice(np.where(skill_factor == skill_factor[ind])[0], size= 1))
        #             # pr2 = int(np.random.choice(np.where(skill_factor == skill_factor[ind])[0], size= 1))
        #             # pr1 = int(np.random.choice(np.array(list((set(np.where(scalar_fitness >= 1/(0.5 * current_size_population[task2]))[0]) & set(np.where(skill_factor == task2)[0])))), size= (1)))

        #             new_ind = de[skill_factor[ind]].DE_cross(population[ind], population[pbest], population[pr1], population[pr2])

        #             new_fcost = tasks[skill_factor[ind]].calculate_fitness(new_ind) 
        #             delta_fcost = factorial_cost[ind] - new_fcost
        #             evaluations[skill_factor[ind]] += 1
        #             # xsuat_lay =  (1/mutation[skill_factor[ind]].tiso) * np.exp(delta_fcost /(max_f - min_f))
        #             # if delta_fcost > 0 : 
        #             danh_gia_DE += 1
        #             if (delta_fcost >=0 and factorial_cost[ind] > 0) or  (min_f > 0 and scalar_fitness[ind] < 1/(0.5* current_size_population[skill_factor[ind]]) and np.random.rand() < min_f / new_fcost ): 
        #             # if delta_fcost > 0 : 
        #                 if delta_fcost != 0:
        #                     count_de += 1       
        #                     de[skill_factor[ind]].update(delta_fcost) 
        #                 population[ind] = new_ind 
        #                 factorial_cost[ind]= factorial_cost[ind] - delta_fcost
                    
        #         else: 
        #             (new_ind, std) = gauss_mu_jam[skill_factor[ind]].mutation(population[ind], population[np.where(skill_factor == subpop)[0]])
        #             danh_gia_Gauss += 1
        #             new_fcost = tasks[skill_factor[ind]].calculate_fitness(new_ind) 
        #             delta_fcost = factorial_cost[ind] - new_fcost
        #             evaluations[skill_factor[ind]] += 1
        #             # xsuat_lay =  (1/mutation[skill_factor[ind]].tiso) * np.exp(delta_fcost /(max_f - min_f))
        #             if (delta_fcost >=0 and factorial_cost[ind] > 0) or (min_f > 0 and scalar_fitness[ind] < 1/(0.5* current_size_population[skill_factor[ind]]) and np.random.rand() < min_f / new_fcost) or std == 0: 
        #                 # de[skill_factor[ind]].update(delta_fcost) 
        #                 if delta_fcost > 0 and factorial_cost[ind] > 0:
        #                     count_mutation += 1 
        #                 # count_mutation += delta_fcost / factorial_cost[ind]
        #                 population[ind] = new_ind 
        #                 factorial_cost[ind]= new_fcost


        #     if danh_gia_DE> 0 and danh_gia_Gauss > 0:
        #         a = count_de / danh_gia_DE
        #         b= count_mutation/ danh_gia_Gauss
        #         if a== b and a == 0: 
        #             ti_le_DE_gauss[subpop] -= (ti_le_DE_gauss[subpop] - 0.5) * 0.2
        #         else: 
        #             x= np.max([a / (a + b), 0.2]) 
        #             x= np.min([x, 0.8]) 
        #             ti_le_DE_gauss[subpop] = ti_le_DE_gauss[subpop]* 0.5+ x * 0.5 
                    
            
        # # ANCHOR: UPDATE DE :))
        # for d in de: 
        #     d.reset()




        if int(evaluations[0] / 100) > len(history_cost):
            
            results = optimize_result(population, skill_factor, factorial_cost, tasks)
            history_cost.append(results)

            #NOTE: new gauss jam: update 
            end = len(history_cost) -1 
            for i in range(len(tasks)):
                gauss_mu_jam[i].update_scale(history_cost[end][i].cost) 


            history_p_matrix.append(np.copy(p_matrix))
            sys.stdout.flush()
            sys.stdout.write("\r")
            from time import sleep
            
            sys.stdout.write(
                "Epoch {}, [%-20s] %3d%% ,pop_size: {},count_de: {},  func_val: {}".format(
                    int(evaluations[0] / 100) + 1,
                    len(population),
                    # [ti_le_DE_gauss[i] for i in range(len(tasks))],
                    [evaluations[i] for i in range(len(tasks))],
                    # [gauss_mu_jam[i].is_jam for i in range(len(tasks))],
                    # [gauss_mu_jam[i].curr_diversity for i in range(len(tasks))],
                    # [i for i in ti_le_dung_de],
                    # count_mutation,
                    # [memory_task[9].success_history[i][memory_task[9].next_position_update-1] for i in range(len(tasks))],
                    # [mutation[i].jam for i in range(NUMBER_TASKS)],
                    # [p_matrix[6][i] for i in range(NUMBER_TASKS)],
                    [results[i].cost for i in range(NUMBER_TASKS)],
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

