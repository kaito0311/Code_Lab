
from re import sub
import matplotlib.pyplot as plt
import numpy as np
import sys
from SaMTPSO.learn_phase import DE, Gauss, learn_phase_task
from SaMTPSO.memory import Memory_SaMTPSO, gauss_mutation
from utils.utils_gauss_adapt import add_coefficient_gauss, gauss_mutation_self_adap
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




def lsa_SaMTPSO_DE_SBX(tasks, lsa=True, seed = 1, rate= 0.5, max_popu = 100, min_popu= 20):

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
    sbx.get_dim_uss(DIMENSIONS + 1)
    ti_le_dung_de = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]) - 0.2
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
        while len(childs) < np.sum(current_size_population):

            for task in range(len(tasks)):
                # if block[task] != 0: 
                #     continue 
                while(number_child_each_task[task] < current_size_population[task]):
                    # random de chon task nao de giao phoi 
                    task2 = None 
                    index_pa= int(np.random.choice(np.where(skill_factor == task)[0], size= 1))
                    
                    # index_pa = int(np.random.choice(np.array(list((set(np.where(scalar_fitness >= 1/(0.5 * current_size_population[task]))[0]) & set(np.where(skill_factor == task)[0])))), size= (1)))
                    index_pb = index_pa 
                    # if memory_task[task].isFocus == False: 
                    if True:
                        task2 = np.random.choice(np.arange(len(tasks)), p= p_matrix[task]) 
                        while index_pa == index_pb:
                            index_pb = int(np.random.choice(np.array(list((set(np.where(scalar_fitness >= 1/(1.0 * current_size_population[task2]))[0]) & set(np.where(skill_factor == task2)[0])))), size= (1)))
                            # index_pb = int(np.random.choice(np.where(skill_factor == task2)[0], size= 1))
                    else: 
                        task2 = task 
                        while index_pa == index_pb:
                            # index_pb = int(np.random.choice(np.where(skill_factor == task2)[0], size= 1))
                            index_pb = int(np.random.choice(np.array(list((set(np.where(scalar_fitness >= 1/(0.25 * current_size_population[task]))[0]) & set(np.where(skill_factor == task2)[0])))), size= (1)))
                    
                    # CHỌN CHA MẸ # TRONG BÀI BÁO LÀ CHỌN CON KHỎE NHẤT. 
                    # CROSSOVER 
                    skf_oa = skf_ob= task
                    # NOTE: NEW SBX CODE KIEN 
                    # if scalar_fitness[index_pa] < scalar_fitness[index_pb] and task == task2: 
                    #     index_pa, index_pb = index_pb, index_pa 

                    # if task == task2 and np.random.rand() < ti_le_dung_de[task]: 
                    #     if np.random.rand() <= 1.0: 
                    #         ind = index_pa
                    #         pbest = pr1 = pr2= index_pa
                    #         while pbest == index_pa or pr1 == index_pa or pr2 == index_pa: 
                    #             pbest= int(np.random.choice(np.array(list((set(np.where(scalar_fitness >= 1/(0.1 * current_size_population[skill_factor[ind]]))[0]) & set(np.where(skill_factor == skill_factor[ind])[0])))), size= (1)))
                    #             pr1 = int(np.random.choice(np.where(skill_factor == skill_factor[ind])[0], size= 1))
                    #             pr2 = int(np.random.choice(np.where(skill_factor == skill_factor[ind])[0], size= 1))

                    #         new_ind = de[skill_factor[ind]].DE_cross(population[ind], population[pbest], population[pr1], population[pr2])

                    #         delta_fcost =  factorial_cost[ind] - tasks[skill_factor[ind]].calculate_fitness(new_ind) 
                            
                    #         de[skill_factor[ind]].update(delta_fcost) 
                    #         oa = np.copy(new_ind)
    
                    #         ind = index_pb
                    #         while pbest == index_pb or pr1 == index_pb or pr2 == index_pb: 
                    #             pbest= int(np.random.choice(np.array(list((set(np.where(scalar_fitness >= 1/(0.1 * current_size_population[skill_factor[ind]]))[0]) & set(np.where(skill_factor == skill_factor[ind])[0])))), size= (1)))
                    #             pr1 = int(np.random.choice(np.where(skill_factor == skill_factor[ind])[0], size= 1))
                    #             pr2 = int(np.random.choice(np.where(skill_factor == skill_factor[ind])[0], size= 1))


                    #         new_ind = de[skill_factor[ind]].DE_cross(population[ind], population[pbest], population[pr1], population[pr2])
                    #         delta_fcost=  factorial_cost[ind] - tasks[skill_factor[ind]].calculate_fitness(new_ind) 
                
                    #         de[skill_factor[ind]].update(delta_fcost)
                    #         ob = np.copy(new_ind)
                    #         dc_sinh_ra_de.append(True)
                    #         dc_sinh_ra_de.append(True)

                            
                    # else:
                    #     oa, ob = sbx(population[index_pa], population[index_pb], (task, task2))
                    #     dc_sinh_ra_de.append(False)
                    #     dc_sinh_ra_de.append(False)
                    oa, ob = sbx(population[index_pa], population[index_pb], (task, task2))
                    
                   
                    # if task == task2: 
                    #     # if factorial_cost[index_pa] > factorial_cost[index_pb]:
                    #     #     index_pa, index_pb = index_pb, index_pa
                    #     oa, ob = sbx_crossover(population[index_pa], population[index_pb],swap=True)
                    #     pa_mutation = int(np.random.choice(np.where(skill_factor == task)[0], size= 1))
                    #     pb_mutation = pa_mutation
                    #     while pb_mutation == pa_mutation:
                    #         pb_mutation = int(np.random.choice(np.where(skill_factor == task)[0], size= 1))
                    #         # pb_mutation = int(np.random.choice(np.array(list((set(np.where(scalar_fitness >= 1/(0.25 * current_size_population[task]))[0]) & set(np.where(skill_factor == task)[0])))), size= (1)))
                    #     scale = np.abs(population[pa_mutation] - population[pb_mutation]) 
                    #     # oa = gauss_mutation_kien(oa, population[pa_mutation], population[pb_mutation], scale)
                    #     # ob = gauss_mutation_kien(ob, population[pa_mutation], population[pb_mutation], scale)
                    

                    #     # oa = gauss_mutation_self_adap(oa,rate= scale)
                    #     # ob = gauss_mutation_self_adap(ob, rate= scale) 
                    # else: 
                    #     oa, ob = sbx_crossover(population[index_pa], population[index_pb],swap=False)
                    #     # if np.random.uniform() > p_matrix[task][task2]:
                    #     #     index_pc=index_pb
                    #     #     while index_pa == index_pc or index_pb == index_pc:
                    #     #     # index_pb = int(np.random.choice(np.where(skill_factor == task2)[0], size= 1))
                    #     #         index_pc = int(np.random.choice(np.array(list((set(np.where(scalar_fitness >= 1/(0.5 * current_size_population[task]))[0]) & set(np.where(skill_factor == task2)[0])))), size= (1)))
                    #     #     ob, _ = sbx_crossover(population[index_pa], population[index_pc], swap= False)
                    #     # ob = mutation[skf_ob].mutation(population[index_pa])
                    
                    # #ANCHOR: MUTATION
                    # oa = mutation[skf_oa].mutation(oa)
                    # ob = mutation[skf_ob].mutation(ob)

                    # oa = gauss_self_adapt(oa)
                    # ob = gauss_self_adapt(ob)

                    # pa_mutation = int(np.random.choice(np.where(skill_factor == task)[0], size= 1))
                    # pb_mutation = pa_mutation
                    # while pb_mutation == pa_mutation:
                    #     pb_mutation = int(np.random.choice(np.where(skill_factor == task)[0], size= 1))
                    #     # pb_mutation = int(np.random.choice(np.array(list((set(np.where(scalar_fitness >= 1/(0.25 * current_size_population[task]))[0]) & set(np.where(skill_factor == task)[0])))), size= (1)))
                    # scale = np.abs(population[pa_mutation] - population[pb_mutation]) 
                    # oa = gauss_mutation_kien(oa, population[pa_mutation], population[pb_mutation], scale)
                    # ob = gauss_mutation_kien(ob, population[pa_mutation], population[pb_mutation], scale)
                    

                    # oa = gauss_mutation_self_adap(oa,rate= scale)
                    # ob = gauss_mutation_self_adap(ob, rate= scale) 

            
                
                    fcost_oa = tasks[skf_oa].calculate_fitness(oa)
                    fcost_ob = tasks[skf_ob].calculate_fitness(ob) 

                    # tinh phan tram cai thien 
                    delta_oa = (factorial_cost[index_pa] - fcost_oa) / (factorial_cost[index_pa] + 1e-10)
                    delta_ob = (factorial_cost[index_pa] - fcost_ob) / (factorial_cost[index_pa] + 1e-10) 



        
                    # if task2 == task: 
                    #     delta_oa = max(delta_oa, (factorial_cost[index_pb] - fcost_oa) / (factorial_cost[index_pb] + 1e-10)) 
                    #     delta_ob = max(delta_ob, (factorial_cost[index_pb] - fcost_ob)/ (factorial_cost[index_pb] + 1e-10))

                    # if delta_oa > 0 or delta_ob > 0: 
                    #     if delta_oa >= delta_ob : 
                    #         # swap 
                    #         population[index_pa], oa = oa, population[index_pa]
                    #         fcost_oa, factorial_cost[index_pa] = factorial_cost[index_pa], fcost_oa
                    #     if delta_ob > delta_oa: 
                    #         population[index_pa], ob = ob, population[index_pa]
                    #         fcost_ob, factorial_cost[index_pa] = factorial_cost[index_pa], fcost_ob

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
        index_child_success = [] 
        index_child_fail = [] 
        delta_improvment = [] 
        delta_decrease = [] 
        for ind in range(len(population)):
            # if ind >= index_child_each_tasks[0]: 
            #     if delta[ind-index_child_each_tasks[0]] >= 0: 
            #         index_child_success.append(ind-index_child_each_tasks[0])
            if(scalar_fitness[ind]) < 1.0 / current_size_population[skill_factor[ind]]:
                delete_index.append(ind) 
                if ind >= index_child_each_tasks[0]:
                    task1 = skill_factor[ind] 
                    task2 = task_partner[ind - index_child_each_tasks[0]]

                    # #NOTE:  cho new sbx fail and success 
                    # index_child_fail.append(ind - index_child_each_tasks[0])
                    # delta_decrease.append(delta[ind-index_child_each_tasks[0]])
                    
                    delta[ind-index_child_each_tasks[0]] = 1 

                    memory_task[task1].update_history_memory(task2, delta[ind-index_child_each_tasks[0]], success=False)
            else: 
                choose_index.append(ind)
                if ind >= index_child_each_tasks[0]:
                    task1 = skill_factor[ind]
                    task2 = task_partner[ind - index_child_each_tasks[0]]

                    #NOTE: cho new sbx fail and success
                    # if dc_sinh_ra_de[ind - index_child_each_tasks[0]] is False: 
                    #     index_child_success.append(ind - index_child_each_tasks[0]- len(np.where(np.array(dc_sinh_ra_de[:(ind - index_child_each_tasks[0])]) == 1)[0]))
                    index_child_success.append(ind- index_child_each_tasks[0])
                    # delta_improvment.append(delta[ind-index_child_each_tasks[0]]) 

                    delta[ind-index_child_each_tasks[0]] = 1 
                    memory_task[task1].update_history_memory(task2, delta[ind-index_child_each_tasks[0]], success=True)
                    
        # Tính toán lại ma trận prob 
        for task in range(len(tasks)):
            p = np.copy(memory_task[task].compute_prob())
            assert p_matrix[task].shape == p.shape
            # p_matrix[task] = p_matrix[task] * 0.9 + p * 0.1
            p_matrix[task] = p 




        #: UPDATE QUẦN THỂ
        np.random.shuffle(choose_index)
        population = population[choose_index]
        scalar_fitness = scalar_fitness[choose_index]
        skill_factor = skill_factor[choose_index]
        factorial_cost = factorial_cost[choose_index]

        assert len(population) == np.sum(current_size_population)

        #NOTE: UPDATE SBX CODE KIEN 
        sbx.update(index_child_success)
        # sbx.update_success_fail(index_child_success, delta_improvment, index_child_fail, delta_decrease, c=2)

        index_population_tasks = [[] for i in range(len(tasks))]
        for ind in range(len(population)): 
            index_population_tasks[skill_factor[ind]].append(ind) 
        

      
        

        xsuat_lay = 0
        # # ANCHOR: learn phase DE
        # for subpop in range(len(tasks)):
        #     count_mutation = 0; 
        #     count_de = 0; 
        #     danh_gia_DE = 0 
        #     danh_gia_Gauss= 0 
            
        #     if block[task] != 0:
        #         continue
        #     # for ind in np.random.shuffle(index_population_tasks[subpop])[:int(len(index_child_each_tasks[subpop]))]:
        #     max_f = np.max(factorial_cost[index_population_tasks[subpop]])
        #     min_f = np.min(factorial_cost[index_population_tasks[subpop]])
        #     np.random.shuffle(index_population_tasks[subpop])
       
        #     for ind in index_population_tasks[subpop][:int(len(index_population_tasks[subpop])/2)]:
        #         # if np.random.uniform() <ti_le_DE_gauss[subpop]:
        #         if np.random.uniform() < 1.0:
        #             pbest= int(np.random.choice(np.array(list((set(np.where(scalar_fitness >= 1/(0.1 * current_size_population[skill_factor[ind]]))[0]) & set(np.where(skill_factor == skill_factor[ind])[0])))), size= (1)))
        #             pr1 = int(np.random.choice(np.where(skill_factor == skill_factor[ind])[0], size= 1))
        #             pr2 = int(np.random.choice(np.where(skill_factor == skill_factor[ind])[0], size= 1))

        #             new_ind = de[skill_factor[ind]].DE_cross(population[ind], population[pbest], population[pr1], population[pr2])

        #             delta_fcost=  factorial_cost[ind] - tasks[skill_factor[ind]].calculate_fitness(new_ind) 
        #             evaluations[skill_factor[ind]] += 1
        #             # xsuat_lay =  (1/mutation[skill_factor[ind]].tiso) * np.exp(delta_fcost /(max_f - min_f))
        #             # if delta_fcost > 0 : 
        #             danh_gia_DE += 1
        #             if delta_fcost > 0 : 
        #             # if delta_fcost > 0 : 
        #                 count_de += 1
        #                 if delta_fcost != 0:
        #                     de[skill_factor[ind]].update(delta_fcost) 
        #                 population[ind] = new_ind 
        #                 factorial_cost[ind]= factorial_cost[ind] - delta_fcost
                    
        #         else: 
        #             # new_ind = gauss_self_adapt(population[ind])
        #             pa_mutation = int(np.random.choice(np.where(skill_factor == subpop)[0], size= 1))
        #             pb_mutation = pa_mutation
        #             while pb_mutation == pa_mutation:
        #                 pb_mutation = int(np.random.choice(np.where(skill_factor == subpop)[0], size= 1))
        #                 # pb_mutation = int(np.random.choice(np.array(list((set(np.where(scalar_fitness >= 1/(0.25 * current_size_population[task]))[0]) & set(np.where(skill_factor == task)[0])))), size= (1)))
        #             scale = np.abs(population[pa_mutation] - population[pb_mutation]) 
        #             # new_ind = gauss_mutation_kien(population[ind], population[pa_mutation], population[pb_mutation], scale)
        #             # new_ind = mutation[subpop].mutation(population[ind])

        #             new_ind = gauss_mutation_self_adap(population[ind], rate = scale)
        #             danh_gia_Gauss += 1
        #             delta_fcost = factorial_cost[ind] - tasks[skill_factor[ind]].calculate_fitness(new_ind) 
        #             evaluations[skill_factor[ind]] += 1
        #             # xsuat_lay =  (1/mutation[skill_factor[ind]].tiso) * np.exp(delta_fcost /(max_f - min_f))
        #             if delta_fcost >=0 : 
        #                 # de[skill_factor[ind]].update(delta_fcost) 
        #                 count_mutation += 1
        #                 population[ind] = new_ind 
        #                 factorial_cost[ind]= factorial_cost[ind] - delta_fcost


        #     if danh_gia_DE> 0 and danh_gia_Gauss > 0:
        #         a = count_de / danh_gia_DE
        #         b= count_mutation/ danh_gia_Gauss
        #         if a== b and a == 0: 
        #             ti_le_DE_gauss[subpop] -= (ti_le_DE_gauss[subpop] - 0.5) * 0.2
        #         else: 
        #             x= np.max([a / (a + b), 0.05]) 
        #             x= np.min([x, 0.95]) 
        #             ti_le_DE_gauss[subpop] = ti_le_DE_gauss[subpop]* 0.5+ x * 0.5 
                    
            
        #ANCHOR: UPDATE DE :))
        for d in de: 
            d.reset()
        # ti_le_dung_de += 0.3 / 1000 




        if int(evaluations[0] / 100) > len(history_cost):
            
            results = optimize_result(population, skill_factor, factorial_cost, tasks)
            history_cost.append(results)

            #ANCHOR: Update Mutation
            # end = len(history_cost) -1 
            # for i in range(len(tasks)):
            #     # mutation[i].update_scale_base_population(population, i, skill_factor, scalar_fitness, factorial_cost)
            #     mutation[i].update_scale(history_cost[end][i].ind, history_cost[end -1][i].ind, history_cost[end][i].cost)
                    
            history_p_matrix.append(np.copy(p_matrix))

            sys.stdout.write("\r")
            sys.stdout.write(
                "Epoch {}, [%-20s] %3d%% ,pop_size: {},count_de: {},  func_val: {}".format(
                    int(evaluations[0] / 100) + 1,
                    len(population),
                    # [ti_le_DE_gauss[i] for i in range(len(tasks))],
                    [i for i in ti_le_dung_de],
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

