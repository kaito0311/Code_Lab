import matplotlib.pyplot as plt
import numpy as np
from numpy.core.fromnumeric import choose
from utils.utils_sa_lsa import *
from config import *
from tqdm import trange

np.random.seed(0)


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


def lsa_mfea(tasks, lsa=True):

    history_memories = [Memory(H) for i in range(NUMBER_RMPS)]

    # Kich thuoc quan the
    initial_size_population = np.zeros((NUMBER_TASKS), dtype=int) + 100
    current_size_population = np.copy(initial_size_population)
    min_size_population = np.zeros((NUMBER_TASKS), dtype=int) + 50

    evaluations = np.zeros((NUMBER_TASKS), dtype=int)
    maxEvals = np.zeros_like(evaluations, dtype=int) + MAXEVALS / NUMBER_TASKS

    # Khoi tao quan the
    population = create_population(
        np.sum(initial_size_population), DIMENSIONS, LOWER_BOUND, UPPER_BOUND
    )

    skill_factor = np.zeros((np.sum(initial_size_population)), dtype=int)
    factorial_cost = np.zeros((np.sum(initial_size_population)), dtype=float)

    skill_factor, factorial_cost = generate_population(
        population, tasks, np.sum(initial_size_population)
    )

    skill_factor = skill_factor_best_task(population, tasks)
    
    factorial_cost = cal_factor_cost(population, tasks, skill_factor)

    scalar_fitness = compute_scalar_fitness(factorial_cost, skill_factor)

    # For visual
    dem = 0
    history_cost = []
    history_rmp = [[] for i in range(NUMBER_RMPS)]

    iterator = trange(LOOP * 2)

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
        while count < np.sum(current_size_population):
            list_population = np.arange(len(population))
            np.random.shuffle(list_population)
            index_parents = choose_parent(list_population)
            # chọn ra 2 index parent
            # index_parents = choose_parent(maxEvals, scalar_fitness, evaluations, skill_factor)
            if index_parents is None:
                break

            index = index_convert_matrix_to_1D(
                skill_factor[index_parents[0]],
                skill_factor[index_parents[1]],
                len(tasks),
            )
            generate_rmp = history_memories[index].random_Gauss()

            if skill_factor[index_parents[0]] != skill_factor[index_parents[1]]:

                generate_rmps[index].append(generate_rmp)
                count_rmp[index] += 1

            child, skf_child, fac_cost_child, S, xichma = create_child_lsa_sa(
                population, index_parents, skill_factor, generate_rmp, S, xichma, tasks
            )

            # for i in range(2):
            #     childs.append(child[i])
            #     if len(np.array(childs).shape) == 1:
            #         print("hmmm")
            #     skill_factor_childs.append(skf_child[i])
            #     factorial_cost_childs.append(fac_cost_child[i])
            evaluations[skf_child[0]] += 1
            evaluations[skf_child[1]] += 1
            if len(childs) == 0:
                childs = child
                skill_factor_childs = skf_child
                factorial_cost_childs = fac_cost_child
            else:
                childs = np.concatenate([childs, child], axis=0)
                skill_factor_childs = np.concatenate([skill_factor_childs, skf_child])
                factorial_cost_childs = np.concatenate(
                    [factorial_cost_childs, fac_cost_child]
                )

            count += 2

        for i in range(NUMBER_RMPS):
            if count_rmp[i] > 0:
                history_rmp[i].append(np.sum(generate_rmps[i]) / count_rmp[i])
        # For visual
        # print(history_memories[1].M)
        history_memories = Update_Success_History_Memory(
            history_memories, S, xichma, NUMBER_TASKS
        )

        # Linear_population_size_reduction()
        if lsa is True:
            current_size_population = Linear_population_size_reduction(
                evaluations,
                current_size_population,
                maxEvals,
                NUMBER_TASKS,
                initial_size_population,
                min_size_population,
            )

        # Combine population parent with childs
        population = np.concatenate([population, np.array(childs)])
        skill_factor = np.concatenate([skill_factor, np.array(skill_factor_childs)])
        factorial_cost = np.concatenate(
            [factorial_cost, np.array(factorial_cost_childs)]
        )
        scalar_fitness = compute_scalar_fitness(factorial_cost, skill_factor)

        # Check
        assert len(population) == len(skill_factor)
        assert len(population) == len(factorial_cost)

        population, skill_factor, scalar_fitness, factorial_cost = update(
            population,
            current_size_population,
            skill_factor,
            scalar_fitness,
            factorial_cost,
        )

        if int(evaluations[0] / 100) > len(history_cost):
            results = optimize_result(population, skill_factor, factorial_cost, tasks)
            history_cost.append(results)
            iterator.set_description(
                f"{[results[i].cost for i in range(NUMBER_TASKS)]}  {[evaluations[k] for k in range(NUMBER_TASKS)]}"
            )

        assert len(population) == np.sum(current_size_population)

        # if dem > LOOP :
        #     break

    for i in range(NUMBER_TASKS):
        print(results[i].cost)
    return history_cost, history_rmp
