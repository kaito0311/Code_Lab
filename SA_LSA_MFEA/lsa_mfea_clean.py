import matplotlib.pyplot as plt
import numpy as np
import sys
from numpy.core.defchararray import find
from numpy.core.fromnumeric import choose, take
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


def lsa_mfea(tasks, lsa=True):

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

    # skill_factor, factorial_cost = generate_population(
    #     population, tasks, np.sum(initial_size_population)
    # )

    skill_factor = skill_factor_best_task(population, tasks)

    factorial_cost = cal_factor_cost(population, tasks, skill_factor)

    scalar_fitness = compute_scalar_fitness(factorial_cost, skill_factor)

    M_rmp = [[Memory(H) for i in range(len(tasks))] for j in range(len(tasks))]

    history_rmp = [[[] for i in range(len(tasks))] for i in range(len(tasks))]

    history_cost = []
    while np.sum(evaluations) <= MAXEVALS:

        S = np.empty((len(tasks), len(tasks), 0)).tolist()
        sigma = np.empty((len(tasks), len(tasks), 0)).tolist()

        childs = []
        skill_factor_childs = []
        factorial_cost_childs = []

        list_generate_rmp = np.empty((len(tasks), len(tasks), 0)).tolist()

        list_population = np.arange(len(population))
        np.random.shuffle(list_population)

        while len(childs) < np.sum(current_size_population):

            index_pa = int(np.random.choice(list_population[: int(len(list_population) / 2)], size=(1)))
            index_pb = int(np.random.choice(list_population[int(len(list_population) / 2) :], size=(1)))

            # NOTE
            # index_pb = int(np.random.choice(list_population, size=(1)))

            if skill_factor[index_pa] > skill_factor[index_pb]:
                tmp = index_pa
                index_pa = index_pb
                index_pb = tmp

            rmp = M_rmp[skill_factor[index_pa]][skill_factor[index_pb]].random_Gauss()
            fcost_oa = None 
            fcost_ob = None
            if skill_factor[index_pa] == skill_factor[index_pb]:
                oa, ob = sbx_crossover(population[index_pa], population[index_pb])

                # TODO
                # POLY MUTATION :)

                skf_oa = skf_ob = skill_factor[index_pa]

                fcost_oa = tasks[skf_oa].calculate_fitness(oa)
                fcost_ob = tasks[skf_ob].calculate_fitness(ob)
            else:
                list_generate_rmp[skill_factor[index_pa]][skill_factor[index_pb]].append(rmp)
                if np.random.uniform() < rmp:
                    oa, ob = sbx_crossover(
                        population[index_pa], population[index_pb], swap= False
                    )

                    skf_oa, skf_ob = np.random.choice(
                        skill_factor[[index_pa, index_pb]], size=2
                    )
                    # skf_ob = int(
                    #     np.random.choice(skill_factor[[index_pa, index_pb]], size=(1))
                    # )

                else:
                    pa1 = find_individual_same_skill(skill_factor, index_pa)
                    oa, _ = sbx_crossover(population[pa1], population[index_pa])

                    pb1 = find_individual_same_skill(skill_factor, index_pb)
                    ob, _ = sbx_crossover(population[pb1], population[index_pb])

                    skf_oa = skill_factor[index_pa]
                    skf_ob = skill_factor[index_pb]

                delta = 0


                
                fcost_oa = tasks[skf_oa].calculate_fitness(oa)
                fcost_ob = tasks[skf_ob].calculate_fitness(ob)

                if skf_oa == skill_factor[index_pa]:
                    fcost_pa = tasks[skill_factor[index_pa]].calculate_fitness(population[index_pa])
                    delta = np.max([delta, (fcost_pa - fcost_oa)/ (fcost_pa + 1e-10)])
                else:
                    fcost_pb = tasks[skill_factor[index_pb]].calculate_fitness(population[index_pb])
                    delta = np.max([delta, (fcost_pb - fcost_oa) /(fcost_pb + 1e-10)])

                if skf_ob == skill_factor[index_pa]:
                    fcost_pa = tasks[skill_factor[index_pa]].calculate_fitness(population[index_pa])
                    delta = np.max([delta, (fcost_pa - fcost_ob)/ (fcost_pa + 1e-10)])
                else:
                    fcost_pb = tasks[skill_factor[index_pb]].calculate_fitness(population[index_pb])
                    delta = np.max([delta, (fcost_pb - fcost_ob) /(fcost_pb + 1e-10)])

                if delta > 0:
                    S[skill_factor[index_pa]][skill_factor[index_pb]].append(rmp)
                    sigma[skill_factor[index_pa]][skill_factor[index_pb]].append(delta)

            evaluations[skf_oa] += 1
            evaluations[skf_ob] += 1

            factorial_cost_childs.append(fcost_oa)
            factorial_cost_childs.append(fcost_ob)

            skill_factor_childs.append(skf_oa)
            skill_factor_childs.append(skf_ob)

            childs.append(oa)
            childs.append(ob)

        M_rmp = Update_History_Memory(M_rmp, S, sigma, len(tasks))

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

        assert len(population) == np.sum(current_size_population)

        if int(evaluations[0] / 100) > len(history_cost):
            for i in range(len(tasks)):
                j = i + 1
                while j < len(tasks):
                    if len(list_generate_rmp[i][j]) > 0:
                        history_rmp[i][j].append(
                            np.sum(list_generate_rmp[i][j])
                            / len(list_generate_rmp[i][j])
                        )
                    j += 1

            results = optimize_result(population, skill_factor, factorial_cost, tasks)
            history_cost.append(results)

            sys.stdout.write("\r")
            sys.stdout.write(
                "Epoch {}, [%-20s] %3d%% ,pop_size: {}, func_val: {}".format(
                    int(evaluations[0] / 100) + 1,
                    len(population),
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
        print(results[i].cost)
    return history_cost, history_rmp


def Update_History_Memory(history_memories, S, sigma, number_tasks):
    for i in range(number_tasks):
        j = i + 1
        while j < number_tasks:
            if len(S[i][j]) != 0:
                history_memories[i][j].update_M(
                    np.sum(np.array(sigma[i][j]) * np.array(S[i][j]) ** 2)
                    / np.sum(np.array(sigma[i][j]) * (np.array(S[i][j])) + 1e-10)
                )

            j += 1

    return history_memories

