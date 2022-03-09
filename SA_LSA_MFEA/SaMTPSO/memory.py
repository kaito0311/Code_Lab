
import numpy as np

from config import DIMENSIONS


class Memory_SaMTPSO:
    def __init__(self, number_tasks, LP=10) -> None:
        self.success_history = np.zeros(shape=(number_tasks, LP), dtype=float)
        self.fail_history = np.zeros(shape=(number_tasks, LP), dtype=float)
        self.next_position_update = 0  # index cột đang được update
        self.isFocus = False
        self.epsilon = 0.001
        self.pb = 0.005
        self.LP = LP
        self.duytri = 10
        self.number_tasks = number_tasks

    def update_history_memory(self, task_partner, delta, success=True, end_generation=False) -> None:
        if success:
            self.success_history[task_partner][self.next_position_update] += delta
        else:
            self.fail_history[task_partner][self.next_position_update] += delta

        # Nếu hết một thế hệ -> tăng vị trí cập nhật success và fail memory lên một ->
        # if end_generation:
            # Nếu trong LP thế hệ gần nhất mà không cải thiện được thì -> tập trung intra

    def compute_prob(self) -> np.ndarray:
        sum = np.clip(np.sum(self.success_history, axis=1) +
                      np.sum(self.fail_history, axis=1), 0, 100000)
        sum_sucess = np.clip(np.sum(self.success_history, axis=1), 0, 10000)

        SRtk = sum_sucess / (sum + self.epsilon) + self.pb
        p = SRtk / np.sum(SRtk)
        # FIXME: LÀM SAO BIẾT ĐƯỢC BAO NHIÊU LÀ ĐỦ ĐỂ CHO FOCUS ?
        if np.sum(self.success_history[:, self.next_position_update]) == 0 and self.duytri <= 0:
            self.isFocus = not self.isFocus
            self.duytri = 10

        if np.sum(self.success_history[:, self.next_position_update]) != 0 and self.isFocus is True and self.duytri <= 0:
            self.isFocus = False
            self.duytri = 10
        self.duytri -= 1
        self.next_position_update = (self.next_position_update + 1) % self.LP
        self.success_history[:, self.next_position_update] = 0
        self.fail_history[:, self.next_position_update] = 0
        return p

    # def compute_prob(self)->np.ndarray:


class gauss_mutation:
    def __init__(self, scale, population, task, skill_factor, scalar_fitness, fcost) -> None:
        self.scale = scale  # scale là một mảng nhé có thể âm dương tùy ý
        self.mean = 0

        self.cost = None
        self.beforecost = None

        self.jam = False
        self.loop = 10
        self.D0 = 0

        self.tiso = 1

        if task != None:
            sub = np.where(skill_factor == task)[0]
            best = None
            for i in range(len(population)):
                if skill_factor[i] == task:
                    if scalar_fitness[i] == 1.0:
                        best = i
                        break

            sum_cost = np.sum(fcost[sub])
            for ind_index in sub:
                w = 1 - fcost[ind_index] / sum_cost
                self.D0 += w * \
                    (np.sqrt(
                        np.sum((population[ind_index] - population[best]) ** 2)))

    # FIXME: LÀM SAO ĐỂ QUYẾT ĐỊNH SCALE LÀ BAO NHIÊU

    def update_scale_base_population(self, population=None, task=None, skill_factor=None, scalar_fitness=None, fcost=None):
        Dt = 0
        if task != None:
            sub = np.where(skill_factor == task)[0]
            best = None
            for i in range(len(population)):
                if skill_factor[i] == task:
                    if scalar_fitness[i] == 1.0:
                        best = i
                        break
            sum_cost = np.sum(fcost[sub])
            if sum_cost == 0:
                sum_cost = 1e10

            for ind_index in sub:
                w = 1 - fcost[ind_index] / sum_cost
                Dt += w * \
                    (np.sqrt(
                        np.sum((population[ind_index] - population[best]) ** 2)))

        a = self.D0 / (self.D0 - Dt)
        self.tiso = a

    def update_scale(self, new_ind, old_ind, newest_cost):
        ind = new_ind - old_ind
        self.scale = self.scale * 0.9 + ind * 0.1
        if self.loop <= 0:
            if self.beforecost - newest_cost < 0.1 * self.beforecost:
                self.jam = True
                # self.scale = self.scale * 10
                self.scale = new_ind * 0.1

            else:
                self.jam = False

            self.loop = 10
            self.beforecost = newest_cost
            self.cost = newest_cost
        else:
            self.cost = newest_cost
            if self.beforecost is None:
                self.beforecost = newest_cost
            self.loop -= 1

    '''
    Kiểm tra tắc: - Nếu đủ loop vòng lặp -> kiểm tra xem sau loop vòng có giảm hơn 10 % không -> không thì cho là tắc 
                                                                                                -> có thì thôi bỏ qua cho tắc = False
                 
    Nếu không tắc -> self.scale = self.scale * 0.9 + ind * 0.1
    '''

    def mutation(self, ind) -> np.ndarray:
        D = ind.shape[0]
        for i in range(D):
            a = 1.0
            if self.jam:
                a = 1
            if self.scale.any() == np.nan:
                break
            if(np.random.uniform() < a / D):
                t = -1
                # while t > 1 or t < 0:
                t = ind[i] + np.random.normal(size=1,
                                              loc=self.mean, scale=np.abs(self.scale[i]))
                if t > 1:
                    t = ind[i] + np.random.uniform() * (1 - ind[i])
                elif t < 0:
                    t = np.random.uniform() * ind[i]
                ind[i] = t

        return ind


class gauss_mutation_jam:
    def __init__(self) -> None:
        self.is_jam = False
        self.scale = None

        self.cost_10_pre_gen = None
        self.curr_cost = None
        self.count_gen = 0

        self.max_log = np.zeros(shape=(DIMENSIONS, ), dtype=int) + 2
        self.min_log = 1

        self.count_mu_each_dim = np.zeros(shape=(DIMENSIONS,), dtype=int)
        self.diversity_begin: float = -1
        self.curr_diversity: float = -1

        self.count_mu_each_scale = np.zeros(shape=(DIMENSIONS, 30), dtype=int)

        self.his_scale_each_dim = [[] for i in range(DIMENSIONS,)]

    def update_scale(self, best_cost):
        if self.cost_10_pre_gen is None and self.curr_cost is None:
            self.curr_cost = best_cost
            self.cost_10_pre_gen = best_cost
        else:
            self.curr_cost = best_cost
            self.count_gen += 1
        if self.count_gen > 10 and self.cost_10_pre_gen > 0:
            delta = (self.cost_10_pre_gen - self.curr_cost) / \
                self.cost_10_pre_gen
            if delta < 0.1:
                self.is_jam = True
            else:
                self.is_jam = False

            self.cost_10_pre_gen = best_cost
            self.count_gen = 0

    def mutation(self, ind, subpopulation=None, skf=0, need_diversity=False):
        D = DIMENSIONS

        p_for_dim = 1 / (self.count_mu_each_dim + 1)
        p_for_dim /= np.sum(p_for_dim)
        i = int(np.random.choice(np.arange(D), size=1, p=p_for_dim))

        # mean = np.mean(subpopulation[:, i])
        std = np.std(subpopulation[:, i])
        e = -1

        if std != 0:
            log = np.log10(std)
            log = int(np.abs(log))+1
            if log+1 > self.max_log[i]:
                self.max_log[i] = log + 1
                # NOTE
                self.count_mu_each_dim = np.zeros_like(self.count_mu_each_dim) + 1
                self.count_mu_each_scale[i] = np.zeros_like(
                    self.count_mu_each_scale[i]) + 1

        rand = np.random.rand()
        if self.is_jam is False:
            return ind, std

        rand_scale = 1 / (self.count_mu_each_scale[i][1:self.max_log[i]] + 1)
        p = np.array(rand_scale / np.sum(rand_scale))

        if std == 0:
            e = int(np.random.choice(
                np.arange(self.max_log[i])[1:], size=1, p=p))
            e = -e
            self.scale = 10 ** e

        if std != 0:
            e = np.random.choice(np.arange(self.max_log[i])[1:], size=1, p=p)
            e = int(e)
            e = -1 * e
            self.scale = 10 ** e

        # if rand <= 0.1 and self.is_jam is False:
        if False:
            pa = np.random.choice(
                np.arange(len(subpopulation)), size=(2,), replace=False)
            scale = np.abs(subpopulation[pa[0]][i] - subpopulation[pa[1]][i])
            t = ind[i] + np.random.normal(loc=0, scale=scale)
        else:
            self.count_mu_each_dim[i] += 1
            self.count_mu_each_scale[i][np.abs(e)] += 1
            self.his_scale_each_dim[i].append(np.abs(e))
            t = ind[i] + np.random.normal(loc=0, scale=self.scale)
        if t > 1:
            t = ind[i] + np.random.rand() * (1 - ind[i])
        elif t < 0:
            t = np.random.rand() * np.abs(ind[i])

        ind[i] = t

        return ind, std
