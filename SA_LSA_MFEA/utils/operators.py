
import numpy as np 


class newSBX():
    '''
    pa, pb in [0, 1]^n
    '''
    def __init__(self, nb_tasks: int, nc = 15, gamma = .9, alpha = 1, *args, **kwargs):
        self.nc = nc
        self.nb_tasks = nb_tasks
        self.gamma = gamma
        self.alpha = alpha

    def get_dim_uss(self, dim_uss):
        self.dim_uss = dim_uss
        self.prob = np.ones((self.nb_tasks, self.nb_tasks, dim_uss))/2
        for i in range(self.nb_tasks):
            self.prob[i, i, :] = 1
        
        #nb all offspring bored by crossover at dimensions d by task x task
        self.count_crossover_each_dimensions = np.zeros((self.nb_tasks, self.nb_tasks, dim_uss))
        #index off offspring
        self.epoch_idx_crossover = []

        #nb inds alive after epoch
        self.success_crossover_each_dimension = np.zeros((self.nb_tasks, self.nb_tasks, dim_uss))
      
        self.skf_parent = np.empty((0, 2), dtype= int)

    def update(self, idx_success):

        # sum success crossover
        for idx in idx_success:
            self.success_crossover_each_dimension[self.skf_parent[idx][0], self.skf_parent[idx][1]] += self.epoch_idx_crossover[idx]

        # percent success: per_success = success / count
        per_success = (self.success_crossover_each_dimension / (self.count_crossover_each_dimensions + 1e-10))** (1/self.alpha)

        # new prob
        new_prob = np.copy(per_success)
        # prob_succes greater than intra -> p = 1 # Nếu task hiện tại thích task khác hơn 

        #===============
            
    

        # ===============
        # tmp_smaller_intra_change = np.empty_like(self.count_crossover_each_dimensions)
        # for i in range(self.nb_tasks):
        #     tmp_smaller_intra_change[i] = (new_prob[i] <= new_prob[i, i])
        # new_prob = np.where(
        #     tmp_smaller_intra_change, 
        #     new_prob, 
        #     1
        # )
        # for i in range(self.nb_tasks): 
        #     for j in range(self.dim_uss) :
        #         a = np.max(new_prob[i,:, j]) 
        #         if a != 0: 
        #         # a = np.sum(new_prob[i, :, j]) 
        #         # new_prob[i, :, j] = (new_prob[i, :, j]) / a 
        #             new_prob[i, :, j] = new_prob[i, :, j] / a
        new_prob = np.where(
            self.count_crossover_each_dimensions != 0, 
            new_prob,
            self.prob
        )

        # update prob 
        self.prob = self.prob * self.gamma + (1 - self.gamma) * new_prob
        self.prob = np.clip(self.prob, 1/self.dim_uss, 1)

        # reset
        self.count_crossover_each_dimensions = np.zeros((self.nb_tasks, self.nb_tasks, self.dim_uss))
        self.success_crossover_each_dimension = np.zeros((self.nb_tasks, self.nb_tasks, self.dim_uss))
        self.epoch_idx_crossover = []
        self.skf_parent = np.empty((0, 2), dtype= int)

    def __call__(self, pa, pb, skf: tuple[int, int], *args, **kwargs):
        '''
        skf = (skf_pa, skf_pb)
        '''

        self.skf_parent = np.append(self.skf_parent, [[skf[0], skf[1]]], axis = 0)
        self.skf_parent = np.append(self.skf_parent, [[skf[0], skf[1]]], axis = 0)

        u = np.random.rand(self.dim_uss)
        
        # ~1
        beta = np.where(u < 0.5, (2*u)**(1/(self.nc +1)), (2 * (1 - u))**(-1 / (1 + self.nc)))

        if skf[0] == skf[1]:
            idx_crossover = np.ones_like(pa)
            self.count_crossover_each_dimensions[skf[0], skf[1]] += 2 * idx_crossover
            self.epoch_idx_crossover.append(idx_crossover)
            self.epoch_idx_crossover.append(idx_crossover)
            #like pa
            c1 = 0.5*((1 + beta) * pa + (1 - beta) * pb)
            #like pb
            c2 = 0.5*((1 - beta) * pa + (1 + beta) * pb)

            #swap
            idx_swap = np.where(np.random.rand(len(pa)) < 0.5)[0]
            c1[idx_swap], c2[idx_swap] = c2[idx_swap], c1[idx_swap]

        else:
            idx_crossover = (np.random.rand(self.dim_uss) < self.prob[skf[0], skf[1]])
            if np.sum(idx_crossover) == 0:
                idx_crossover[np.random.randint(0, self.dim_uss)] = True
            self.count_crossover_each_dimensions[skf[0], skf[1]] += 2 * idx_crossover
            self.epoch_idx_crossover.append(idx_crossover)
            self.epoch_idx_crossover.append(idx_crossover)
            #like pa
            c1 = np.where(idx_crossover, 0.5*((1 + beta) * pa + (1 - beta) * pb), pa)
            #like pb
            c2 = np.where(idx_crossover, 0.5*((1 - beta) * pa + (1 + beta) * pb), pa)

            #swap
            idx_swap = np.where((np.random.rand(len(pa)) < 0.5) * (self.prob[skf[0], skf[1]] > 0.6))
            c1[idx_swap], c2[idx_swap] = c2[idx_swap], c1[idx_swap]

        c1, c2 = np.clip(c1, 0, 1), np.clip(c2, 0, 1)
        
        return c1, c2

def gauss_self_adapt(p):
    scale= np.zeros_like(p) + np.max(p) * 0.1 
    ind = np.copy(p)
    pm = 1/len(ind) 
    idx_mutation= np.where(np.random.rand(len(ind)) < pm)[0] 

    t = ind[idx_mutation] + np.random.normal(0, scale[idx_mutation], size = len(idx_mutation))
    
    t = np.where(t > 1, ind[idx_mutation] + np.random.rand() * (1 - ind[idx_mutation]), t)
    t = np.where(t < 0, np.random.rand() * ind[idx_mutation], t)

    ind[idx_mutation] = t

    return ind
    

def sbx_crossover(p1, p2, nc = 2, swap = True):
    SBXDI = nc
    D = p1.shape[0]
    cf = np.empty([D])
    u = np.random.rand(D)        

    cf[u <= 0.5] = np.power((2 * u[u <= 0.5]), (1 / (SBXDI + 1)))
    cf[u > 0.5] = np.power((2 * (1 - u[u > 0.5])), (-1 / (SBXDI + 1)))

    c1 = 0.5 * ((1 + cf) * p1 + (1 - cf) * p2)
    c2 = 0.5 * ((1 + cf) * p2 + (1 - cf) * p1)
    c1 = np.clip(c1, 0, 1)
    c2 = np.clip(c2, 0, 1)
    if swap is True: 
        c1, c2 = variable_swap(c1,c2, 0.5)

    return c1, c2

def Gauss_mutation(ind, mean= 0, sigma = 0.001):
    D = ind.shape[0] 
    sigma = min(ind) 
    rand = np.random.normal(size= (D), loc= mean, scale= sigma) 
    for i in range(D):
        if(np.random.uniform() < 1.0 / D):
            t = ind[i] + rand[i] 
            if(t > 1):
                t = ind[i] + np.random.uniform() * (1-ind[i]) 
            elif t < 0:
                t = np.random.uniform() * ind[i] 
            ind[i] = t
    
    return ind

def variable_swap(p1, p2, probswap):
    D = p1.shape[0]
    swap_indicator = np.random.rand(D) <= probswap
    c1, c2 = p1.copy(), p2.copy()
    c1[np.where(swap_indicator)] = p2[np.where(swap_indicator)]
    c2[np.where(swap_indicator)] = p1[np.where(swap_indicator)]
    return c1, c2

def poly_mutation(p, pmdi=5):

    # u = np.random.rand() 
    # nm = pmdi 
    # parent = p 
    # child = None 

    # r = 1 - np.power(2.0 * (1.0 - u), 1.0 / (nm + 1.0))
    # l = np.power(2.0 * u, 1.0 / (nm + 1.0)) - 1
    # # p = 1.0 / float(len(parent))
    # child = np.zeros_like(parent)
    # if u <= 0.5:
    #     child = parent + l * parent
    # else:
    #     child = parent + r * (1 - parent)
    
    # tmp = child 


    mp = float(1. / p.shape[0])
    u = np.random.uniform(size=[p.shape[0]])
    r = np.random.uniform(size=[p.shape[0]])
    tmp = np.copy(p)
    for i in range(p.shape[0]):
        v = 0
        if r[i] < mp:
            if u[i] < 0.5:
                delta = (2*u[i]) ** (1/(1+pmdi)) - 1
                v = p[i] + delta * p[i]
            else:
                delta = 1 - (2 * (1 - u[i])) ** (1/(1+pmdi))
                v = p[i] + delta * (1 - p[i])
        if v > 1: 
            print("hmmmmm")
            tmp[i] = tmp[i] + np.random.rand() * (1 - tmp[i]) 
        elif  v < 0: 
            print("ầy")
            tmp[i] = tmp[i] * np.random.rand(); 

    tmp = np.clip(tmp, 0, 1)

    return tmp 



def gauss_mutation_kien(p, pa, pb, scale, rate= 0.5):
    ind = np.copy(p)
    pm = 1/len(ind) 
    idx_mutation= np.where(np.random.rand(len(ind)) < pm)[0] 

    t = ind[idx_mutation] + np.random.normal(0, scale[idx_mutation], size = len(idx_mutation))
    
    t = np.where(t > 1, ind[idx_mutation] + np.random.rand() * (1 - ind[idx_mutation]), t)
    t = np.where(t < 0, np.random.rand() * ind[idx_mutation], t)

    ind[idx_mutation] = t

    return ind


# def sbx_crossover(parent1, parent2):
#     # rand = random.random()
#     # beta = compute_beta_for_sbx(rand)

#     # child1 = 0.5 * ((1.0 + beta) * parent1 + (1.0 - beta) * parent2)
#     # child2 = 0.5 * ((1.0 - beta) * parent1 + (1.0 + beta) * parent2)

#     rand = np.random.rand(len(parent1))
#     beta = np.zeros_like(rand) 
#     # for i in range(len(rand)):
#     #     beta[i] = compute_beta_for_sbx(rand[i])
#     nc = 2 
#     beta = np.where(rand <= 0.5, np.power(2.0 * rand, 1.0 / float(nc + 1)), np.power(1.0 / (2.0 * (1 - rand)), 1.0 / (nc + 1)))

#     child1 = 0.5 * ((1.0 + beta) * parent1 + (1.0 - beta) * parent2)
#     child2 = 0.5 * ((1.0 - beta) * parent1 + (1.0 + beta) * parent2)

#     child1, child2 = np.clip(child1, 0, 1), np.clip(child2, 0, 1)
#     return child1, child2


# def poly_mutation(parent, nm=5):
#     # u = random.random()

#     # r = 1 - np.power(2.0 * (1.0 - u), 1.0 / (nm + 1.0))
#     # l = np.power(2.0 * u, 1.0 / (nm + 1.0)) - 1
#     # # p = 1.0 / float(len(parent))
#     # child = np.zeros_like(parent)
#     # if u <= 0.5:
#     #     child = parent + l * parent
#     # else:
#     #     child = parent + r * (1 - parent)


#     child = np.copy(parent)
#     u = np.random.rand(len(parent))
#     r = 1 - np.power(2.0 * (1.0 - u), 1.0 / (nm + 1.0))
#     l = np.power(2.0 * u, 1.0 / (nm + 1.0)) - 1
#     rand = np.random.rand(len(parent))
#     a = np.where(u <= 0.5, parent *(1+l) , parent + r * (1-parent))
#     child = np.where(rand < 1.0/len(parent), a, parent)
    
#     return child