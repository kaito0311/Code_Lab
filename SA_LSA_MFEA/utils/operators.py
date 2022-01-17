from random import random
import numpy as np 

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