import numpy as np

from config import DIMENSIONS 


def add_coefficient_gauss(population): 
    # Thêm vào cuối mỗi phần tử là một số random từ 0 -> 0.1 
    gauss_add_element= np.abs(np.random.normal(loc= 0, scale=0.1, size= (len(population),1)))
    population = np.append(population, gauss_add_element, axis= 1) 
    return population 


def gauss_mutation_self_adap(pa, rate, scale = 0.013):
    #TODO: CONVERT SANG 0.0002 -> 0.2
    if pa[len(pa) - 1] > 0.2: 
        pa[len(pa) - 1] = 0.2 
    elif pa[len(pa) - 1] <= 2e-5: 
        pa[len(pa) - 1] = 2e-5
    converted_scale= pa[len(pa) -1] * (0.2 - 2e-5) / 0.1  + 2e-5
    converted_scale = np.random.normal(loc= 0, scale= scale)
    converted_scale = np.abs(converted_scale)
    ind = np.copy(pa)
    D = DIMENSIONS
    ind[D] = converted_scale
    i = int(np.random.choice(np.arange(D), size= 1)) 
    t = ind[i] +  np.random.normal(scale= converted_scale) 

    if t >1: 
        t = ind[i] + np.random.rand() * (1- ind[i])
    elif t < 0: 
        t = np.random.rand() * ind[i]  
    
    ind[i] = t
    
    return ind  

def gauss_base_population(subpopulation, ind, scale = 1): 
    """
    subpopulation: subpopulation correspond with task of ind. 
    ind: ind need mutation
    """
    D = len(ind)
    if D > DIMENSIONS:
        D = DIMENSIONS  

    p = 1 / D 
    i = int(np.random.choice(np.arange(D), size= 1)) 
    mean = np.mean(subpopulation[:, i]) 
    std = np.std(subpopulation[:, i]) 
    t = np.random.normal(loc = ind[i], scale= std * scale) 
    if t >1: 
        t = ind[i] + np.random.rand() * (1- ind[i])
    elif t < 0: 
        t = np.random.rand() * np.abs(ind[i] - mean)  + ind[i]  
    
    ind[i] = t 
    
    return ind 
            


    
     





