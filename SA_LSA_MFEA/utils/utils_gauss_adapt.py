import numpy as np 


def add_coefficient_gauss(population): 
    # Thêm vào cuối mỗi phần tử là một số random từ 0 -> 0.1 
    gauss_add_element= np.abs(np.random.normal(loc= 0, scale=0.1, size= (len(population),1)))
    population = np.append(population, gauss_add_element, axis= 1) 
    return population 


def gauss_mutation_self_adap(pa, rate, scale = 0.013):
    #TODO: CONVERT SANG 0.0002 -> 0.2
    if pa[len(pa) - 1] > 0.2: 
        pa[len(pa) - 1] = 0.2 
    elif pa[len(pa) - 1] <= 0.0002: 
        pa[len(pa) - 1] = 0.0002
    converted_scale= pa[len(pa) -1] * (0.2 - 0.0002) / 0.1  + 0.0002
    converted_scale = np.random.normal(loc= 0, scale= scale)
    converted_scale = np.abs(converted_scale)
    converted_scale = np.max(rate[:len(rate) - 1]) * converted_scale 
    ind = np.copy(pa)
    for i in range(len(pa) -1 ): 
        if np.random.uniform() < 1/(len(pa) - 1): 
            t = pa[i] + np.random.normal(loc= 0, scale= converted_scale) 
            if t > 1: 
                t = ind[i]  + np.random.rand() * (1- ind[i])
            elif t < 0: 
                t = np.random.rand() * ind[i]
            
            ind[i] = t 
    
    return ind  



    
     





