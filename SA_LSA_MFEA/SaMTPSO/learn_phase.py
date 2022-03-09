from matplotlib import scale
import numpy as np
import scipy.stats
from config import DIMENSIONS

class DE: 
    def __init__(self) -> None:
        # DE
        self.Mcr = np.zeros(shape= (30), dtype= float) + 0.5
        self.Mf = np.zeros(shape= (30), dtype= float) + 0.5
        self.index_update = 0 
        # cr and r temp 
        self.Scr = [] 
        self.Sf = [] 
        self.w = [] 

        self.rate_improve = 0 
        self.func_eval = 0 

        self.name = "DE"

        # luu tam thoi 
        self.cr_tmp = 0 
        self.f_tmp = 0 

    def DE_cross(self,p, pbest, pr1, pr2)->np.array: 
        '''
        pbest: chon random tu 10% on top 
        pr1, pr2: chon random tu quan the voi task tuong ung thoi 
        '''
        D = len(pbest) - 1
        jrand = np.random.choice(np.arange(D), size = 1) 
        y = np.zeros_like(pbest) 
        k = np.random.choice(np.arange(len(self.Mcr)), size= 1)
        cr = np.random.normal(loc= self.Mcr[k], scale = 0.1) 
        if cr > 1:
            cr = 1 
        if cr <=0: 
            cr = 0
        # cr = scipy.stats.cauchy.rvs(loc= self.Mcr[k], scale= 0.1)
        F = 0
        while F <= 0: 
            F = scipy.stats.cauchy.rvs(loc= self.Mf[k], scale= 0.1) 
        if F > 1: 
            F = 1 
        for i in range(D):
            if np.random.uniform() < cr or i == jrand: 
                y[i] = pbest[i] + F * (pr1[i] - pr2[i])
            else: 
                y[i] =  p[i] 
        self.cr_tmp = cr 
        self.f_tmp = F 
        y = np.clip(y, 0,1)
        return y 
    def performance(self)->float:
        if self.func_eval == 0:
            return 1 
        else: 
            return self.rate_improve / self.func_eval 
    def update(self, delta_fcost):
        if delta_fcost > 0: 
            self.rate_improve += delta_fcost 

            self.Scr.append(float(self.cr_tmp))
            self.Sf.append(float(self.f_tmp))
            self.w.append(float(delta_fcost))
        self.func_eval += 1 
    
    def reset(self): 
        sum_w = np.sum(self.w) 
        new_cr = 0 
        new_f = 0 
        new_index = (self.index_update +1) % len(self.Mcr) 
        if len(self.Scr) > 0:
            # for i in range(len(self.Scr)): 
            #     new_cr += self.Scr[i] * self.w[i] / sum_w 
            #     new_f += self.Sf[i] * self.w[i] / sum_w 
            new_cr = np.sum(np.array(self.Scr) * (np.array(self.w) / sum_w) )
            new_f = (np.sum(np.array(self.w) * np.array(self.Sf) ** 2)) / (np.sum(np.array(self.w )*np.array(self.Sf)))
            self.Mcr[new_index] = new_cr 
            self.Mf[new_index] = new_f 
            
        else: 
            self.Mcr[new_index] = np.copy(self.Mcr[self.index_update])
            self.Mf[new_index] = np.copy(self.Mf[self.index_update])
        
        self.index_update = new_index 
        self.Scr.clear() 
        self.Sf.clear() 
        self.w.clear() 
        self.func_eval = 0 
        self.rate_improve = 0 
    
    


class Gauss:
    def __init__(self) -> None:
        # Gauss 
        self.scale = np.zeros(shape=(DIMENSIONS,)) + 0.1
        self.mean= 0 

        self.rate_improve = 0 
        self.func_eval = 0 

        self.name = "gauss"
    def gauss_mutation(self, ind) ->np.ndarray: 
        D = ind.shape[0] 
        for i in range(D):
            a = 1.0 
            if(np.random.uniform() < a / D ):
                t = -1 
                # while t > 1 or t < 0: 
                t = ind[i] + np.random.normal(size= 1, loc= self.mean, scale= np.abs(self.scale[i])) 
                if t > 1: 
                    t = ind[i] + np.random.uniform() * (1 - ind[i]) 
                elif t < 0: 
                    t = np.random.uniform() * ind[i] 
                ind[i] = t 
        
        return ind 
    
    def performance(self)->float:
        if self.func_eval == 0:
            return 1 
        else: 
            return self.rate_improve / self.func_eval 
    
    def update(self, delta_fcost):
        if delta_fcost > 0: 
            self.rate_improve += delta_fcost 
        self.func_eval += 1 
        

class learn_phase_task:
    def __init__(self) -> None:
        self.operator1 = Gauss() 
    
        self.operator2 = DE() 
    def update_operator(self, swap = False) : 
        if self.operator1.performance() < self.operator2.performance() or swap:
            self.operator1, self.operator2 = self.operator2, self.operator1 
    
    

    
    
        