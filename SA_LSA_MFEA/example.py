import numpy as np 

number_child = np.array([2,3])

top_parent =   np.array([0,1,2,3,4,5,6,7])

# skill_factor = np.array([0,1,1,1,2,2,2,0])
# number_subpopulation = np.array([4,4,4])

# index_parent_not_enough_child = np.where(number_child[skill_factor[top_parent]] < number_subpopulation[skill_factor[top_parent]])[0] 


# parent = np.random.choice(top_parent[index_parent_not_enough_child], size= (2))
# print(parent)
# number_child[skill_factor[parent]] += 1 
# print(index_parent_not_enough_child)
# print(number_child)

# print(np.random.choice([1.1,  1.2,1.3, 1.4], size= (2))) 

# a = np.array([1,2,3,4])
# print(np.prod(a))

# import chime 
# chime.success() 

# import pygame 

# pygame.mixer.init()
# pygame.mixer.music.load("D:/LinhTinh/warning_audio.mp3")
# pygame.mixer.music.play()

# import time 
# time.sleep(6) 

a = np.array([1, 0.2, 0.6, 0.7])

b = np.where(a > 0.5, a+ 1, a -1)
print(a)
print(b)

import random
random.seed(1)
def poly_mutation(parent, nm=5):
    child = np.copy(parent) 
    u_ = [0.6, 0.8, 0.2, 0.2]
    rand = [ 0.89294695, 0.89629309, 0.12558531, 0.20724288 ]

    for i in range(len(parent)):
        if(rand[i] < 1.0/len(parent)):
            # u = random.random()
            u = u_[i]
            
            r = 1 - np.power(2.0 * (1.0 - u), 1.0 / (nm + 1.0))
            l = np.power(2.0 * u, 1.0 / (nm + 1.0)) - 1
            # p = 1.0 / float(len(parent))
            if u <= 0.5:
                child[i] = parent[i] + l * parent[i]
            else:
                child[i] = parent[i] + r * (1 - parent[i])
        if(child[i] > 1):
             child[i] = parent[i] + random.random()* (1 - parent[i]) 
        elif child[i] < 0: 
            child[i] = parent[i]* random.random() 
    return child
def poly_mutation_v2(parent, nm=5):
    child = np.copy(parent)
    u = np.random.rand(len(parent))
    r = 1 - np.power(2.0 * (1.0 - u), 1.0 / (nm + 1.0))
    l = np.power(2.0 * u, 1.0 / (nm + 1.0)) - 1
    rand = np.random.rand(len(parent))
    a = np.where(u <= 0.5,parent *(1+l), parent + r * (1-parent))
    child = np.where(rand < 1.0/len(parent), a, parent)

    return child

np.random.seed(3) 
print(a)

a = np.array([2,2,2,2])
k = np.array([1,1,1,2])

print(np.power(a,k))
print(np.zeros(3))