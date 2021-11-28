from scipy.stats import norm
import numpy as np 

a = np.array([1,1,2,2,3,4,1,2,1])
p = norm.pdf(a) 
print(p) 
print(a.shape)