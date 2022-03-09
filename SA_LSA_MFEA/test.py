from utils.benchmark_function import *
from SaMTPSO.lsa_new_gauss import *

tasks = getManyTasks10()

history_cost, history_p_matrix = [], [] 
for i in range(1):
 
    history, p_matrix = lsa_SaMTPSO_DE_SBX_new_gauss(tasks, lsa= True, seed = 2, ti_le_giu_lai= 0.9, min_popu= 20)
    history_cost.append(history)
    history_p_matrix.append(p_matrix) 