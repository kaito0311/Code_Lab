from utils.benchmark_function import *
from SaMTPSO.lsa_pso_de_sbx import * 

tasks = getManyTasks10()

history_cost, history_p_matrix = [], [] 
for i in range(1):

    history, p_matrix = lsa_SaMTPSO_DE_SBX(tasks, lsa= True)
    history_cost.append(history)
    history_p_matrix.append(p_matrix) 