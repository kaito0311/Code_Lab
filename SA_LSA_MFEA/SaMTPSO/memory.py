import numpy as np 

class Memory_SaMTPSO: 
    def __init__(self, number_tasks, LP = 10) -> None:
        self.success_history = np.zeros(shape = (number_tasks, LP), dtype = int) 
        self.fail_history = np.zeros(shape = (number_tasks, LP), dtype= int) 
        self.next_position_update = 0  # index cột đang được update 
        self.isFocus = False 
        self.epsilon = 0.001
        self.pb = 0.001  
        self.LP = LP
        self.duytri = 10
    
    def update_history_memory(self,task_partner,success = True, end_generation= False)->None:
        if success : 
            self.success_history[task_partner][self.next_position_update] += 1
        else: 
            self.fail_history[task_partner][self.next_position_update] += 1 
        
        # Nếu hết một thế hệ -> tăng vị trí cập nhật success và fail memory lên một -> 
        # if end_generation: 
            # Nếu trong LP thế hệ gần nhất mà không cải thiện được thì -> tập trung intra 
            
    
    def compute_prob(self)->np.ndarray: 
        SRtk = np.sum(self.success_history, axis = 1) / (np.sum(self.success_history, axis = 1) + np.sum(self.fail_history, axis = 1) + self.epsilon) + self.pb
        p = SRtk/ np.sum(SRtk) 
        if np.sum(self.success_history).all() < 100 and self.duytri <= 0: 
            self.isFocus = not self.isFocus
            self.duytri = 5

        if np.sum(self.success_history).any() > 100 and self.isFocus is True and self.duytri <= 0: 
            self.isFocus = False
            self.duytri = 5
        self.duytri -=1 
        self.next_position_update = (self.next_position_update + 1) % self.LP 
        self.success_history[:, self.next_position_update] = 0
        self.fail_history[:, self.next_position_update] = 0 
        return p 