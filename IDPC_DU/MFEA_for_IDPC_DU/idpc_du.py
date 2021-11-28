import numpy as np 

class graph:
    def __init__(self, path):
        self.path = path 
        self.number_nodes = -1 
        self.number_domains = -1 
        self.soure_node = -1 
        self.destination_node = -1 
        self.E = self.read_data() 

    def read_data(self): 
        data = []
        with open(self.path) as f: 
            data = f.readlines()
     
        count = 0 
        self.number_nodes, self.number_domains = list(map(int, data[count].strip("\n").split()))
        count +=1 
        self.soure_node , self.destination_node = list(map(int, data[count].strip("\n").split()))
        count += 1 
        self.soure_node -= 1 
        self.destination_node -= 1 

        E = np.zeros(shape = (self.number_domains, self.number_nodes, self.number_nodes), dtype= int) 
        for line in data[2:]: 
            u, v, w, d = list(map(int, line.strip("\n").split()))
            E[d-1][u-1][v-1] = w 
        return E

    def decode(self, individual):
        """
        individual: một cá thể ở không gian chung 
        """
        node_priority = np.zeros((self.number_nodes), dtype = int) 
        edge_index = np.copy(individual.edge_index[:self.number_nodes])
        index = 0 
        for i in range(self.number_nodes):
            while(individual.node_priority[index] >= self.number_nodes):
                index += 1
            node_priority[i] = individual.node_priority[index] 
            index += 1 
        return node_priority, edge_index 
    
    def take_edge_connected_to(self, node_i, H, visited): 
        """
        H: Domains was visited. Type 1D- array
        visited: array of nodes. value = 0 : node wasn't visited value = 1: node was visited. type 1D -- array
        """
        Adj = [] 
        node_j = [] 
        for node in range(self.number_nodes):
            if visited[node] == 0:
                for domain in range(self.number_domains):
                    if domain not in H and self.E[domain][node_i][node] > 0: 

                        Adj.append((node_i, node,self.E[domain][node_i][node] , domain))
                        if node not in node_j: 
                            node_j.append(node) 
        
        return Adj, node_j

    def take_next_node(self,list_next_node, node_priority, curr_node):
        """
        next_node: list of node can visit
        node_priority: 
        """
        next_node = curr_node
        priority = node_priority[curr_node] -1
        while( next_node not in list_next_node):
            a = np.where(node_priority == priority)[0]
            if len(a) <=0:
                break 
            if len(a) >= 1: 
                next_node = a[0] 
            

            priority -= 1 
            if priority < 0: 
                break
        return next_node 

    def set_edges_connect(self,Adj, node_begin, node_end, H):
        E = [] 
        for domain in range(self.number_domains):
            if domain not in H: 
                if((node_begin, node_end, self.E[domain][node_begin][node_end], domain)) in Adj: 
                    E.append((node_begin, node_end, self.E[domain][node_begin][node_end], domain))
        
        return E 

    def grow_path_alogrithms(self, individual):
        """
        s: source node
        t: destination node
        """
        node_priority, edge_index = self.decode(individual) 
        H = [] 
        d = -1 
        cost = np.zeros((self.number_nodes), dtype = int) 
        cost[self.destination_node] = 1000000 
        curr_node = self.soure_node
        visited = np.zeros((self.number_nodes), dtype = int) 
        path = []

 
        while curr_node != self.destination_node: 
            visited[curr_node] = 1 
            Adj, list_next_node = self.take_edge_connected_to(curr_node, H, visited)
            if len(Adj) == 0: 
                break 
            next_node = self.take_next_node(list_next_node, node_priority, curr_node)
            E = self.set_edges_connect(Adj, curr_node, next_node, H) 
            if len(E) == 0: 
                break 
            k = edge_index[next_node] % len(E) 
            path.append(E[k])
            _, _, weight, domain = E[k] 

            if d != -1 and d != domain: 
                H.append(d) 
            
            d = domain 
            cost[next_node] = cost[curr_node] + weight  
            curr_node = next_node 
        return path, cost[self.destination_node]
    

class individual: 
    def __init__(self, number_nodes, S, init = 1):
        if init:
            self.node_priority = np.random.permutation(number_nodes)
            self.edge_index = np.zeros((number_nodes), dtype = int) 
            for i in range(number_nodes): 
                if S[i] == 0 : 
                    self.edge_index[i] = None 
                else: 
                    self.edge_index[i] = int(np.random.randint(low = 0, high = S[i] - 1, size = 1))
        else: 
            self.node_priority = np.zeros((number_nodes), dtype = int)
            self.edge_index = np.zeros((number_nodes), dtype = int)