import numpy as np 
from config import * 
class function: 
    def __init__(self,  matrix, bias,dimension = 50, lower = -50, upper = 50):
        self.dimension= dimension 
        self.matrix = matrix
        self.bias = bias 
        self.lower = lower 
        self.upper = upper 
        self.name = self.__class__.__name__ + " _dimension: " + str(dimension)
    

    
    def decode(self, array_value):
        array_value = array_value[: self.dimension]
        x = array_value
        x = x * (self.upper - self.lower) + self.lower
        # x = (np.dot(np.linalg.inv(self.matrix), (x - self.bias).reshape((self.dimension, 1)))); 
        x = (np.dot((self.matrix), (x - self.bias).reshape((self.dimension, 1)))); 
        return x

class Rosenbrock(function):
    def calculate_fitness(self, array_value):
        x = self.decode(array_value) 
        sum = 0

        l = 100*np.sum((np.delete(x, 0, 0) - np.delete(x, -1, 0 )**2) ** 2)
        r = np.sum((np.delete(x, -1, 0) - 1) ** 2)
        sum = l + r
        return sum 

class Ackley(function):
    def calculate_fitness(self, array_value): 
        x = self.decode(array_value)
        sum1, sum2 = 0,0
        # for i in range(self.dimension):
        #     sum1 += x[i] ** 2
        #     sum2 += np.cos(2 * np.pi * x[i]) 
        sum1 = np.sum(x**2) 
        sum2 = np.sum(np.cos(2 * np.pi * x))
        
        avg = sum1/self.dimension 
        avg2 = sum2/self.dimension 
    
        return -20 * np.exp(-0.2*np.sqrt(avg)) - np.exp(avg2) + 20 + np.exp(1)

class Girewank(function):
    def calculate_fitness(self, array_value):
        x = self.decode(array_value) 
        sum1 = 0
        sum2 = 1 
        sum1 = np.sum(x ** 2)

        # for i in range(self.dimension):
        #     sum2 *= np.cos(x[i]/np.sqrt(i+1))
        sum2 = np.prod(np.cos(x/np.sqrt(np.arange(self.dimension) + 1)))
        
        return 1 + 1/4000 * sum1 - sum2


class Rastrigin(function): 
    def calculate_fitness(self, array_value): 
        x = self.decode(array_value)
        sum = 10 * self.dimension

        # for i in range(self.dimension): 
        #     sum += (x[i] ** 2 - 10 * np.cos(2 * np.pi * x[i])) 
        sum += np.sum(x**2)
        sum += np.sum(-10 * np.cos(2 * np.pi * x))
        
        return sum; 


class Schewefel(function): 
    def calculate_fitness(self, array_value): 
        x = self.decode(array_value) 
        sum = 0 
        # for i in range(self.dimension): 
        #     sum += x[i] * np.sin(np.sqrt(np.abs(x[i])))
        sum = np.sum(x * np.sin(np.sqrt(np.abs(x))))
        return 418.9829 * self.dimension - sum 
        

class Sphere(function):
    def calculate_fitness(self, array_value): 
        x = self.decode(array_value) 
        sum = 0; 
        x = x**2
        sum = np.sum(x)
        sum = float(sum) 
        return sum; 

class Weierstrass(function):
    kmax = 21
    a = 0.5 
    b = 3 
    vt1 = np.power(np.zeros(kmax) + a, np.arange(kmax))
    vt2 = np.power(np.zeros(kmax) + b, np.arange(kmax))
    
    def calculate_fitness(self, array_value): 
        x = self.decode(array_value) 
        sum = 0 

       
        # for i in range(self.dimension): 
        #     for k in range(kmax): 
        #         sum += np.power(a, k) *  np.cos(2 * np.pi * np.power(b, k) * (x[i] + 0.5))


        # for i in range(self.dimension):
        #     for k in range(kmax): 
        #         sum += vt1[k] * np.cos(2 * np.pi * vt2[k] * (x[i] + 0.5))
        
        # for k in range(kmax):
        #     sum -= self.dimension * vt1[k] * np.cos(2 * np.pi * vt2[k] * 0.5) 
        # return sum
        left = 0
        for i in range(self.dimension):
            left += np.sum(self.a ** np.arange(self.kmax) * \
                np.cos(2*np.pi * self.b ** np.arange(self.kmax) * (x[i]  + 0.5)))
            
        right = self.dimension * np.sum(self.a ** np.arange(self.kmax) * \
            np.cos(2 * np.pi * self.b ** np.arange(self.kmax) * 0.5)
        )
        return left - right




def switch_func(i):
    switcher = {
        0: [1],
        1: [2],
        2: [4],
        3: [1, 2, 3],
        4: [4,5,6],
        5: [2,5,7],
        6: [3,4,6],
        7: [2,3,4,5,6],
        8: [2,3,4,5,6,7],
        9: [3,4,5,6,7]

    }
    assert i <10 and i >= 0
    return switcher.get(i)


def switch_task(i, matrix, bias, dim):
    switcher = {
        1: Sphere(matrix, bias, dimension= dim, lower= -100, upper = 100),
        2: Rosenbrock(matrix, bias, dimension= dim, lower= -50, upper= 50),
        3: Ackley(matrix, bias, dimension= dim, lower= -50, upper= 50),
        4: Rastrigin(matrix, bias, dimension= dim, lower = -50, upper = 50),
        5: Girewank(matrix, bias, dimension= dim, lower = -100, upper= 100),
        6: Weierstrass(matrix, bias, dimension= dim, lower = -0.5, upper= 0.5),
        7: Schewefel(matrix, bias, dimension= dim, lower = -500, upper= 500)
    }
    # print(i)
    # print(Sphere(matrix, bias, dimension= dim, lower= -100, upper = 100).name)

    assert i >= 1 and i <= 7
    return switcher.get(i)

def load_file(path_file):
    input = np.loadtxt(path_file)
    return input

def switch_task_CEC(i, matrix, bias, dim, lower, upper):
    switcher = {
        1: Sphere(matrix, bias, dim, lower,upper),
        2: Sphere(matrix, bias, dim, lower,upper),
        3: Sphere(matrix, bias, dim, lower,upper),
        4: Weierstrass(matrix, bias, dim, lower,upper),
        5: Rosenbrock(matrix, bias, dim, lower,upper),
        6: Ackley(matrix, bias, dim, lower,upper),
        7: Weierstrass(matrix, bias, dim, lower,upper),
        8: Schewefel(matrix, bias, dim, lower,upper),
        9: Girewank(matrix, bias, dim, lower,upper),
        10: Rastrigin(matrix, bias, dim, lower,upper),
    }
    assert i >= 1 and i <= 10 
    return switcher.get(i) 

def getManyTask50():
    tasks = [] 
    number = 1
    tasks_size = int(NUMBER_TASKS/number); 
    for index in range(number): 
        dim = 50
        # index += 5
        choice_functions = switch_func(index) 
        for task_id in range(tasks_size):
            task_id += 1 

            function_id = choice_functions[(task_id -1) % len(choice_functions)]
            file_dir = "../GECCO/Tasks/benchmark_" + str(index+1); 
            file_matrix = file_dir + "/matrix_" + str(task_id) 
            file_bias = file_dir + "/bias_" + str(task_id)
            matrix = load_file(file_matrix)
            bias = load_file(file_bias)
            assert matrix.shape == (50,50)
            assert bias.shape == (50,)
            tasks.append(switch_task(function_id, matrix, bias, dim))
    
    return tasks 

def getManyTasks10():
    dim = 50 
    tasks = [] 
    # task 1 
    matrix = np.eye(dim) 
    bias = np.zeros(dim) 
    tasks.append(Sphere(matrix, bias, dim, -100, 100)) 

    #task 2 
    matrix = np.eye(dim) 
    bias = np.zeros(dim) + 80; 
    tasks.append(Sphere(matrix, bias, dim, -100, 100)) 

    # tasks 3 
    matrix = np.eye(dim) 
    bias = np.zeros(dim) - 80; 
    tasks.append(Sphere(matrix, bias, dim, -100, 100)) 

    # task 4 
    dim = 25
    bias = np.zeros(dim) - 0.4 
    matrix = np.eye(dim) 
    tasks.append(Weierstrass(matrix, bias,dim, -0.5, 0.5))

    # task 5
    dim = 50
    matrix = np.eye(dim) 
    bias = np.zeros(dim) -1
    tasks.append(Rosenbrock(matrix, bias, dim, lower= -50, upper = 50))

    # task 6 
    dim = 50 
    matrix = np.eye(dim) 
    bias = np.zeros(dim) + 40
    tasks.append(Ackley(matrix, bias, dim, -50, 50))

    # task 7 
    dim = 50 
    bias = np.zeros(dim) - 0.4 
    matrix = np.eye(dim) 
    tasks.append(Weierstrass(matrix, bias,dim, -0.5, 0.5))

    # task 8 
    dim = 50 
    bias = np.zeros(dim) 
    matrix = np.eye(dim) 
    tasks.append(Schewefel(matrix, bias, dim, -500, 500))

    # tasks 9 
    dim = 50
    bias = np.zeros(dim) 
    bias[:int(dim/2)] += -80 
    bias[int(dim/2):] += 80 
    tasks.append(Girewank(matrix, bias, dim, lower=-100, upper=100))

    # task 10 
    dim = 50 
    bias = np.zeros(dim) 
    bias[:int(dim/2)] += 40 
    bias[int(dim/2):] += -40 
    tasks.append(Rastrigin(matrix, bias, dim, lower = -50, upper= 50))
    return tasks
    