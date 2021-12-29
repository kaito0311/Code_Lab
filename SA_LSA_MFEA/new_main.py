# %%
from numpy.core.defchararray import upper
from utils_mf import * 
from lsa_mfea import * 
from mfea import * 


# %%
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

# %%
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

# %%
def load_file(path_file):
    input = np.loadtxt(path_file)
    return input

# %%
def getManyTask50():
    tasks = [] 
    tasks_size = 2; 
    for index in range(1): 
        dim = 50; 
        choice_functions = switch_func(index) 
        for task_id in range(tasks_size):
            task_id += 1 

            function_id = choice_functions[(task_id -1) % len(choice_functions)]
            file_dir = "./GECCO/Tasks/benchmark_" + str(index+1); 
            file_matrix = file_dir + "/matrix_" + str(task_id) 
            file_bias = file_dir + "/bias_" + str(task_id)
            matrix = load_file(file_matrix)
            bias = load_file(file_bias)
            assert matrix.shape == (50,50)
            assert bias.shape == (50,)
            tasks.append(switch_task(function_id, matrix, bias, dim))
    
    return tasks 


# %%
def visual_result(history, name, tasks):
    _, bieudo = plt.subplots(1, NUMBER_TASKS)
    color = ['r', 'b', 'y']

    for task in range(NUMBER_TASKS):
        for version in range(len(history)):
            x_axis = np.arange(len(history[version]))
            y_axis =  [x[task].cost for x in history[version]]

            bieudo[task].plot(x_axis,y_axis, color[version], label = name[version])
            ymin, y_max = min(y_axis), max(y_axis)
            plt.yscale('log')
            # plt.ylim(ymin, 1)
            bieudo[task].set (title= tasks[task].name)
    plt.legend(loc = 'best')
    plt.show()


# %%
def main() :
    
    tasks = getManyTask50()    
    history= []
    print("lsa_mfea")
    hi, history_rmp = lsa_mfea(tasks)
    # for number_rmp in range(len(history_rmp[0])):
    history.append(hi)
    print("mfea")
    history.append(mfea2(tasks))
    

    # print(len(history_rmp[0]))

    # import pygame 

    # pygame.mixer.init()
    # pygame.mixer.music.load("D:/LinhTinh/warning_audio.mp3")
    # pygame.mixer.music.play()

    # import time 
    # time.sleep(1) 
    visual_result(history, ["lsa_mfea", "mfea"], tasks)


# %%
main()


