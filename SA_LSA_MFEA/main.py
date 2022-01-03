from numpy.core.defchararray import lower, upper
from utils.benchmark_function import *  
from lsa_mfea import * 
from mfea import * 
np.random.seed(0)


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


def main() :
    
    # tasks = getManyTask50()  
    tasks = getManyTasks10()   
    history= []
    print("lsa_mfea")
    hi, history_rmp = lsa_mfea(tasks)
    # for number_rmp in range(len(history_rmp[0])):
    history.append(hi)
    print("mfea")
    history.append(mfea(tasks))

    _, bieudo = plt.subplots(1, int(NUMBER_RMPS/2))
    for rmp in range(int(NUMBER_RMPS/2)):
        x = np.arange(len(history_rmp[rmp]))
        y = [i for i in history_rmp[rmp]]
        bieudo[rmp].plot(x, y, label = rmp)
    plt.show() 


    

    # print(len(history_rmp[0]))

    # import pygame 

    # pygame.mixer.init()
    # pygame.mixer.music.load("D:/LinhTinh/warning_audio.mp3")
    # pygame.mixer.music.play()

    # import time 
    # time.sleep(1) 
    visual_result(history, ["lsa_mfea", "mfea"], tasks)


main()