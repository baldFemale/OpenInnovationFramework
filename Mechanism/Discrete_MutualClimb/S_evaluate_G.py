# -*- coding: utf-8 -*-
# @Time     : 9/26/2022 20:23
# @Author   : Junyi
# @FileName: run.py
# @Software  : PyCharm
# Observing PEP 8 coding style
import numpy as np
from Generalist import Generalist
from Specialist import Specialist
from Landscape import Landscape
from Crowd import Crowd
import multiprocessing as mp
import time
from multiprocessing import Semaphore
import pickle


# mp version
def func(N=None, K=None, state_num=None, generalist_expertise=None, agent_num=None,
         search_iteration=None, loop=None, return_dict=None, sema=None):
    np.random.seed(None)
    landscape = Landscape(N=N, K=K, state_num=state_num, alpha=0.25)
    # Evaluator Crowd
    crowd = Crowd(N=N, agent_num=50, landscape=landscape, state_num=state_num,
                           generalist_expertise=0, specialist_expertise=12, label="S")
    mutual_climb_rate_list = []
    for _ in range(agent_num):
        generalist = Generalist(N=N, landscape=landscape, state_num=state_num, crowd=crowd, generalist_expertise=generalist_expertise)
        for _ in range(search_iteration):
            generalist.search()
        # Mutual Climb
        reached_solution = generalist.state.copy()
        count = 0
        for agent in crowd.agents:
            suggestion = agent.suggest_better_state(state=reached_solution)
            if len(suggestion) != 0:
                climb = generalist.suggest_better_state(state=suggestion)
                if len(climb) != 0:
                    if landscape.query_second_fitness(state=climb) > landscape.query_second_fitness(state=reached_solution):
                        count += 1
        mutual_climb_rate = count / 50
        mutual_climb_rate_list.append(mutual_climb_rate)
    final_mutual_climb_rate = sum(mutual_climb_rate_list) / len(mutual_climb_rate_list)
    return_dict[loop] = [final_mutual_climb_rate]
    sema.release()


if __name__ == '__main__':
    t0 = time.time()
    landscape_iteration = 400
    agent_num = 100
    search_iteration = 200
    N = 9
    state_num = 4
    generalist_expertise = 12
    # K_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    K_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    concurrency = 50
    # DVs
    joint_confusion_across_K = []
    for K in K_list:
        manager = mp.Manager()
        return_dict = manager.dict()
        sema = Semaphore(concurrency)
        jobs = []
        for loop in range(landscape_iteration):
            sema.acquire()
            p = mp.Process(target=func, args=(N, K, state_num, generalist_expertise,
                                              agent_num, search_iteration, loop, return_dict, sema))
            jobs.append(p)
            p.start()
        for proc in jobs:
            proc.join()
        returns = return_dict.values()  # Don't need dict index, since it is repetition.

        temp_joint_confusion = []
        for result in returns:  # 50 landscape repetitions
            temp_joint_confusion.append(result[0])

        joint_confusion_across_K.append(sum(temp_joint_confusion) / len(temp_joint_confusion))

    # remove time dimension
    with open("sg_mutual_climb_across_K", 'wb') as out_file:
        pickle.dump(joint_confusion_across_K, out_file)

    t1 = time.time()
    print("Evaluating SG: ", time.strftime("%H:%M:%S", time.gmtime(t1-t0)))


