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
def func(N=None, alpha=None, state_num=None, agent_num=None,
         search_iteration=None, loop=None, return_dict=None, sema=None):
    np.random.seed(None)
    landscape = Landscape(N=N, K=5, state_num=state_num, alpha=alpha)
    # Evaluator Crowd
    crowd = Crowd(N=N, agent_num=agent_num, landscape=landscape, state_num=state_num,
                           generalist_expertise=0, specialist_expertise=12, label="S")
    joint_confusion_rate_list = []
    for _ in range(agent_num):
        specialist = Specialist(N=N, landscape=landscape, state_num=state_num, crowd=crowd, specialist_expertise=12)
        for _ in range(search_iteration):
            specialist.search()
        # Joint local optima
        reached_solution = specialist.state
        count = 0
        for agent in crowd.agents:
            if landscape.query_second_fitness(state=reached_solution) < 1:
                if agent.is_local_optima(state=reached_solution):
                    count += 1
        joint_confusion_rate = count / agent_num
        joint_confusion_rate_list.append(joint_confusion_rate)
    final_joint_confusion_rate = sum(joint_confusion_rate_list) / len(joint_confusion_rate_list)
    return_dict[loop] = [final_joint_confusion_rate]
    sema.release()


if __name__ == '__main__':
    t0 = time.time()
    landscape_iteration = 200
    agent_num = 500
    search_iteration = 200
    N = 9
    state_num = 4
    specialist_expertise = 12
    alpha_list = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
    concurrency = 40
    # DVs
    joint_confusion_across_K = []
    for alpha in alpha_list:
        manager = mp.Manager()
        return_dict = manager.dict()
        sema = Semaphore(concurrency)
        jobs = []
        for loop in range(landscape_iteration):
            sema.acquire()
            p = mp.Process(target=func, args=(N, alpha, state_num, agent_num, search_iteration, loop, return_dict, sema))
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
    with open("ss_joint_confusion_across_alpha", 'wb') as out_file:
        pickle.dump(joint_confusion_across_K, out_file)

    t1 = time.time()
    print("SS: ", time.strftime("%H:%M:%S", time.gmtime(t1-t0)))


