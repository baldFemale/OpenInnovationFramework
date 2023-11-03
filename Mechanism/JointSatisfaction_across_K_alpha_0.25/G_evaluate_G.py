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
    # Sender Crowd
    crowd = Crowd(N=N, agent_num=state_num, landscape=landscape, state_num=state_num,
                           generalist_expertise=12, specialist_expertise=0, label="G")
    joint_confusion_rate_list = []
    for _ in range(agent_num):
        # Focal Agent
        generalist = Generalist(N=N, landscape=landscape, state_num=state_num, crowd=crowd, generalist_expertise=generalist_expertise)
        for _ in range(search_iteration):
            generalist.search()
        # Joint satisfaction
        sender_solution = generalist.state.copy()
        sender_domain = generalist.generalist_domain.copy()
        count = 0
        for agent in crowd.agents:
            learnt_solution = agent.state.copy()
            for index in sender_domain:
                learnt_solution[index] = sender_solution[index]
            suggestions = agent.suggest_better_state_from_expertise(state=learnt_solution)
            for each_suggestion in suggestions:
                climbs = generalist.suggest_better_state_from_expertise(state=each_suggestion)
                climbs.remove(reached_solution)
                count += len(climbs)
        joint_confusion_rate = count / state_num
        joint_confusion_rate_list.append(joint_confusion_rate)
    final_joint_confusion_rate = sum(joint_confusion_rate_list) / len(joint_confusion_rate_list)
    return_dict[loop] = [final_joint_confusion_rate]
    sema.release()


if __name__ == '__main__':
    t0 = time.time()
    landscape_iteration = 400
    agent_num = 500
    search_iteration = 200
    N = 9
    state_num = 4
    generalist_expertise = 12
    # K_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    K_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    concurrency = 40
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
    with open("gg_joint_confusion_across_K", 'wb') as out_file:
        pickle.dump(joint_confusion_across_K, out_file)
    t1 = time.time()
    print("Evaluating GG: ", time.strftime("%H:%M:%S", time.gmtime(t1-t0)))


