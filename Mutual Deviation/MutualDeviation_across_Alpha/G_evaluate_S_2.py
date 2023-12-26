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
def func(N=None, K=None, agent_num=None, alpha=None,
         search_iteration=None, loop=None, return_dict=None, sema=None):
    np.random.seed(None)
    landscape = Landscape(N=N, K=K, state_num=4, alpha=alpha)
    # Evaluator Crowd
    crowd = Crowd(N=N, agent_num=agent_num, landscape=landscape, state_num=4,
                           generalist_expertise=12, specialist_expertise=0, label="G")
    mutual_climb_rate_list = []
    for _ in range(agent_num):
        # Focal Agent
        specialist = Specialist(N=N, landscape=landscape, state_num=4, crowd=crowd, specialist_expertise=12)
        for _ in range(search_iteration):
            specialist.search()
        # Mutual Climb
        reached_solution = specialist.state.copy()
        count = 0
        for agent in crowd.agents:
            suggestions = agent.suggest_better_state_from_expertise(state=reached_solution)
            for each_suggestion in suggestions:
                climbs = specialist.suggest_better_state_from_expertise(state=each_suggestion)
                if reached_solution in climbs:
                    climbs.remove(reached_solution)
                if len(climbs) != 0:
                    count += 1
                    break
        mutual_climb_rate = count / agent_num
        mutual_climb_rate_list.append(mutual_climb_rate)
    final_mutual_climb_rate = sum(mutual_climb_rate_list) / len(mutual_climb_rate_list)
    return_dict[loop] = [final_mutual_climb_rate]
    sema.release()


if __name__ == '__main__':
    t0 = time.time()
    landscape_iteration = 200
    agent_num = 500
    search_iteration = 200
    N = 9
    K_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    # alpha_list = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
    alpha_list = [0.25, 0.30, 0.35, 0.40, 0.45]
    concurrency = 100
    for alpha in alpha_list:
        joint_confusion_across_K = []
        for K in K_list:
            manager = mp.Manager()
            return_dict = manager.dict()
            sema = Semaphore(concurrency)
            jobs = []
            for loop in range(landscape_iteration):
                sema.acquire()
                p = mp.Process(target=func, args=(N, K, agent_num, alpha, search_iteration, loop, return_dict, sema))
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
        with open("gs_mutual_deviation_across_K_alpha_{0}".format(alpha), 'wb') as out_file:
            pickle.dump(joint_confusion_across_K, out_file)

    t1 = time.time()
    print("GS: ", time.strftime("%H:%M:%S", time.gmtime(t1-t0)))
