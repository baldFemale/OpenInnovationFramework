# -*- coding: utf-8 -*-
# @Time     : 9/26/2022 20:23
# @Author   : Junyi
# @FileName: run.py
# @Software  : PyCharm
# Observing PEP 8 coding style
import numpy as np
from Generalist import Generalist
from Specialist import Specialist
from Tshape import Tshape
from Landscape import Landscape
import multiprocessing as mp
import time
from multiprocessing import Pool
from multiprocessing import Semaphore
import pickle
import math


# mp version
def func(N=None, K=None, state_num=None, expertise_amount=None, agent_num=None,
         search_iteration=None, loop=None, return_dict=None, sema=None):
    np.random.seed(None)
    landscape = Landscape(N=N, state_num=state_num)
    landscape.type(IM_type="Traditional Directed", K=K, k=0)
    landscape.initialize()
    crowd = []
    for _ in range(agent_num):
        generalist = Generalist(N=N, landscape=landscape, state_num=state_num, expertise_amount=expertise_amount)
        crowd.append(generalist)
    for agent in crowd:
        for _ in range(search_iteration):
            agent.search()

    diversity = 0
    state_pool = [agent.cog_state for agent in crowd]
    for index, agent in enumerate(crowd):
        if index >= agent_num - 1:
            break
        selected_pool = state_pool[index+1::]
        for cog_state in selected_pool:
            for i in range(N):
                if agent.cog_state[i] == cog_state[i]:
                    continue
                else:
                    diversity += 1
    diversity = diversity * 2 / (N * agent_num * (agent_num - 1))
    performance_across_agent = [agent.cog_fitness for agent in crowd]
    performance_deviation = np.std(performance_across_agent)
    return_dict[loop] = [performance_across_agent, performance_deviation, diversity]
    sema.release()


if __name__ == '__main__':
    t0 = time.time()
    landscape_iteration = 100
    agent_num = 400
    search_iteration = 200  # In pre-test, 200 is quite enough for convergence
    hyper_iteration = 10
    N = 10
    state_num = 4
    expertise_amount = 20
    K_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    performance_across_K = []
    jump_count_across_K = []
    deviation_across_K = []
    diversity_across_K = []
    concurrency = 25
    original_performance_data_across_K = []
    original_deviation_data_across_K = []
    for K in K_list:
        temp_1, temp_2, temp_3 = [], [], []
        for hyper_loop in range(hyper_iteration):
            manager = mp.Manager()
            return_dict = manager.dict()
            sema = Semaphore(concurrency)
            jobs = []
            for loop in range(landscape_iteration):
                sema.acquire()
                p = mp.Process(target=func, args=(N, K, state_num, expertise_amount, agent_num, search_iteration, loop, return_dict, sema))
                jobs.append(p)
                p.start()
            for proc in jobs:
                proc.join()
            performance_across_landscape = return_dict.values()  # Don't need dict index, since it is repetition.
            for result in performance_across_landscape:
                # using += means we don't differentiate different landscapes
                temp_1.append(sum(result[0]) / len(result[0]))  # result[0] is a list across agents, take an average-> landscape level
                temp_2.append(result[1])  # result[1] is the standard deviation
                temp_3
        result_1 = sum(temp_1) / len(temp_1)
        result_2 = math.sqrt(sum([sd ** 2 for sd in temp_2]) / (hyper_iteration * landscape_iteration))
        performance_across_K.append(result_1)
        deviation_across_K.append(result_2)
        original_performance_data_across_K.append(temp_1)  # every element: a list of values across landscape, in which one value refer to one landscape
        original_deviation_data_across_K.append(temp_2)  # shape: K * {hyper_iteration * landscape_iteration}
    with open("g_performance_across_K", 'wb') as out_file:
        pickle.dump(performance_across_K, out_file)
    with open("g_deviation_across_K", 'wb') as out_file:
        pickle.dump(deviation_across_K, out_file)
    with open("g_original_performance_data_across_K", "wb") as out_file:
        pickle.dump(original_performance_data_across_K, out_file)
    with open("g_original_deviation_data_across_K", "wb") as out_file:
        pickle.dump(original_deviation_data_across_K, out_file)
    t1 = time.time()
    print(time.strftime("%H:%M:%S", time.gmtime(t1-t0)))

