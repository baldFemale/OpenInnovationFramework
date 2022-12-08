# -*- coding: utf-8 -*-
# @Time     : 9/26/2022 20:23
# @Author   : Junyi
# @FileName: run.py
# @Software  : PyCharm
# Observing PEP 8 coding style
from Generalist import Generalist
from Specialist import Specialist
import numpy as np
from Tshape import Tshape
from Landscape import Landscape
import multiprocessing as mp
import time
from multiprocessing import Pool
from multiprocessing import Semaphore
import pickle
import math


def func(N=None, K=None, state_num=None, generalist_expertise=None, specialist_expertise=None, agent_num=None,
         search_iteration=None, loop=None, return_dict=None, sema=None):
    np.random.seed(None)
    landscape = Landscape(N=N, state_num=state_num)
    landscape.type(K=K)
    landscape.initialize()
    for state, value in landscape.cache.items():
        if value == 1:
            initial_state = list(state)
    crowd = []
    for _ in range(agent_num):
        t_shape = Tshape(N=N, landscape=landscape, state_num=state_num, generalist_expertise=generalist_expertise, specialist_expertise=specialist_expertise)
        t_shape.align_default_state(state=initial_state)
        crowd.append(t_shape)
    for agent in crowd:
        for _ in range(search_iteration):
            agent.search()
    performance_across_agent = [agent.fitness for agent in crowd]
    potential_across_agent = [agent.potential_fitness for agent in crowd]
    return_dict[loop] = [performance_across_agent, potential_across_agent]
    sema.release()


if __name__ == '__main__':
    t0 = time.time()
    landscape_iteration = 50
    agent_num = 100
    search_iteration = 200
    hyper_iteration = 4
    N = 12
    state_num = 4
    generalist_expertise = 8  # 2 * 4: four G domains
    specialist_expertise = 8    # 4 * 2: three S domains
    K_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    performance_across_K = []
    potential_across_K = []
    concurrency = 50
    original_performance_across_K = []
    original_potential_across_K = []
    for K in K_list:
        temp_1, temp_2 = [], []
        for hyper_loop in range(hyper_iteration):
            manager = mp.Manager()
            return_dict = manager.dict()
            sema = Semaphore(concurrency)
            jobs = []
            for loop in range(landscape_iteration):
                sema.acquire()
                p = mp.Process(target=func, args=(N, K, state_num, generalist_expertise, specialist_expertise, agent_num, search_iteration, loop, return_dict, sema))
                jobs.append(p)
                p.start()
            for proc in jobs:
                proc.join()
            performance_across_landscape = return_dict.values()  # Don't need dict index, since it is repetition.
            for result in performance_across_landscape:
                # using += means we don't differentiate different landscapes
                temp_1 += result[0]  # result[0] is a list across agents, take an average-> landscape level
                temp_2 += result[1]   # potential
        result_1 = sum(temp_1) / len(temp_1)
        result_2 = sum(temp_2) / len(temp_2)
        performance_across_K.append(result_1)
        potential_across_K.append(result_2)
        original_performance_across_K.append(temp_1)  # every element: a list of values across landscape, in which one value refer to one landscape
        original_potential_across_K.append(temp_2)  # shape: K * {hyper_iteration * landscape_iteration}
    with open("t_performance_across_K", 'wb') as out_file:
        pickle.dump(performance_across_K, out_file)
    with open("t_potential_across_K", 'wb') as out_file:
        pickle.dump(potential_across_K, out_file)
    with open("t_original_performance_across_K", "wb") as out_file:
        pickle.dump(original_performance_across_K, out_file)
    with open("t_original_potential_across_K", "wb") as out_file:
        pickle.dump(original_potential_across_K, out_file)
    t1 = time.time()
    print(time.strftime("%H:%M:%S", time.gmtime(t1-t0)))
