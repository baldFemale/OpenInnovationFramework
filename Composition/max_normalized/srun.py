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


def func(N=None, K=None, state_num=None, expertise_amount=None, agent_num=None,
         search_iteration=None, loop=None, hyper_loop=None, hyper_iteration=None, return_dict=None, sema=None):
    landscape = Landscape(N=N, state_num=state_num)
    landscape.type(IM_type="Traditional Directed", K=K, k=0)
    landscape.initialize(norm=False)
    crowd = []
    for _ in range(agent_num):
        specialist = Specialist(N=N, landscape=landscape, state_num=state_num, expertise_amount=expertise_amount)
        specialist.align_default_state(loop=hyper_loop*hyper_iteration+loop)
        crowd.append(specialist)
    for agent in crowd:
        for _ in range(search_iteration):
            agent.search()
    performance_across_agent = [agent.fitness for agent in crowd]  # only the last fitness value (convergence)
    cog_performance_across_agent = [agent.cog_fitness for agent in crowd]
    performance_deviation = np.std(performance_across_agent)
    return_dict[loop] = [performance_across_agent, performance_deviation, cog_performance_across_agent]
    sema.release()


if __name__ == '__main__':
    t0 = time.time()
    landscape_iteration = 100
    agent_num = 400
    search_iteration = 100
    hyper_iteration = 10
    N = 9
    state_num = 4
    expertise_amount = 12  # C_9_3
    K_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    performance_across_K = []
    cog_performance_across_K = []
    deviation_across_K = []
    concurrency = 25
    original_performance_data_across_K = []
    original_cog_performance_data_across_K = []
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
                p = mp.Process(target=func, args=(N, K, state_num, expertise_amount, agent_num, search_iteration, loop, hyper_loop, hyper_iteration, return_dict, sema))
                jobs.append(p)
                p.start()
            for proc in jobs:
                proc.join()
            performance_across_landscape = return_dict.values()  # Don't need dict index, since it is repetition.
            for result in performance_across_landscape:
                # we don't differentiate different landscapes
                temp_1.append(sum(result[0]) / len(result[0]))  # result[0] is a list across agents, take an average-> landscape level
                temp_2.append(result[1])  # result[1] is the standard deviation
                temp_3.append(sum(result[2]) / len(result[2]))
        result_1 = sum(temp_1) / len(temp_1)
        result_2 = math.sqrt(sum([sd ** 2 for sd in temp_2]) / (hyper_iteration * landscape_iteration))
        result_3 = sum(temp_3) / len(temp_3)

        performance_across_K.append(result_1)
        deviation_across_K.append(result_2)
        cog_performance_across_K.append(result_3)

        original_performance_data_across_K.append(temp_1)  # every element: a list of values across landscape, in which one value refer to one landscape
        original_deviation_data_across_K.append(temp_2)  # shape: K * {hyper_iteration * landscape_iteration}
        original_cog_performance_data_across_K.append(temp_3)  # shape: K * {hyper_iteration * landscape_iteration}

    with open("s_performance_across_K", 'wb') as out_file:
        pickle.dump(performance_across_K, out_file)
    with open("s_deviation_across_K", 'wb') as out_file:
        pickle.dump(deviation_across_K, out_file)

    with open("s_original_performance_data_across_K", "wb") as out_file:
        pickle.dump(original_performance_data_across_K, out_file)
    with open("s_original_cog_performance_data_across_K", "wb") as out_file:
        pickle.dump(original_cog_performance_data_across_K, out_file)
    with open("s_original_deviation_data_across_K", "wb") as out_file:
        pickle.dump(original_deviation_data_across_K, out_file)
    t1 = time.time()
    print(time.strftime("%H:%M:%S", time.gmtime(t1-t0)))
