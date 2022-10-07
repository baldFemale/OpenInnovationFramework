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


def func_2(N=None, K=None, state_num=None, expertise_amount=None, agent_num=None,
         search_iteration=None, loop=None, return_dict=None, sema=None, quality=None):
    landscape = Landscape(N=N, state_num=state_num)
    landscape.type(IM_type="Traditional Directed", K=K, k=0)
    landscape.initialize()
    crowd = []
    for _ in range(agent_num):
        specialist = Specialist(N=N, landscape=landscape, state_num=state_num, expertise_amount=expertise_amount)
        crowd.append(specialist)
    pool = landscape.generate_quality_pool(quality_percentage=quality)
    performance_across_pool = []
    deviation_across_pool = []
    for state in pool:
        performance_across_agent = []
        for agent in crowd:
            for _ in range(search_iteration):
                agent.elaborate_from_state(state=state)
            # agent.state = agent.cog_state_2_state(cog_state=agent.cog_state)
            # agent.fitness = landscape.query_fitness(state=agent.state)
            performance_across_agent.append(agent.cog_fitness)
        agent_average = sum(performance_across_agent) / len(performance_across_agent)
        agent_deviation = np.std(performance_across_agent)
        performance_across_pool.append(agent_average)
        deviation_across_pool.append(agent_deviation)
    final_average = sum(performance_across_pool) / len(performance_across_pool)
    final_deviation = math.sqrt(sum([each ** 2 for each in deviation_across_pool]) / len(deviation_across_pool))
    return_dict[loop] = [final_average, final_deviation]
    sema.release()


if __name__ == '__main__':
    t0 = time.time()
    landscape_iteration = 200
    agent_num = 400
    search_iteration = 100
    N = 6
    state_num = 4
    expertise_amount = 12
    K = 3
    quality_list = [0.2, 0.4, 0.6, 0.8]
    performance_across_K = []
    learn_count_across_K = []
    deviation_across_K = []
    concurrency = 30
    sema = Semaphore(concurrency)
    for quality in quality_list:
        temp_1, temp_2, temp_3 = [], [], []
        for _ in range(20):
            manager = mp.Manager()
            return_dict = manager.dict()
            jobs = []
            for loop in range(landscape_iteration):
                sema.acquire()  # !!!!!!!!!!!!!!!!!!!!!!
                p = mp.Process(target=func_2, args=(N, K, state_num, expertise_amount, agent_num, search_iteration, loop, return_dict, sema, quality))
                jobs.append(p)
                p.start()
            for proc in jobs:
                proc.join()
            performance_across_landscape = return_dict.values()  # Don't need dict index, since it is repetition.
            temp_1 += [result[0] for result in performance_across_landscape]  # performance
            # temp_2 += [result[1] for result in performance_across_landscape]  # learn count
            temp_3 += [result[1] ** 2 for result in performance_across_landscape]  # deviation has a formula to take average
        result_1 = sum(temp_1) / len(temp_1)
        # result_2 = sum(temp_2) / len(temp_2)
        result_3 = math.sqrt(sum(temp_3) / len(temp_3))
        performance_across_K.append(result_1)
        # learn_count_across_K.append(result_2)
        deviation_across_K.append(result_3)
    with open("s_performance_across_quality", 'wb') as out_file:
        pickle.dump(performance_across_K, out_file)
    # with open("s_learn_across_quality", 'wb') as out_file:
    #     pickle.dump(learn_count_across_K, out_file)
    with open("s_deviation_across_quality", 'wb') as out_file:
        pickle.dump(deviation_across_K, out_file)
    t1 = time.time()
    print(time.strftime("%H:%M:%S", time.gmtime(t1-t0)))
