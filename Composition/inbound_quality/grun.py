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
def func_2(N=None, K=None, state_num=None, expertise_amount=None, agent_num=None,
         search_iteration=None, loop=None, return_dict=None, sema=None, quality=None):
    landscape = Landscape(N=N, state_num=state_num)
    landscape.type(IM_type="Traditional Directed", K=K, k=0)
    landscape.initialize()
    crowd = []
    for _ in range(agent_num):
        generalist = Generalist(N=N, landscape=landscape, state_num=state_num, expertise_amount=expertise_amount)
        crowd.append(generalist)
    pool = landscape.generate_quality_pool(quality_percentage=quality)
    learn_count_across_agent = []
    performance_across_agent = []
    for agent in crowd:
        learn_count = 0
        for _ in range(search_iteration):
            agent.search()
            if agent.learn_from_pool(pool=pool):
                learn_count += 1
        agent.state = agent.cog_state_2_state(cog_state=agent.cog_state)
        agent.fitness = landscape.query_fitness(state=agent.state)
        learn_count_across_agent.append(learn_count)
        performance_across_agent.append(agent.fitness)
    performance_average = sum(performance_across_agent) / len(performance_across_agent)
    learn_average = sum(learn_count_across_agent) / len(learn_count_across_agent)
    performance_deviation = np.std(performance_across_agent)
    return_dict[loop] = [performance_average, learn_average, performance_deviation]
    sema.release()


if __name__ == '__main__':
    t0 = time.time()
    landscape_iteration = 200
    agent_num = 400
    search_iteration = 100
    N = 6
    state_num = 4
    expertise_amount = 8
    K = 3
    quality_list = [0.2, 0.4, 0.6, 0.8]
    performance_across_K = []
    learn_count_across_K = []
    deviation_across_K = []
    concurrency = 24
    sema = Semaphore(concurrency)
    for quality in quality_list:
        temp_1, temp_2, temp_3 = [], [], []
        for _ in range(10):
            manager = mp.Manager()
            return_dict = manager.dict()
            jobs = []
            for loop in range(landscape_iteration):
                sema.acquire()
                p = mp.Process(target=func_2, args=(N, K, state_num, expertise_amount, agent_num, search_iteration, loop, return_dict, sema, quality))
                jobs.append(p)
                p.start()
            for proc in jobs:
                proc.join()
            performance_across_landscape = return_dict.values()  # Don't need dict index, since it is repetition.
            temp_1 += [result[0] for result in performance_across_landscape]  # performance
            temp_2 += [result[1] for result in performance_across_landscape]  # learn count
            temp_3 += [result[2] ** 2 for result in performance_across_landscape]  # deviation has a formula to take average
        result_1 = sum(temp_1) / len(temp_1)
        result_2 = sum(temp_2) / len(temp_2)
        result_3 = math.sqrt(sum(temp_3) / len(temp_3))
        performance_across_K.append(result_1)
        learn_count_across_K.append(result_2)
        deviation_across_K.append(result_3)
    with open("g_performance_across_K", 'wb') as out_file:
        pickle.dump(performance_across_K, out_file)
    with open("g_learn_across_K", 'wb') as out_file:
        pickle.dump(learn_count_across_K, out_file)
    with open("g_deviation_across_K", 'wb') as out_file:
        pickle.dump(deviation_across_K, out_file)
    t1 = time.time()
    print(time.strftime("%H:%M:%S", time.gmtime(t1-t0)))
