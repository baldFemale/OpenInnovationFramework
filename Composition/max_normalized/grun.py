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
def fun(N=None, state_num=None, expertise_amount=None, generalist_expertise=None,
        specialist_expertise=None, agent_num=None, landscape=None, initial_state=None,
         search_iteration=None, loop=None, return_dict=None, sema=None):
    g_crowd = []
    for _ in range(agent_num):
        generalist = Generalist(N=N, landscape=landscape, state_num=state_num, expertise_amount=expertise_amount)
        generalist.align_default_state(initial_state=initial_state)
        g_crowd.append(generalist)
    g_performance_across_agent = []
    g_cog_performance_across_agent = []
    g_potential_performance_across_agent = []
    for agent in g_crowd:
        for _ in range(search_iteration):
            agent.search()
        g_performance_across_agent.append(agent.fitness)
        g_cog_performance_across_agent.append(agent.cog_fitness)
        g_potential_performance_across_agent.append(agent.potential_fitness)
        g_deviation = np.std(g_performance_across_agent)

    s_crowd = []
    for _ in range(agent_num):
        specialist = Specialist(N=N, landscape=landscape, state_num=state_num, expertise_amount=expertise_amount)
        specialist.align_default_state(initial_state=initial_state)
        s_crowd.append(specialist)
    s_performance_across_agent = []
    s_cog_performance_across_agent = []
    s_potential_performance_across_agent = []
    for agent in s_crowd:
        for _ in range(search_iteration):
            agent.search()
        s_performance_across_agent.append(agent.fitness)
        s_cog_performance_across_agent.append(agent.cog_fitness)
        s_potential_performance_across_agent.append(agent.potential_fitness)
        s_deviation = np.std(s_performance_across_agent)

    t_crowd = []
    for _ in range(agent_num):
        t_shape = Tshape(N=N, landscape=landscape, state_num=state_num, generalist_expertise=generalist_expertise, specialist_expertise=specialist_expertise)
        t_shape.align_default_state(initial_state=initial_state)
        t_crowd.append(t_shape)
    t_performance_across_agent = []
    t_cog_performance_across_agent = []
    t_potential_performance_across_agent = []
    for agent in t_crowd:
        for _ in range(search_iteration):
            agent.search()
        t_performance_across_agent.append(agent.fitness)
        t_cog_performance_across_agent.append(agent.cog_fitness)
        t_potential_performance_across_agent.append(agent.potential_fitness)
        t_deviation = np.std(t_performance_across_agent)

    return_dict[loop] = [g_performance_across_agent, g_cog_performance_across_agent, g_potential_performance_across_agent, g_deviation,
                         s_performance_across_agent, s_cog_performance_across_agent, s_potential_performance_across_agent, s_deviation,
                         t_performance_across_agent, t_cog_performance_across_agent, t_potential_performance_across_agent, t_deviation]
    sema.release()


if __name__ == '__main__':
    t0 = time.time()
    landscape_iteration = 50
    agent_num = 400
    search_iteration = 100  # In pre-test, 200 is quite enough for convergence
    hyper_iteration = 20
    N = 9
    state_num = 4
    expertise_amount = 12
    generalist_expertise = 4
    specialist_expertise = 8
    K_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    performance_across_para = []
    cog_performance_across_para = []
    potential_performance_across_para = []
    deviation_across_para = []
    concurrency = 25

    original_performance_across_para = []
    original_cog_performance_across_para = []
    for K in K_list:
        temp_1, temp_2, temp_3, temp_4 = [], [], [], []
        original_performance = []
        original_cog_performance = []
        for hyper_loop in range(hyper_iteration):
            manager = mp.Manager()
            return_dict = manager.dict()
            sema = Semaphore(concurrency)
            jobs = []
            for loop in range(landscape_iteration):
                # align the landscape
                landscape = Landscape(N=N, state_num=state_num)
                landscape.type(IM_type="Traditional Directed", K=K, k=0)
                landscape.initialize(norm=True)  # with the normalization
                # align the initial state
                initial_state = np.random.choice(range(state_num), N).tolist()
                sema.acquire()
                p = mp.Process(target=fun, args=(N, state_num, expertise_amount, generalist_expertise, specialist_expertise,
                                                   agent_num, landscape, initial_state,
                                                  search_iteration, loop, return_dict, sema))
                jobs.append(p)
                p.start()
            # wait for all the new child processes to terminate
            for proc in jobs:
                proc.join()
            ress = return_dict.values()  # Don't need dict index, since it is repetition.
            for result in ress:
                # using += means we don't differentiate different landscapes
                temp_1.append(sum(result[0]) / len(result[0]))  # performance; landscape level
                temp_2.append(sum(result[1]) / len(result[1]))  # cog_performance; landscape level
                temp_3.append(sum(result[2]) / len(result[2]))  # potential_performance; landscape level
                temp_4.append(result[3])  # deviation; landscape level

                original_performance += result[0]  # not differentiate across landscapes
                original_cog_performance += result[1]  # not differentiate across landscapes: Agent_repeat * Landscape repeat
        result_1 = sum(temp_1) / len(temp_1)  # performance; landscape level
        result_2 = sum(temp_2) / len(temp_2)  # cog_performance; landscape level
        result_3 = sum(temp_3) / len(temp_3)  # potential_performance; landscape level
        result_4 = math.sqrt(sum([sd ** 2 for sd in temp_4]) / len(temp_4))

        performance_across_para.append(result_1)
        cog_performance_across_para.append(result_2)
        potential_performance_across_para.append(result_3)
        deviation_across_para.append(result_4)

        original_performance_across_para.append(original_performance)
        original_cog_performance_across_para.append(original_cog_performance)

    with open("g_performance_across_K", 'wb') as out_file:
        pickle.dump(performance_across_para, out_file)
    with open("g_cog_performance_across_K", 'wb') as out_file:
        pickle.dump(cog_performance_across_para, out_file)
    with open("g_potential_performance_across_K", 'wb') as out_file:
        pickle.dump(potential_performance_across_para, out_file)
    with open("g_deviation_across_K", 'wb') as out_file:
        pickle.dump(deviation_across_para, out_file)
    with open("g_original_performance_data_across_K", "wb") as out_file:
        pickle.dump(original_performance_across_para, out_file)
    with open("g_original_cog_performance_data_across_K", "wb") as out_file:
        pickle.dump(original_cog_performance_across_para, out_file)


    t1 = time.time()
    print(time.strftime("%H:%M:%S", time.gmtime(t1-t0)))

