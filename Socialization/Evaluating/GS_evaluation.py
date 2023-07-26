# -*- coding: utf-8 -*-
# @Time     : 9/26/2022 20:23
# @Author   : Junyi
# @FileName: run.py
# @Software  : PyCharm
# Observing PEP 8 coding style
import numpy as np
from Agent import Agent
from Landscape import Landscape
from Crowd import Crowd
import multiprocessing as mp
import time
from multiprocessing import Semaphore
import pickle
import statistics


# mp version
def func(N=None, K=None, state_num=None, generalist_expertise=None, specialist_expertise=None, agent_num=None,
         search_iteration=None, loop=None, return_dict=None, sema=None):
    np.random.seed(None)
    landscape = Landscape(N=N, K=K, state_num=state_num)

    g_performance_across_agent_time = []
    g_cog_performance_across_agent_time = []

    s_performance_across_agent_time = []
    s_cog_performance_across_agent_time = []
    # Within the same crowd, individuals boost themselves
    G_crowd = Crowd(N=N, agent_num=agent_num, landscape=landscape, state_num=state_num,
                           generalist_expertise=12, specialist_expertise=0)
    S_crowd = Crowd(N=N, agent_num=agent_num, landscape=landscape, state_num=state_num,
                           generalist_expertise=0, specialist_expertise=12)
    G_crowd.form_connections(group_size=7)
    S_crowd.form_connections(group_size=7)
    for _ in range(search_iteration):
        for agent in G_crowd.agents:
            agent.search()
            G_crowd.imitate_from_external_connections(lr=0.3, crowd=S_crowd)
        for agent in S_crowd.agents:
            agent.search()
            S_crowd.imitate_from_external_connections(lr=0.3, crowd=G_crowd)

    for agent in G_crowd.agents:
        g_performance_across_agent_time.append(agent.fitness_across_time)
        g_cog_performance_across_agent_time.append(agent.cog_fitness_across_time)

    for agent in S_crowd.agents:
        s_performance_across_agent_time.append(agent.fitness_across_time)
        s_cog_performance_across_agent_time.append(agent.cog_fitness_across_time)

    g_performance_across_time = []
    g_cog_performance_across_time = []
    g_variance_across_time = []
    for period in range(search_iteration):
        temp_1 = [performance_list[period] for performance_list in g_performance_across_agent_time]
        temp_2 = [performance_list[period] for performance_list in g_cog_performance_across_agent_time]
        g_performance_across_time.append(sum(temp_1) / len(temp_1))
        g_cog_performance_across_time.append(sum(temp_2) / len(temp_2))
        g_variance_across_time.append(np.std(temp_1))

    s_performance_across_time = []
    s_cog_performance_across_time = []
    s_variance_across_time = []
    for period in range(search_iteration):
        temp_1 = [performance_list[period] for performance_list in s_performance_across_agent_time]
        temp_2 = [performance_list[period] for performance_list in s_cog_performance_across_agent_time]
        s_performance_across_time.append(sum(temp_1) / len(temp_1))
        s_cog_performance_across_time.append(sum(temp_2) / len(temp_2))
        s_variance_across_time.append(np.std(temp_1))

    return_dict[loop] = [g_performance_across_time, g_cog_performance_across_time, g_variance_across_time,
                         s_performance_across_time, s_cog_performance_across_time, s_variance_across_time]
    sema.release()


if __name__ == '__main__':
    t0 = time.time()
    landscape_iteration = 300
    agent_num = 100
    search_iteration = 200
    N = 9
    state_num = 4
    generalist_expertise = 12
    specialist_expertise = 0
    K_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    concurrency = 50
    # DVs
    g_performance_across_K = []
    g_variance_across_K = []

    s_performance_across_K = []
    s_variance_across_K = []

    g_performance_across_K_time = []
    g_cog_performance_across_K_time = []
    g_variance_across_K_time = []

    s_performance_across_K_time = []
    s_cog_performance_across_K_time = []
    s_variance_across_K_time = []
    for K in K_list:
        manager = mp.Manager()
        return_dict = manager.dict()
        sema = Semaphore(concurrency)
        jobs = []
        for loop in range(landscape_iteration):
            sema.acquire()
            p = mp.Process(target=func, args=(N, K, state_num, generalist_expertise, specialist_expertise,
                                              agent_num, search_iteration, loop, return_dict, sema))
            jobs.append(p)
            p.start()
        for proc in jobs:
            proc.join()
        returns = return_dict.values()  # Don't need dict index, since it is repetition.

        g_temp_fitness_time, g_temp_cog_time, g_temp_var_time = [], [], []
        s_temp_fitness_time, s_temp_cog_time, s_temp_var_time = [], [], []
        g_temp_fitness, g_temp_cog, g_temp_var = [], [], []
        s_temp_fitness, s_temp_cog, s_temp_var = [], [], []
        for result in returns:  # 50 landscape repetitions
            g_temp_fitness_time.append(result[0])
            g_temp_cog_time.append(result[1])
            g_temp_var_time.append(result[2])

            s_temp_fitness_time.append(result[3])
            s_temp_cog_time.append(result[4])
            s_temp_var_time.append(result[5])

            g_temp_fitness.append(result[0][-1])
            g_temp_cog.append(result[1][-1])
            g_temp_var.append(result[2][-1])

            s_temp_fitness.append(result[3][-1])
            s_temp_cog.append(result[4][-1])
            s_temp_var.append(result[5][-1])

        g_performance_across_K.append(sum(g_temp_fitness) / len(g_temp_fitness))
        g_variance_across_K.append(sum(g_temp_var) / len(g_temp_var))

        s_performance_across_K.append(sum(s_temp_fitness) / len(s_temp_fitness))
        s_variance_across_K.append(sum(s_temp_var) / len(s_temp_var))

        g_performance_across_time, g_cog_performance_across_time, g_variance_across_time = [], [], []
        s_performance_across_time, s_cog_performance_across_time, s_variance_across_time = [], [], []
        for period in range(search_iteration):
            temp_1 = [performance_list[period] for performance_list in g_temp_fitness_time]
            g_performance_across_time.append(sum(temp_1) / len(temp_1))

            temp_2 = [performance_list[period] for performance_list in g_temp_cog_time]
            g_cog_performance_across_time.append(sum(temp_2) / len(temp_2))

            temp_3 = [performance_list[period] for performance_list in g_temp_var_time]
            g_variance_across_time.append(sum(temp_3) / len(temp_3))

            temp_4 = [performance_list[period] for performance_list in s_temp_fitness_time]
            s_performance_across_time.append(sum(temp_4) / len(temp_4))

            temp_5 = [performance_list[period] for performance_list in s_temp_cog_time]
            s_cog_performance_across_time.append(sum(temp_5) / len(temp_5))

            temp_6 = [performance_list[period] for performance_list in s_temp_var_time]
            s_variance_across_time.append(sum(temp_6) / len(temp_6))

        g_performance_across_K_time.append(g_performance_across_time)
        g_cog_performance_across_K_time.append(g_cog_performance_across_time)
        g_variance_across_K_time.append(g_variance_across_time)

        s_performance_across_K_time.append(s_performance_across_time)
        s_cog_performance_across_K_time.append(s_cog_performance_across_time)
        s_variance_across_K_time.append(s_variance_across_time)
    # remove time dimension
    with open("g_performance_across_K", 'wb') as out_file:
        pickle.dump(g_performance_across_K, out_file)
    with open("g_variance_across_K", 'wb') as out_file:
        pickle.dump(g_variance_across_K, out_file)
    # retain time dimension
    with open("g_performance_across_K_time", 'wb') as out_file:
        pickle.dump(g_performance_across_K_time, out_file)
    with open("g_cog_performance_across_K_time", 'wb') as out_file:
        pickle.dump(g_cog_performance_across_K_time, out_file)
    with open("g_variance_across_K_time", 'wb') as out_file:
        pickle.dump(g_variance_across_K_time, out_file)

    # remove time dimension
    with open("s_performance_across_K", 'wb') as out_file:
        pickle.dump(s_performance_across_K, out_file)
    with open("s_variance_across_K", 'wb') as out_file:
        pickle.dump(s_variance_across_K, out_file)
    # retain time dimension
    with open("s_performance_across_K_time", 'wb') as out_file:
        pickle.dump(s_performance_across_K_time, out_file)
    with open("s_cog_performance_across_K_time", 'wb') as out_file:
        pickle.dump(s_cog_performance_across_K_time, out_file)
    with open("s_variance_across_K_time", 'wb') as out_file:
        pickle.dump(s_variance_across_K_time, out_file)

    t1 = time.time()
    print("GS: ", time.strftime("%H:%M:%S", time.gmtime(t1-t0)))


