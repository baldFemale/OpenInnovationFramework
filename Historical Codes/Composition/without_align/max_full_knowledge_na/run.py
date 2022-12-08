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
def fun(N=None, K=None, state_num=None, expertise_amount=None, generalist_expertise=None,
        specialist_expertise=None, agent_num=None, search_iteration=None, loop=None, return_dict=None, sema=None):
    # align the landscape
    landscape = Landscape(N=N, state_num=state_num)
    landscape.type(IM_type="Traditional Directed", K=K, k=0)
    landscape.initialize(norm=True)  # with the normalization
    # align the initial state
    # initial_state = np.random.choice(range(state_num), N).tolist()
    # initial_state = [str(i) for i in initial_state]

    g_crowd = []
    for _ in range(agent_num):
        generalist = Generalist(N=N, landscape=landscape, state_num=state_num, expertise_amount=20)
        # generalist.align_default_state(initial_state=initial_state)
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
        specialist = Specialist(N=N, landscape=landscape, state_num=state_num, expertise_amount=40)
        # specialist.align_default_state(initial_state=initial_state)
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
        t_shape = Tshape(N=N, landscape=landscape, state_num=state_num, generalist_expertise=10, specialist_expertise=20)
        # t_shape.align_default_state(initial_state=initial_state)
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
    hyper_iteration = 40
    N = 10
    state_num = 4
    expertise_amount = 12
    generalist_expertise = 4
    specialist_expertise = 8
    concurrency = 50
    K_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    g_performance_across_para, g_cog_performance_across_para, g_potential_performance_across_para, g_deviation_across_para = [], [], [], []
    s_performance_across_para, s_cog_performance_across_para, s_potential_performance_across_para, s_deviation_across_para = [], [], [], []
    t_performance_across_para, t_cog_performance_across_para, t_potential_performance_across_para, t_deviation_across_para = [], [], [], []

    g_original_performance_across_para, g_original_cog_performance_across_para = [], []
    s_original_performance_across_para, s_original_cog_performance_across_para = [], []
    t_original_performance_across_para, t_original_cog_performance_across_para = [], []
    for K in K_list:
        g_temp_1, g_temp_2, g_temp_3, g_temp_4 = [], [], [], []
        s_temp_1, s_temp_2, s_temp_3, s_temp_4 = [], [], [], []
        t_temp_1, t_temp_2, t_temp_3, t_temp_4 = [], [], [], []
        g_original_performance, g_original_cog_performance = [], []
        s_original_performance, s_original_cog_performance = [], []
        t_original_performance, t_original_cog_performance = [], []
        for hyper_loop in range(hyper_iteration):
            manager = mp.Manager()
            return_dict = manager.dict()
            sema = Semaphore(concurrency)
            jobs = []
            for loop in range(landscape_iteration):
                sema.acquire()
                p = mp.Process(target=fun, args=(N, K, state_num, expertise_amount, generalist_expertise, specialist_expertise,
                                                   agent_num, search_iteration, loop, return_dict, sema))
                jobs.append(p)
                p.start()
            # wait for all the new child processes to terminate
            for proc in jobs:
                proc.join()
            ress = return_dict.values()  # Don't need dict index, since it is repetition.
            for result in ress:
                # using += means we don't differentiate different landscapes
                g_temp_1.append(sum(result[0]) / len(result[0]))  # performance; landscape level
                g_temp_2.append(sum(result[1]) / len(result[1]))  # cog_performance; landscape level
                g_temp_3.append(sum(result[2]) / len(result[2]))  # potential_performance; landscape level
                g_temp_4.append(result[3])  # deviation; landscape level

                s_temp_1.append(sum(result[4]) / len(result[4]))  # performance; landscape level
                s_temp_2.append(sum(result[5]) / len(result[5]))  # cog_performance; landscape level
                s_temp_3.append(sum(result[6]) / len(result[6]))  # potential_performance; landscape level
                s_temp_4.append(result[7])  # deviation; landscape level

                t_temp_1.append(sum(result[8]) / len(result[8]))  # performance; landscape level
                t_temp_2.append(sum(result[9]) / len(result[9]))  # cog_performance; landscape level
                t_temp_3.append(sum(result[10]) / len(result[10]))  # potential_performance; landscape level
                t_temp_4.append(result[11])  # deviation; landscape level

                g_original_performance += result[0]  # not differentiate across landscapes
                g_original_cog_performance += result[1]  # not differentiate across landscapes: Agent_repeat * Landscape repeat

                s_original_performance += result[4]  # not differentiate across landscapes
                s_original_cog_performance += result[5]  # not differentiate across landscapes: Agent_repeat * Landscape repeat

                t_original_performance += result[8]  # not differentiate across landscapes
                t_original_cog_performance += result[9]  # not differentiate across landscapes: Agent_repeat * Landscape repeat

        g_result_1 = sum(g_temp_1) / len(g_temp_1)  # performance; landscape level
        g_result_2 = sum(g_temp_2) / len(g_temp_2)  # cog_performance; landscape level
        g_result_3 = sum(g_temp_3) / len(g_temp_3)  # potential_performance; landscape level
        g_result_4 = math.sqrt(sum([sd ** 2 for sd in g_temp_4]) / len(g_temp_4))

        s_result_1 = sum(s_temp_1) / len(s_temp_1)  # performance; landscape level
        s_result_2 = sum(s_temp_2) / len(s_temp_2)  # cog_performance; landscape level
        s_result_3 = sum(s_temp_3) / len(s_temp_3)  # potential_performance; landscape level
        s_result_4 = math.sqrt(sum([sd ** 2 for sd in s_temp_4]) / len(s_temp_4))

        t_result_1 = sum(t_temp_1) / len(t_temp_1)  # performance; landscape level
        t_result_2 = sum(t_temp_2) / len(t_temp_2)  # cog_performance; landscape level
        t_result_3 = sum(t_temp_3) / len(t_temp_3)  # potential_performance; landscape level
        t_result_4 = math.sqrt(sum([sd ** 2 for sd in t_temp_4]) / len(t_temp_4))

        g_performance_across_para.append(g_result_1)
        g_cog_performance_across_para.append(g_result_2)
        g_potential_performance_across_para.append(g_result_3)
        g_deviation_across_para.append(g_result_4)

        s_performance_across_para.append(s_result_1)
        s_cog_performance_across_para.append(s_result_2)
        s_potential_performance_across_para.append(s_result_3)
        s_deviation_across_para.append(s_result_4)

        t_performance_across_para.append(t_result_1)
        t_cog_performance_across_para.append(t_result_2)
        t_potential_performance_across_para.append(t_result_3)
        t_deviation_across_para.append(t_result_4)


        g_original_performance_across_para.append(g_original_performance)
        g_original_cog_performance_across_para.append(g_original_cog_performance)
        s_original_performance_across_para.append(s_original_performance)
        s_original_cog_performance_across_para.append(s_original_cog_performance)
        t_original_performance_across_para.append(t_original_performance)
        t_original_cog_performance_across_para.append(t_original_cog_performance)

    with open("g_performance_across_K", 'wb') as out_file:
        pickle.dump(g_performance_across_para, out_file)
    with open("g_cog_performance_across_K", 'wb') as out_file:
        pickle.dump(g_cog_performance_across_para, out_file)
    with open("g_potential_performance_across_K", 'wb') as out_file:
        pickle.dump(g_potential_performance_across_para, out_file)
    with open("g_deviation_across_K", 'wb') as out_file:
        pickle.dump(g_deviation_across_para, out_file)
    with open("g_original_performance_data_across_K", "wb") as out_file:
        pickle.dump(g_original_performance_across_para, out_file)
    with open("g_original_cog_performance_data_across_K", "wb") as out_file:
        pickle.dump(g_original_cog_performance_across_para, out_file)

    with open("s_performance_across_K", 'wb') as out_file:
        pickle.dump(s_performance_across_para, out_file)
    with open("s_cog_performance_across_K", 'wb') as out_file:
        pickle.dump(s_cog_performance_across_para, out_file)
    with open("s_potential_performance_across_K", 'wb') as out_file:
        pickle.dump(s_potential_performance_across_para, out_file)
    with open("s_deviation_across_K", 'wb') as out_file:
        pickle.dump(s_deviation_across_para, out_file)
    with open("s_original_performance_data_across_K", "wb") as out_file:
        pickle.dump(s_original_performance_across_para, out_file)
    with open("s_original_cog_performance_data_across_K", "wb") as out_file:
        pickle.dump(s_original_cog_performance_across_para, out_file)

    with open("t_performance_across_K", 'wb') as out_file:
        pickle.dump(t_performance_across_para, out_file)
    with open("t_cog_performance_across_K", 'wb') as out_file:
        pickle.dump(t_cog_performance_across_para, out_file)
    with open("t_potential_performance_across_K", 'wb') as out_file:
        pickle.dump(t_potential_performance_across_para, out_file)
    with open("t_deviation_across_K", 'wb') as out_file:
        pickle.dump(t_deviation_across_para, out_file)
    with open("t_original_performance_data_across_K", "wb") as out_file:
        pickle.dump(t_original_performance_across_para, out_file)
    with open("t_original_cog_performance_data_across_K", "wb") as out_file:
        pickle.dump(t_original_cog_performance_across_para, out_file)

    t1 = time.time()
    print(time.strftime("%H:%M:%S", time.gmtime(t1-t0)))

