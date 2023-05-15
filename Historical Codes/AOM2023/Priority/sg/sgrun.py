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


# G + S
def func(N=None, K=None, state_num=None, g_expertise_amount=None,s_expertise_amount=None, agent_num=None,
         search_iteration=None, overlap=None, loop=None, return_dict=None, sema=None):
    np.random.seed(None)
    landscape = Landscape(N=N, state_num=state_num)
    landscape.type(K=K)
    landscape.initialize(norm=True)
    crowd_1, crowd_2 = [], []
    for _ in range(agent_num):
        agent_1 = Generalist(N=N, landscape=landscape, state_num=state_num, expertise_amount=g_expertise_amount)
        crowd_1.append(agent_1)
        agent_2 = Specialist(N=N, landscape=landscape, state_num=state_num, expertise_amount=s_expertise_amount)
        overlap_domains = np.random.choice(agent_1.expertise_domain, overlap, replace=False).tolist()
        free_domains = [each for each in range(N) if each not in agent_1.expertise_domain]
        other_domains = np.random.choice(free_domains, s_expertise_amount // 4 - overlap, replace=False).tolist()
        agent_2.expertise_domain = overlap_domains + other_domains
        agent_2.cog_state = agent_2.state_2_cog_state(state=agent_2.state)
        agent_2.cog_fitness = landscape.query_cog_fitness_partial(cog_state=agent_2.cog_state,
                                                                  expertise_domain=agent_2.expertise_domain)
        crowd_2.append(agent_2)
    for index in range(agent_num):
        for _ in range(search_iteration):
            crowd_2[index].search()
            # the first agent, generalist, will always follow the solution of specialist
            crowd_1[index].priority_search(co_state=crowd_2[index].cog_state,
                                           co_expertise_domain=crowd_2[index].expertise_domain)
    performance_across_agent_1 = [agent.fitness for agent in crowd_1]
    performance_across_agent_2 = [agent.fitness for agent in crowd_2]
    return_dict[loop] = [performance_across_agent_1, performance_across_agent_2]
    sema.release()


if __name__ == '__main__':
    t0 = time.time()
    landscape_iteration = 50
    agent_num = 50
    search_iteration = 200  # In pre-test, 200 is quite enough for convergence
    hyper_iteration = 4
    N = 12
    state_num = 4
    g_expertise_amount = 12
    s_expertise_amount = 12
    K_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    concurrency = 50
    for overlap in [3]:
        performance1_across_K = []
        performance2_across_K = []
        original1_across_K = []
        original2_across_K = []
        for K in K_list:
            temp_1, temp_2 = [], []
            for hyper_loop in range(hyper_iteration):
                manager = mp.Manager()
                return_dict = manager.dict()
                sema = Semaphore(concurrency)
                jobs = []
                for loop in range(landscape_iteration):
                    sema.acquire()
                    p = mp.Process(target=func, args=(N, K, state_num, g_expertise_amount, s_expertise_amount, agent_num, search_iteration, overlap, loop, return_dict, sema))
                    jobs.append(p)
                    p.start()
                for proc in jobs:
                    proc.join()
                performance_across_landscape = return_dict.values()  # Don't need dict index, since it is repetition.
                for result in performance_across_landscape:
                    # using += means we don't differentiate different landscapes
                    temp_1 += result[0]  # g1
                    temp_2 += result[1]   # g2
            result_1 = sum(temp_1) / len(temp_1)
            result_2 = sum(temp_2) / len(temp_2)
            performance1_across_K.append(result_1)
            performance2_across_K.append(result_2)
            original1_across_K.append(temp_1)  # every element: a list of values across landscape, in which one value refer to one landscape
            original2_across_K.append(temp_2)  # shape: K * {hyper_iteration * landscape_iteration}
        with open("g_performance_across_K_{0}".format(overlap), 'wb') as out_file:
            pickle.dump(performance1_across_K, out_file)
        with open("s_performance_across_K_{0}".format(overlap), 'wb') as out_file:
            pickle.dump(performance2_across_K, out_file)
        with open("g_original_performance_across_K_{0}".format(overlap), "wb") as out_file:
            pickle.dump(original1_across_K, out_file)
        with open("s_original_performance_across_K_{0}".format(overlap), "wb") as out_file:
            pickle.dump(original2_across_K, out_file)
    t1 = time.time()
    print(time.strftime("%H:%M:%S", time.gmtime(t1-t0)))


