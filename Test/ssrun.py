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
from Team import Team
from Landscape import Landscape
import multiprocessing as mp
import time
from multiprocessing import Pool
from multiprocessing import Semaphore
import pickle
import gc
import sys
import psutil
import math


# S + S
def func(N=None, K=None, state_num=None, expertise_amount=None, agent_num=None,
         search_iteration=None, s_overlap=None, loop=None, return_dict=None, sema=None):
    t0 = time.time()
    np.random.seed(None)
    landscape = Landscape(N=N, state_num=state_num)
    landscape.type(K=K)
    landscape.initialize(norm=True)
    t1 = time.time()
    print("landscape time: ", time.strftime("%H:%M:%S", time.gmtime(t1-t0)))
    team_list = []
    for _ in range(agent_num):
        agent_1 = Specialist(N=N, landscape=landscape, state_num=state_num, expertise_amount=expertise_amount)
        agent_2 = Specialist(N=N, landscape=landscape, state_num=state_num, expertise_amount=expertise_amount)
        overlap_domains = np.random.choice(agent_1.expertise_domain, s_overlap, replace=False).tolist()
        free_domains = [each for each in range(N) if each not in agent_1.expertise_domain]
        other_domains = np.random.choice(free_domains, expertise_amount // 4 - s_overlap, replace=False).tolist()
        agent_2.expertise_domain = overlap_domains + other_domains
        team = Team(agent_1=agent_1, agent_2=agent_2, state_num=state_num, N=N)
        team_list.append(team)
    for team in team_list:
        for _ in range(search_iteration):
            team.search()
    # need to query the full fitnee after convergence
    for team in team_list:
        team.agent_1.fitness, team.agent_1.potential_fitness = \
            landscape.query_cog_fitness_full(cog_state=team.agent_1.cog_state)
        team.agent_2.fitness, team.agent_2.potential_fitness = \
            landscape.query_cog_fitness_full(cog_state=team.agent_2.cog_state)
    performance_across_agent_1 = [team.agent_1.fitness for team in team_list]
    performance_across_agent_2 = [team.agent_2.fitness for team in team_list]
    return_dict[loop] = [performance_across_agent_1, performance_across_agent_2]
    sema.release()
    t2 = time.time()
    print("process time: ", time.strftime("%H:%M:%S", time.gmtime(t2-t0)))
    mem = psutil.virtual_memory()
    print("total memory 2: ", float(mem.total) / 1024 / 1024 / 1024)
    print("used memory 2: ", float(mem.used) / 1024 / 1024 / 1024)
    print("free memory 2: ", float(mem.free) / 1024 / 1024 / 1024)


if __name__ == '__main__':
    t0 = time.time()
    landscape_iteration = 5
    agent_num = 100
    search_iteration = 200  # In pre-test, 200 is quite enough for convergence
    hyper_iteration = 2
    N = 12
    state_num = 4
    expertise_amount = 12
    K_list = [0]
    concurrency = 5
    for s_overlap in [3]:
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
                    p = mp.Process(target=func, args=(N, K, state_num, expertise_amount, agent_num, search_iteration, s_overlap, loop, return_dict, sema))
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
            mem = psutil.virtual_memory()
        mem = psutil.virtual_memory()
        print("total memory 3: ", float(mem.total) / 1024 / 1024 / 1024)
        print("used memory 3: ", float(mem.used) / 1024 / 1024 / 1024)
        print("free memory 3: ", float(mem.free) / 1024 / 1024 / 1024)
        with open("s1_performance_across_K_{0}".format(s_overlap), 'wb') as out_file:
            pickle.dump(performance1_across_K, out_file)
        with open("s2_performance_across_K_{0}".format(s_overlap), 'wb') as out_file:
            pickle.dump(performance2_across_K, out_file)
        with open("s1_original_performance_across_K_{0}".format(s_overlap), "wb") as out_file:
            pickle.dump(original1_across_K, out_file)
        with open("s2_original_performance_across_K_{0}".format(s_overlap), "wb") as out_file:
            pickle.dump(original2_across_K, out_file)
    mem = psutil.virtual_memory()
    print("total memory 4: ", float(mem.total) / 1024 / 1024 / 1024)
    print("used memory 4: ", float(mem.used) / 1024 / 1024 / 1024)
    print("free memory 4: ", float(mem.free) / 1024 / 1024 / 1024)
    t1 = time.time()
    print(time.strftime("%H:%M:%S", time.gmtime(t1-t0)))

