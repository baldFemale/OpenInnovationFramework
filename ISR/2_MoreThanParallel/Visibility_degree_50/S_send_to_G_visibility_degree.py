#!/usr/bin/env py39
# -*- coding: utf-8 -*-
# @Time     : 9/26/2022 20:23
# @Author   : Junyi
# @FileName: S_send_to_G_visibility_degree.py
# @Software : PyCharm
# Observing PEP 8 coding style

import numpy as np
from Generalist import Generalist
from Specialist import Specialist
from Landscape import Landscape
from Crowd import Crowd
import multiprocessing as mp
import time
from multiprocessing import Semaphore
import pickle


# mp version
def func(N=None, K=None, agent_num=None, search_iteration=None, visibility_prob=None,
         visibility_interval=1, loop=None, return_dict=None, sema=None):
    """
    Visibility-degree experiment with separated sender and receiver crowds.

    - Two independent G crowds are created on the same landscape.
    - The sender crowd only searches and shares visible solutions.
    - The receiver crowd searches and learns from sender's visible solutions.
    - The receiver crowd's learned solutions do not feed back into the visible pool.

    Visibility condition:
        visibility is activated every visibility_interval periods;
        when activated, share if random_draw < visibility_prob.

    Interpretation:
        visibility_prob = visibility intensity
        visibility_interval = visibility frequency
            visibility_interval = 1 means visible every period, same as the original design.
            visibility_interval = 5 means visible at periods 5, 10, 15, ...
    """
    np.random.seed(None)

    if visibility_interval is None:
        visibility_interval = 1
    visibility_interval = int(visibility_interval)
    if visibility_interval < 1:
        raise ValueError("visibility_interval must be a positive integer.")

    landscape = Landscape(N=N, K=K, state_num=4, alpha=0.1)

    # Sender crowd: Generalists who only search and share
    crowd_sender = Crowd(N=N, agent_num=agent_num, landscape=landscape, state_num=4,
                         generalist_expertise=0, specialist_expertise=12, label="S")

    # Receiver crowd: Generalists who search and learn from sender's visible solutions
    crowd_receiver = Crowd(N=N, agent_num=agent_num, landscape=landscape, state_num=4,
                           generalist_expertise=12, specialist_expertise=0, label="G")

    crowd_sender.share_prob_list = [visibility_prob] * agent_num

    for period in range(search_iteration):
        # Both crowds conduct their own independent search.
        crowd_sender.search()
        crowd_receiver.search()

        if (period + 1) % visibility_interval == 0:
            crowd_sender.solution_pool = []
            for agent, share_prob in zip(crowd_sender.agents, crowd_sender.share_prob_list ):
                if np.random.uniform(0, 1) < share_prob:
                    domains = (agent.generalist_domain.copy() + agent.specialist_domain.copy())
                    partial_solution = [agent.state[index] for index in domains]
                    crowd_sender.solution_pool.append([domains, partial_solution])
            np.random.shuffle(crowd_sender.solution_pool)

            crowd_receiver.solution_pool = [[domains.copy(), partial_solution.copy()] for domains, partial_solution in crowd_sender.solution_pool]
            crowd_receiver.learn_from_shared_pool()

    # DVs are measured only on the receiver crowd.
    performance_list = [agent.fitness for agent in crowd_receiver.agents]
    fitness_rank_list = [
        landscape.query_second_fitness_rank(state=agent.state)
        for agent in crowd_receiver.agents
    ]

    breakthrough_fitness = max(performance_list)
    breakthrough_rank = min(fitness_rank_list)  # smaller rank means better solution; rank 1 is global best

    # Calculate diversity among receiver agents.
    domain_solution_dict = {}
    for agent in crowd_receiver.agents:
        domains = agent.generalist_domain.copy() + agent.specialist_domain.copy()
        domains.sort()
        domain_str = "".join([str(i) for i in domains])

        solution_str = [agent.state[index] for index in domains]
        solution_str = "".join(solution_str)

        if domain_str not in domain_solution_dict.keys():
            domain_solution_dict[domain_str] = [solution_str]
        else:
            if solution_str not in domain_solution_dict[domain_str]:
                domain_solution_dict[domain_str].append(solution_str)

    diversity = 0
    for key, value in domain_solution_dict.items():
        diversity += len(value)

    return_dict[loop] = [breakthrough_fitness, breakthrough_rank, diversity]
    sema.release()


if __name__ == '__main__':
    import datetime
    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))
    t0 = time.time()

    landscape_iteration = 200
    search_iteration = 300
    N = 9
    K_list = [1, 2, 3, 4, 5, 6, 7, 8]

    # Visibility degree p_v: probability that an S solution becomes visible to G
    # during a visibility period.
    # p_v = 0.0 means no S solutions are visible to G.
    # p_v = 1.0 means all S solutions are visible to G during each visibility period.
    visibility_prob_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # Visibility frequency: S solutions become visible to G every x periods.
    visibility_interval = 50

    agent_num = 200
    concurrency = 100

    for visibility_prob in visibility_prob_list:
        # DVs
        breakthrough_fitness_across_K = []
        breakthrough_rank_across_K = []
        diversity_across_K = []

        for K in K_list:
            manager = mp.Manager()
            return_dict = manager.dict()
            sema = Semaphore(concurrency)
            jobs = []

            for loop in range(landscape_iteration):
                sema.acquire()
                p = mp.Process(target=func, args=(N, K, agent_num, search_iteration,
                                                  visibility_prob, visibility_interval,
                                                  loop, return_dict, sema))
                jobs.append(p)
                p.start()

            for proc in jobs:
                proc.join()

            returns = return_dict.values()  # Don't need dict index, since it is repetition.
            arr = np.asarray(list(returns))  # shape: (n_runs, 3)
            means = arr.mean(axis=0)

            breakthrough_fitness_across_K.append(means[0])
            breakthrough_rank_across_K.append(means[1])
            diversity_across_K.append(means[2])

        # Save results across K for each visibility probability and visibility interval.
        with open("sg_visibility_prob_{0}_interval_{1}_breakthrough_fitness_across_K_size_{2}".format(
                visibility_prob, visibility_interval, agent_num), 'wb') as out_file:
            pickle.dump(breakthrough_fitness_across_K, out_file)

        with open("sg_visibility_prob_{0}_interval_{1}_breakthrough_rank_across_K_size_{2}".format(
                visibility_prob, visibility_interval, agent_num), 'wb') as out_file:
            pickle.dump(breakthrough_rank_across_K, out_file)

        with open("sg_visibility_prob_{0}_interval_{1}_diversity_across_K_size_{2}".format(
                visibility_prob, visibility_interval, agent_num), 'wb') as out_file:
            pickle.dump(diversity_across_K, out_file)

    t1 = time.time()
    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))
    print("SG Visibility Degree with Interval {0}: ".format(visibility_interval),
          time.strftime("%H:%M:%S", time.gmtime(t1 - t0)))