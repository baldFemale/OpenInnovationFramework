#!/usr/bin/env py39
# -*- coding: utf-8 -*-
# @Author   : Junyi
# @FileName: Visibility_maturity.py
# @Software : PyCharm
# Observing PEP 8 coding style

import numpy as np
from Landscape import Landscape
from Crowd import Crowd
import multiprocessing as mp
import time
from multiprocessing import Semaphore
import pickle


# mp version
def func(N=None, K=None, agent_num=None, knowledge_breadth=None, search_iteration=None,
         uniform_sharing_prob=None, maturity_threshold=None, loop=None, return_dict=None, sema=None):
    """
    Maturity-based visibility experiment with separated sender and receiver crowds.

    Both crowds conduct ordinary local search every period. A sender discloses
    its complete current solution when a fresh random draw is below
    uniform_sharing_prob and its cog_fitness is at least maturity_threshold.

    Maturity is evaluated using the sender's perceived fitness. Receivers
    evaluate disclosed solutions using their own bounded knowledge.
    Receiver solutions do not feed back into the sender pool.

    uniform_sharing_prob is fixed across conditions; maturity_threshold is the
    focal running parameter. All dependent variables concern the receiver crowd.
    Adoption-quality trajectories contain one cumulative mean per period,
    including np.nan until the first adoption.
    """
    try:
        np.random.seed(None)

        landscape = Landscape(N=N, K=K)

        # Sender crowd: solvers who independently search and make solutions visible
        crowd_sender = Crowd(N=N, agent_num=agent_num, knowledge_breadth=knowledge_breadth,
                             landscape=landscape)

        # Receiver crowd: solvers who independently search and learn from visible sender solutions
        crowd_receiver = Crowd(N=N, agent_num=agent_num, knowledge_breadth=knowledge_breadth,
                               landscape=landscape)

        # Search-process trajectory: cumulative mean true fitness of adopted solutions.
        adopted_solution_fitness_across_time = []

        for period in range(search_iteration):
            # Both crowds conduct their own independent search.
            crowd_sender.search()
            crowd_receiver.search()

            # Disclose only sender solutions meeting the perceived maturity threshold.
            crowd_sender.solution_pool = []
            for agent in crowd_sender.agents:
                if (np.random.uniform(0, 1) < uniform_sharing_prob) and (agent.cog_fitness >= maturity_threshold):
                    crowd_sender.solution_pool.append(agent.state.copy())

            np.random.shuffle(crowd_sender.solution_pool)

            # Receiver solutions do not feed back into the sender pool.
            crowd_receiver.solution_pool = [
                solution.copy()
                for solution in crowd_sender.solution_pool
            ]
            crowd_receiver.learn_from_visible_pool()

            adopted_solution_fitness_across_time.append(
                np.mean(crowd_receiver.adopted_solution_fitness_history)
                if crowd_receiver.adopted_solution_fitness_history else np.nan
            )

        # DVs are measured only on the receiver crowd.
        performance_list = [agent.fitness for agent in crowd_receiver.agents]
        fitness_rank_list = [
            landscape.query_fitness_rank(state=agent.state)
            for agent in crowd_receiver.agents
        ]

        breakthrough_fitness = max(performance_list)
        breakthrough_rank = min(fitness_rank_list)  # smaller rank means better solution; rank 1 is global best

        # Calculate the number of unique complete solutions among receiver agents.
        full_solution_set = set()
        for agent in crowd_receiver.agents:
            solution_str = "".join([str(bit) for bit in agent.state])
            full_solution_set.add(solution_str)

        unique_solution_count = len(full_solution_set)

        # Average pairwise normalized Hamming distance among receiver agents.
        pairwise_diversity = crowd_receiver.calculate_pairwise_solution_distance()

        # Search-process measure for visibility.
        # Use true fitness rather than perceived fitness.
        adopted_solution_fitness = (
            np.mean(crowd_receiver.adopted_solution_fitness_history)
            if crowd_receiver.adopted_solution_fitness_history else np.nan
        )

        return_dict[loop] = [
            breakthrough_fitness,
            breakthrough_rank,
            unique_solution_count,
            pairwise_diversity,
            adopted_solution_fitness,
            adopted_solution_fitness_across_time,
        ]
    finally:
        sema.release()


if __name__ == '__main__':
    import datetime
    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))
    t0 = time.time()

    landscape_iteration = 400
    search_iteration = 300
    N = 20
    K_list = [1, 2, 3, 4, 5, 6, 7, 8]

    # Each solver can deliberately evaluate and modify knowledge_breadth dimensions.
    knowledge_breadth = 10

    # Sharing probability is fixed; maturity_threshold is the focal running parameter.
    uniform_sharing_prob = 1

    # Minimum sender perceived fitness required for disclosure.
    maturity_threshold_list = [0.0, 0.1, 0.2, 0.3, 0.4,
                               0.5, 0.6, 0.7, 0.8, 0.9]

    agent_num = 200
    concurrency = 100

    for maturity_threshold in maturity_threshold_list:
        # DVs
        breakthrough_fitness_across_K = []
        breakthrough_rank_across_K = []
        unique_solution_count_across_K = []
        pairwise_diversity_across_K = []
        adopted_solution_fitness_across_K = []
        adopted_solution_fitness_across_time_across_K = []

        for K in K_list:
            manager = mp.Manager()
            return_dict = manager.dict()
            sema = Semaphore(concurrency)
            jobs = []

            try:
                for loop in range(landscape_iteration):
                    # Check failed workers while waiting, including workers that
                    # were killed before they could release their semaphore.
                    while True:
                        acquired = sema.acquire(timeout=0.5)
                        failed_jobs = [(index, proc.exitcode)
                                       for index, proc in enumerate(jobs)
                                       if proc.exitcode not in (None, 0)]
                        if failed_jobs:
                            raise RuntimeError("Failed repetitions: {0}".format(failed_jobs))
                        if acquired:
                            break

                    p = mp.Process(target=func, args=(N, K, agent_num, knowledge_breadth, search_iteration,
                                                      uniform_sharing_prob, maturity_threshold,
                                                      loop, return_dict, sema))
                    p.start()
                    jobs.append(p)

                for proc in jobs:
                    while True:
                        proc.join(timeout=0.5)
                        failed_jobs = [(index, job.exitcode)
                                       for index, job in enumerate(jobs)
                                       if job.exitcode not in (None, 0)]
                        if failed_jobs:
                            raise RuntimeError("Failed repetitions: {0}".format(failed_jobs))
                        if not proc.is_alive():
                            break

                returns = list(return_dict.values())  # Repetition order does not affect the averages.
            finally:
                for proc in jobs:
                    if proc.is_alive():
                        proc.terminate()
                for proc in jobs:
                    proc.join()
                    proc.close()
                manager.shutdown()

            arr = np.asarray([item[:5] for item in returns], dtype=float)  # shape: (n_runs, 5)
            time_series_arr = np.asarray([item[5] for item in returns], dtype=float)

            # The first four DVs are always defined. The visibility-process
            # measure can be np.nan when a run contains no adoption.
            means = arr[:, :4].mean(axis=0)

            adopted_solution_fitness_mean = (
                np.nan
                if np.all(np.isnan(arr[:, 4]))
                else np.nanmean(arr[:, 4])
            )

            # Average the cumulative adoption-quality trajectory across landscape repetitions.
            if np.all(np.isnan(time_series_arr)):
                adopted_solution_fitness_across_time_mean = time_series_arr[0].tolist()
            else:
                adopted_solution_fitness_across_time_mean = np.nanmean(time_series_arr, axis=0).tolist()

            breakthrough_fitness_across_K.append(means[0])
            breakthrough_rank_across_K.append(means[1])
            unique_solution_count_across_K.append(means[2])
            pairwise_diversity_across_K.append(means[3])
            adopted_solution_fitness_across_K.append(adopted_solution_fitness_mean)
            adopted_solution_fitness_across_time_across_K.append(
                adopted_solution_fitness_across_time_mean
            )

        # Save results across K for each maturity threshold.
        with open("maturity_threshold_{0}_breakthrough_fitness_across_K_size_{1}".format(
                maturity_threshold, agent_num), 'wb') as out_file:
            pickle.dump(breakthrough_fitness_across_K, out_file)

        with open("maturity_threshold_{0}_breakthrough_rank_across_K_size_{1}".format(
                maturity_threshold, agent_num), 'wb') as out_file:
            pickle.dump(breakthrough_rank_across_K, out_file)

        with open("maturity_threshold_{0}_unique_solution_count_across_K_size_{1}".format(
                maturity_threshold, agent_num), 'wb') as out_file:
            pickle.dump(unique_solution_count_across_K, out_file)

        with open("maturity_threshold_{0}_pairwise_diversity_across_K_size_{1}".format(
                maturity_threshold, agent_num), 'wb') as out_file:
            pickle.dump(pairwise_diversity_across_K, out_file)

        with open("maturity_threshold_{0}_adopted_solution_fitness_across_K_size_{1}".format(
                maturity_threshold, agent_num), 'wb') as out_file:
            pickle.dump(adopted_solution_fitness_across_K, out_file)

        with open("maturity_threshold_{0}_adopted_solution_fitness_across_time_across_K_size_{1}".format(
                maturity_threshold, agent_num), 'wb') as out_file:
            pickle.dump(adopted_solution_fitness_across_time_across_K, out_file)

    t1 = time.time()
    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))
    print("Maturity-Based Visibility: ",
          time.strftime("%H:%M:%S", time.gmtime(t1 - t0)))
