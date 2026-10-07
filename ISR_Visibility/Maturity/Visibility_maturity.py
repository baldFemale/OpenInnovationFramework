#!/usr/bin/env py39
# -*- coding: utf-8 -*-
# @Author   : Junyi
# @FileName: Visibility_timing.py
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
         visibility_extent=None, visibility_start=0, visibility_interval=10, loop=None, return_dict=None, sema=None):
    """
    Visibility-timing experiment with separated sender and receiver crowds.

    visibility_extent is fixed across timing conditions and determines the
    proportion of sender agents whose visibility_status is True for the run.

    Visibility condition:
        visibility starts at visibility_start and repeats every
        visibility_interval periods thereafter, following ordinary local search.

    Interpretation:
        visibility_start is a zero-based period index.
            visibility_start = 0 means first visibility follows the first search step.
            visibility_start = 10 means first visibility follows the eleventh search step.
        visibility_interval = 5 means visible at visibility_start,
            visibility_start + 5, visibility_start + 10, ...

    Both crowds search throughout. Receiver solutions do not feed back into the
    sender pool. All dependent variables are measured only on the receiver crowd.
    Adoption-quality trajectories contain one entry per visibility event.
    """
    try:
        np.random.seed(None)

        if visibility_interval is None:
            visibility_interval = 10
        visibility_interval = int(visibility_interval)
        if visibility_interval < 1:
            raise ValueError("visibility_interval must be a positive integer.")

        landscape = Landscape(N=N, K=K)

        # Sender crowd: solvers who independently search and make solutions visible
        crowd_sender = Crowd(N=N, agent_num=agent_num, knowledge_breadth=knowledge_breadth,
                             landscape=landscape)

        # Receiver crowd: solvers who independently search and learn from visible sender solutions
        crowd_receiver = Crowd(N=N, agent_num=agent_num, knowledge_breadth=knowledge_breadth,
                               landscape=landscape)

        crowd_sender.set_visibility_status(visibility_extent=visibility_extent)

        # Search-process trajectory: cumulative mean true fitness of adopted solutions.
        adopted_solution_fitness_across_time = []

        for period in range(search_iteration):
            # Both crowds conduct their own independent search.
            crowd_sender.search()
            crowd_receiver.search()

            if (period >= visibility_start) and ((period - visibility_start) % visibility_interval == 0):
                # Visible senders disclose their complete current solutions.
                crowd_sender.get_visible_pool()

                # Receiver crowd learns only from sender's visible solutions.
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

    # Visibility extent is fixed; visibility_start is the focal running parameter.
    visibility_extent = 1.0
    visibility_start_list = [0, 10, 20, 30, 40, 50, 75, 100, 200, 250, 260, 270, 280, 290]

    # Visibility starts at visibility_start, then repeats at the specified interval.
    visibility_interval = 10

    agent_num = 200
    concurrency = 100

    for visibility_start in visibility_start_list:
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
                                                      visibility_extent, visibility_start, visibility_interval,
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

        # Save results across K for each visibility start and visibility interval.
        with open("visibility_start_{0}_interval_{1}_breakthrough_fitness_across_K_size_{2}".format(
                visibility_start, visibility_interval, agent_num), 'wb') as out_file:
            pickle.dump(breakthrough_fitness_across_K, out_file)

        with open("visibility_start_{0}_interval_{1}_breakthrough_rank_across_K_size_{2}".format(
                visibility_start, visibility_interval, agent_num), 'wb') as out_file:
            pickle.dump(breakthrough_rank_across_K, out_file)

        with open("visibility_start_{0}_interval_{1}_unique_solution_count_across_K_size_{2}".format(
                visibility_start, visibility_interval, agent_num), 'wb') as out_file:
            pickle.dump(unique_solution_count_across_K, out_file)

        with open("visibility_start_{0}_interval_{1}_pairwise_diversity_across_K_size_{2}".format(
                visibility_start, visibility_interval, agent_num), 'wb') as out_file:
            pickle.dump(pairwise_diversity_across_K, out_file)

        with open("visibility_start_{0}_interval_{1}_adopted_solution_fitness_across_K_size_{2}".format(
                visibility_start, visibility_interval, agent_num), 'wb') as out_file:
            pickle.dump(adopted_solution_fitness_across_K, out_file)

        with open("visibility_start_{0}_interval_{1}_adopted_solution_fitness_across_time_across_K_size_{2}".format(
                visibility_start, visibility_interval, agent_num), 'wb') as out_file:
            pickle.dump(adopted_solution_fitness_across_time_across_K, out_file)

    t1 = time.time()
    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))
    print("Visibility Timing with Interval {0}: ".format(visibility_interval),
          time.strftime("%H:%M:%S", time.gmtime(t1 - t0)))
