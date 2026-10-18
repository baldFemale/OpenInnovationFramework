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
         uniform_sharing_prob=None, maturity_threshold=None, visibility_interval=10, solver_reset_num=1,
         loop=None, return_dict=None, sema=None):
    """
    Maturity-based visibility experiment with separated sender and receiver crowds.

    Both crowds conduct ordinary local search every period. A sender discloses
    its complete current solution every visibility_interval periods when
    a fresh random draw is below uniform_sharing_prob and its cog_fitness
    is at least maturity_threshold.

    Maturity is evaluated using the sender's perceived fitness. Receivers
    evaluate disclosed solutions using their own bounded knowledge.
    Receiver solutions do not feed back into the sender pool.

    uniform_sharing_prob is fixed across conditions; maturity_threshold is the
    focal running parameter. All dependent variables concern the receiver crowd.
    Adoption-quality trajectories contain one cumulative mean per visibility
    interval, including np.nan until the first adoption.

    solver_reset_num controls how many independent sender/receiver crowds
    are generated and run on the same landscape. The landscape is generated
    only once per landscape repetition.
    """
    try:
        np.random.seed(None)

        if visibility_interval is None:
            visibility_interval = 10
        visibility_interval = int(visibility_interval)
        if visibility_interval < 1:
            raise ValueError("visibility_interval must be a positive integer.")

        solver_reset_num = int(solver_reset_num)
        if solver_reset_num < 1:
            raise ValueError("solver_reset_num must be a positive integer.")

        # Generate the landscape once, then restart the crowds on the same landscape.
        landscape = Landscape(N=N, K=K)
        solver_reset_returns = []

        for solver_reset in range(solver_reset_num):
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

                if (period + 1) % visibility_interval == 0:
                    # Disclose only sender solutions meeting the perceived maturity threshold.
                    crowd_sender.solution_pool = []
                    for sender_id, agent in enumerate(crowd_sender.agents):
                        if (np.random.uniform(0, 1) < uniform_sharing_prob) and (agent.cog_fitness >= maturity_threshold):
                            crowd_sender.solution_pool.append(agent.state.copy())

                    np.random.shuffle(crowd_sender.solution_pool)

                    # Receiver solutions do not feed back into the sender pool.
                    crowd_receiver.solution_pool = [
                        solution.copy() for solution in crowd_sender.solution_pool
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

            solver_reset_returns.append([
                breakthrough_fitness,
                breakthrough_rank,
                unique_solution_count,
                pairwise_diversity,
                adopted_solution_fitness,
                adopted_solution_fitness_across_time,
            ])

        # One landscape repetition returns solver_reset_num sets of outcomes.
        return_dict[loop] = solver_reset_returns
    finally:
        sema.release()


if __name__ == '__main__':
    import datetime
    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))
    t0 = time.time()

    landscape_iteration = 600
    solver_reset_num = 10  # Fresh sender/receiver crowds per landscape. Set to 1 for the original design.
    search_iteration = 300

    # Binary NK landscape: 2^18 = 4^9, matching the old problem-space size.
    N = 18
    K_list = list(range(9, N))  # K = 9, 10, ..., 17; Upper-complexity half

    # 12/18 = 6/9 = 2/3, matching the old generalists' knowledge coverage.
    # The expected pairwise knowledge overlap is 12^2 / 18 = 8 dimensions.
    knowledge_breadth = 12

    # Sharing probability is fixed; maturity_threshold is the focal running parameter.
    uniform_sharing_prob = 1

    # Visibility frequency: sender solutions become visible every x periods.
    visibility_interval = 10

    # Minimum sender perceived fitness required for disclosure.
    maturity_threshold_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.725, 0.75, 0.775, 0.8, 0.825, 0.85, 0.875, 0.9]

    # Restore the original crowd size of 200 agents per crowd.
    agent_num = 100
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
                    # Close completed processes as the experiment runs.
                    # Otherwise, keeping all 600 Process objects open can
                    # exhaust the operating system's file-descriptor limit.
                    while True:
                        acquired = sema.acquire(timeout=0.5)

                        for index, proc in jobs[:]:
                            if proc.exitcode is not None:
                                proc.join()
                                exitcode = proc.exitcode
                                proc.close()
                                jobs.remove((index, proc))

                                if exitcode != 0:
                                    raise RuntimeError(
                                        "Repetition {0} failed with exit code {1}".format(index, exitcode)
                                    )

                        if acquired:
                            break

                    p = mp.Process(target=func, args=(N, K, agent_num, knowledge_breadth, search_iteration,
                                                      uniform_sharing_prob, maturity_threshold, visibility_interval,
                                                      solver_reset_num, loop, return_dict, sema))
                    p.start()
                    jobs.append((loop, p))

                # Collect and close the processes that are still running.
                while jobs:
                    for index, proc in jobs[:]:
                        if proc.exitcode is not None:
                            proc.join()
                            exitcode = proc.exitcode
                            proc.close()
                            jobs.remove((index, proc))

                            if exitcode != 0:
                                raise RuntimeError(
                                    "Repetition {0} failed with exit code {1}".format(index, exitcode)
                                )

                    if jobs:
                        time.sleep(0.5)

                if len(return_dict) != landscape_iteration:
                    raise RuntimeError("Missing results: expected {0}, received {1}".format(
                        landscape_iteration, len(return_dict)))

                landscape_returns = [return_dict[index] for index in range(landscape_iteration)]
            finally:
                for index, proc in jobs:
                    if proc.is_alive():
                        proc.terminate()
                for index, proc in jobs:
                    proc.join()
                    proc.close()
                manager.shutdown()

            # Flatten the two repetition layers into individual runs for averaging.
            # Each landscape contributes the same number of solver resets.
            returns = [item for landscape_results in landscape_returns for item in landscape_results]
            if len(returns) != landscape_iteration * solver_reset_num:
                raise RuntimeError("Missing solver resets: expected {0}, received {1}".format(
                    landscape_iteration * solver_reset_num, len(returns)))

            arr = np.asarray([item[:5] for item in returns], dtype=float)  # shape: (landscape_iteration * solver_reset_num, 5)
            time_series_arr = np.asarray([item[5] for item in returns], dtype=float)

            # The first four DVs are always defined. The visibility-process
            # measure can be np.nan when a run contains no adoption.
            means = arr[:, :4].mean(axis=0)

            adopted_solution_fitness_mean = (
                np.nan
                if np.all(np.isnan(arr[:, 4]))
                else np.nanmean(arr[:, 4])
            )

            # Average the cumulative adoption-quality trajectory across landscapes and solver resets.
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
