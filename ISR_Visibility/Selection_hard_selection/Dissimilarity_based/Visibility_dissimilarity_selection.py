#!/usr/bin/env py39
# -*- coding: utf-8 -*-
# @Author   : Junyi
# @FileName: Visibility_dissimilarity_selection_hard.py
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
         visibility_extent=None, visibility_interval=10, solver_reset_num=1,
         loop=None, return_dict=None, sema=None,
         generative_horizon=30, generative_replications=30,
         generative_sample_size=5, generative_seed=20261119):
    """
    Dissimilarity-based hard selection with separated sender and receiver crowds.

    At each visibility event, each recipient receives the most distant proportion
    (visibility_extent) of all senders' current complete solutions, measured
    by Hamming distance to the recipient's current complete configuration.
    The number of visible sender entries is fixed, but their identities can
    vary across recipients and visibility events. No fixed visible subset is
    imposed. All sender solutions are eligible, not necessarily accessible.

    A common initial shuffle breaks distance ties at the selection cutoff.
    An independent common random sender order determines evaluation order
    within each selected subset, so dissimilarity determines membership only.
    Recipients adopt at most the first perceived improvement. At full
    visibility, all recipients evaluate the same randomly ordered full pool.

    All dependent variables and aggregation procedures match the extent
    experiment. No selection-strength or recommendation-count parameter is used.
    """
    try:
        np.random.seed(None)

        if visibility_interval is None:
            visibility_interval = 10
        visibility_interval = int(visibility_interval)
        if visibility_interval < 1:
            raise ValueError("visibility_interval must be a positive integer.")

        if visibility_extent is None or not 0 <= visibility_extent <= 1:
            raise ValueError("visibility_extent must be between 0 and 1.")
        visible_num = int(round(visibility_extent * agent_num))

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
            # Separate RNG: generative measurement cannot change search trajectories.
            generative_rng = np.random.default_rng(
                [generative_seed, K, loop if loop is not None else 0, solver_reset]
            )
            crowd_receiver = Crowd(N=N, agent_num=agent_num, knowledge_breadth=knowledge_breadth,
                                   landscape=landscape, generative_sample_size=generative_sample_size,
                                   generative_rng=generative_rng)

            # Search-process trajectory: cumulative mean true fitness of adopted solutions.
            adopted_solution_fitness_across_time = []

            for period in range(search_iteration):
                # Both crowds conduct their own independent search.
                crowd_sender.search()
                crowd_receiver.search()

                if (period + 1) % visibility_interval == 0:
                    # Snapshot all candidate solutions, not just visible ones.
                    # Personalized visibility cannot be represented by a single
                    # sender-level visibility_status flag.
                    if visible_num > 0:
                        sender_pool = [agent.state.copy() for agent in crowd_sender.agents]
                        # Common initial shuffle for randomized cutoff ties.
                        np.random.shuffle(sender_pool)

                        if visible_num < agent_num:
                            sender_states = np.asarray(sender_pool, dtype=np.uint8)
                            recipient_states = np.asarray(
                                [agent.state for agent in crowd_receiver.agents], dtype=np.uint8
                            )
                            distances = np.count_nonzero(
                                recipient_states[:, None, :] != sender_states[None, :, :], axis=2
                            )

                            # Hard selection: keep only the farthest visible_num entries.
                            selected_indices = np.argsort(
                                -distances, axis=1, kind='stable'
                            )[:, :visible_num]

                            # Independently randomize evaluation priority once per event.
                            # Recipients share the random relative order of overlapping senders.
                            # Dissimilarity determines membership, not evaluation order.
                            random_priority = np.random.permutation(agent_num)
                            evaluation_orders = np.take_along_axis(
                                selected_indices,
                                np.argsort(random_priority[selected_indices], axis=1),
                                axis=1
                            )
                        else:
                            # At full visibility, all recipients receive the same
                            # randomly ordered full pool, as in the extent experiment.
                            evaluation_orders = np.broadcast_to(
                                np.arange(agent_num), (len(crowd_receiver.agents), agent_num)
                            )

                        for recipient_index, recipient in enumerate(crowd_receiver.agents):
                            for index in evaluation_orders[recipient_index]:
                                if recipient.search(solution=sender_pool[index]):
                                    crowd_receiver.adopted_solution_fitness_history.append(recipient.fitness)
                                    if crowd_receiver.generative_sample_size:
                                        # Uniformly sample successful adoptions without affecting search.
                                        count = len(crowd_receiver.adopted_solution_fitness_history)
                                        event = (recipient, recipient.state.copy())
                                        if len(crowd_receiver.generative_samples) < crowd_receiver.generative_sample_size:
                                            crowd_receiver.generative_samples.append(event)
                                        else:
                                            sample_index = crowd_receiver.generative_rng.integers(count)
                                            if sample_index < crowd_receiver.generative_sample_size:
                                                crowd_receiver.generative_samples[sample_index] = event
                                    break

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

            # Post-run independent continuations do not modify the observed paths.
            generative_potential = crowd_receiver.calculate_generative_potential(
                horizon=generative_horizon, replications=generative_replications
            )

            solver_reset_returns.append([
                breakthrough_fitness,
                breakthrough_rank,
                unique_solution_count,
                pairwise_diversity,
                adopted_solution_fitness,
                adopted_solution_fitness_across_time,
                generative_potential,
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

    landscape_iteration = 400
    solver_reset_num = 5  # Fresh sender/receiver crowds per landscape. Set to 1 for the original design.
    search_iteration = 300

    # Generative potential: H independent local-search opportunities per continuation.
    generative_horizon = 30
    generative_replications = 30  # Independent Monte Carlo continuations per sampled event.
    generative_sample_size = 5  # Uniformly sampled adoptions per run (all if fewer).
    generative_seed = 20261119  # Dedicated RNG; does not seed the original simulation.

    # Binary NK landscape: 2^18 = 4^9, matching the old problem-space size.
    N = 18
    K_list = list(range(9, N))  # K = 9, 10, ..., 17; Upper-complexity half

    # 12/18 = 6/9 = 2/3, matching the old generalists' knowledge coverage.
    # The expected pairwise knowledge overlap is 12^2 / 18 = 8 dimensions.
    knowledge_breadth = 12

    # Match the extent and rank-based hard-selection grids. At v=0.01,
    # the farthest sender is selected dynamically, so this case must be run.
    # At v=0, reuse search-apart results; retain v=1 as a boundary check.
    visibility_extent_list = [0.01, 0.02, 0.04, 0.08,
                              0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # Visibility frequency: sender solutions become visible every x periods.
    visibility_interval = 10

    # With 100 senders, extent 0.01 represents one visible sender.
    agent_num = 100

    # Limit simultaneous workers because high-K landscapes may be memory-heavy.
    # Increase this only if the compute node has sufficient available memory.
    concurrency = 100

    for visibility_extent in visibility_extent_list:
        # DVs
        breakthrough_fitness_across_K = []
        breakthrough_rank_across_K = []
        unique_solution_count_across_K = []
        pairwise_diversity_across_K = []
        adopted_solution_fitness_across_K = []
        adopted_solution_fitness_across_time_across_K = []
        generative_potential_across_K = []

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
                                                      visibility_extent, visibility_interval, solver_reset_num,
                                                      loop, return_dict, sema),
                                   kwargs=dict(generative_horizon=generative_horizon,
                                               generative_replications=generative_replications,
                                               generative_sample_size=generative_sample_size,
                                               generative_seed=generative_seed))
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
            generative_potential_arr = np.asarray([item[6] for item in returns], dtype=float)

            # The first four DVs are always defined. The visibility-process
            # measure can be np.nan when a run contains no adoption.
            means = arr[:, :4].mean(axis=0)

            adopted_solution_fitness_mean = (
                np.nan
                if np.all(np.isnan(arr[:, 4]))
                else np.nanmean(arr[:, 4])
            )
            # Same aggregation as adopted quality: event mean within each run,
            # then equal-weight mean across runs with at least one adoption.
            generative_potential_mean = (
                np.nan if np.all(np.isnan(generative_potential_arr))
                else np.nanmean(generative_potential_arr)
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
            generative_potential_across_K.append(generative_potential_mean)

        # Save dissimilarity-based hard-selection results using the extent experiment's metric suffixes.
        with open("visibility_dissimilarity_selection_hard_extent_{0}_interval_{1}_breakthrough_fitness_across_K_size_{2}".format(
                visibility_extent, visibility_interval, agent_num), 'wb') as out_file:
            pickle.dump(breakthrough_fitness_across_K, out_file)

        with open("visibility_dissimilarity_selection_hard_extent_{0}_interval_{1}_breakthrough_rank_across_K_size_{2}".format(
                visibility_extent, visibility_interval, agent_num), 'wb') as out_file:
            pickle.dump(breakthrough_rank_across_K, out_file)

        with open("visibility_dissimilarity_selection_hard_extent_{0}_interval_{1}_unique_solution_count_across_K_size_{2}".format(
                visibility_extent, visibility_interval, agent_num), 'wb') as out_file:
            pickle.dump(unique_solution_count_across_K, out_file)

        with open("visibility_dissimilarity_selection_hard_extent_{0}_interval_{1}_pairwise_diversity_across_K_size_{2}".format(
                visibility_extent, visibility_interval, agent_num), 'wb') as out_file:
            pickle.dump(pairwise_diversity_across_K, out_file)

        with open("visibility_dissimilarity_selection_hard_extent_{0}_interval_{1}_adopted_solution_fitness_across_K_size_{2}".format(
                visibility_extent, visibility_interval, agent_num), 'wb') as out_file:
            pickle.dump(adopted_solution_fitness_across_K, out_file)

        with open("visibility_dissimilarity_selection_hard_extent_{0}_interval_{1}_adopted_solution_fitness_across_time_across_K_size_{2}".format(
                visibility_extent, visibility_interval, agent_num), 'wb') as out_file:
            pickle.dump(adopted_solution_fitness_across_time_across_K, out_file)

        with open("visibility_dissimilarity_selection_hard_extent_{0}_interval_{1}_generative_potential_across_K_size_{2}".format(
                visibility_extent, visibility_interval, agent_num), 'wb') as out_file:
            pickle.dump(generative_potential_across_K, out_file)

    t1 = time.time()
    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))
    print("Dissimilarity-Based Hard Selection across Visibility Extent with Interval {0}: ".format(visibility_interval),
          time.strftime("%H:%M:%S", time.gmtime(t1 - t0)))
