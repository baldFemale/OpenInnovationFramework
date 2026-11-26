#!/usr/bin/env py39
# -*- coding: utf-8 -*-
# @Author   : Junyi
# @FileName: Visibility_rank_selection_strength.py
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
         visibility_extent=None, rank_selection_strength=None, recommendation_num=10,
         visibility_interval=10, solver_reset_num=1,
         loop=None, return_dict=None, sema=None,
         generative_horizon=30, generative_replications=30,
         generative_sample_size=5, generative_seed=20261119):
    """
    Performance-based selection with separated sender and recipient crowds.

    visibility_extent = 1.0 makes all senders eligible for rule-based selection.
    At each visibility event, all recipients receive one common pool of
    recommendation_num sender entries (10 by default). A fraction is selected by true NK landscape fitness
    rank (lower numerical ranks are better), and the remainder comes from a
    common random ordering of sender identities fixed at each crowd reset.
    Ranked entries are skipped when filling the remaining slots, without duplicates.

        rank_selection_strength = 0.0: the same random senders throughout a run.
        rank_selection_strength = 1.0: the highest-performing senders at each event.

    Both crowds conduct local search every period. Every visibility_interval
    periods, sender ranks are recomputed after local search. Fitness-rank ties
    are broken randomly once per event; the common recommendation pool is
    shuffled once per event and evaluated in the same order by all recipients.
    Recipients adopt at most the first perceived improvement, without feedback
    to senders. Search-apart (zero exposure) is not part of this experiment.

    All DVs concern the recipient crowd. Adoption-fitness trajectories contain
    a cumulative mean at each visibility event, or np.nan before any adoption.
    solver_reset_num reruns independent crowds on each shared NK landscape.
    """
    try:
        np.random.seed(None)

        if visibility_interval is None:
            visibility_interval = 10
        visibility_interval = int(visibility_interval)
        if visibility_interval < 1:
            raise ValueError("visibility_interval must be a positive integer.")
        if rank_selection_strength is None or not 0 <= rank_selection_strength <= 1:
            raise ValueError("rank_selection_strength must be between 0 and 1.")
        if visibility_extent != 1.0:
            raise ValueError("visibility_extent must equal 1.0 for this selection experiment.")

        if not 1 <= recommendation_num <= agent_num:
            raise ValueError("recommendation_num must be between 1 and agent_num.")
        rank_selection_num = int(round(rank_selection_strength * recommendation_num))
        random_selection_num = recommendation_num - rank_selection_num

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
            # Separate measurement RNG preserves the original search trajectory.
            generative_rng = np.random.default_rng(
                [generative_seed, K, loop if loop is not None else 0, solver_reset]
            )
            crowd_receiver = Crowd(N=N, agent_num=agent_num, knowledge_breadth=knowledge_breadth,
                                   landscape=landscape, generative_sample_size=generative_sample_size,
                                   generative_rng=generative_rng)

            crowd_sender.set_visibility_status(visibility_extent=visibility_extent)

            # Fix a common random ordering of sender identities for this run.
            # The first recommendation_num senders form the s = 0 baseline.
            eligible_senders = [agent for agent in crowd_sender.agents
                                if agent.visibility_status]
            if len(eligible_senders) < recommendation_num:
                raise ValueError("Eligible sender pool is smaller than recommendation_num.")
            fixed_random_order = np.random.permutation(len(eligible_senders))

            # Search-process trajectory: cumulative mean true fitness of adopted solutions.
            adopted_solution_fitness_across_time = []

            for period in range(search_iteration):
                # Both crowds conduct their own independent search.
                crowd_sender.search()
                crowd_receiver.search()

                if (period + 1) % visibility_interval == 0:
                    # Refresh solutions, not the identities or fixed random ordering.
                    sender_pool = [agent.state.copy() for agent in eligible_senders]

                    # Construct one common recommendation pool per visibility event.
                    # Lower true NK fitness ranks indicate better solutions.
                    if rank_selection_num > 0:
                        fitness_ranks = np.asarray([
                            landscape.query_fitness_rank(state=state) for state in sender_pool
                        ])
                        # Break fitness-rank ties randomly once for all recipients.
                        candidate_indices = np.random.permutation(len(sender_pool))
                        candidate_indices = candidate_indices[np.argsort(
                            fitness_ranks[candidate_indices], kind='stable'
                        )]
                        selected_indices = candidate_indices[:rank_selection_num].copy()
                    else:
                        selected_indices = np.empty(0, dtype=int)

                    # Fill non-ranked slots from the same fixed random ordering.
                    # Skip ranked senders to preserve the recommendation count.
                    if random_selection_num > 0:
                        remaining_indices = fixed_random_order[
                            ~np.isin(fixed_random_order, selected_indices)
                        ]
                        random_indices = remaining_indices[:random_selection_num]
                        selected_indices = np.concatenate((selected_indices, random_indices))

                    if len(selected_indices) != recommendation_num:
                        raise RuntimeError("Recommendation count must remain exactly recommendation_num.")

                    # Shuffle once per event, not once per recipient, as in the extent experiment.
                    # Selection determines pool membership, not evaluation order.
                    np.random.shuffle(selected_indices)
                    crowd_receiver.solution_pool = [
                        sender_pool[index].copy() for index in selected_indices
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

            # Independent post-run continuations measure recipient-specific potential.
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

    # Generative potential: independent local-search continuations after adoption.
    generative_horizon = 30  # Number of additional search opportunities per continuation.
    generative_replications = 30  # Monte Carlo continuations per sampled adoption.
    generative_sample_size = 5  # Uniformly sampled adoptions per run (all if fewer).
    generative_seed = 20261119  # Independent measurement RNG; original search is unchanged.

    # Binary NK landscape: 2^18 = 4^9, matching the old problem-space size.
    N = 18
    K_list = list(range(9, N))  # K = 9, 10, ..., 17; Upper-complexity half

    # 12/18 = 6/9 = 2/3, matching the old generalists' knowledge coverage.
    # The expected pairwise knowledge overlap is 12^2 / 18 = 8 dimensions.
    knowledge_breadth = 12

    # All sender paths remain eligible for rule-based selection.
    visibility_extent = 1.0

    # With 100 senders, 10 recommendations correspond to effective extent 0.1.
    recommendation_num = 10
    # Select this fraction by true fitness rank; fill the rest from the fixed order.
    # Strength 0.0 reproduces the fixed-sender visibility design (not search-apart).
    rank_selection_strength_list = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    # Sender solutions are disclosed every visibility_interval periods.
    visibility_interval = 10

    # Sender ranks, selected pool, and evaluation order are shared across recipients.
    agent_num = 100
    concurrency = 100

    for rank_selection_strength in rank_selection_strength_list:
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
                                                      visibility_extent, rank_selection_strength, recommendation_num,
                                                      visibility_interval, solver_reset_num,
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

            # Equal-weight mean across runs, ignoring runs without adoptions.
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

        # Save existing DVs plus generative potential for each selection condition.
        filename_prefix = "visibility_rank_selection_fixed_random_strength_{0}_recommendation_num_{1}_interval_{2}".format(
            rank_selection_strength, recommendation_num, visibility_interval
        )

        with open("{0}_breakthrough_fitness_across_K_size_{1}".format(
                filename_prefix, agent_num), 'wb') as out_file:
            pickle.dump(breakthrough_fitness_across_K, out_file)

        with open("{0}_breakthrough_rank_across_K_size_{1}".format(
                filename_prefix, agent_num), 'wb') as out_file:
            pickle.dump(breakthrough_rank_across_K, out_file)

        with open("{0}_unique_solution_count_across_K_size_{1}".format(
                filename_prefix, agent_num), 'wb') as out_file:
            pickle.dump(unique_solution_count_across_K, out_file)

        with open("{0}_pairwise_diversity_across_K_size_{1}".format(
                filename_prefix, agent_num), 'wb') as out_file:
            pickle.dump(pairwise_diversity_across_K, out_file)

        with open("{0}_adopted_solution_fitness_across_K_size_{1}".format(
                filename_prefix, agent_num), 'wb') as out_file:
            pickle.dump(adopted_solution_fitness_across_K, out_file)

        with open("{0}_adopted_solution_fitness_across_time_across_K_size_{1}".format(
                filename_prefix, agent_num), 'wb') as out_file:
            pickle.dump(adopted_solution_fitness_across_time_across_K, out_file)

        with open("{0}_generative_potential_across_K_size_{1}".format(
                filename_prefix, agent_num), 'wb') as out_file:
            pickle.dump(generative_potential_across_K, out_file)

    t1 = time.time()
    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))
    print("Rank-Based Selection with {0} Recommendations and Interval {1}: ".format(
        recommendation_num, visibility_interval),
          time.strftime("%H:%M:%S", time.gmtime(t1 - t0)))
