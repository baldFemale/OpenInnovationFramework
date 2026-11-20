#!/usr/bin/env py39
# -*- coding: utf-8 -*-
# @Author   : Junyi
# @FileName: Visibility_extent_bidirectional.py
# @Software : PyCharm
# Observing PEP 8 coding style

import csv
import multiprocessing as mp
import pickle
import time
from multiprocessing import Semaphore

import numpy as np
from Landscape import Landscape
from Crowd import Crowd


# False reproduces the sender -> receiver mechanism.
# True adds the receiver -> sender mechanism, without changing the solver rules.
BIDIRECTIONAL_VISIBILITY_MODES = (False, True)

# Original seven outcomes (the first seven positions in every run record).
OUTCOME_NAMES = (
    'breakthrough_fitness',
    'breakthrough_rank',
    'unique_solution_count',
    'pairwise_diversity',
    'adopted_solution_fitness',
    'adopted_solution_fitness_across_time',
    'generative_potential',
)


def exchange_visible_solutions(crowd_sender, crowd_receiver,
                               bidirectional_visibility=False, reverse_rng=None):
    """Exchange pre-exposure snapshots; no same-event feedback or cascades.

    The visible sender pool is constructed exactly as in the original model.
    In the extension the visible receiver pool is constructed independently,
    before *either* crowd learns. Both crowds then apply the original
    first-strictly-improving-candidate adoption rule to the opposite pool.
    """
    crowd_sender.get_visible_pool()
    sender_snapshot = [solution.copy() for solution in crowd_sender.solution_pool]

    if bidirectional_visibility:
        crowd_receiver.get_visible_pool(rng=reverse_rng)
        receiver_snapshot = [solution.copy() for solution in crowd_receiver.solution_pool]

    # Assign all exposure pools before allowing either population to adopt.
    crowd_receiver.solution_pool = sender_snapshot
    if bidirectional_visibility:
        crowd_sender.solution_pool = receiver_snapshot

    crowd_receiver.learn_from_visible_pool()
    if bidirectional_visibility:
        crowd_sender.learn_from_visible_pool()


def func(N=None, K=None, agent_num=None, knowledge_breadth=None,
         search_iteration=None, visibility_extent=None, visibility_interval=10,
         solver_reset_num=1, loop=None, return_dict=None, sema=None,
         generative_horizon=30, generative_replications=30,
         generative_sample_size=5, generative_seed=20261119,
         reverse_visibility_seed=20261120, experiment_seed=None,
         bidirectional_visibility_modes=BIDIRECTIONAL_VISIBILITY_MODES):
    """Run paired one-way and two-way simulations on each NK landscape.

    Sender/receiver crowds both contain agent_num conventional solvers.
    On each landscape, both visibility modes start from exactly the same
    solver knowledge, initial states, and local-search random-number draws.

    In both modes visibility_extent selects a fixed proportion of senders.
    In bidirectional mode the same fraction of receivers is also visible.
    Reverse visibility sampling and pool shuffling use a dedicated RNG, so
    adding that sampling does not perturb original local-search randomness.

    Results are measured among receivers; sender adoption counts are saved
    only as a diagnostic for reverse information propagation.
    """
    try:
        if experiment_seed is None:
            np.random.seed(None)  # Original stochastic landscape initialization.
        else:
            # A reproducible but distinct random landscape for each (K, loop).
            landscape_seed = np.random.SeedSequence(
                [experiment_seed, N, K, loop if loop is not None else 0]
            ).generate_state(1)[0]
            np.random.seed(int(landscape_seed))

        if visibility_interval is None:
            visibility_interval = 10
        visibility_interval = int(visibility_interval)
        if visibility_interval < 1:
            raise ValueError('visibility_interval must be a positive integer.')

        solver_reset_num = int(solver_reset_num)
        if solver_reset_num < 1:
            raise ValueError('solver_reset_num must be a positive integer.')

        modes = tuple(bidirectional_visibility_modes)
        if not modes or len(set(modes)) != len(modes) or any(
                type(mode) is not bool for mode in modes):
            raise ValueError('Visibility modes must be a nonempty sequence of distinct booleans.')

        # One landscape per worker/repetition; shared across both modes and resets.
        landscape = Landscape(N=N, K=K)
        solver_reset_seeds = np.random.randint(
            0, 2 ** 32, size=solver_reset_num, dtype=np.uint32
        )
        solver_reset_returns = {mode: [] for mode in modes}

        for solver_reset, reset_seed in enumerate(solver_reset_seeds):
            for bidirectional_visibility in modes:
                # Paired initial conditions and common random numbers across modes.
                np.random.seed(int(reset_seed))
                crowd_sender = Crowd(
                    N=N, agent_num=agent_num, knowledge_breadth=knowledge_breadth,
                    landscape=landscape
                )
                generative_rng = np.random.default_rng(
                    [generative_seed, K, loop if loop is not None else 0, solver_reset]
                )
                crowd_receiver = Crowd(
                    N=N, agent_num=agent_num, knowledge_breadth=knowledge_breadth,
                    landscape=landscape, generative_sample_size=generative_sample_size,
                    generative_rng=generative_rng
                )

                # Original outward visibility assignment (global search RNG).
                crowd_sender.set_visibility_status(visibility_extent=visibility_extent)

                if bidirectional_visibility:
                    # Independently select visible receivers. The number visible
                    # equals the number of visible senders (same extent and size).
                    reverse_rng = np.random.default_rng(
                        [reverse_visibility_seed, K,
                         loop if loop is not None else 0, solver_reset]
                    )
                    crowd_receiver.set_visibility_status(
                        visibility_extent=visibility_extent, rng=reverse_rng
                    )
                else:
                    reverse_rng = None

                adopted_solution_fitness_across_time = []

                for period in range(search_iteration):
                    # Existing independent local-search moves, in original order.
                    crowd_sender.search()
                    crowd_receiver.search()

                    if (period + 1) % visibility_interval == 0:
                        exchange_visible_solutions(
                            crowd_sender=crowd_sender,
                            crowd_receiver=crowd_receiver,
                            bidirectional_visibility=bidirectional_visibility,
                            reverse_rng=reverse_rng
                        )
                        adopted_solution_fitness_across_time.append(
                            np.mean(crowd_receiver.adopted_solution_fitness_history)
                            if crowd_receiver.adopted_solution_fitness_history else np.nan
                        )

                # The seven existing receiver-only outcome measures are unchanged.
                performance_list = [agent.fitness for agent in crowd_receiver.agents]
                fitness_rank_list = [
                    landscape.query_fitness_rank(state=agent.state)
                    for agent in crowd_receiver.agents
                ]
                breakthrough_fitness = max(performance_list)
                breakthrough_rank = min(fitness_rank_list)  # Rank 1 is best.

                full_solution_set = set()
                for agent in crowd_receiver.agents:
                    solution_str = ''.join([str(bit) for bit in agent.state])
                    full_solution_set.add(solution_str)
                unique_solution_count = len(full_solution_set)

                pairwise_diversity = crowd_receiver.calculate_pairwise_solution_distance()
                adopted_solution_fitness = (
                    np.mean(crowd_receiver.adopted_solution_fitness_history)
                    if crowd_receiver.adopted_solution_fitness_history else np.nan
                )
                generative_potential = crowd_receiver.calculate_generative_potential(
                    horizon=generative_horizon, replications=generative_replications
                )

                solver_reset_returns[bidirectional_visibility].append([
                    breakthrough_fitness,
                    breakthrough_rank,
                    unique_solution_count,
                    pairwise_diversity,
                    adopted_solution_fitness,
                    adopted_solution_fitness_across_time,
                    generative_potential,
                    len(crowd_receiver.adopted_solution_fitness_history),
                    len(crowd_sender.adopted_solution_fitness_history),
                ])

        return_dict[loop] = solver_reset_returns
    finally:
        if sema is not None:
            sema.release()


def aggregate_condition(returns):
    """Preserve the original across-run averaging for all seven outcomes."""
    arr = np.asarray([item[:5] for item in returns], dtype=float)
    time_series_arr = np.asarray([item[5] for item in returns], dtype=float)
    generative_potential_arr = np.asarray([item[6] for item in returns], dtype=float)
    means = arr[:, :4].mean(axis=0)

    adopted_mean = (
        np.nan if np.all(np.isnan(arr[:, 4])) else np.nanmean(arr[:, 4])
    )
    generative_mean = (
        np.nan if np.all(np.isnan(generative_potential_arr))
        else np.nanmean(generative_potential_arr)
    )

    # Use per-column available-case means, preserving NaN when no adoptions.
    if np.all(np.isnan(time_series_arr)):
        trajectory = time_series_arr[0].tolist()
    else:
        with np.errstate(invalid='ignore'):
            valid_counts = np.sum(~np.isnan(time_series_arr), axis=0)
            trajectory = np.divide(
                np.nansum(time_series_arr, axis=0),
                valid_counts,
                out=np.full(time_series_arr.shape[1], np.nan),
                where=valid_counts > 0
            ).tolist()

    return dict(zip(OUTCOME_NAMES, [
        means[0], means[1], means[2], means[3], adopted_mean,
        trajectory, generative_mean
    ]))


def main():
    import datetime
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    t0 = time.time()

    # Original experiment settings; each landscape is evaluated under both modes.
    landscape_iteration = 400
    solver_reset_num = 5
    search_iteration = 300

    generative_horizon = 30
    generative_replications = 30
    generative_sample_size = 5
    generative_seed = 20261119

    N = 18
    K_list = list(range(9, N))
    knowledge_breadth = 12
    agent_num = 100  # 100 senders and 100 receivers; each pool can have 0..100 solutions.
    visibility_extent_list = [0.0, 0.01, 0.02, 0.04, 0.08,
                              0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    visibility_interval = 10
    concurrency = 100

    # Reproducible landscapes, matched solver initialization and search draws.
    # Use None to restore the original nonreproducible landscape initialization.
    experiment_seed = 20261120
    reverse_visibility_seed = 20261120
    bidirectional_visibility_modes = BIDIRECTIONAL_VISIBILITY_MODES

    for visibility_extent in visibility_extent_list:
        results_across_K = {
            mode: {outcome: [] for outcome in OUTCOME_NAMES}
            for mode in bidirectional_visibility_modes
        }

        # Run-level records enable paired contrasts and mode-by-extent regression.
        raw_filename = (
            'visibility_extent_{0}_interval_{1}_bidirectional_comparison_size_{2}_runs.csv'
            .format(visibility_extent, visibility_interval, agent_num)
        )
        with open(raw_filename, 'w', newline='') as raw_file:
            writer = csv.writer(raw_file)
            writer.writerow([
                'visibility_extent', 'visibility_interval', 'K',
                'landscape_replication', 'solver_reset', 'bidirectional_visibility',
                'breakthrough_fitness', 'breakthrough_rank',
                'unique_solution_count', 'pairwise_diversity',
                'adopted_solution_fitness', 'generative_potential',
                'receiver_adoption_count', 'sender_adoption_count',
            ])

            for K in K_list:
                manager = mp.Manager()
                return_dict = manager.dict()
                sema = Semaphore(concurrency)
                jobs = []

                try:
                    for loop in range(landscape_iteration):
                        # Reap exited workers throughout the run to avoid OS FD limits.
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
                                            'Repetition {0} failed with exit code {1}'
                                            .format(index, exitcode)
                                        )

                            if acquired:
                                break

                        p = mp.Process(
                            target=func,
                            args=(N, K, agent_num, knowledge_breadth,
                                  search_iteration, visibility_extent,
                                  visibility_interval, solver_reset_num,
                                  loop, return_dict, sema),
                            kwargs=dict(
                                generative_horizon=generative_horizon,
                                generative_replications=generative_replications,
                                generative_sample_size=generative_sample_size,
                                generative_seed=generative_seed,
                                reverse_visibility_seed=reverse_visibility_seed,
                                experiment_seed=experiment_seed,
                                bidirectional_visibility_modes=bidirectional_visibility_modes,
                            )
                        )
                        p.start()
                        jobs.append((loop, p))

                    while jobs:
                        for index, proc in jobs[:]:
                            if proc.exitcode is not None:
                                proc.join()
                                exitcode = proc.exitcode
                                proc.close()
                                jobs.remove((index, proc))
                                if exitcode != 0:
                                    raise RuntimeError(
                                        'Repetition {0} failed with exit code {1}'
                                        .format(index, exitcode)
                                    )
                        if jobs:
                            time.sleep(0.5)

                    if len(return_dict) != landscape_iteration:
                        raise RuntimeError('Missing results: expected {0}, received {1}'.format(
                            landscape_iteration, len(return_dict)
                        ))
                    landscape_returns = [
                        return_dict[index] for index in range(landscape_iteration)
                    ]
                finally:
                    for index, proc in jobs:
                        if proc.is_alive():
                            proc.terminate()
                    for index, proc in jobs:
                        proc.join()
                        proc.close()
                    manager.shutdown()

                for bidirectional_visibility in bidirectional_visibility_modes:
                    returns = [
                        result for landscape_results in landscape_returns
                        for result in landscape_results[bidirectional_visibility]
                    ]
                    expected = landscape_iteration * solver_reset_num
                    if len(returns) != expected:
                        raise RuntimeError('Missing solver resets: expected {0}, received {1}'.format(
                            expected, len(returns)
                        ))
                    aggregated = aggregate_condition(returns)
                    for outcome in OUTCOME_NAMES:
                        results_across_K[bidirectional_visibility][outcome].append(
                            aggregated[outcome]
                        )

                # Preserve the matching identifiers for the paired mode outcomes.
                for loop, landscape_results in enumerate(landscape_returns):
                    for solver_reset in range(solver_reset_num):
                        for bidirectional_visibility in bidirectional_visibility_modes:
                            item = landscape_results[bidirectional_visibility][solver_reset]
                            writer.writerow([
                                visibility_extent, visibility_interval, K,
                                loop, solver_reset, int(bidirectional_visibility),
                                item[0], item[1], item[2], item[3],
                                item[4], item[6], item[7], item[8],
                            ])
                raw_file.flush()

        # Same seven across-K pickles, now with suffixes to avoid overwriting
        # files produced by the original unidirectional experiment.
        for bidirectional_visibility in bidirectional_visibility_modes:
            for outcome in OUTCOME_NAMES:
                output_filename = (
                    'visibility_extent_{0}_interval_{1}_{2}_across_K_size_{3}_bidirectional_{4}'
                    .format(visibility_extent, visibility_interval, outcome,
                            agent_num, int(bidirectional_visibility))
                )
                with open(output_filename, 'wb') as out_file:
                    pickle.dump(
                        results_across_K[bidirectional_visibility][outcome], out_file
                    )

    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print('Visibility Extent, Interval {0}, paired directions: '.format(
        visibility_interval
    ), time.strftime('%H:%M:%S', time.gmtime(time.time() - t0)))


if __name__ == '__main__':
    main()
