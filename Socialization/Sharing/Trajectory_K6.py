# -*- coding: utf-8 -*-
# @Time     : 7/1/2023 13:56
# @Author   : Junyi
# @FileName: Trajectory.py
# @Software  : PyCharm
# Observing PEP 8 coding style
import numpy as np
from Agent import Agent
from Landscape import Landscape
import time
import pickle


def func(N=None, K=None, state_num=None, search_iteration=None):
    np.random.seed(None)
    landscape = Landscape(N=N, K=K, state_num=state_num)
    gg_positions, gg_fitness_values, gg_cog_fitness_values = [], [], []
    gs_positions, gs_fitness_values, gs_cog_fitness_values = [], [], []
    sg_positions, sg_fitness_values, sg_cog_fitness_values = [], [], []
    ss_positions, ss_fitness_values, ss_cog_fitness_values = [], [], []
    generalist_1 = Agent(N=N, landscape=landscape, state_num=state_num,
                       generalist_expertise=20, specialist_expertise=0)
    generalist_2 = Agent(N=N, landscape=landscape, state_num=state_num,
                       generalist_expertise=20, specialist_expertise=0)
    specialist_1 = Agent(N=N, landscape=landscape, state_num=state_num,
                       generalist_expertise=0, specialist_expertise=20)
    specialist_2 = Agent(N=N, landscape=landscape, state_num=state_num,
                       generalist_expertise=0, specialist_expertise=20)
    for _ in range(search_iteration):
        generalist_1.search()
        gg_positions.append(generalist_1.state)
        gg_fitness_values.append(generalist_1.fitness)
        gg_cog_fitness_values.append(generalist_1.cog_fitness)

        gs_positions.append(generalist_1.state)
        gs_fitness_values.append(generalist_1.fitness)
        gs_cog_fitness_values.append(generalist_1.cog_fitness)

        specialist_1.search()
        sg_positions.append(specialist_1.state)
        sg_fitness_values.append(specialist_1.fitness)
        sg_cog_fitness_values.append(specialist_1.cog_fitness)

        ss_positions.append(specialist_1.state)
        ss_fitness_values.append(specialist_1.fitness)
        ss_cog_fitness_values.append(specialist_1.cog_fitness)

    # Generalist -> Generalist
    generalist_2.state = generalist_1.state.copy()
    generalist_2.fitness = landscape.query_second_fitness(state=generalist_2.state)
    generalist_2.cog_fitness = generalist_2.get_cog_fitness(state=generalist_2.state)
    for _ in range(search_iteration):
        generalist_2.search()
        gg_positions.append(generalist_2.state)
        gg_fitness_values.append(generalist_2.fitness)
        gg_cog_fitness_values.append(generalist_2.cog_fitness)

    # Specialist -> Generalist
    generalist_2.state = specialist_1.state.copy()
    generalist_2.fitness = landscape.query_second_fitness(state=generalist_2.state)
    generalist_2.cog_fitness = generalist_2.get_cog_fitness(state=generalist_2.state)
    for _ in range(search_iteration):
        generalist_2.search()
        sg_positions.append(generalist_2.state)
        sg_fitness_values.append(generalist_2.fitness)
        sg_cog_fitness_values.append(generalist_2.cog_fitness)

    # Specialist -> Specialist
    specialist_2.state = specialist_1.state.copy()
    specialist_2.fitness = landscape.query_second_fitness(state=specialist_2.state)
    specialist_2.cog_fitness = specialist_2.get_cog_fitness(state=specialist_2.state)
    for _ in range(search_iteration):
        specialist_2.search()
        ss_positions.append(specialist_2.state)
        ss_fitness_values.append(specialist_2.fitness)
        ss_cog_fitness_values.append(specialist_2.cog_fitness)

    # Generalist -> Specialist
    specialist_2.state = generalist_1.state.copy()
    specialist_2.fitness = landscape.query_second_fitness(state=specialist_2.state)
    specialist_2.cog_fitness = specialist_2.get_cog_fitness(state=specialist_2.state)
    for _ in range(search_iteration):
        specialist_2.search()
        gs_positions.append(specialist_2.state)
        gs_fitness_values.append(specialist_2.fitness)
        gs_cog_fitness_values.append(specialist_2.cog_fitness)

    return [gg_positions, gg_fitness_values, gg_cog_fitness_values,
             gs_positions, gs_fitness_values, gs_cog_fitness_values,
             ss_positions, ss_fitness_values, ss_cog_fitness_values,
             sg_positions, sg_fitness_values, sg_cog_fitness_values]


if __name__ == '__main__':
    t0 = time.time()
    search_iteration = 100
    N = 10
    state_num = 4
    K = 6
    results = func(N=N, K=K, state_num=4, search_iteration=search_iteration)

    with open("gg_position_K_{0}".format(K), 'wb') as out_file:
        pickle.dump(results[0], out_file)
    with open("gg_fitness_values_K_{0}".format(K), 'wb') as out_file:
        pickle.dump(results[1], out_file)
    with open("gg_cog_fitness_values_K_{0}".format(K), 'wb') as out_file:
        pickle.dump(results[2], out_file)

    with open("gs_position_K_{0}".format(K), 'wb') as out_file:
        pickle.dump(results[3], out_file)
    with open("gs_fitness_values_K_{0}".format(K), 'wb') as out_file:
        pickle.dump(results[4], out_file)
    with open("gs_cog_fitness_values_K_{0}".format(K), 'wb') as out_file:
        pickle.dump(results[5], out_file)

    with open("ss_position_K_{0}".format(K), 'wb') as out_file:
        pickle.dump(results[6], out_file)
    with open("ss_fitness_values_K_{0}".format(K), 'wb') as out_file:
        pickle.dump(results[7], out_file)
    with open("ss_cog_fitness_values_K_{0}".format(K), 'wb') as out_file:
        pickle.dump(results[8], out_file)

    with open("sg_position_K_{0}".format(K), 'wb') as out_file:
        pickle.dump(results[9], out_file)
    with open("sg_fitness_values_K_{0}".format(K), 'wb') as out_file:
        pickle.dump(results[10], out_file)
    with open("sg_cog_fitness_values_K_{0}".format(K), 'wb') as out_file:
        pickle.dump(results[11], out_file)

    t1 = time.time()
    print(time.strftime("%H:%M:%S", time.gmtime(t1-t0)))
