# -*- coding: utf-8 -*-
# @Time     : 7/1/2023 13:56
# @Author   : Junyi
# @FileName: Trajectory.py
# @Software  : PyCharm
# Observing PEP 8 coding style
import numpy as np
from Agent import Agent
from Landscape import Landscape
from Crowd import Crowd
import time
import pickle


def func(N=None, K=None, state_num=None, search_iteration=None):
    np.random.seed(None)
    landscape = Landscape(N=N, K=K, state_num=state_num)
    gg_positions, gg_fitness_values, gg_cog_fitness_values = [], [], []
    gs_positions, gs_fitness_values, gs_cog_fitness_values = [], [], []
    sg_positions, sg_fitness_values, sg_cog_fitness_values = [], [], []
    ss_positions, ss_fitness_values, ss_cog_fitness_values = [], [], []
    G_crowd = Crowd(N=N, agent_num=50, landscape=landscape, state_num=state_num,
                           generalist_expertise=20, specialist_expertise=0)
    S_crowd = Crowd(N=N, agent_num=50, landscape=landscape, state_num=state_num,
                           generalist_expertise=0, specialist_expertise=20)

    generalist_G = Agent(N=N, landscape=landscape, state_num=state_num, crowd=G_crowd,
                       generalist_expertise=20, specialist_expertise=0)
    generalist_S = Agent(N=N, landscape=landscape, state_num=state_num, crowd=S_crowd,
                       generalist_expertise=20, specialist_expertise=0)

    specialist_G = Agent(N=N, landscape=landscape, state_num=state_num, crowd=G_crowd,
                       generalist_expertise=0, specialist_expertise=20)
    specialist_S = Agent(N=N, landscape=landscape, state_num=state_num, crowd=S_crowd,
                       generalist_expertise=0, specialist_expertise=20)
    for _ in range(search_iteration):
        # G -> G
        generalist_G.feedback_search(roll_back_ratio=0.5, roll_forward_ratio=0.5)
        gg_positions.append(generalist_G.state)
        gg_fitness_values.append(generalist_G.fitness)
        gg_cog_fitness_values.append(generalist_G.cog_fitness)
        # S -> G
        generalist_S.feedback_search(roll_back_ratio=0.5, roll_forward_ratio=0.5)
        sg_positions.append(generalist_S.state)
        sg_fitness_values.append(generalist_S.fitness)
        sg_cog_fitness_values.append(generalist_S.cog_fitness)
        # G -> S
        specialist_G.feedback_search(roll_back_ratio=0.5, roll_forward_ratio=0.5)
        gs_positions.append(specialist_G.state)
        gs_fitness_values.append(specialist_G.fitness)
        gs_cog_fitness_values.append(specialist_G.cog_fitness)
        # S -> S
        specialist_S.feedback_search(roll_back_ratio=0.5, roll_forward_ratio=0.5)
        ss_positions.append(specialist_S.state)
        ss_fitness_values.append(specialist_S.fitness)
        ss_cog_fitness_values.append(specialist_S.cog_fitness)

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
