# -*- coding: utf-8 -*-
# @Time     : 7/1/2023 13:56
# @Author   : Junyi
# @FileName: Trajectory.py
# @Software  : PyCharm
# Observing PEP 8 coding style
import numpy as np
from Landscape import Landscape
from Agent import Agent
import pickle
import time
import multiprocessing as mp


def func(N=None, K=None, state_num=None):
    np.random.seed(1000)
    landscape = Landscape(N=N, K=K, state_num=state_num, alpha=0.5)
    agent = Agent(N=N, landscape=landscape, state_num=state_num,
                  generalist_expertise=12, specialist_expertise=0)
    # State: list of int;  Node: string
    fresh_states = [agent.state.copy()]
    visited_position = {}
    nodes_relation = {}  # focal node: following nodes
    layer_info = {"".join(agent.state): 0}
    while len(fresh_states) != 0:
        state = fresh_states.pop()
        if "".join(state) in visited_position.keys():
            continue
        else:
            visited_position["".join(state)] = True
        node_cog_fitness = agent.get_cog_fitness(state=state)
        neighbor_states = get_neighbor_list(state=state)  # return list of states
        neighbor_states = [one for one in neighbor_states if "".join(one) not in visited_position.keys()]
        neighbors_fitness = [agent.get_cog_fitness(state=neighbor) for neighbor in neighbor_states]
        next_nodes = []
        for index, value in enumerate(neighbors_fitness):
            if value >= node_cog_fitness:
                next_nodes.append("".join(neighbor_states[index]))
                fresh_states.append(neighbor_states[index])
        nodes_relation["".join(state)] = next_nodes
        for each in next_nodes:
            if each in layer_info.keys():
                continue
            layer_info[each] = layer_info["".join(state)] + 1
    # print(result_dict)
    # print(layer_info)

    # change into int value as the index (bit index is for the neighbor generation)
    layer_info_int = {}
    for key in layer_info.keys():
        layer_info_int[int(key, 4)] = layer_info[key]
    # print(layer_info_int)

    nodes_relation_int = {}
    for key in nodes_relation.keys():
        nodes_relation_int[int(key, 4)] = [int(node, 4) for node in nodes_relation[key]]
    # print(result_dict_int)

    with open("G_nodes_relation_K_{0}".format(K), 'wb') as out_file:
        pickle.dump(nodes_relation, out_file)
    with open("G_nodes_relation_int_K_{0}".format(K), 'wb') as out_file:
        pickle.dump(nodes_relation_int, out_file)
    with open("G_layer_info_K_{0}".format(K), 'wb') as out_file:
        pickle.dump(layer_info, out_file)
    with open("G_layer_info_int_K_{0}".format(K), 'wb') as out_file:
        pickle.dump(layer_info_int, out_file)


def get_neighbor_list(state: list) -> list:
    neighbor_states = []
    for i, char in enumerate(state):
        neighbors = []
        for alt_bit in range(4):
            if alt_bit != int(char):
                new_state = state.copy()
                new_state[i] = str(alt_bit)
                neighbors.append(new_state)
        neighbor_states.extend(neighbors)
    return neighbor_states

if __name__ == '__main__':
    t0 = time.time()
    N = 9
    state_num = 4
    for K in range(N):
        manager = mp.Manager()
        return_dict = manager.dict()
        jobs = []
        p = mp.Process(target=func, args=(N, K, state_num))
        jobs.append(p)
        p.start()

    t1 = time.time()
    print(time.strftime("%H:%M:%S", time.gmtime(t1 - t0)))

