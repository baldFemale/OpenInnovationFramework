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

search_iteration = 50
N = 10
state_num = 4
generalist_expertise = 0
specialist_expertise = 20
for K in range(10):
    np.random.seed(1000)
    landscape = Landscape(N=N, K=K, state_num=state_num, alpha=0.5)
    agent = Agent(N=N, landscape=landscape, state_num=state_num,
                    generalist_expertise=generalist_expertise, specialist_expertise=specialist_expertise)
    # agent.describe()
    trajectory = []
    nodes = [agent.state]
    nodes_relation = {}  # focal node: following nodes
    layer_info = {"".join(agent.state): 0}
    visited_position = []
    while len(nodes) != 0:
        node = nodes.pop()
        if node in visited_position:
            continue
        iteration = layer_info["".join(node)]
        node_cog_fitness = agent.get_cog_fitness(state=node)
        neighbors = get_neighbor_list(state=node)
        neighbors_fitness = [agent.get_cog_fitness(state=neighbor) for neighbor in neighbors]
        next_nodes = []
        for index, value in enumerate(neighbors_fitness):
            if value >= node_cog_fitness:
                next_nodes.append("".join(neighbors[index]))
                nodes.append(neighbors[index])
        nodes_relation["".join(node)] = next_nodes
        for each in next_nodes:
            layer_info[each] = iteration + 1
        visited_position.append(node)
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
