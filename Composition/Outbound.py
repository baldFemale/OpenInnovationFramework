# -*- coding: utf-8 -*-
# @Time     : 12/14/2021 20:16
# @Author   : Junyi
# @FileName: Simulator.py
# @Software  : PyCharm
# Observing PEP 8 coding style
from Landscape import Landscape
from Socialized_Agent import Agent
import numpy as np


class Simulator:

    def __init__(self, N=10, state_num=4, agent_num=500, search_iteration=100, IM_type=None,
                 K=0, k=0, gs_proportion=0.5, knowledge_num=20,
                 exposure_type="Self-interested", openness=None, frequency=None,
                 quality=1.0, S_exposed_to_S=None, G_exposed_to_G=None):
        self.N = N
        self.state_num = state_num
        # Landscape
        self.landscapes = None


    def set_landscape(self):
        pass

    def set_agent(self):
        pass

    def process(self, socialization_freq=1, footprint=False):
        pass

    def pair_wise_distance(self, state_pool=None):
        distance = 0
        for state in state_pool:
            distance += sum([self.count_divergence(state, next_) for next_ in state_pool]) / len(state_pool)
        distance = distance/len(state_pool)
        return distance

    def count_divergence(self, state_1=None, state_2=None):
        divergence = 0
        for i in range(len(state_1)):
            if state_1[i] != state_2[i]:
                divergence += 1
        return divergence


if __name__ == '__main__':
    # Test Example
    N = 6
    state_num = 4
    K = 2
    k = 0
    IM_type = "Traditional Directed"
    openness = 0.5
    quality = 0.5
    S_exposed_to_S = 0
    G_exposed_to_G = 0.5
    agent_num = 500
    search_iteration = 50
    knowledge_num = 12
    # exposure_type = "Overall-ranking"
    exposure_type = "Self-interested"
    # exposure_type = "Random"
    # if S_exposed_to_S and G_exposed_to_G are None, then it refers to whole state pool,
    # could be either self-interested rank or overall rank on the whole state pool
    simulator = Simulator(N=N, state_num=state_num, agent_num=agent_num, search_iteration=search_iteration, IM_type=IM_type,
                 K=K, k=k, gs_proportion=0.5, knowledge_num=knowledge_num,
                 exposure_type=exposure_type, openness=openness, quality=quality,
                          S_exposed_to_S=S_exposed_to_S, G_exposed_to_G=G_exposed_to_G)
    simulator.process(socialization_freq=1, footprint=False)
    count_GS = 0
    count_GG = 0
    count_SS = 0
    count_SG = 0
    count_open_G, count_open_S = 0, 0
    for agent in simulator.agents:
        if agent.name == "Generalist":
            if agent.fixed_state_pool == 1:
                count_GS += 1
            else:
                count_GG += 1
            if agent.fixed_openness_flag == 1:
                count_open_G += 1
        else:
            if agent.fixed_state_pool == 1:
                count_SS += 1
            else:
                count_SG += 1
            if agent.fixed_openness_flag == 1:
                count_open_S += 1
    print("GG, GS: ", count_GG, count_GS)
    print("SS, SG", count_SS, count_SG)
    print("Openness G, S: ", count_open_G, count_open_S)
    surface_quality_G, surface_quality_S = [], []
    for each_qualities in simulator.surface_quality_G_landscape:
        surface_quality_G.append(np.mean(np.array(each_qualities, dtype=object), axis=0))
    print("surface_quality_G: ", surface_quality_G)
    for each_qualities in simulator.surface_quality_S_landscape:
        surface_quality_S.append(np.mean(np.array(each_qualities, dtype=object), axis=0))
    print("surface_quality_S: ", surface_quality_S)
    print("END")

