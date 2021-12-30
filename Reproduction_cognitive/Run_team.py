# -*- coding: utf-8 -*-
# @Time     : 12/26/2021 20:30
# @Author   : Junyi
# @FileName: Run_team.py
# @Software  : PyCharm
# Observing PEP 8 coding style
import random
from Simulator import Simulator
from MultiStateInfluentialLandscape import LandScape
from Agent import Agent
from Team import Team
import time
import pickle
import numpy as np

random.seed(1024)
np.random.seed(1024)
# start_time = time.time()
N = 10
state_num = 4
landscape_iteration = 5
agent_iteration = 200
search_iteration = 100
k_list = [23, 33, 43]
K_list = [2, 4, 6, 8, 10]  # for HPC
agent_name = ["Generalist", "Specialist", "T shape", "T shape"]
IM_type = ["Traditional Mutual", "Factor Directed", "Influential Directed", "Random Directed"]
generalist_list = [6, 0, 4, 2]
specialist_list = [0, 3, 1, 2]

# state = 10, E = 12 (=2*6; 4*3; 2*4+ 4*1; 2*2+4*2)
# fix the task feature (i.e., N, K, and types); compare the team type (i.e,  GS, SG, GT. TG. ST, TS) -> heterogeneous team
# (GG, SS, TT) -> homogeneous team
teams_list = ['GS', "SG", "GT41", "T41G", "GT22", "T22G", "ST41", "T41S", "ST22", "T22S"]

# fitness_K_team = []
K = 6
# for K in K_list:  # for HPC
fitness_team = []
for team_type in teams_list:
    fitness_landscape = []
    for landscape_loop in range(landscape_iteration):
        landscape = LandScape(N=N, state_num=state_num)
        landscape.type(IM_type="Traditional Directed", K=K)
        landscape.initialize()
        fitness_agent = []
        for agent_loop in range(agent_iteration):
            if team_type == "GS":
                agent_s = Agent(N=N, lr=0, landscape=landscape, state_num=state_num)
                agent_s.type(name="Specialist", generalist_num=0, specialist_num=3)
                agent_g = Agent(N=N, lr=0, landscape=landscape, state_num=state_num)
                agent_g.type(name="Generalist", generalist_num=6, specialist_num=0)
                team = Team(members=[agent_g, agent_s])
            elif team_type == "SG":
                agent_s = Agent(N=N, lr=0, landscape=landscape, state_num=state_num)
                agent_s.type(name="Specialist", generalist_num=0, specialist_num=3)
                agent_g = Agent(N=N, lr=0, landscape=landscape, state_num=state_num)
                agent_g.type(name="Generalist", generalist_num=6, specialist_num=0)
                team = Team(members=[agent_s, agent_g])
            elif team_type == "GT41":
                agent_g = Agent(N=N, lr=0, landscape=landscape, state_num=state_num)
                agent_g.type(name="Generalist", generalist_num=6, specialist_num=0)
                agent_t41 = Agent(N=N, lr=0, landscape=landscape, state_num=state_num)
                agent_t41.type(name="T shape", generalist_num=4, specialist_num=1)
                team = Team(members=[agent_g, agent_t41])
            elif team_type == "T41G":
                agent_t41 = Agent(N=N, lr=0, landscape=landscape, state_num=state_num)
                agent_t41.type(name="T shape", generalist_num=4, specialist_num=1)
                agent_g = Agent(N=N, lr=0, landscape=landscape, state_num=state_num)
                agent_g.type(name="Generalist", generalist_num=6, specialist_num=0)
                team = Team(members=[agent_t41, agent_g])
            elif team_type == "GT22":
                agent_g = Agent(N=N, lr=0, landscape=landscape, state_num=state_num)
                agent_g.type(name="Generalist", generalist_num=6, specialist_num=0)
                agent_t22 = Agent(N=N, lr=0, landscape=landscape, state_num=state_num)
                agent_t22.type(name="T shape", generalist_num=2, specialist_num=2)
                team = Team(members=[agent_g, agent_t22])
            elif team_type == "T22G":
                agent_t22 = Agent(N=N, lr=0, landscape=landscape, state_num=state_num)
                agent_t22.type(name="T shape", generalist_num=2, specialist_num=2)
                agent_g = Agent(N=N, lr=0, landscape=landscape, state_num=state_num)
                agent_g.type(name="Generalist", generalist_num=6, specialist_num=0)
                team = Team(members=[agent_t22, agent_g])
            elif team_type == "ST41":
                agent_s = Agent(N=N, lr=0, landscape=landscape, state_num=state_num)
                agent_s.type(name="Specialist", generalist_num=0, specialist_num=3)
                agent_t41 = Agent(N=N, lr=0, landscape=landscape, state_num=state_num)
                agent_t41.type(name="T shape", generalist_num=4, specialist_num=1)
                team = Team(members=[agent_s, agent_t41])
            elif team_type == "T41S":
                agent_t41 = Agent(N=N, lr=0, landscape=landscape, state_num=state_num)
                agent_t41.type(name="T shape", generalist_num=4, specialist_num=1)
                agent_s = Agent(N=N, lr=0, landscape=landscape, state_num=state_num)
                agent_s.type(name="Specialist", generalist_num=0, specialist_num=3)
                team = Team(members=[agent_t41, agent_s])
            elif team_type == "ST22":
                agent_s = Agent(N=N, lr=0, landscape=landscape, state_num=state_num)
                agent_s.type(name="Specialist", generalist_num=0, specialist_num=3)
                agent_t22 = Agent(N=N, lr=0, landscape=landscape, state_num=state_num)
                agent_t22.type(name="T shape", generalist_num=2, specialist_num=2)
                team = Team(members=[agent_s, agent_t22])
            elif team_type == "T22S":
                agent_t22 = Agent(N=N, lr=0, landscape=landscape, state_num=state_num)
                agent_t22.type(name="T shape", generalist_num=2, specialist_num=2)
                agent_s = Agent(N=N, lr=0, landscape=landscape, state_num=state_num)
                agent_s.type(name="Specialist", generalist_num=0, specialist_num=3)
                team = Team(members=[agent_t22, agent_s])
            else:
                raise ValueError("Unknown team type")

            fitness_agent += team.serial_search(search_iteration=search_iteration)

            # record the match between team knowledge and task

            print("Current landscape iteration: {0}; Agent iteration: {1}; Team type: {2}".format(landscape_loop, agent_loop, team_type))
        fitness_landscape.append(fitness_agent)
    fitness_team.append(fitness_landscape)
# fitness_K_team.append(fitness_team)

    file_name = "HeteroTeamSerial" + '_' + "Traditional Directed" + '_N' + str(N) + \
                '_K' + str(K) + '_E12_' + team_type
    with open(file_name, 'wb') as out_file:
            pickle.dump(fitness_team, out_file)
# end_time = time.time()
# print("Time used: ", end_time - start_time)