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
start_time = time.time()
N = 10
state_num = 4
landscape_iteration = 5
agent_iteration = 200
search_iteration = 100
k_list = [24, 34, 44]
K_list = [2, 4, 6, 8, 10, 12, 14]
agent_name = ["Generalist", "Specialist", "T shape", "T shape"]
IM_type = ["Traditional Mutual", "Factor Directed", "Influential Directed", "Random Directed"]
generalist_list = [6, 0, 4, 2]
specialist_list = [0, 3, 1, 2]

# state = 10, E = 12 (=2*6; 4*3; 2*4+ 4*1; 2*2+4*2)
# fix the task feature (i.e., N, K, and types); compare the team type (i.e,  GS, SG, GT. TG. ST, TS) -> heterogeneous team
# (GG, SS, TT) -> homogeneous team
teams_list = ['GS', "SG", "GT41", "T41G", "GT22", "T22G", "ST41", "T41S", "ST22", "T22S"]
fitness_team = []
k = 33
for team_type in teams_list:
    fitness_landscape = []
    for landscape_loop in range(landscape_iteration):
        landscape = LandScape(N=N, state_num=state_num)
        landscape.type(IM_type="Factor Directed", k=k)
        landscape.initialize()
        fitness_agent = []
        for agent_loop in range(agent_iteration):
            agent_s = Agent(N=N, lr=0, landscape=landscape, state_num=state_num)
            agent_s.type(name="Specialist", generalist_num=0, specialist_num=3)
            agent_g = Agent(N=N, lr=0, landscape=landscape, state_num=state_num)
            agent_g.type(name="Generalist", generalist_num=6, specialist_num=0)
            agent_t41 = Agent(N=N, lr=0,landscape=landscape, state_num=state_num)
            agent_t41.type(name="T shape", generalist_num=4, specialist_num=1)
            agent_t22 = Agent(N=N, lr=0, landscape=landscape,state_num=state_num)
            agent_t22.type(name="T shape", generalist_num=2, specialist_num=2)
            if team_type == "GS":
                team = Team(members=[agent_g, agent_s])
            elif team_type == "SG":
                team = Team(members=[agent_s, agent_g])
            elif team_type == "GT41":
                team = Team(members=[agent_g, agent_t41])
            elif team_type == "T41G":
                team = Team(members=[agent_t41, agent_g])
            elif team_type == "GT22":
                team = Team(members=[agent_g, agent_t22])
            elif team_type == "T22G":
                team = Team(members=[agent_t22, agent_g])
            elif team_type == "ST41":
                team = Team(members=[agent_s, agent_t41])
            elif team_type == "T41S":
                team = Team(members=[agent_t41, agent_g])
            elif team_type == "ST22":
                team = Team(members=[agent_s, agent_t22])
            elif team_type == "T22S":
                team = Team(members=[agent_t22, agent_s])
            else:
                raise ValueError("Unknown team type")

            fitness_search = team.parallel_search(search_iteration=search_iteration)
            fitness_agent.append(fitness_search)
            print("Current landscape iteration: {0}; Agent iteration: {1}; Team type: {2}".format(landscape_loop, agent_loop, team_type))
        fitness_landscape.append(fitness_agent)
    fitness_team.append(fitness_landscape)

    file_name = "HeteroTeam" + '_' + "Factor" + '_N' + str(N) + \
                '_k' + str(k) + '_E12' + team_type
    with open(file_name, 'wb') as out_file:
        pickle.dump(fitness_landscape, out_file)
end_time = time.time()
print("Time used: ", end_time - start_time)