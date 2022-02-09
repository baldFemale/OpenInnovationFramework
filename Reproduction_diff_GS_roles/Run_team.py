# -*- coding: utf-8 -*-
# @Time     : 12/26/2021 20:30
# @Author   : Junyi
# @FileName: Run_team.py
# @Software  : PyCharm
# Observing PEP 8 coding style
import random
from Landscape import Landscape
from Agent import Agent
from Team import Team
import multiprocessing as mp
import pickle


N = 10
state_num = 4
landscape_iteration = 1000
agent_iteration = 1000
search_iteration = 100
k_list = [4, 14, 24, 34, 44, 54, 64, 74, 84, 94]
K_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
agent_name = ["Generalist", "Specialist", "T shape", "T shape"]
IM_type = ["Traditional Mutual", "Factor Directed", "Influential Directed", "Random Directed"]
generalist_list = [6, 0, 4, 2]
specialist_list = [0, 3, 1, 2]

# fix the task feature (i.e., N, K, and types);
# compare the team type (i.e,  GS, SG, GT. TG. ST, TS) -> heterogeneous team
# (GG, SS, TT) -> homogeneous team
teams_list = ['GS', "SG", "GT41", "T41G", "GT22", "T22G", "ST41", "T41S", "ST22", "T22S"]


def loop(k=0, K=0):
    for team_type in teams_list:
        A_fitness_landscape = []  # for the detailed fitness dynamic during search
        B_converged_fitness_landscape = []  # for the final converged fitness after search
        C_row_match_landscape = []  # for the weighted sum according to IM row
        C_column_match_landscape = [] # for the weighted sum according to IM column
        D_landscape_IM_list = []  # for the landscape IM
        E_knowledge_list1_landscape = []  # for the agent 1 knowledge in case we will change the weighted sum algorithm
        E_knowledge_list2_landscape = []  # for the agent 2 knowledge in case we will change the weighted sum algorithm
        for landscape_loop in range(landscape_iteration):
            landscape = Landscape(N=N, state_num=state_num)
            landscape.type(IM_type="Traditional Directed", K=K)
            landscape.initialize()
            A_fitness_agent = []
            B_converged_fitness_agent = []
            C_row_match_agent = []
            C_column_match_agent = []
            IM = landscape.IM.tolist()
            D_landscape_IM_list.append(IM)
            E_knowledge_list_agent1 = []
            E_knowledge_list_agent2 = []
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
                temp_fitness_list = team.serial_search(search_iteration=search_iteration)
                A_fitness_agent += temp_fitness_list
                B_converged_fitness_agent.append(temp_fitness_list[-1])
                C_row_match_temp = [0] * team.agent_num
                for i in range(team.agent_num):
                    for row in range(team.N):
                        if row in team.members[i].specialist_knowledge_domain:
                            C_row_match_temp[i] += sum(team.landscape.IM[row]) * team.state_num
                        if row in team.members[i].generalist_knowledge_domain:
                            C_row_match_temp += sum(team.landscape.IM[row]) * team.state_num * team.gs_ratio
                C_column_match_temp = [0] * team.agent_num
                for j in range(team.agent_num):
                    for column in range(team.N):
                        if column in team.members[j].specialist_knowledge_domain:
                            C_column_match_temp[j] += sum(team.landscape.IM[:][column]) * team.state_num
                        if column in team.members[j].generalist_knowledge_domain:
                            C_column_match_temp[j] += sum(team.landscape.IM[:][column]) * team.state_num * team.gs_ratio
                C_row_match_agent.append(C_row_match_temp)
                C_column_match_agent.append(C_column_match_temp)
                E_knowledge_list_agent1.append([team.members[0].specialist_knowledge_domain, team[0].agent.generalist_knowledge_domain])
                E_knowledge_list_agent2.append([team.members[1].specialist_knowledge_domain, team[1].agent.generalist_knowledge_domain])

            A_fitness_landscape.append(A_fitness_agent)
            B_converged_fitness_landscape.append(B_converged_fitness_agent)
            C_row_match_landscape.append(C_row_match_agent)
            C_column_match_landscape.append(C_column_match_agent)
            E_knowledge_list1_landscape.append(E_knowledge_list_agent1)
            E_knowledge_list2_landscape.append(E_knowledge_list_agent2)

            A_file_name = "HeteroSerial" + '_' + "Traditional Directed" + '_N' + str(N) + \
                        '_K' + str(K) + '_E12_' + team_type
            B_file_name_convergence = "Convergence_" + A_file_name
            C_file_name_row_match = "RowMatch_" + A_file_name
            C_file_name_column_match = "ColumnMatch_" + A_file_name
            D_file_name_IM_information = "IM_" + A_file_name
            E_file_name_agent1_knowledge = "KnowledgeofAgent1_" + A_file_name
            E_file_name_agent2_knowledge = "KnowledgeofAgent2_" + A_file_name

            with open(A_file_name, 'wb') as out_file:
                pickle.dump(A_fitness_landscape, out_file)
            with open(B_file_name_convergence, 'wb') as out_file:
                pickle.dump(B_converged_fitness_landscape, out_file)
            with open(C_file_name_row_match, 'wb') as out_file:
                pickle.dump(C_row_match_landscape, out_file)
            with open(C_file_name_column_match, 'wb') as out_file:
                pickle.dump(C_column_match_landscape, out_file)
            with open(D_file_name_IM_information, 'wb') as out_file:
                pickle.dump(D_landscape_IM_list, out_file)
            with open(E_file_name_agent1_knowledge, 'wb') as out_file:
                pickle.dump(E_knowledge_list1_landscape, out_file)
            with open(E_file_name_agent2_knowledge, 'wb') as out_file:
                pickle.dump(E_knowledge_list2_landscape, out_file)

if __name__ == '__main__':
    for K in K_list:
        k = 0
        p = mp.Process(target=loop, args=(k, K))
        p.start()