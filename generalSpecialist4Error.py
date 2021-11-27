# Landscape with some variance
from MultiStateInfluentialLandscape import *
import numpy as np


class Agent:

    def __init__(self, N, state_list, landscape=None, state_num=2, fixed_variance=True, variance=1):

        self.N = N
        self.state_num = state_num
        self.state = np.random.choice([cur for cur in range(self.state_num)], self.N).tolist()
        self.decision_space = np.random.choice(self.N, len(state_list), replace=False).tolist()
        self.state_list = []
        for cur in range(self.N):
            if cur not in self.decision_space:
                self.state_list.append(0)
            else:
                index = self.decision_space.index(cur)
                self.state_list.append(state_list[index])
        self.landscape = landscape
        self.variance = variance
        self.fixed_variance = fixed_variance

        self.cache = self.landscape.contribution_cache
        self.cog_cache = {}

        for i in range(pow(self.state_num,self.N)):
            bit = numberToBase(i, self.state_num)
            if len(bit)<self.N:
                bit = "0"*(self.N-len(bit))+bit
            contribution_list = self.cache[bit]
            temp_contribution_list = []

            for cur in range(self.N):
                denominator = self.state_list[cur]+1
                temp_contribution_list.append(np.random.normal(contribution_list[cur], self.variance/denominator))

            self.cog_cache[bit] = temp_contribution_list

    def query_cog_fitness(self, state):
        bit = "".join([str(cur) for cur in state])

        res_fitness = []
        if self.fixed_variance:
            contribution = self.cog_cache[bit]
            for cur in self.decision_space:
                res_fitness.append(contribution[cur])
            return np.mean(res_fitness)
        else:
            contribution = self.cache[bit]
            for cur in self.decision_space:
                res_fitness.append(np.random.normal(contribution[cur], self.variance/(1+self.state_list[cur])))
            return np.mean(res_fitness)

    def independent_search(self, ):

        # local area

        temp_state = list(self.state)

        c = np.random.choice(self.decision_space)
        temp_state[c] = temp_state[c] ^ 1

        if self.query_cog_fitness(temp_state) > self.query_cog_fitness(self.state):
            return list(temp_state)
        else:
            return list(self.state)


def simulation(
        return_dic, idx, N, k, IM_type, land_num, period, agentNum, knowledge_num, state_list, state_num=2,
        fixed_variance=True, variance=1
):

    ress_fitness = []
    knowledge_list = []
    all_state_list = []

    for repeat in range(land_num):

        print(repeat)

        res_fitness = []

        np.random.seed(None)

        landscape = LandScape(N, k, IM_type, state_num=state_num)
        landscape.initialize()

        agents = []

        for cur in range(agentNum):
            agents.append(Agent(N, state_list, landscape, state_num, fixed_variance, variance))

        # print([agents[i].decision_space for i in range(agentNum)])
        # print([agents[i].specialist_decision_space for i in range(agentNum)])
        knowledge_list.append([list(agent.decision_space) for agent in agents])
        all_state_list.append([list(agent.state_list) for agent in agents])

        for step in range(period):

            for i in range(agentNum):

                # as individuals
                print(step)
                temp_state = agents[i].independent_search()
                agents[i].state = list(temp_state)

            tempFitness = [landscape.query_fitness(agents[i].state) for i in range(agentNum)]
            print(np.mean(tempFitness))

            res_fitness.append(tempFitness)

        ress_fitness.append(res_fitness)

    return_dic[idx] = (ress_fitness, knowledge_list, all_state_list)