from LandScape import *
from tools import *

# this might be ok, the analysis code is wrong


class Agent:

    def __init__(self, N, knowledge_num, specialist_num, lr=0, landscape=None):

        self.N = N
        self.state = np.random.choice([0, 1], self.N).tolist()
        self.decision_space = np.random.choice(self.N, knowledge_num, replace=False).tolist()
        self.knowledge_space = list(self.decision_space)
        self.specialist_decision_space = np.random.choice(self.decision_space, specialist_num, replace=False).tolist()
        self.lr = lr

        self.landscape = landscape

    def independent_search(self, ):

        # local area

        base_state = list(self.state)

        for c in self.decision_space:
            if c in self.specialist_decision_space:
                continue
            temp_state = list(self.state)
            temp_state[c] ^= 1

            if self.landscape.query_cog_fitness(
                    temp_state, self.knowledge_space
            ) > self.landscape.query_cog_fitness(
                base_state, self.knowledge_space
            ):
                base_state = list(temp_state)

        # distant area
        temp_state = list(self.state)
        for c in self.specialist_decision_space:
            temp_state[c]^=1
            if self.landscape.query_cog_fitness(
                temp_state, self.knowledge_space
            ) > self.landscape.query_cog_fitness(
                base_state, self.knowledge_space
            ):
                base_state = list(temp_state)

        return base_state

    def learn(self, target_):
        pass


def simulation(return_dic, idx, N, k, land_num, period, agentNum, teamup, teamup_timing, knowledge_num, lr=0.1):
    """
    default: random formation
    three types of agents:
        0-1/3 G
        1/3-2/3 S
        2/3-1 T

    """

    ress_fitness = []
    team_list = []
    knowledge_list = []

    for repeat in range(land_num):

        print(repeat)

        res_fitness = []

        np.random.seed(None)

        landscape = LandScape(N, k, None, None)
        landscape.initialize()

        agents = []

        for cur in range(agentNum):
            if cur < agentNum//3:
                agents.append(Agent(N, knowledge_num, 0, lr, landscape))
            elif cur < 2*agentNum//3:
                agents.append(Agent(N, knowledge_num, knowledge_num, lr, landscape))
            else:
                agents.append(Agent(N, knowledge_num, knowledge_num//2, lr, landscape))

        teams = {i: i for i in range(agentNum * 2)}

        for step in range(period):

            if teamup and step == teamup_timing:

                rank = np.random.choice([cur for cur in range(agentNum)], agentNum, replace=False)
                for i in range(agentNum):

                    if teams[rank[i]] is None or teams[rank[i]] != rank[i]:
                        continue

                    for j in range(agentNum):
                        if i == j or teams[rank[j]] is None or teams[rank[j]] != rank[j]:
                            continue

                        teams[rank[i]] = rank[j]
                        teams[rank[j]] = None

                        integrated_solution = solutuionIntegration(
                            agents[rank[i]].state, agents[rank[j]].state, agents[rank[i]].decision_space,
                            agents[rank[j]].decision_space, landscape
                        )

                        agents[rank[i]].state = list(integrated_solution)
                        agents[rank[j]].state = list(integrated_solution)
                        break

            for i in range(agentNum):

                if teams[i] is None:
                    continue

                elif teams[i] == i:

                    # as individuals

                    temp_state = agents[i].independent_search()
                    agents[i].state = list(temp_state)

                elif teams[i] != i:

                    # learning

                    overlap = list(set(agents[i].decision_space) & set(agents[teams[i]].decision_space))

                    p = lr * len(overlap)

                    if np.random.uniform(0, 1) < p:

                        if (
                            len(agents[i].knowledge_space)-len(agents[i].decision_space) < len(overlap) and len(
                            agents[i].knowledge_space) < len(agents[i].decision_space) + len(agents[teams[i]].decision_space) - len(overlap)
                        ):
                            new_knowledge_A = np.random.choice(
                                [cur for cur in agents[teams[i]].decision_space if cur not in agents[i].knowledge_space]
                            )
                            agents[i].knowledge_space.append(new_knowledge_A)

                    if np.random.uniform(0, 1) < p:

                        if (
                            len(agents[teams[i]].knowledge_space) - len(agents[teams[i]].decision_space) < len(
                            overlap) and len(agents[teams[i]].knowledge_space) < len(
                            agents[teams[i]].decision_space) + len(agents[i].decision_space) - len(overlap)
                        ):

                            new_knowledge_B = np.random.choice(
                                [cur for cur in agents[i].decision_space if cur not in agents[teams[i]].knowledge_space]
                            )
                            agents[teams[i]].knowledge_space.append(new_knowledge_B)

                    # A's proposal
                    temp_state = agents[i].independent_search()

                    # B's evaluation
                    if landscape.query_cog_fitness(
                            temp_state, agents[teams[i]].knowledge_space
                    ) > landscape.query_cog_fitness(
                        agents[i].state, agents[teams[i]].knowledge_space
                    ):
                        pass
                    else:
                        temp_state = list(agents[i].state)

                    # B's proposal
                    agents[teams[i]].state = list(temp_state)
                    B_temp_state = agents[teams[i]].independent_search()

                    # A's evaluation
                    if landscape.query_cog_fitness(
                        B_temp_state, agents[i].knowledge_space
                    ) > landscape.query_cog_fitness(
                        temp_state, agents[i].knowledge_space
                    ):
                        pass

                    else:
                        B_temp_state = list(temp_state)

                    agents[i].state = list(B_temp_state)
                    agents[teams[i]].state = list(B_temp_state)

            tempFitness = [landscape.query_fitness(agents[i].state) for i in range(agentNum)]

            res_fitness.append(tempFitness)

        ress_fitness.append(res_fitness)
        team_list.append(teams)
        knowledge_list.append([agents[i].decision_space for i in range(agentNum)])

    return_dic[idx] = (ress_fitness, team_list, knowledge_list)









