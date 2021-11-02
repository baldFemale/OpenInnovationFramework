from MultiStateLandScape import *
from tools import *

# the results might be determined by the way we calculate cognition
# we overvalue generalist
# I want to make it a fair comparison & learning from specialist and generalist should be different -> need to specify
# bit different in MultiStateLandScape class
# target -> specialist works well in complex problems


class Agent:

    def __init__(self, N, knowledge_num, specialist_num, lr=0, landscape=None, state_num=2):

        self.N = N
        self.state_num = state_num
        self.state = np.random.choice([cur for cur in range(2)], self.N).tolist()
        self.decision_space = np.random.choice(self.N, knowledge_num, replace=False).tolist()
        self.knowledge_space = list(self.decision_space)
        self.specialist_decision_space = np.random.choice(self.decision_space, specialist_num, replace=False).tolist()
        self.specialist_knowledge_space = list(self.specialist_decision_space)
        self.generalist_knowledge_space = [
            cur for cur in self.decision_space if cur not in self.specialist_knowledge_space
        ]

        self.generalist_map_dic = defaultdict(lambda: defaultdict(int))

        for cur in self.generalist_knowledge_space:
            self.generalist_map_dic[cur][0] = np.random.choice([0, 1])
            self.generalist_map_dic[cur][1] = np.random.choice([2, 3])

        self.lr = lr

        self.landscape = landscape

    def independent_search(self, ):

        # local area

        temp_state = list(self.state)

        c = np.random.choice(self.decision_space)

        if c in self.specialist_knowledge_space:
            current_state = temp_state[c]
            new_state = np.random.choice([cur for cur in range(self.state_num) if cur != current_state])
            temp_state[c] = new_state

        else:
            focal_flag = temp_state[c]//2
            focal_flag = focal_flag^1
            temp_state[c] = self.generalist_map_dic[c][focal_flag]

        cognitive_state = self.change_state_to_cog_state(self.state)
        # print("knowledge", self.generalist_knowledge_space, self.specialist_knowledge_space)
        # print("current", self.state, cognitive_state)
        cognitive_temp_state = self.change_state_to_cog_state(temp_state)
        # print("next", temp_state, cognitive_temp_state)

        if self.landscape.query_cog_fitness_gst(
            cognitive_state, self.generalist_knowledge_space, self.specialist_knowledge_space
        ) > self.landscape.query_cog_fitness_gst(
            cognitive_temp_state, self.generalist_knowledge_space, self.specialist_knowledge_space
        ):
            return list(self.state)
        else:
            return list(temp_state)

    def change_state_to_cog_state(self, state):
        temp_state = []
        for cur in range(len(state)):
            if cur in self.generalist_knowledge_space:
                temp_state.append(state[cur]//2)
            else:
                temp_state.append(state[cur])
        return temp_state

    def learn(self, target_):
        pass


def simulation(return_dic, idx, N, k, land_num, period, agentNum, teamup, teamup_timing, knowledge_num, specialist_num,
               lr=0.1, state_num=2, ):
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

        landscape = LandScape(N, k, None, None, state_num=state_num)
        landscape.initialize()

        agents = []

        if not teamup:
            for cur in range(agentNum):
                agents.append(Agent(N, knowledge_num, specialist_num, lr, landscape, state_num))
        else:
            for cur in range(agentNum):
                if cur < agentNum//3:
                    agents.append(Agent(N, knowledge_num[0], specialist_num[0], lr, landscape, state_num))
                elif cur < 2*agentNum//3:
                    agents.append(Agent(N, knowledge_num[1], specialist_num[1], lr, landscape, state_num))
                else:
                    agents.append(Agent(N, knowledge_num[2], specialist_num[2], lr, landscape, state_num))

        # print([agents[i].decision_space for i in range(agentNum)])
        # print([agents[i].specialist_decision_space for i in range(agentNum)])

        teams = {i: i for i in range(agentNum)}

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
                            agents[rank[i]].state, agents[rank[j]].state, agents[rank[i]].decision_space, agents[rank[j]].decision_space, landscape
                        )

                        agents[rank[i]].state = list(integrated_solution)
                        agents[rank[j]].state = list(integrated_solution)
                        break

            for i in range(agentNum):

                if teams[i] is None:
                    continue

                elif teams[i] == i:

                    # as individuals
                    print(step)
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
                                [cur for cur in agents[teams[i]].decision_space]
                            )

                            if new_knowledge_A in agents[teams[i]].generalist_knowledge_space:
                                if (
                                        new_knowledge_A not in agents[i].specialist_knowledge_space
                                ) and (
                                        new_knowledge_A not in agents[i].generalist_knowledge_space
                                ):
                                    agents[i].generalist_knowledge_space.append(new_knowledge_A)
                                    agents[i].generalist_map_dic[new_knowledge_A][0] = \
                                        agents[teams[i]].generalist_map_dic[new_knowledge_A][0]
                                    agents[i].generalist_map_dic[new_knowledge_A][1] = \
                                        agents[teams[i]].generalist_map_dic[new_knowledge_A][1]
                            elif new_knowledge_A in agents[teams[i]].specialist_knowledge_space:
                                if new_knowledge_A not in agents[i].specialist_knowledge_space:
                                    if new_knowledge_A not in agents[i].generalist_knowledge_space:
                                        agents[i].specialist_knowledge_space.append(new_knowledge_A)
                                    else:
                                        focal_index = agents[i].generalist_knowledge_space.index(new_knowledge_A)
                                        agents[i].generalist_knowledge_space.pop(focal_index)
                                        agents[i].specialist_knowledge_space.append(new_knowledge_A)

                            if new_knowledge_A not in agents[i].knowledge_space:
                                agents[i].knowledge_space.append(new_knowledge_A)

                    if np.random.uniform(0, 1) < p:

                        if (
                            len(agents[teams[i]].knowledge_space) - len(agents[teams[i]].decision_space) < len(
                            overlap) and len(agents[teams[i]].knowledge_space) < len(
                            agents[teams[i]].decision_space) + len(agents[i].decision_space) - len(overlap)
                        ):

                            new_knowledge_B = np.random.choice(
                                [cur for cur in agents[i].decision_space]
                            )

                            if new_knowledge_B in agents[i].generalist_knowledge_space:
                                if (
                                        new_knowledge_B not in agents[teams[i]].specialist_knowledge_space
                                ) and (
                                        new_knowledge_B not in agents[teams[i]].generalist_knowledge_space
                                ):
                                    agents[teams[i]].generalist_knowledge_space.append(new_knowledge_B)
                                    agents[teams[i]].generalist_map_dic[new_knowledge_B][0] = \
                                        agents[i].generalist_map_dic[new_knowledge_B][0]
                                    agents[teams[i]].generalist_map_dic[new_knowledge_B][1] = \
                                        agents[i].generalist_map_dic[new_knowledge_B][1]
                            elif new_knowledge_B in agents[i].specialist_knowledge_space:
                                if new_knowledge_B not in agents[teams[i]].specialist_knowledge_space:
                                    if new_knowledge_B not in agents[teams[i]].generalist_knowledge_space:
                                        agents[teams[i]].specialist_knowledge_space.append(new_knowledge_B)
                                    else:
                                        focal_index = agents[teams[i]].generalist_knowledge_space.index(new_knowledge_B)
                                        agents[teams[i]].generalist_knowledge_space.pop(focal_index)
                                        agents[teams[i]].specialist_knowledge_space.append(new_knowledge_B)

                            if new_knowledge_B not in agents[teams[i]].knowledge_space:
                                agents[teams[i]].knowledge_space.append(new_knowledge_B)

                    # A's proposal
                    temp_state = agents[i].independent_search()

                    cognitive_temp_state = agents[teams[i]].change_state_to_cog_state(temp_state)
                    cognitive_state = agents[teams[i]].change_state_to_cog_state(agents[teams[i]].state)

                    # B's evaluation
                    if landscape.query_cog_fitness_gst(
                        cognitive_temp_state,
                        agents[teams[i]].generalist_knowledge_space,
                        agents[teams[i]].specialist_knowledge_space,
                    ) > landscape.query_cog_fitness_gst(
                        cognitive_state,
                        agents[teams[i]].generalist_knowledge_space,
                        agents[teams[i]].specialist_knowledge_space,
                    ):
                        pass
                    else:
                        temp_state = list(agents[i].state)

                    # B's proposal
                    agents[teams[i]].state = list(temp_state)
                    B_temp_state = agents[teams[i]].independent_search()

                    # A's evaluation
                    cognitive_temp_state = agents[i].change_state_to_cog_state(B_temp_state)
                    cognitive_state = agents[i].change_state_to_cog_state(temp_state)
                    if landscape.query_cog_fitness_gst(
                        cognitive_temp_state, agents[i].generalist_knowledge_space, agents[i].specialist_knowledge_space
                    ) > landscape.query_cog_fitness_gst(
                        cognitive_state, agents[i].generalist_knowledge_space, agents[i].specialist_knowledge_space
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









