from MultiStateInfluentialLandscape import *
from tools import *
import time
from generalSpecialist7 import Agent


class Platform:

    """
    generalist -> 2
    specialist -> 4
    """

    def __init__(self, N, k, agentNum, knowledgeNum, generalistProportion, defaultValues=True, state_num=4):

        self.N = N
        self.landscape = LandScape(N, k, IM_type="random")
        self.landscape.initialize()

        self.agents = []
        self.agent_types = []

        default_state = np.random.choice(state_num, N).tolist()

        for cur in range(agentNum):
            if np.random.uniform(0, 1) < generalistProportion:
                agent = Agent(self.N, [2 for cur in range(knowledgeNum//2)], self.landscape, state_num=state_num)
                self.agent_types.append("g")
            else:
                agent = Agent(self.N, [4 for cur in range(knowledgeNum//4)], self.landscape, state_num=state_num)
                self.agent_types.append("s")

            if defaultValues:

                agent.state = adoptDefault(agent.state, agent.decision_space, default_state)
                agent.cog_fitness = self.landscape.query_partial_fitness_tree(
                    agent.state, agent.decision_space, agent.knowledge_tree_list, agent.tree_depth, None, None
                )

            self.agents.append(agent)
            # print(agent.tree_depth)

    def extract_results(self, cutoff_threshold=0):
        # platform level -> overall fitness, average fitness, unique solutions
        # g & s level -> average fitness, unique_solutions

        platform_fitness = []
        platform_unique = set([])
        platform_overall_fitness = []

        g_fitness = []
        g_unique = set([])
        s_fitness = []
        s_unique = set([])

        for cur in range(len(self.agents)):

            if self.landscape.query_fitness(self.agents[cur].state) < cutoff_threshold:
                pass

            bit_str = "".join([str(x) for x in self.agents[cur].state])
            if bit_str not in platform_unique:
                platform_unique.add(bit_str)
                platform_overall_fitness.append(self.landscape.query_fitness(self.agents[cur].state))
            platform_fitness.append(self.landscape.query_fitness(self.agents[cur].state))

            if self.agent_types[cur] == "g":
                g_fitness.append(self.landscape.query_fitness(self.agents[cur].state))
                if bit_str not in g_unique:
                    g_unique.add(bit_str)
            else:
                s_fitness.append(self.landscape.query_fitness(self.agents[cur].state))
                if bit_str not in s_unique:
                    s_unique.add(bit_str)
        return (
            np.mean(platform_fitness), len(platform_unique), np.sum(platform_overall_fitness),
            np.max(platform_fitness) if len(platform_fitness)>0 else 0,
            np.mean(g_fitness), len(g_unique), np.max(g_fitness) if len(g_fitness)>0 else 0,
            np.mean(s_fitness), len(s_unique), np.max(s_fitness) if len(s_fitness)>0 else 0
        )

    def aggregate_vote_number(self, vote_list):
        vote_count = np.zeros(len(self.agent_types))
        for cur in range(len(vote_list)):
            for vote_index in vote_list[cur]:
                vote_count[vote_index] += 1
        return vote_count.tolist()

    def adaptation(self, learn, learn_method="random", vote_percentage=0.1):
        """
        :param learn: bool -> whether learn
        :param learn_method: random, aggregate vote, personal vote
        :return:
        """
        for cur in range(len(self.agents)):
            self.agents[cur].state, self.agents[cur].cog_fitness = self.agents[cur].independent_search()

        if learn:
            if learn_method=="random":
                for cur in range(len(self.agents)):
                    target = np.random.choice([x for x in range(len(self.agents)) if x != cur])

                    self.agents[cur].state, self.agents[cur].cog_fitness = self.agents[cur].adopt_existing_solution(
                        self.agents[target].decision_space, self.agents[target].state, self.agents[target].knowledge_tree_list,
                        self.agents[target].tree_depth
                    )
            elif learn_method=="personal_vote":
                for cur in range(len(self.agents)):
                    personal_vote_list = self.agents[cur].evaluate_all_solution(
                        [self.agents[x].state for x in range(len(self.agents)) if x != cur], vote_percentage
                    )
                    target = personal_vote_list[0]

                    self.agents[cur].state, self.agents[cur].cog_fitness = self.agents[cur].adopt_existing_solution(
                        self.agents[target].decision_space, self.agents[target].state,
                        self.agents[target].knowledge_tree_list,
                        self.agents[target].tree_depth
                    )
            elif learn_method=="aggregate_vote":
                aggregate_vote_list = []
                for cur in range(len(self.agents)):
                    personal_vote_list = self.agents[cur].evaluate_all_solution(
                        [self.agents[x].state for x in range(len(self.agents)) if x != cur], vote_percentage
                    )
                    aggregate_vote_list.append(personal_vote_list)
                aggregate_vote_count = self.aggregate_vote_number(aggregate_vote_list)

                for cur in range(len(self.agents)):
                    ps = [aggregate_vote_count[x] for x in range(len(self.agents)) if x != cur]
                    ps = [x/np.sum(ps) for x in ps]
                    target = np.random.choice([x for x in range(len(self.agents)) if x != cur], 1, p=ps)[0]

                    self.agents[cur].state, self.agents[cur].cog_fitness = self.agents[cur].adopt_existing_solution(
                        self.agents[target].decision_space, self.agents[target].state,
                        self.agents[target].knowledge_tree_list,
                        self.agents[target].tree_depth
                    )


def simulation(return_dic, idx, repeat, period, N, k, agentNum, knowledgeNum, generalistProportion,
               defaultValues, learn, learn_method, vote_percentage=0.1):

    platform_average_fitness = []
    platform_unique_solution = []
    platform_overall_fitness = []
    platform_max_fitness = []
    generalist_average_fitness = []
    generalist_unique_solution = []
    generalist_max_fitness = []
    specialist_average_fitness = []
    specialist_unique_solution = []
    specialist_max_fitness = []

    for r in range(repeat):

        print(r, time.asctime())
        # print(time.asctime())

        np.random.seed(None)

        temp_platform_average_fitness = []
        temp_platform_unique_solution = []
        temp_platform_overall_fitness = []
        temp_platform_max_fitness = []
        temp_generalist_average_fitness = []
        temp_generalist_unique_solution = []
        temp_generalist_max_fitness = []
        temp_specialist_average_fitness = []
        temp_specialist_unique_solution = []
        temp_specialist_max_fitness = []

        platform = Platform(N, k, agentNum, knowledgeNum, generalistProportion, defaultValues)

        for step in range(period):

            # print(step)

            platform.adaptation(learn, learn_method, vote_percentage)

            # for cutoff_threshold in [0, 0.4, 0.6, 0.8, 1.0]:
            #     results = platform.extract_results(cutoff_threshold)

            results_list = [platform.extract_results(cutoff_threshold) for cutoff_threshold in [0, 0.4, 0.6, 0.8]]

            temp_platform_average_fitness.append([results[0] for results in results_list])
            temp_platform_unique_solution.append([results[1] for results in results_list])
            temp_platform_overall_fitness.append([results[2] for results in results_list])
            temp_platform_max_fitness.append([results[3] for results in results_list])
            temp_generalist_average_fitness.append([results[4] for results in results_list])
            temp_generalist_unique_solution.append([results[5] for results in results_list])
            temp_generalist_max_fitness.append([results[6] for results in results_list])
            temp_specialist_average_fitness.append([results[7] for results in results_list])
            temp_specialist_unique_solution.append([results[8] for results in results_list])
            temp_specialist_max_fitness.append([results[9] for results in results_list])

        platform_average_fitness.append(temp_platform_average_fitness)
        platform_unique_solution.append(temp_platform_unique_solution)
        platform_overall_fitness.append(temp_platform_overall_fitness)
        platform_max_fitness.append(temp_platform_max_fitness)
        generalist_average_fitness.append(temp_generalist_average_fitness)
        generalist_unique_solution.append(temp_generalist_unique_solution)
        generalist_max_fitness.append(temp_generalist_max_fitness)
        specialist_average_fitness.append(temp_specialist_average_fitness)
        specialist_unique_solution.append(temp_specialist_unique_solution)
        specialist_max_fitness.append(temp_specialist_max_fitness)

        # print(np.mean(np.array(platform_average_fitness), axis=0), "zhou junjie sb!")

    return_dic[idx] = (
        np.mean(np.array(platform_average_fitness), axis=0),
        np.mean(np.array(platform_unique_solution), axis=0),
        np.mean(np.array(platform_overall_fitness), axis=0),
        np.mean(np.array(platform_max_fitness), axis=0),
        np.mean(np.array(generalist_average_fitness), axis=0),
        np.mean(np.array(generalist_unique_solution), axis=0),
        np.mean(np.array(generalist_max_fitness), axis=0),
        np.mean(np.array(specialist_average_fitness), axis=0),
        np.mean(np.array(specialist_unique_solution), axis=0),
        np.mean(np.array(specialist_max_fitness), axis=0),
    )






