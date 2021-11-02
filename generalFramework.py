from LandScape import *

"""
The probability based model doesn't work 
Manually specify time interval 
Even for a simple refinement there could be many variation in its duration, location (i.e., at the beginning or at the end) 
"""


class Firm:

    def __init__(self, N, decision_num, landscape=None):

        self.N = N
        self.state = np.random.choice([0, 1], N).tolist()
        self.decision_space = np.random.choice(N, decision_num, replace=False).tolist()
        self.knowledge_space = list(self.decision_space)
        self.landscape = landscape

    def independent_search(self, ):
        base_state = list(self.state)

        c = np.random.choice(self.decision_space)

        base_state[c] ^= 1

        if self.landscape.query_cog_fitness(
                base_state, self.knowledge_space
        ) > self.landscape.query_cog_fitness(
            self.state, self.knowledge_space
        ):
            self.state = list(base_state)

    def pickup(self, alternatives):

        base_state = list(self.state)

        for alter in alternatives:
            if self.landscape.query_cog_fitness(
                alter, self.knowledge_space
            ) > self.landscape.query_cog_fitness(
                base_state, self.knowledge_space
            ):
                base_state = list(alter)

        self.state = list(base_state)


class Agent:

    def __init__(self, N, knowledge_num, landscape=None):

        self.N = N
        self.state = np.random.choice([0, 1], self.N).tolist()
        self.decision_space = np.random.choice(self.N, knowledge_num, replace=False).tolist()
        self.knowledge_space = list(self.decision_space)

        self.landscape = landscape

    def independent_search(self, ):

        base_state = list(self.state)

        c = np.random.choice(self.decision_space)

        base_state[c] ^= 1

        if self.landscape.query_cog_fitness(
            base_state, self.knowledge_space
        ) > self.landscape.query_cog_fitness(
            self.state, self.knowledge_space
        ):
            self.state = list(base_state)


class Industry:

    def __init__(self, N, firm_knowledge, agent_knowledge, agent_num, landscape,
                 time_interval
                 ):

        self.firm = Firm(N, firm_knowledge, landscape)
        self.agents = [Agent(N, agent_knowledge, landscape) for cur in range(agent_num)]
        for agent in self.agents:
            agent.state = list(self.firm.state)

        self.time_interval = time_interval
        self.tag = 0

    def get_index(self, step):

        if step+1 > self.time_interval[-1]:
            return -1

        for time_index, time in enumerate(self.time_interval):

            if step+1 <= time:
                return time_index

    def adaptation(self, step, ):

        index = self.get_index(step)

        if index % 2 == 0:
            self.firm.independent_search()
            next_index = self.get_index(step+1)
            if next_index != -1 and next_index % 2 == 1:
                for agent in self.agents:
                    agent.state = list(self.firm.state)
        else:
            for agent in self.agents:
                agent.independent_search()
            next_index = self.get_index(step+1)
            if next_index != -1 and next_index % 2 == 0:
                self.firm.pickup([agent.state for agent in self.agents])
            elif next_index == -1:
                self.firm.pickup([agent.state for agent in self.agents])


def simulation(N, k, land_num, agent_num, period, firm_knowledge, agent_knowledge,
               time_interval
               ):

    ress_fitness = []

    for repeat in range(land_num):

        print(repeat)

        np.random.seed(None)

        res_fitness = []

        np.random.seed(None)

        landscape = LandScape(N, k, None, None)
        landscape.initialize()

        industry = Industry(
            N, firm_knowledge, agent_knowledge, agent_num, landscape, time_interval
        )

        for step in range(period):
            industry.adaptation(step)
            print(firm_knowledge, industry.firm.landscape.query_fitness(industry.firm.state))
            res_fitness.append(industry.firm.landscape.query_fitness(industry.firm.state))

        ress_fitness.append(res_fitness)

    return ress_fitness











