from MultiStateInfluentialLandscape import *
from tools import *

# an extension of generalFramework

# let's first ignore specialist vs. generalist

# firms' policy
# 1. open vs. closed: whether other crowd's configuration is visible -> openness
# 1.1 probability or proportion, two extreme cases would be identical but not for middle stages
# 2. feedback vs. no feedback: in progress feedback frequency
# 2.1 feedback about your performance vs. feedback about the whole population
# 3. incentives -> aspiration level: 1st/10%/50%/don't care
# 3.1 assumption -> agents wouldn't do long jumps if they are above aspiration

# firm's property
# 1. knowledge number

# crowd's property
# 1. knowledge number
# 2. information collection


class Crowd:

    def __init__(self, N, knowledge_num, aspiration, landscape=None, state_num=2,
                 opensource_probability=None, information_ratio=None
                 ):

        self.N = N
        self.state_num = state_num
        self.state = np.random.choice([cur for cur in range(state_num)], self.N).tolist()
        self.decision_space = np.random.choice(self.N, knowledge_num, replace=False).tolist()

        self.landscape = landscape

        self.aspiration = aspiration
        self.rank = aspiration
        self.opensource_probability = opensource_probability
        self.opensourced_state = None
        self.feedback_in_current_period = False

        self.information_ratio = information_ratio

    def local_search(self, ):

        temp_state = list(self.state)

        c = np.random.choice(self.decision_space)

        temp_state[c] ^= 1
        if self.landscape.query_cog_fitness(
            temp_state, self.decision_space
        ) > self.landscape.query_cog_fitness(
            self.state, self.decision_space
        ):
            return list(temp_state)
        else:
            return list(self.state)

    def distant_search(self, step_list=[2, 3, 4]):

        step = np.random.choice(step_list)

        temp_state = list(self.state)
        choices = np.random.choice(self.decision_space, step, replace=False).tolist()
        for c in choices:
            temp_state[c] ^= 1
        return temp_state

    def mimic(self, target, first_time=False):

        if first_time:
            temp_state = np.random.choice([cur for cur in range(self.state_num)], self.N).tolist()
        else:
            temp_state = list(self.state)

        for cur in range(self.N):
            if cur in self.decision_space:
                temp_state[cur] = target.opensourced_state[cur]
        return list(temp_state)

    def check_cog_optimal(self, ):

        current_cog = self.landscape.query_cog_fitness(self.state, self.decision_space)

        for cur in self.decision_space:
            temp_state = list(self.state)
            temp_state[cur] ^= 1
            if self.landscape.query_cog_fitness(temp_state, self.decision_space) >= current_cog:
                return False
        return True

    def select_imitation_target(self, reference, rank=False):
        crowd_num = len(reference)
        reference = [ref for ref in reference if ref.opensourced_state is not None]
        if len(reference) == 0:
            return None
        if not rank:
            sorted_reference = [(ref, self.landscape.query_cog_fitness(ref.state, self.decision_space)) for ref in reference]
            sorted_reference.sort(key=lambda x: -x[1])
            sorted_reference = sorted_reference[:min(len(sorted_reference), int(crowd_num*self.information_ratio))]
            p = [x[1] for x in sorted_reference]
            p = [x / np.sum(p) for x in p]
            return np.random.choice([x[0] for x in sorted_reference], p=p)
        else:
            sorted_reference = sorted(reference, key=lambda x: x.rank)
            sorted_reference = sorted_reference[:min(len(sorted_reference), int(crowd_num*self.information_ratio))]
            p = [self.landscape.query_fitness(x.state) for x in sorted_reference]
            p = [x/np.sum(p) for x in p]
            return np.random.choice(sorted_reference, p=p)

    def adaptation(self, reference, imitation_prob=0.5, rank=False):

        if self.rank <= self.aspiration and self.feedback_in_current_period:
            self.state = self.local_search()
        else:
            if self.check_cog_optimal():
                if np.random.uniform(0, 1) < imitation_prob:
                    target = self.select_imitation_target(reference, rank)
                    if target is None:
                        self.state = self.distant_search()
                    else:
                        self.state = self.mimic(target, first_time=False)
                else:
                    self.state = self.distant_search()
            else:
                self.state = self.local_search()


class Firm:

    # firms are also with limited cognition
    def __init__(self, N, decision_num, state_num=2, landscape=None):
        self.N = N
        self.decision_space = np.random.choice(self.N, decision_num, replace=False).tolist()
        self.state_num = state_num
        self.state = np.random.choice([cur for cur in range(state_num)], self.N).tolist()
        self.opensourced_state = list(self.state)
        self.landscape = landscape

    def search(self):
        c = np.random.choice(self.decision_space)
        temp_state = list(self.state)
        temp_state[c] ^= 1

        if self.landscape.query_cog_fitness(
            temp_state, self.decision_space
        ) > self.landscape.query_cog_fitness(
            self.state, self.decision_space
        ):
            self.state = list(temp_state)
        self.opensourced_state = list(self.state)

    def provide_feedback(self, crowds):
        fitness = []
        for crowd_index, crowd in enumerate(crowds):
            fitness.append((self.landscape.query_cog_fitness(crowd.state, self.decision_space), crowd_index))
        fitness.sort(key=lambda x: -x[0])
        return [x[1] for x in fitness]

    def pick_up(self, crowds):
        temp_state = list(self.state)
        for crowd in crowds:
            if self.landscape.query_cog_fitness(
                crowd.state, self.decision_space
            ) > self.landscape.query_cog_fitness(
                temp_state, self.decision_space
            ):
                temp_state = list(crowd.state)
        self.state = list(temp_state)
        self.opensourced_state = list(self.state)


class Industry:

    def __init__(self, N, state_num, firm_knowledge_num, crowd_num, crowd_knowledge_num, crowd_information_ratio,
                 aspiration_ratio, landscape, time_interval, feedback_frequency, openness, crowd_imitation_prob=0.5,
                 population_feedback=False
                 ):
        """
        """

        self.firm = Firm(N, firm_knowledge_num, state_num, landscape)
        self.crowd_num = crowd_num

        self.aspiration = int(self.crowd_num*aspiration_ratio)

        self.crowds = []
        for cur in range(crowd_num):
            crowd = Crowd(
                N, crowd_knowledge_num, self.aspiration, landscape, state_num, openness, crowd_information_ratio
            )
            self.crowds.append(crowd)

        self.time_interval = time_interval
        self.tag = 0

        self.feedback_frequency = feedback_frequency
        self.crowd_imitation_prob = crowd_imitation_prob
        self.population_feedback = population_feedback

    def get_index(self, step):

        if step + 1 > self.time_interval[-1]:
            return -1

        for time_index, time in enumerate(self.time_interval):

            if step + 1 <= time:
                return time_index

    def adaptation(self, step, ):

        index = self.get_index(step)

        if index % 2 == 0:
            self.firm.search()
            next_index = self.get_index(step + 1)
            if next_index != -1 and next_index % 2 == 1:
                for crowd in self.crowds:
                    crowd.state = crowd.mimic(target=self.firm, first_time=True)
        else:

            for crowd in self.crowds:
                crowd.adaptation(self.crowds, self.crowd_imitation_prob, self.population_feedback)
                if np.random.uniform(0, 1) < crowd.opensource_probability:
                    crowd.opensourced_state = list(crowd.state)
            if np.random.uniform(0, 1) < self.feedback_frequency:
                crowd_rank = self.firm.provide_feedback(self.crowds)

                for rank, crowd_index in enumerate(crowd_rank):
                    self.crowds[crowd_index].rank = rank
                    self.crowds[crowd_index].feedback_in_current_period = True
            else:
                for crowd in self.crowds:
                    crowd.feedback_in_current_period = False

            next_index = self.get_index(step + 1)
            if next_index != -1 and next_index % 2 == 0:
                self.firm.pick_up(self.crowds)
            elif next_index == -1:
                self.firm.pick_up(self.crowds)


def simulation(return_dic, idx, N, k, land_num, firm_knowledge_num, crowd_num, period, crowd_knowledge_num,
               aspiration_ratio, time_interval, feedback_frequency, population_feedback,openness, state_num=2,
               crowd_information_ratio=0.25, crowd_imitation_prob=0.5, IM_type="random",
               ):

    ress_firm_fitness = []
    ress_firm_rank = []

    ress_crowd_fitness = []
    ress_crowd_rank = []

    for repeat in range(land_num):

        print(repeat)

        np.random.seed(None)

        res_firm_fitness = []
        res_firm_rank = []
        res_crowd_fitness = []
        res_crowd_rank = []

        landscape = LandScape(N, k, IM_type, state_num=state_num)
        landscape.initialize()

        industry = Industry(
            N, state_num, firm_knowledge_num, crowd_num, crowd_knowledge_num, crowd_information_ratio, aspiration_ratio,
            landscape, time_interval, feedback_frequency, openness, crowd_imitation_prob, population_feedback,
        )

        for step in range(period):
            industry.adaptation(step)
            res_firm_fitness.append(industry.firm.landscape.query_fitness(industry.firm.state))
            print(res_firm_fitness[-1])
            res_firm_rank.append(industry.firm.landscape.fitness_to_rank_dict[res_firm_fitness[-1]])
            res_crowd_fitness.append([industry.firm.landscape.query_fitness(crowd.state) for crowd in industry.crowds])
            res_crowd_rank.append([industry.firm.landscape.fitness_to_rank_dict[x] for x in res_crowd_fitness[-1]])

        ress_firm_fitness.append(res_firm_fitness)
        ress_firm_rank.append(res_firm_rank)
        ress_crowd_fitness.append(res_crowd_fitness)
        ress_crowd_rank.append(res_crowd_rank)

    return_dic[idx] = (
        ress_firm_fitness, ress_firm_rank, ress_crowd_fitness, ress_crowd_rank,
    )

