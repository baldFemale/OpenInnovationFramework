from MultiStateInfluentialLandscape import *
from tools import *


# a triangle of GST distribution in different crowd-based tasks
# when there is no aspiration -> firms provide no in-progress feedback to crowds

class Crowd:

    def __init__(self, N, knowledge_num, specialist_num, aspiration, lr=0, landscape=None, state_num=4):

        self.N = N
        self.state_num = state_num
        self.state = np.random.choice([cur for cur in range(state_num)], self.N).tolist()
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

        self.aspiration = aspiration
        self.rank = aspiration

    def local_search(self, ):

        # local area

        temp_state = list(self.state)

        c = np.random.choice(self.decision_space)

        if c in self.specialist_knowledge_space:
            current_state = temp_state[c]
            new_state = np.random.choice([cur for cur in range(self.state_num) if cur != current_state])
            temp_state[c] = new_state

        else:
            focal_flag = temp_state[c] // 2
            focal_flag = focal_flag ^ 1
            temp_state[c] = self.generalist_map_dic[c][focal_flag]

        cognitive_state = self.change_state_to_cog_state(self.state)
        cognitive_temp_state = self.change_state_to_cog_state(temp_state)

        if self.landscape.query_cog_fitness_gst(
            cognitive_state, self.generalist_knowledge_space, self.specialist_knowledge_space
        ) > self.landscape.query_cog_fitness_gst(
            cognitive_temp_state, self.generalist_knowledge_space, self.specialist_knowledge_space
        ):
            return list(self.state)
        else:
            return list(temp_state)

    def distant_search(self, step=3):
        temp_state = list(self.state)
        choices = np.random.choice(self.decision_space, step, replace=False).tolist()
        for c in choices:

            if c in self.specialist_knowledge_space:
                current_state = temp_state[c]
                new_state = np.random.choice([cur for cur in range(self.state_num) if cur != current_state])
                temp_state[c] = new_state
            else:
                focal_flag = temp_state[c] // 2
                focal_flag = focal_flag ^ 1
                temp_state[c] = self.generalist_map_dic[c][focal_flag]

        return temp_state

    def change_state_to_cog_state(self, state):
        temp_state = []
        for cur in range(len(state)):
            if cur in self.generalist_knowledge_space:
                temp_state.append(state[cur] // 2)
            else:
                temp_state.append(state[cur])
        return temp_state

    def mimic(self, target, first_time=False):

        if first_time:
            temp_state = np.random.choice([cur for cur in range(self.state_num)], self.N).tolist()
        else:
            temp_state = list(self.state)

        for cur in range(self.N):
            if cur in self.specialist_knowledge_space:
                temp_state[cur] = target.state[cur]
            elif cur in self.generalist_knowledge_space:
                focal_index = target.state[cur]//2
                temp_state[cur] = self.generalist_map_dic[cur][focal_index]
        return list(temp_state)

    def check_cog_optimal(self, ):

        current_cog = self.landscape.query_cog_fitness_gst(
            self.change_state_to_cog_state(self.state),
            self.generalist_knowledge_space,
            self.specialist_knowledge_space
        )

        for cur in self.decision_space:
            temp_state = list(self.state)

            if cur in self.generalist_knowledge_space:

                focal_flag = temp_state[cur] // 2
                focal_flag = focal_flag ^ 1
                temp_state[cur] = self.generalist_map_dic[cur][focal_flag]

                if self.landscape.query_cog_fitness_gst(
                    self.change_state_to_cog_state(temp_state),
                    self.generalist_knowledge_space,
                    self.specialist_knowledge_space
                ) >= current_cog:
                    return False
            else:
                for new_state in [cur for cur in range(self.state_num) if cur != temp_state[cur]]:
                    temp_state[cur] = new_state

                    if self.landscape.query_cog_fitness_gst(
                        self.change_state_to_cog_state(temp_state),
                        self.generalist_knowledge_space,
                        self.specialist_knowledge_space
                    ) >= current_cog:
                        return False
        return True

    def adaptation_with_imitation(self, reference, imitation_prob):
        # above aspiration
        if self.rank <= self.aspiration:
            self.state = self.local_search()
        # below aspiration
        else:
            if self.check_cog_optimal():
                if np.random.uniform(0, 1) < imitation_prob:
                    p = [self.landscape.query_fitness(firm.state) for firm in reference]
                    p = [x / np.sum(p) for x in p]
                    choice = np.random.choice(len(p), 1, p=p)[0]
                    target = reference[choice]
                    self.state = self.mimic(target, first_time=False)
                else:
                    self.state = self.distant_search()
            else:
                self.state = self.local_search()

    def adaptation_without_reference_with_imitation(self, reference, imitation_prob):

        if self.check_cog_optimal():
            if np.random.uniform(0, 1) < imitation_prob:

                current_state = list(self.state)
                current_cog_state = self.change_state_to_cog_state(current_state)
                current_cog_fitness = self.landscape.query_cog_fitness_gst(
                    current_cog_state,
                    self.generalist_knowledge_space,
                    self.specialist_knowledge_space
                )

                for ref in reference:
                    temp_state = self.mimic(ref, first_time=False)
                    temp_cog_state = self.change_state_to_cog_state(temp_state)
                    temp_cog_fitness = self.landscape.query_cog_fitness_gst(
                        temp_cog_state,
                        self.generalist_knowledge_space,
                        self.specialist_knowledge_space,
                    )
                    if temp_cog_fitness > current_cog_fitness:
                        current_state, current_cog_state, current_cog_fitness = temp_state, temp_cog_state, temp_cog_fitness
                self.state = current_state
            else:
                self.state = self.distant_search()
        else:
            self.state = self.local_search()


class Firm:

    # suppose firms have all knowledge -> utlize crowd as a distant search
    def __init__(self, N, state_num=4, landscape=None):
        self.N = N
        self.state_num = state_num
        self.state = np.random.choice([cur for cur in range(state_num)], self.N).tolist()
        self.landscape = landscape

    def search(self):
        c = np.random.choice(self.N)

        temp_state = list(self.state)

        current_state = temp_state[c]
        new_state = np.random.choice([cur for cur in range(self.state_num) if cur != current_state])
        temp_state[c] = new_state

        if self.landscape.query_fitness(temp_state) > self.landscape.query_fitness(self.state):
            self.state = list(temp_state)

    def pick_up(self, crowds):

        temp_state = list(self.state)
        for crowd in crowds:

            if self.landscape.query_fitness(crowd.state) > self.landscape.query_fitness(temp_state):
                temp_state = crowd.state
        self.state = list(temp_state)


class Industry:

    def __init__(self, N, state_num, crowd_num, crowd_proportion, crowd_knowledge, crowd_special_knowledge,
                 aspiration_ratio, landscape, time_interval, in_progress_feedback, open_env,
                 ):
        """

        :param N:
        :param state_num:
        :param crowd_num:
        :param crowd_proportion: for simplification, let's first ignore T shape [0.5, 1.0] -> G 50%, S 50%
        :param crowd_knowledge:
        :param crowd_special_knowledge:
        :param aspiration_ratio: assume homogeneous aspiration
        :param landscape:
        :param time_interval: [10, 20, 50] -> total 50 periods, first 10 firms, second 10 crowd, last 30 firms
        :param in_progress_feedback:
        :param open_env
        """

        self.firm = Firm(N, state_num, landscape)
        self.crowd_num = crowd_num

        self.aspiration = int(self.crowd_num*aspiration_ratio)

        self.crowds = []
        for cur in range(crowd_num):

            if cur < int(crowd_proportion[0]*self.crowd_num):
                crowd = Crowd(N, crowd_knowledge[0], crowd_special_knowledge[0], self.aspiration, 0, landscape, state_num)
            elif cur < int(crowd_proportion[1]*self.crowd_num):
                crowd = Crowd(N, crowd_knowledge[1], crowd_special_knowledge[1], self.aspiration, 0, landscape, state_num)

            self.crowds.append(crowd)

        self.time_interval = time_interval
        self.tag = 0

        self.in_progress_feedback = in_progress_feedback
        self.open_env = open_env

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

            # assume in-progress feedback

            if self.in_progress_feedback:
                rank = [(crowd, self.firm.landscape.query_fitness(crowd.state)) for crowd in self.crowds]
                rank.sort(key=lambda x: -x[1])
                reference = [rank[cur][0] for cur in range(self.aspiration)]
                for cur in range(len(rank)):
                    rank[cur][0].rank = cur + 1

                if self.open_env:
                    for crowd in self.crowds:
                        crowd.adaptation_with_imitation(reference, imitation_prob=0.5)
                else:
                    for crowd in self.crowds:
                        crowd.adaptation_with_imitation(reference, imitation_prob=0.0)
            else:
                rank = [(crowd, self.firm.landscape.query_fitness(crowd.state)) for crowd in self.crowds]
                reference = [rank[cur][0] for cur in range(self.aspiration)]
                for cur in range(len(rank)):
                    rank[cur][0].rank = cur + 1

                if self.open_env:
                    for crowd in self.crowds:
                        crowd.adaptation_without_reference_with_imitation(reference, imitation_prob=0.5)
                else:
                    for crowd in self.crowds:
                        crowd.adaptation_without_reference_with_imitation(reference, imitation_prob=0.0)

            next_index = self.get_index(step + 1)
            if next_index != -1 and next_index % 2 == 0:
                self.firm.pick_up(self.crowds)
            elif next_index == -1:
                self.firm.pick_up(self.crowds)


def simulation(return_dic, idx, N, k, land_num, crowd_num, period, crowd_proportion, crowd_knowledge,
               crowd_special_knowledge, aspiration_ratio, time_interval, state_num=4, IM_type="random",
               in_progress_feedback=False, open_env=False
               ):

    ress_fitness = []
    ress_rank = []

    for repeat in range(land_num):

        print(repeat)

        np.random.seed(None)

        res_fitness = []
        res_rank = []

        landscape = LandScape(N, k, IM_type, state_num=state_num)
        landscape.initialize()

        industry = Industry(
            N, state_num, crowd_num, crowd_proportion, crowd_knowledge,
            crowd_special_knowledge, aspiration_ratio, landscape, time_interval, in_progress_feedback, open_env
        )

        for step in range(period):
            industry.adaptation(step)
            res_fitness.append(industry.firm.landscape.query_fitness(industry.firm.state))
            res_rank.append(industry.firm.landscape.fitness_to_rank_dict[res_fitness[-1]])

        ress_fitness.append(res_fitness)
        ress_rank.append(res_rank)

    return_dic[idx] = ress_fitness, ress_rank

