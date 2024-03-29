from MultiStateInfluentialLandscape import *
from tools import *
import time

# no default value and random state across G domain
# always learning & fixed openness


class TreeNode:
    def __init__(self, val, left, right):
        self.val = val
        self.left = left
        self.right = right


class KnowledgeBinaryTree:
    """
    There will be more than one tree satisfy depth and stateNum constrains with depth is larger than 3
    We use FIFO method, so knowledge will distribute evenly and be as closed as possible to the leaf node
    """

    def __init__(self, depth, stateNum):
        """
        :param depth: the depth of the binary tree
        :param stateNum: how many states the agent know about the focal decision
        """
        self.depth = depth
        self.stateNum = stateNum
        self.head = self.initial_tree()
        self.node_list = []
        self.node_map_leaf_list = {}
        self.node_map_leaves_list = {}
        self.leaf_map_node_list = {}
        self.initial_knowledge_structure()

    def initial_tree(self):

        cur = 0
        head = TreeNode(cur, None, None)
        queue = [head]
        last_width = pow(2, self.depth-1)
        random_val_list = np.arange(last_width)

        while cur < pow(2, self.depth)-1-1:
            node = queue.pop(0)
            cur += 1
            if cur >= pow(2, self.depth-1)-1:
                left_child = TreeNode(pow(2, self.depth-1)-1+random_val_list[cur-pow(2, self.depth-1)+1], None, None)
                node.left = left_child
                queue.append(left_child)
                cur += 1
                right_child = TreeNode(pow(2, self.depth-1)-1+random_val_list[cur-pow(2, self.depth-1)+1], None, None)
                node.right = right_child
                queue.append(right_child)
            else:
                left_child = TreeNode(cur, None, None)
                node.left = left_child
                queue.append(left_child)
                cur += 1
                right_child = TreeNode(cur, None, None)
                node.right = right_child
                queue.append(right_child)
        return head

    def search_map_leaf(self, node_list, remain_depth=0):

        next_node_list = []

        for node in node_list:
            if node.left is None and node.right is None:
                return node_list, remain_depth
            else:
                next_node_list.append(node.left)
                next_node_list.append(node.right)
        return self.search_map_leaf(next_node_list, int(remain_depth+1))

    def initial_knowledge_structure(self):

        queue = [self.head]
        while len(queue)!=self.stateNum:
            node = queue.pop(0)
            queue.append(node.left)
            queue.append(node.right)
        self.node_list = [node.val for node in queue]

        for node in queue:
            leaf_nodes, remain_depth = self.search_map_leaf([node], 0)
            self.node_map_leaves_list[node.val] = [leaf.val for leaf in leaf_nodes]
            map_leaf_node = np.random.choice(leaf_nodes)
            #map_leaf_node = leaf_nodes[0]
            self.node_map_leaf_list[node.val] = map_leaf_node.val

            for leaf in leaf_nodes:
                self.leaf_map_node_list[leaf.val] = node.val


class Agent:

    def __init__(self, N, changeable_state_list, decision_order, beta_v, landscape=None, state_num=2, openness=1):

        """
        :param N: param N in NK model
        :param changeable_state_list: A list of state number an agent know, e.g., [4, 4], knowledge outside this list takes 0 as default
        :param landscape: landscape
        :param state_num: state number in landscape, 2^depth=state_num
        """

        self.N = N
        self.state_num = state_num
        self.decision_space = generate_knowledge(decision_order, beta_v, len(changeable_state_list))
        # self.decision_space = np.random.choice(self.N, len(changeable_state_list), replace=False).tolist()
        self.changeable_state = changeable_state_list
        self.changeable_probability = [
            (x - 1) / (sum(self.changeable_state) - len(self.changeable_state)) for x in self.changeable_state
        ]

        self.tree_depth = stateNumToDepth(self.state_num)

        self.knowledge_tree_list = [
            KnowledgeBinaryTree(self.tree_depth, self.changeable_state[cur]) for cur in
            range(len(self.changeable_state))
        ]

        # initialize state
        self.state = []
        for cur in range(self.N):
            if cur not in self.decision_space:
                self.state.append(np.random.choice(self.state_num))
            else:
                index = self.decision_space.index(cur)
                val = np.random.choice(list(self.knowledge_tree_list[index].node_map_leaf_list.values()))
                val -= (pow(2, self.tree_depth - 1) - 1)
                self.state.append(val)
        self.landscape = landscape

        self.openness = openness
        self.open_tag = True if np.random.uniform(0, 1) < openness else False

        self.cache = {}

    def independent_search(self,):
        """
        random choice from all possible alternatives
        :return:
        """

        d = np.random.choice(self.decision_space, 1, p=self.changeable_probability)[0]
        index = self.decision_space.index(d)
        node_list = self.knowledge_tree_list[index].node_list
        alternatives = [
            cur for cur in node_list if self.knowledge_tree_list[index].node_map_leaf_list[cur] - (
                    pow(2, self.tree_depth - 1) - 1) != self.state[d]
        ]

        # print(len(self.decision_space), len(alternatives))

        alter = np.random.choice(alternatives)
        temp_state = list(self.state)
        temp_state[d] = self.knowledge_tree_list[index].node_map_leaf_list[alter] - (pow(2, self.tree_depth - 1) - 1)
        temp_bit = "".join([str(x) for x in temp_state])
        if temp_bit in self.cache:
            new_cog_fitness = self.cache[temp_bit]
        else:
            new_cog_fitness = self.landscape.query_partial_fitness_tree(
                temp_state, self.decision_space, self.knowledge_tree_list, self.tree_depth
            )
            self.cache[temp_bit] = float(new_cog_fitness)

        current_bit = "".join([str(x) for x in self.state])
        if current_bit in self.cache:
            current_cog_fitness = self.cache[current_bit]
        else:
            current_cog_fitness = self.landscape.query_partial_fitness_tree(
                self.state, self.decision_space, self.knowledge_tree_list, self.tree_depth
            )
            self.cache[current_bit] = float(current_cog_fitness)
        if new_cog_fitness >= current_cog_fitness:
            return list(temp_state), new_cog_fitness
        else:
            return list(self.state), current_cog_fitness

    def overlap_calculator(self, target_decision, ):
        overlap_decision = []
        for d in self.decision_space:
            if d in target_decision:
                overlap_decision.append(d)
        return len(overlap_decision)

    def evaluate_all_solution(self, target_state_list):

        if len(target_state_list)==0:
            return []

        perceived_fitness = []

        for state_index, state in enumerate(target_state_list):
            temp_state = list(state)
            temp_bit = "".join([str(x) for x in temp_state])
            if temp_bit in self.cache:
                cog_fitness = self.cache[temp_bit]
            else:
                cog_fitness = self.landscape.query_partial_fitness_tree(temp_state, self.decision_space, self.knowledge_tree_list, self.tree_depth)
                self.cache[temp_bit] = float(cog_fitness)

            perceived_fitness.append(
                (
                    state_index, cog_fitness

                )
            )
        perceived_fitness.sort(key=lambda x: -x[1])
        return perceived_fitness

    def adopt_existing_solution(self, target_decision, target_state, target_knowledge_structure, target_tree_depth):

        # temp_state = list(self.state)
        # for d in target_decision:
        #     v = target_state[d]
        #     d_index = target_decision.index(d)
        #     node = target_knowledge_structure[d_index].leaf_map_node_list[v+(pow(2, target_tree_depth-1)-1)]
        #     node_alternative = target_knowledge_structure[d_index].node_map_leaves_list[node]
        #     node_alternative = [x-(pow(2, target_tree_depth-1)-1) for x in node_alternative]
        #     # print(node_alternative)
        #     temp_state[d] = np.random.choice(node_alternative)

        temp_state = list(target_state)
        temp_bit = "".join([str(x) for x in temp_state])
        if temp_bit in self.cache:
            new_cog_fitness = self.cache[temp_bit]
        else:
            new_cog_fitness = self.landscape.query_partial_fitness_tree(
                temp_state, self.decision_space, self.knowledge_tree_list, self.tree_depth
            )
            self.cache[temp_bit] = float(new_cog_fitness)

        current_bit = "".join([str(x) for x in self.state])
        if current_bit in self.cache:
            current_cog_fitness = self.cache[current_bit]
        else:
            current_cog_fitness = self.landscape.query_partial_fitness_tree(
                self.state, self.decision_space, self.knowledge_tree_list, self.tree_depth
            )
            self.cache[current_bit] = float(current_cog_fitness)

        if new_cog_fitness >= current_cog_fitness:
            # print("learning is happening:", self.decision_space, target_decision)
            return list(temp_state), new_cog_fitness
        else:
            return list(self.state), current_cog_fitness

    def measure_surface_quality(self):
        return_state = list(self.state)
        potential_list = []

        unknown_decision = [cur for cur in range(self.N) if cur not in self.decision_space]

        for cur in range(pow(4, self.N-len(self.decision_space))):
            bit = numberToBase(cur, 4)
            bit = "0"*(self.N-len(self.decision_space)-len(bit)) + bit

            temp_state = list(self.state)
            for inner_cur in range(len(unknown_decision)):
                temp_state[unknown_decision[inner_cur]] = int(bit[inner_cur])
            potential_list.append(self.landscape.query_fitness(temp_state))

        exploitation_potential_list = []
        for cur in range(pow(4, len(self.decision_space))):
            bit = numberToBase(cur, 4)
            bit = "0" * (len(self.decision_space) - len(bit)) + bit

            temp_state = list(self.state)

            for inner_cur in range(len(self.decision_space)):
                temp_state[self.decision_space[inner_cur]] = int(bit[inner_cur])
            exploitation_potential_list.append(self.landscape.query_fitness(temp_state))

        print(self.landscape.query_fitness(return_state), np.max(exploitation_potential_list))

        return (
            return_state,
            np.max(exploitation_potential_list),
            np.mean(potential_list),
            self.landscape.fitness_to_rank_dict[self.landscape.query_fitness(return_state)]
        )


class Platform:

    """
    generalist -> 2
    specialist -> 4
    """

    def __init__(
            self, N, k, agentNum, knowledgeNum, generalistProportion, beta_v,
            defaultValues=True, state_num=2, openness=1, g_p=0.5, s_p=0.5, error_probability=0,
    ):

        self.N = N
        self.landscape = LandScape(N, k, IM_type="random", state_num=state_num)
        self.state_num = state_num
        self.landscape.initialize()

        self.agents = []
        self.agent_types = []

        self.g_p = g_p
        self.s_p = s_p

        self.error_probability=error_probability

        default_state = np.random.choice(state_num, N).tolist()

        decision_order = np.random.choice(N, N, replace=False).tolist()

        for cur in range(agentNum):
            if cur < int(generalistProportion*agentNum):
                agent = Agent(
                    self.N, [2 for cur in range(knowledgeNum//2)], decision_order,
                    beta_v, self.landscape, state_num=state_num, openness=openness
                )
                self.agent_types.append("g")
            else:
                agent = Agent(
                    self.N, [4 for cur in range(knowledgeNum//4)], decision_order,
                    beta_v, self.landscape, state_num=state_num, openness=openness
                )
                self.agent_types.append("s")

            if defaultValues:

                agent.state = adoptDefault(agent.state, agent.decision_space, default_state)

            self.agents.append(agent)

    def extract_results(self):
        # platform level -> average fitness
        # platform level -> unique fitness
        # platform level -> unique number
        # g & s

        fitness = []
        aggregation = []
        uniqueness = []

        g_fitness = []
        g_aggregation = []
        g_uniqueness = []

        s_fitness = []
        s_aggregation = []
        s_uniqueness = []

        platform_unique_set = set([])
        s_unique_set = set([])
        g_unique_set = set([])

        all_product_best = {}

        data = [
            (
                self.landscape.query_fitness(self.agents[cur].state), self.agents[cur].state, cur
            ) for cur in range(len(self.agents))
        ]

        data.sort(key=lambda x:-x[0])

        s_count = 0
        s_unique_count = 0
        s_all_fitness = 0
        s_unique_fitness = 0
        g_count = 0
        g_unique_count = 0
        g_all_fitness = 0
        g_unique_fitness = 0
        platform_count = 0
        platform_unique_count = 0
        platform_all_fitness = 0
        platform_unique_fitness = 0

        for cur in range(len(data)):

            bit_str = "".join([str(x) for x in data[cur][1]])
            if bit_str not in platform_unique_set:
                platform_unique_count += 1
                platform_unique_fitness += data[cur][0]
                platform_unique_set.add(bit_str)
            platform_count += 1
            platform_all_fitness += data[cur][0]

            agent_type = self.agent_types[data[cur][2]]
            if agent_type == "g":
                if bit_str not in g_unique_set:
                    g_unique_count += 1
                    g_unique_fitness += data[cur][0]
                    g_unique_set.add(bit_str)
                g_count += 1
                g_all_fitness += data[cur][0]

                g_fitness.append(g_all_fitness/g_count)
                g_aggregation.append(g_unique_fitness)
                g_uniqueness.append(g_unique_count)
            else:
                if bit_str not in s_unique_set:
                    s_unique_count += 1
                    s_unique_fitness += data[cur][0]
                    s_unique_set.add(bit_str)
                s_count += 1
                s_all_fitness += data[cur][0]

                s_fitness.append(s_all_fitness/s_count)
                s_aggregation.append(s_unique_fitness)
                s_uniqueness.append(s_unique_count)

            fitness.append(platform_all_fitness/platform_count)
            aggregation.append(platform_unique_fitness)
            uniqueness.append(platform_unique_count)

            product_bit = "".join([str(x) for x in sorted(self.agents[data[cur][2]].decision_space)])
            if product_bit not in all_product_best:
                all_product_best[product_bit] = data[cur][0]
            else:
                all_product_best[product_bit] = max(all_product_best[product_bit], data[cur][0])

        return (
            fitness, aggregation, uniqueness,
            s_fitness, s_aggregation, s_uniqueness,
            g_fitness, g_aggregation, g_uniqueness, all_product_best
        )

    def calculate_error_state(self, state):
        temp_state = list(state)
        for cur in range(len(state)):
            if np.random.uniform(0, 1) < self.error_probability:
                temp_state[cur] = np.random.choice(self.state_num)
        return temp_state

    def adaptation(self, learn_frequency, learn_method="random",):

        for cur in range(len(self.agents)):
            self.agents[cur].state, _ = self.agents[cur].independent_search()
            # self.agents[cur].open_tag = True if np.random.uniform(0, 1) < self.agents[cur].openness else False

        if np.random.uniform(0, 1) < learn_frequency:
            if learn_method=="random":
                for cur in range(len(self.agents)):

                    if self.agent_types[cur] == "g":
                        if np.random.uniform(0, 1) < self.g_p:
                            alternatives = [
                                x for x in range(len(self.agents)) if
                                x != cur and self.agents[x].open_tag is True and self.agent_types[x] == "s"
                            ]
                        else:
                            alternatives = [
                                x for x in range(len(self.agents)) if
                                x != cur and self.agents[x].open_tag is True and self.agent_types[x] == "g"
                            ]
                    elif self.agent_types[cur] == "s":
                        if np.random.uniform(0, 1) < self.s_p:
                            alternatives = [
                                x for x in range(len(self.agents)) if
                                x != cur and self.agents[x].open_tag is True and self.agent_types[x] == "s"
                            ]
                        else:
                            alternatives = [
                                x for x in range(len(self.agents)) if
                                x != cur and self.agents[x].open_tag is True and self.agent_types[x] == "g"
                            ]
                    else:
                        print("zjj sb!!!!!")

                    if len(alternatives) == 0:
                        continue

                    target = np.random.choice(alternatives)
                    target_state = self.calculate_error_state(list(self.agents[target].state))

                    self.agents[cur].state, _ = self.agents[cur].adopt_existing_solution(
                        None, target_state, None, None
                    )

            elif learn_method == "personal":

                for cur in range(len(self.agents)):

                    if self.agent_types[cur] == "g":
                        if np.random.uniform(0, 1) < self.g_p:
                            state_list = [
                                self.agents[x].state for x in range(len(self.agents)) if
                                x!= cur and self.agents[x].open_tag is True and self.agent_types[x] == "s"
                            ]
                            index_list = [
                                x for x in range(len(self.agents)) if
                                x != cur and self.agents[x].open_tag is True and self.agent_types[x] == "s"
                            ]
                        else:
                            state_list = [
                                self.agents[x].state for x in range(len(self.agents)) if
                                x != cur and self.agents[x].open_tag is True and self.agent_types[x] == "g"
                            ]
                            index_list = [
                                x for x in range(len(self.agents)) if
                                x != cur and self.agents[x].open_tag is True and self.agent_types[x] == "g"
                            ]
                    elif self.agent_types[cur] == "s":
                        if np.random.uniform(0, 1) < self.s_p:
                            state_list = [
                                self.agents[x].state for x in range(len(self.agents)) if
                                x != cur and self.agents[x].open_tag is True and self.agent_types[x] == "s"
                            ]
                            index_list = [
                                x for x in range(len(self.agents)) if
                                x != cur and self.agents[x].open_tag is True and self.agent_types[x] == "s"
                            ]
                        else:
                            state_list = [
                                self.agents[x].state for x in range(len(self.agents)) if
                                x != cur and self.agents[x].open_tag is True and self.agent_types[x] == "g"
                            ]
                            index_list = [
                                x for x in range(len(self.agents)) if
                                x != cur and self.agents[x].open_tag is True and self.agent_types[x] == "g"
                            ]
                    else:
                        print("zjj nb!!!!")

                    input_state = []
                    for idx in range(len(state_list)):
                        input_state.append(self.calculate_error_state(list(state_list[idx])))

                    personal_vote_list = self.agents[cur].evaluate_all_solution(
                        input_state
                    )

                    if len(personal_vote_list) == 0:
                        continue

                    perceived_fitness_list = [x[1] for x in personal_vote_list]
                    perceived_fitness_list = [x/np.sum(perceived_fitness_list) for x in perceived_fitness_list]
                    choice = np.random.choice(
                        [cur for cur in range(len(perceived_fitness_list))], 1, p=perceived_fitness_list
                    ).tolist()[0]

                    target_state = input_state[choice]

                    self.agents[cur].state, _ = self.agents[cur].adopt_existing_solution(
                        None, target_state, None, None
                    )
            else:
                print("wrong")


def simulation(return_dic, idx, repeat, period, N, k, agentNum, knowledgeNum, generalistProportion, beta_v,
               defaultValues, learn_frequency, state_num=4, openness=1, g_p=0.5, s_p=0.5, error_probability=0
               ):

    platform_fitness = []
    platform_aggregation = []
    platform_uniqueness = []
    platform_solution = []
    platform_mean_potential = []
    platform_max_potential = []
    platform_coverage = []
    g_fitness = []
    g_aggregation = []
    g_uniqueness = []
    s_fitness = []
    s_aggregation = []
    s_uniqueness = []

    for r in range(repeat):

        print(r, time.asctime())
        # print(time.asctime())

        np.random.seed(None)

        temp_platform_fitness = []
        temp_platform_aggregation = []
        temp_platform_uniqueness = []
        temp_g_fitness = []
        temp_g_aggregation = []
        temp_g_uniqueness = []
        temp_s_fitness = []
        temp_s_aggregation = []
        temp_s_uniqueness = []

        temp_platform_solution = []
        temp_platform_max_potential = []
        temp_platform_mean_potential = []
        temp_platform_coverage = []

        platform = Platform(
            N, k, agentNum, knowledgeNum, generalistProportion, beta_v, defaultValues, state_num, openness, g_p, s_p,
            error_probability,
        )

        for step in range(period):

            platform.adaptation(learn_frequency, "personal")
            results = platform.extract_results()
            temp_platform_fitness.append(list(results[0]))
            temp_platform_aggregation.append(results[1])
            temp_platform_uniqueness.append(results[2])
            temp_s_fitness.append(results[3])
            temp_s_aggregation.append(results[4])
            temp_s_uniqueness.append(results[5])
            temp_g_fitness.append(results[6])
            temp_g_aggregation.append(results[7])
            temp_g_uniqueness.append(results[8])

            if step==period-1:
                temp_solution = []
                temp_mean_potential = []
                temp_max_potential = []
                temp_coverage = []

                for cur in range(agentNum):
                    solution, max_potential, mean_potential, rank = platform.agents[cur].measure_surface_quality()
                    temp_solution.append(solution)
                    temp_mean_potential.append(mean_potential)
                    temp_max_potential.append(max_potential)
                    temp_coverage.append(rank)
                temp_platform_solution.append(temp_solution)
                temp_platform_max_potential.append(temp_max_potential)
                temp_platform_mean_potential.append(temp_mean_potential)
                temp_platform_coverage.append(temp_coverage)

        platform_fitness.append(temp_platform_fitness)
        platform_aggregation.append(temp_platform_aggregation)
        platform_uniqueness.append(temp_platform_uniqueness)
        g_fitness.append(temp_g_fitness)
        g_aggregation.append(temp_g_aggregation)
        g_uniqueness.append(temp_g_uniqueness)
        s_fitness.append(temp_s_fitness)
        s_aggregation.append(temp_s_aggregation)
        s_uniqueness.append(temp_s_uniqueness)

        platform_solution.append(temp_platform_solution)
        platform_max_potential.append(temp_platform_max_potential)
        platform_mean_potential.append(temp_platform_mean_potential)
        platform_coverage.append(temp_coverage)

        # print(np.mean(np.array(platform_fitness)[:, :, -1],axis=0))

        # repeat, step, agentNum

    return_dic[idx] = (
        np.mean(np.array(platform_fitness), axis=0),
        np.mean(np.array(platform_aggregation), axis=0),
        np.mean(np.array(platform_uniqueness), axis=0),
        np.mean(np.array(s_fitness), axis=0),
        np.mean(np.array(s_aggregation), axis=0),
        np.mean(np.array(s_uniqueness), axis=0),
        np.mean(np.array(g_fitness), axis=0),
        np.mean(np.array(g_aggregation), axis=0),
        np.mean(np.array(g_uniqueness), axis=0),
        platform_solution,
        platform_max_potential,
        platform_mean_potential,
        platform_coverage,
    )
