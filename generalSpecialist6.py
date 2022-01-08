# An extension of generalSpecialist5.py
# Generalist and Specialist are different in search
# A general domain -> long jump // A special domain -> local search

from MultiStateInfluentialLandscape import *
from tools import *
import time


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
        last_width = pow(2, self.depth - 1)
        random_val_list = np.random.choice(last_width, last_width, replace=False).tolist()

        while cur < pow(2, self.depth) - 1 - 1:
            node = queue.pop(0)
            cur += 1
            if cur >= pow(2, self.depth - 1) - 1:
                left_child = TreeNode(pow(2, self.depth - 1) - 1 + random_val_list[cur - pow(2, self.depth - 1) + 1],
                                      None, None)
                node.left = left_child
                queue.append(left_child)
                cur += 1
                right_child = TreeNode(pow(2, self.depth - 1) - 1 + random_val_list[cur - pow(2, self.depth - 1) + 1],
                                       None, None)
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
        return self.search_map_leaf(next_node_list, int(remain_depth + 1))

    def initial_knowledge_structure(self):

        queue = [self.head]
        while len(queue) != self.stateNum:
            node = queue.pop(0)
            queue.append(node.left)
            queue.append(node.right)
        self.node_list = [node.val for node in queue]

        for node in queue:
            leaf_nodes, remain_depth = self.search_map_leaf([node], 0)
            self.node_map_leaves_list[node.val] = [leaf.val for leaf in leaf_nodes]
            map_leaf_node = np.random.choice(leaf_nodes)
            self.node_map_leaf_list[node.val] = map_leaf_node.val

            for leaf in leaf_nodes:
                self.leaf_map_node_list[leaf.val] = node.val


class Agent:

    def __init__(self, N, changeable_state_list, landscape=None, state_num=2):

        """
        :param N: param N in NK model
        :param changeable_state_list: A list of state number an agent know, e.g., [4, 4], knowledge outside this list takes 0 as default
        :param landscape: landscape
        :param state_num: state number in landscape, 2^depth=state_num
        """

        self.N = N
        self.state_num = state_num
        self.decision_space = np.random.choice(self.N, len(changeable_state_list), replace=False).tolist()
        self.changeable_state = changeable_state_list

        # choose decision according to state number
        # self.changeable_probability = [
        #     x / sum(self.changeable_state) for x in self.changeable_state
        # ]

        # choose g & s decision according to sum(g state number):sum(s state number)
        self.changeable_probability = [
            np.sum([cur for cur in self.changeable_state if cur != state_num])/np.sum(self.changeable_state),
            np.sum([cur for cur in self.changeable_state if cur == state_num])/np.sum(self.changeable_state)
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

        self.learned_decision = []

        self.cog_fitness = self.landscape.query_cog_fitness_tree(
            self.state, self.decision_space, self.knowledge_tree_list, self.tree_depth, self.learned_decision,
            None, None
        )

    def independent_search(self, teammate_decision=None, teammate_knowledge_tree_list=None):
        """
        random choice from all possible alternatives
        :return:
        """

        gs_choice = np.random.choice([0, 1], 1, p=self.changeable_probability, replace=False)[0]

        temp_state = list(self.state)

        if gs_choice==0:
            # g
            index = [cur for cur in range(len(self.decision_space)) if self.changeable_state[cur] != self.state_num]
            ds = [self.decision_space[cur] for cur in index]

            for cur in range(len(index)):
                node_list = self.knowledge_tree_list[index[cur]].node_list

                alternatives = [
                    cur for cur in node_list
                ]
                alter = np.random.choice(alternatives)
                temp_state[ds[cur]] = self.knowledge_tree_list[index[cur]].node_map_leaf_list[alter] - (
                    pow(2, self.tree_depth-1) -1
                )
        else:
            # s
            index = np.random.choice(
                [cur for cur in range(len(self.decision_space)) if self.changeable_state[cur] == self.state_num]
            )

            d = self.decision_space[index]

            node_list = self.knowledge_tree_list[index].node_list
            alternatives = [
                cur for cur in node_list if self.knowledge_tree_list[index].node_map_leaf_list[cur] - (
                        pow(2, self.tree_depth - 1) - 1) != self.state[d]
            ]

            alter = np.random.choice(alternatives)
            temp_state[d] = self.knowledge_tree_list[index].node_map_leaf_list[alter] - (
                        pow(2, self.tree_depth - 1) - 1)

        new_cog_fitness = self.landscape.query_cog_fitness_tree(
            temp_state, self.decision_space, self.knowledge_tree_list, self.tree_depth, self.learned_decision,
            teammate_decision, teammate_knowledge_tree_list
        )
        if new_cog_fitness > self.cog_fitness:
            return list(temp_state), new_cog_fitness
        else:
            return list(self.state), self.cog_fitness


def simulation(return_dic, idx, N, k, IM_type, IM_random_ratio, land_num, period, agentNum, teamup, teamup_timing,
               knowledge_list, lr=0.1, state_num=4, negotiation_round=2, negotiation_priority=["s", "g", "t"]):
    """
    :param return_dic:
    :param idx:
    :param N:
    :param k:
    :param IM_type:
    :param IM_random_ratio:
    :param land_num:
    :param period:
    :param agentNum:
    :param teamup:
    :param teamup_timing: as the current focus is not timing, maybe set as 0
    :param knowledge_list: a list of changeable state list, e.g., [[4, 4], [2, 2, 2, 2]
    :param state_num:
    :param negotiation_round: liaison, decision power, no negotiation
    :return:
    """

    ress_fitness = []
    ress_rank = []
    team_list = []
    ress_knowledge = []
    ress_decision = []
    ress_im = []

    for repeat in range(land_num):

        print(repeat)

        res_fitness = []
        res_rank = []

        np.random.seed(None)

        landscape = LandScape(N, k, IM_type, IM_random_ratio, state_num=state_num)
        landscape.initialize()

        ress_im.append(landscape.IM)

        defaultstate = np.random.choice(state_num, N).tolist()

        agents = []

        # print(len(knowledge_list))

        unit_length = agentNum // len(knowledge_list)

        for cur in range(agentNum):
            index = cur // unit_length
            agents.append(Agent(N, knowledge_list[index], landscape, state_num))
            # agents[-1].state = adoptDefault(agents[-1].state, agents[-1].decision_space, defaultstate)

        ress_knowledge.append([list(agent.changeable_state) for agent in agents])
        ress_decision.append([list(agent.decision_space) for agent in agents])

        teams = {i: i for i in range(agentNum)}

        for step in range(period):

            # if len(knowledge_list[0])==2 and k==56:
            #    print("start an iteration", step, time.asctime())

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
                        agents[rank[i]].cog_fitness = landscape.query_cog_fitness_tree(
                            integrated_solution, agents[rank[i]].decision_space, agents[rank[i]].knowledge_tree_list,
                            agents[rank[i]].tree_depth, agents[rank[i]].learned_decision, None, None
                        )
                        agents[rank[j]].cog_fitness = landscape.query_cog_fitness_tree(
                            integrated_solution, agents[rank[j]].decision_space, agents[rank[j]].knowledge_tree_list,
                            agents[rank[j]].tree_depth, agents[rank[j]].learned_decision, None, None
                        )
                        break

            # print("start search", time.asctime())

            for i in range(agentNum):

                if teams[i] is None:
                    continue

                elif teams[i] == i:

                    temp_state, temp_cog_fitness = agents[i].independent_search()
                    agents[i].state = list(temp_state)
                    agents[i].cog_fitness = float(temp_cog_fitness)

                elif teams[i] != i:

                    # learning

                    overlap = list(set(agents[i].decision_space) & set(agents[teams[i]].decision_space))

                    p = lr * len(overlap)

                    if np.random.uniform(0, 1) < p:

                        learned_unoverlap_decision = list(set(agents[i].learned_decision) - set(overlap))

                        if len(learned_unoverlap_decision) < len(overlap):
                            new_knowledge_A = np.random.choice(
                                agents[teams[i]].decision_space
                            )

                            agents[i].learned_decision.append(new_knowledge_A)

                    if np.random.uniform(0, 1) < p:

                        learned_unoverlap_decision = list(set(agents[teams[i]].learned_decision) - set(overlap))

                        if len(learned_unoverlap_decision) < len(overlap):
                            new_knowledge_B = np.random.choice(
                                agents[i].decision_space
                            )
                            agents[teams[i]].learned_decision.append(new_knowledge_B)

                    if negotiation_round == 2:

                        # A's proposal
                        temp_state, temp_cog_fitness = agents[i].independent_search(
                            agents[teams[i]].decision_space, agents[teams[i]].knowledge_tree_list
                        )

                        # B's evaluation

                        B_evaluated_fitness = landscape.query_cog_fitness_tree(
                            temp_state,
                            agents[teams[i]].decision_space,
                            agents[teams[i]].knowledge_tree_list,
                            agents[teams[i]].tree_depth,
                            agents[teams[i]].learned_decision,
                            agents[i].decision_space,
                            agents[i].knowledge_tree_list,
                        )

                        if B_evaluated_fitness >= agents[teams[i]].cog_fitness:
                            # accept proposal
                            agents[i].state = list(temp_state)
                            agents[i].cog_fitness = float(temp_cog_fitness)
                            agents[teams[i]].state = list(temp_state)
                            agents[teams[i]].cog_fitness = float(B_evaluated_fitness)
                        else:
                            # reject proposal -> nothing happens
                            pass

                        # B's proposal
                        B_temp_state, B_temp_cog_fitness = agents[teams[i]].independent_search(
                            agents[i].decision_space, agents[i].knowledge_tree_list,
                        )

                        # A's evaluation
                        A_evaluated_fitness = landscape.query_cog_fitness_tree(
                            B_temp_state,
                            agents[i].decision_space,
                            agents[i].knowledge_tree_list,
                            agents[i].tree_depth,
                            agents[i].learned_decision,
                            agents[teams[i]].decision_space,
                            agents[teams[i]].knowledge_tree_list,
                        )

                        if A_evaluated_fitness >= agents[i].cog_fitness:
                            # accept
                            agents[i].state = list(B_temp_state)
                            agents[i].cog_fitness = float(A_evaluated_fitness)
                            agents[teams[i]].state = list(B_temp_state)
                            agents[teams[i]].cog_fitness = float(B_temp_cog_fitness)
                        else:
                            pass

                    elif negotiation_round == 0:
                        temp_state_i, _ = agents[i].independent_search(
                            agents[teams[i]].decision_space, agents[teams[i]].knowledge_tree_list
                        )
                        temp_state_j, _ = agents[teams[i]].independent_search(
                            agents[i].decision_space, agents[i].knowledge_tree_list
                        )

                        integrated_solution = random_combine_proposal(temp_state_i, temp_state_j)
                        agents[i].state = list(integrated_solution)
                        agents[teams[i]].state = list(integrated_solution)

                        agents[i].cog_fitness = landscape.query_cog_fitness_tree(
                            integrated_solution, agents[i].decision_space, agents[i].knowledge_tree_list,
                            agents[i].tree_depth, agents[i].learned_decision, agents[teams[i]].decision_space,
                            agents[teams[i]].knowledge_tree_list
                        )

                        agents[teams[i]].cog_fitness = landscape.query_cog_fitness_tree(
                            integrated_solution, agents[teams[i]].decision_space, agents[teams[i]].knowledge_tree_list,
                            agents[teams[i]].tree_depth, agents[teams[i]].learned_decision, agents[i].decision_space,
                            agents[i].knowledge_tree_list,
                        )

                    elif negotiation_round == 1:
                        # where priority comes from
                        index_i = int(i // (agentNum // len(knowledge_list)))
                        index_j = int(teams[i] // (agentNum // len(knowledge_list)))
                        type_i = ["s", "g", "t"][index_i]
                        type_j = ["s", "g", "t"][index_j]

                        if negotiation_priority.index(type_i) >= negotiation_priority.index(type_j):
                            # i should obey j
                            # A's proposal
                            temp_state, temp_cog_fitness = agents[i].independent_search(
                                agents[teams[i]].decision_space, agents[teams[i]].knowledge_tree_list
                            )

                            # B's evaluation

                            B_evaluated_fitness = landscape.query_cog_fitness_tree(
                                temp_state,
                                agents[teams[i]].decision_space,
                                agents[teams[i]].knowledge_tree_list,
                                agents[teams[i]].tree_depth,
                                agents[teams[i]].learned_decision,
                                agents[i].decision_space,
                                agents[i].knowledge_tree_list,
                            )

                            if B_evaluated_fitness >= agents[teams[i]].cog_fitness:
                                agents[i].state = list(temp_state)
                                agents[i].cog_fitness = float(temp_cog_fitness)
                                agents[teams[i]].state = list(temp_state)
                                agents[teams[i]].cog_fitness = float(temp_cog_fitness)
                            else:
                                pass

                            # B's proposal
                            B_temp_state, B_temp_cog_fitness = agents[teams[i]].independent_search(
                                agents[i].decision_space, agents[i].knowledge_tree_list,
                            )
                            agents[i].state = list(B_temp_state)
                            agents[i].cog_fitness = landscape.query_cog_fitness_tree(
                                B_temp_state, agents[i].decision_space, agents[i].knowledge_tree_list,
                                agents[i].tree_depth, agents[i].learned_decision, agents[teams[i]].decision_space,
                                agents[teams[i]].knowledge_tree_list
                            )
                            agents[teams[i]].state = list(B_temp_state)
                            agents[teams[i]].cog_fitness = float(B_temp_cog_fitness)
                        else:
                            # j should follow i
                            temp_state, temp_cog_fitness = agents[teams[i]].independent_search(
                                agents[i].decision_space, agents[i].knowledge_tree_list
                            )

                            # A's evaluation

                            A_evaluated_fitness = landscape.query_cog_fitness_tree(
                                temp_state,
                                agents[i].decision_space,
                                agents[i].knowledge_tree_list,
                                agents[i].tree_depth,
                                agents[i].learned_decision,
                                agents[teams[i]].decision_space,
                                agents[teams[i]].knowledge_tree_list,
                            )

                            if A_evaluated_fitness >= agents[i].cog_fitness:
                                agents[i].state = list(temp_state)
                                agents[i].cog_fitness = float(A_evaluated_fitness)
                                agents[teams[i]].state = list(temp_state)
                                agents[teams[i]].cog_fitness = float(temp_cog_fitness)
                            else:
                                pass

                            # A's proposal
                            A_temp_state, A_temp_cog_fitness = agents[i].independent_search(
                                agents[teams[i]].decision_space, agents[teams[i]].knowledge_tree_list,
                            )
                            agents[i].state = list(A_temp_state)
                            agents[i].cog_fitness = float(A_temp_cog_fitness)
                            agents[teams[i]].state = list(A_temp_state)
                            agents[teams[i]].cog_fitness = landscape.query_cog_fitness_tree(
                                A_temp_state, agents[teams[i]].decision_space, agents[teams[i]].knowledge_tree_list,
                                agents[teams[i]].tree_depth, agents[teams[i]].learned_decision,
                                agents[i].decision_space, agents[i].knowledge_tree_list,
                            )
            # print("start calculate fitness", time.asctime())

            tempFitness = [landscape.query_fitness(agents[i].state) for i in range(agentNum)]
            tempRank = [landscape.fitness_to_rank_dict[x] for x in tempFitness]
            # print(np.mean(tempFitness))

            res_fitness.append(tempFitness)
            res_rank.append(tempRank)

        ress_fitness.append(res_fitness)
        ress_rank.append(res_rank)
        team_list.append(teams)

    return_dic[idx] = (ress_fitness, ress_rank, team_list, ress_knowledge, ress_decision, ress_im)
