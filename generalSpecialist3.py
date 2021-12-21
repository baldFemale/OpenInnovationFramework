# from MultiStateLandScape import *
from MultiStateInfluentialLandscape import *
from generalSpecialist2 import Agent
from tools import *

# specialist and generalist have different roles
# or temporary task division
# A first search the landscape and let B get involved


def simulation(return_dic, idx, N, k, IM_type, land_num, period, agentNum, teamup, teamup_timing, knowledge_num,
               specialist_num, lr=0.1, state_num=2, overlap=1):
    """
    default: random formation
    two types of agents:
        0-1/2 G
        1/2-1 S

    """

    ress_fitness = []
    team_list = []
    general_knowledge_list = []
    special_knowledge_list = []

    for repeat in range(land_num):

        print(repeat)

        res_fitness = []

        np.random.seed(None)

        # landscape = LandScape(N, k, None, None, state_num=state_num)
        landscape = LandScape(N, k, IM_type, state_num=state_num)
        landscape.initialize()

        agents = []

        if not teamup:
            for cur in range(agentNum):
                agents.append(Agent(N, knowledge_num, specialist_num, lr, landscape, state_num))
        else:
            for cur in range(agentNum):
                if cur < agentNum//2:
                    agents.append(Agent(N, knowledge_num[0], specialist_num[0], lr, landscape, state_num))
                else:
                    agents.append(Agent(N, knowledge_num[1], specialist_num[1], lr, landscape, state_num))

        # print([agents[i].decision_space for i in range(agentNum)])
        # print([agents[i].specialist_decision_space for i in range(agentNum)])

        special_knowledge_list.append([list(agent.specialist_knowledge_space) for agent in agents])
        general_knowledge_list.append([list(agent.generalist_knowledge_space) for agent in agents])

        teams = {i: i for i in range(agentNum)}

        for step in range(period):

            if teamup and step == teamup_timing:

                rank = np.random.choice([cur for cur in range(agentNum)], agentNum, replace=False)
                for i in range(agentNum):

                    if rank[i] >= agentNum//2:
                        continue

                    if teams[rank[i]] is None or teams[rank[i]] != rank[i]:
                        continue

                    cognitive_temp_state = agents[rank[i]].change_state_to_cog_state(agents[rank[i]].state)

                    fitness_contribution = agents[rank[i]].landscape.query_cog_fitness_contribution_gst(
                        cognitive_temp_state,
                        agents[rank[i]].generalist_knowledge_space,
                        agents[rank[i]].specialist_knowledge_space,
                    )

                    decision_contribution = [
                        (fitness_contribution[cur], cur) for cur in agents[rank[i]].decision_space
                    ]
                    decision_contribution.sort(key=lambda x: -x[0])

                    sorted_decision = [d[1] for d in decision_contribution]

                    for j in range(agentNum):
                        if i == j or teams[rank[j]] is None or teams[rank[j]] != rank[j]:
                            continue

                        if rank[j] < agentNum//2:
                            continue

                        if not overlap_calculation(sorted_decision, agents[rank[j]].decision_space, overlap):
                            continue

                        teams[rank[i]] = rank[j]
                        teams[rank[j]] = None

                        # no integration & keep using the current state
                        integrated_solution = agents[rank[i]].state

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

    return_dic[idx] = (ress_fitness, team_list, general_knowledge_list, special_knowledge_list)






