from scipy.stats import beta
import numpy as np


def solutuionIntegration(stateA, stateB, decisionA, decisionB, landscape):

    if landscape.query_fitness(stateA)>landscape.query_fitness(stateB):

        result = list(stateA)

        for cur in range(len(stateA)):

            if cur in decisionA:
                continue
            else:
                if cur in decisionB:
                    result[cur] = stateB[cur]
    else:
        result = list(stateB)

        for cur in range(len(stateB)):

            if cur in decisionB:
                continue
            else:
                if cur in decisionA:
                    result[cur] = stateA[cur]
    return result

def numberToBase(n, b):
    if n == 0:
        return "0"
    digits = []
    while n:
        digits.append(int(n % b))
        n //= b
    return "".join([str(cur) for cur in digits[::-1]])

# Generates the full permutation of bytecode: [000], [001], ..., [111]
# from itertools import product
# state_num = 4
# remain_length = 4
# x1 = []
# x2 = [i for i in product(range(state_num), repeat=remain_length)]
# for each in x2:
#     bit = ''.join([str(i) for i in each])
#     print(bit)
# for i in range(pow(state_num, remain_length)):
#     x1.append(numberToBase(i, state_num))
# print(x1, '\n', x2)



def overlap_calculation(sorted_decision, teammate_decision, overlap=1):

    tag = True
    for cur in range(overlap):

        if sorted_decision[cur] not in teammate_decision:
            tag = False

    return tag


def stateNumToDepth(stateNum, ):

    dic = {pow(2, cur): cur for cur in range(10)}
    return dic[stateNum]+1


def generate_knowledge(decision_order, b, decision_num):
    n = len(decision_order)
    bin_width = 1.0 / n
    cdf_v = [cur * bin_width for cur in range(n + 1)]

    cdf = [beta.ppf(x, 1, b) for x in cdf_v][::-1]
    ppf = [cdf[cur] - cdf[cur + 1] for cur in range(n)]

    decision = np.random.choice(decision_order, decision_num, replace=False, p=ppf).tolist()
    return decision


def random_combine_proposal(temp_state_i, temp_state_j):
    res = []
    for cur in range(len(temp_state_i)):
        res.append(np.random.choice([temp_state_i[cur], temp_state_j[cur]]))
    return list(res)


def adoptDefault(state, decision, defaultState):
    res = list(state)
    for i in range(len(state)):
        if i in decision:
            continue
        else:
            res[i] = defaultState[i]
    return res
