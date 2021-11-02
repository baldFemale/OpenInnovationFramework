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
