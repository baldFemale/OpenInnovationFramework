"""Conventional binary NK landscape with exhaustive fitness and rank caches."""

from numbers import Integral
import numpy as np


def require_integer(name, value, minimum, maximum=None):
    if (isinstance(value, bool) or not isinstance(value, Integral)
            or value < minimum or (maximum is not None and value > maximum)):
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}].")


class Landscape:
    def __init__(self, N: int, K: int, norm="MaxMin", rng=None):
        require_integer("N", N, 1)
        require_integer("K", K, 0, N - 1)
        if norm not in (None, "MaxMin"):
            raise ValueError("norm must be None or 'MaxMin'.")
        self.N, self.K, self.norm = N, K, norm
        self.rng = rng if rng is not None else np.random.default_rng()
        self.IM = np.eye(N, dtype=int)
        self.dependency_map = []
        for row in range(N):
            candidates = [column for column in range(N) if column != row]
            dependencies = sorted(self.rng.choice(candidates, K, replace=False).tolist())
            self.IM[row, dependencies] = 1
            self.dependency_map.append(dependencies)
        self.FC = self.rng.uniform(0, 1, size=(N, 2 ** (K + 1)))
        self.fitness_cache = {}
        self.fitness_rank_cache = {}
        self._store_caches()

    def _store_caches(self):
        # Enumerate configurations as integers to avoid a large list of bit tuples.
        configurations = np.arange(2 ** self.N, dtype=np.uint64)
        fitness = np.zeros(len(configurations), dtype=float)
        for row, dependencies in enumerate(self.dependency_map):
            indices = np.zeros(len(configurations), dtype=np.int64)
            for dimension in [row] + dependencies:
                bits = (configurations >> (self.N - 1 - dimension)) & 1
                indices = 2 * indices + bits.astype(np.int64)
            fitness += self.FC[row, indices]
        fitness /= self.N
        self.min_normalizer = float(fitness.min())
        self.max_normalizer = float(fitness.max())
        if self.norm == "MaxMin":
            span = self.max_normalizer - self.min_normalizer
            fitness = (fitness - self.min_normalizer) / span if span else np.zeros_like(fitness)
        self.fitness_cache = {
            format(index, f"0{self.N}b"): float(value)
            for index, value in enumerate(fitness)
        }
        # Competition ranks: tied solutions share ranks, e.g. 1, 2, 2, 4.
        previous = None
        rank = 0
        for position, (state, value) in enumerate(
                sorted(self.fitness_cache.items(), key=lambda item: item[1], reverse=True), 1):
            if value != previous:
                rank, previous = position, value
            self.fitness_rank_cache[state] = rank

    def _state_key(self, state):
        if len(state) != self.N or any(bit not in ("0", "1") for bit in state):
            raise ValueError("state must contain exactly N binary string values.")
        return "".join(state)

    def query_fitness(self, state):
        """Mean of all contributions, optionally rescaled across the landscape."""
        return self.fitness_cache[self._state_key(state)]

    def query_fitness_rank(self, state):
        return self.fitness_rank_cache[self._state_key(state)]

    def query_scoped_fitness(self, state, knowledge_domain):
        """Raw mean of known contributions, using the complete state contingencies.

        Perception is always on the raw contribution scale. Optional objective
        normalization is a positive affine transform and does not affect choices.
        """
        key = self._state_key(state)
        domain = tuple(knowledge_domain)
        if not domain or len(set(domain)) != len(domain):
            raise ValueError("knowledge_domain must be nonempty and contain distinct dimensions.")
        total = 0.0
        for row in domain:
            require_integer("knowledge dimension", row, 0, self.N - 1)
            index = int(key[row] + "".join(key[j] for j in self.dependency_map[row]), 2)
            total += self.FC[row, index]
        return float(total / len(domain))
