from abc import ABC, abstractmethod
import numpy as np
import math


class AbstractGameState(ABC):

    def __init__(self):
        self._tick = 0

        super().__init__()

    def get_tick(self):
        return self._tick

    @abstractmethod
    def next(self, action):
        pass

    @abstractmethod
    def get_actions(self):
        pass

    @abstractmethod
    def get_score(self):
        pass

    @abstractmethod
    def is_terminal(self):
        pass

    @abstractmethod
    def deep_copy(self):
        pass

    def evaluate_rollouts(self, candidate_solutions, discount_factor):
        scores = []
        for solution in candidate_solutions:
            scores.append(self.deep_copy().evaluate_rollout(solution, discount_factor))

        return scores

    def evaluate_rollout(self, action_sequence, discount_factor):
        discounted_return = 0
        if discount_factor is None:
            for idx, action in enumerate(action_sequence):
                self.next(action)
                discounted_return += self.get_score()
        else:
            for idx, action in enumerate(action_sequence):
                self.next(action)
                discounted_return += self.get_score() * math.pow(discount_factor, idx)

        return discounted_return


class GridGameState:

    @abstractmethod
    def get_tile_map(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_width(self) -> int:
        pass

    @abstractmethod
    def get_height(self) -> int:
        pass
