import numpy as np

from abc import ABC, abstractmethod


class Replay(ABC):

    def __init__(self, game: str, level: str, initial_frame: np.array):
        self._game = game
        self._level = level

        self.length = 1
        if len(initial_frame.shape) == 3:
            self._height, self._width, self._tiles = initial_frame.shape
        else:
            self._height, self._width = initial_frame.shape
            self._tiles = None

        self._actions = []
        self._rewards = []
        self._result = 0
        pass

    @classmethod
    def load_from_file(cls, file):
        raise NotImplementedError

    @abstractmethod
    def write_to_file(self, file):
        raise NotImplementedError

    @abstractmethod
    def add_frame(self, action, resulting_frame, reward=0, result=0):
        pass

    @abstractmethod
    def get_transition(self, tick):
        pass

    @abstractmethod
    def get_transitions(self):
        pass
