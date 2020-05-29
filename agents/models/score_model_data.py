import numpy as np
from numba import njit
from agents.models.tile_map import TileSet
import random
import pickle


class ScoreModelData:

    def __init__(self, tile_set: TileSet = None, initial_tile_size=10, max_tiles=1000,
                 initial_pattern_count=1000, max_size=10000):
        self.tile_set = tile_set

        # data storage
        if tile_set is not None:
            if tile_set.nr_of_known_tiles() > initial_tile_size:
                initial_tile_size = tile_set.nr_of_known_tiles()

        self.patterns = np.zeros((initial_pattern_count, initial_tile_size*4))  # mask + action and target columns
        self.reward = np.zeros(initial_pattern_count)  # mask + action and target columns

        # set maximum for rows and columns, current_max values represent the current numpy limits which can be extended
        self.current_max_rows = initial_pattern_count
        self.max_rows = max_size
        self.current_max_tiles = initial_tile_size
        self.max_tile_columns = max_tiles*4

        self.known_patterns = set()

    def set_tile_set(self, tile_set: TileSet):
        if self.tile_set is None:
            self.tile_set = tile_set
        else:
            raise ValueError("Tileset was already defined. Create a new ScoreModel object if used for another tileset")

    def add_observation(self, prev_observation, observation, reward):
        while self.tile_set.nr_of_known_tiles() > self.current_max_tiles:
            self.increase_nr_of_columns()
        if len(self.known_patterns) < self.max_rows:
            pattern = extract_observed_patterns(prev_observation, observation, self.tile_set.nr_of_known_tiles())
            el = tuple(pattern.tolist())
            if el not in self.known_patterns:
                self.patterns[len(self.known_patterns), :len(pattern)] = pattern
                self.reward[len(self.known_patterns)] = reward
                self.known_patterns.add(el)
                if len(self.known_patterns) == self.current_max_rows:
                    self.increase_nr_of_rows()
        else:
            pattern = extract_observed_patterns(prev_observation, observation, self.tile_set.nr_of_known_tiles())
        return pattern

    def get_pattern(self, prev_observation, observation):
        return extract_observed_patterns(prev_observation, observation, self.tile_set.nr_of_known_tiles())

    def get_pattern_lenth(self):
        return self.tile_set.nr_of_known_tiles()*4

    def increase_nr_of_rows(self):
        self.reward.resize((min(self.patterns.shape[0] * 2, self.max_rows)), refcheck=False)

        try:
            self.patterns.resize((min(self.patterns.shape[0] * 2, self.max_rows),
                                  self.patterns.shape[1]), refcheck=False)
        except:
            tmp = np.zeros((min(self.patterns.shape[0] * 2, self.max_rows), self.patterns.shape[1]))
            tmp[:self.patterns.shape[0], :self.patterns.shape[1]] = self.patterns
            self.patterns = tmp

        if self.current_max_rows != self.patterns.shape[0]:
            self.current_max_rows = self.patterns.shape[0]
            return True
        else:
            self.current_max_rows = self.patterns.shape[0]
            return False

    def increase_nr_of_columns(self):
        self.patterns.resize((self.patterns.shape[0],
                              min(self.patterns.shape[1] * 2, self.max_tile_columns)), refcheck=False)
        if self.current_max_tiles != (self.patterns.shape[1]//4):
            self.current_max_tiles = (self.patterns.shape[1]//4)
            return True
        else:
            return False

    def get_data_set(self):
        return self.patterns[:len(self.known_patterns), :self.tile_set.nr_of_known_tiles()*4], self.reward[:len(self.known_patterns)]

    def get_random_batch(self, batch_size):
        idx = random.sample(range(len(self.known_patterns)), min(len(self.known_patterns), batch_size))
        return self.patterns[idx, :]

    def write_to_file(self, filename):
        with open(filename, "wb") as file:

            pickle.dump(self.patterns, file)
            pickle.dump(self.reward, file)

            pickle.dump(self.current_max_rows, file)
            pickle.dump(self.max_rows, file)
            pickle.dump(self.current_max_tiles, file)
            pickle.dump(self.max_tile_columns, file)

            pickle.dump(self.known_patterns, file)

    @classmethod
    def load_from_file(cls, filename, tile_set):
        with open(filename, "rb") as file:
            patterns = pickle.load(file)
            reward = pickle.load(file)

            current_max_rows = pickle.load(file)
            max_rows = pickle.load(file)
            current_max_tiles = pickle.load(file)
            max_tile_columns = pickle.load(file)

            known_patterns = pickle.load(file)

        sm_data = cls(None, 1, 1, 1, 1)
        sm_data.set_tile_set(tile_set)

        sm_data.patterns = patterns
        sm_data.reward = reward

        sm_data.current_max_rows = current_max_rows
        sm_data.max_rows = max_rows
        sm_data.current_max_tiles = current_max_tiles
        sm_data.max_tile_columns = max_tile_columns

        sm_data.known_patterns = known_patterns

        return sm_data


@njit
def extract_observed_patterns(prev_observation, next_observation, nr_of_known_tiles):
    data_row = np.zeros((nr_of_known_tiles*4))

    # the sorting order is important to add additional tile types without changing the meaning of previous rows
    for i in range(nr_of_known_tiles):
        data_row[(i*4)] = np.sum(prev_observation == i)     # how many tiles of type i existed before
        data_row[(i*4)+1] = np.sum(next_observation == i)   # how many tiles of type i exist now
        data_row[(i*4)+2] = np.sum(np.logical_and(next_observation == i, prev_observation != i))  # how many became type i
        data_row[(i*4)+3] = np.sum(np.logical_and(next_observation != i, prev_observation == i))  # how many were of type i

    return data_row
