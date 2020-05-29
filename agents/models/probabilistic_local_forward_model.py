import numpy as np
from numba import njit, prange
import pickle
import random

from agents.models.tile_map import TileSet


class ProbabilisticLearningLocalForwardModelData:

    def __init__(self,  mask, observation=None, initial_pattern_count=100, max_size=100000, tile_set: TileSet = None,
                 initial_tile_size=10, max_tiles=100):
        self.base_mask = mask
        self.observation_shape, self.mask_array, self.span = None, None, None
        self.padded_array, self.flattened_padded_array, self.observed_patterns = None, None, None

        if observation is not None:
            self.initialize(observation)

        if tile_set is not None:
            if tile_set.nr_of_known_tiles() > initial_tile_size:
                initial_tile_size = tile_set.nr_of_known_tiles()

        # data storage
        self.patterns = np.zeros((initial_pattern_count, np.sum(mask)+1))  # mask + action
        self.pattern_results = np.zeros((initial_pattern_count, initial_tile_size))  # nr of observations per result
        # todo (to not compute arg_max() for every prediction): store  most likely tile and its number of occurrences
        # self.most_likely_result = np.zeros(initial_pattern_count)
        # self.occurrences_of_most_likely_result = np.zeros(initial_tile_size)

        self.current_max_patterns = initial_pattern_count
        self.total_max_patterns = max_size

        self.tile_set = tile_set
        self.current_max_tiles = initial_tile_size
        self.total_max_tiles = max_tiles

        self.known_patterns = dict()

    def set_tile_set(self, tile_set: TileSet):
        if self.tile_set is None:
            self.tile_set = tile_set
        else:
            raise ValueError("Tileset was already defined. Create a new ProbabilisticLearningLocalForwardModelData "
                             "object if used for another tileset")

    def initialize(self, observation):
        if self.observation_shape is None or not np.all(observation.shape == self.observation_shape):
            self.observation_shape = observation.shape
            self.span = np.floor_divide(self.base_mask.shape, 2)
            padded_size = observation.shape + self.span * 2
            padded_index_array = np.arange(padded_size.prod()).reshape(padded_size)

            # create the array used for fast extraction of all patterns from a flattened padded array
            self.mask_array = np.zeros((observation.size, np.sum(self.base_mask)), dtype=np.int)
            self.mask_array[0] = padded_index_array[0:self.base_mask.shape[0], 0:self.base_mask.shape[1]][self.base_mask]
            for i in range(1, observation.size):
                if i % observation.shape[1] == 0:
                    self.mask_array[i, :] = self.mask_array[i-1] + self.span[0]*2+1
                else:
                    self.mask_array[i, :] = self.mask_array[i-1] + 1

            # create placeholder arrays used during the pattern extraction
            self.padded_array = padded_index_array
            self.padded_array[:, :] = -1
            self.flattened_padded_array = self.padded_array.flatten()
            self.observed_patterns = np.zeros((observation.size, np.sum(self.base_mask)+1))

        # else: do nothing in case the observation shape stays the same

    def add_observation(self, prev_observation, action, observation):
        if self.tile_set.nr_of_known_tiles() > self.current_max_tiles:
            self.increase_nr_of_columns()

        if self.observation_shape != prev_observation.shape:
            raise AssertionError("LocalForwardModelData has not been correctly initialized. "
                                 "Please call initialize everytime the observation shape changes.")
        if len(self.known_patterns) < self.current_max_patterns:
            extract_observed_patterns(prev_observation, self.padded_array, self.span, self.mask_array,
                                      self.observed_patterns, action)
            observation = observation.flatten()
            for i, pattern in enumerate(self.observed_patterns):
                el = tuple(pattern)
                if el not in self.known_patterns:
                    self.patterns[len(self.known_patterns), :] = pattern
                    self.pattern_results[len(self.known_patterns), observation[i]] += 1  # increase occurrence counter
                    self.known_patterns[el] = len(self.known_patterns)                   # store index of new pattern
                    if len(self.known_patterns) == self.current_max_patterns:
                        if not self.increase_nr_of_rows():
                            break
                else:
                    self.pattern_results[self.known_patterns[el], observation[i]] += 1   # increase occurrence counter
        return self.observed_patterns

    def get_patterns(self, observation, action):
        try:
            return extract_observed_patterns(observation, self.padded_array, self.span, self.mask_array, self.observed_patterns, action)
        except Exception:
            print()

    def increase_nr_of_rows(self):
        self.patterns.resize((min(self.patterns.shape[0] * 2, self.total_max_patterns),
                              self.patterns.shape[1]), refcheck=False)
        self.pattern_results.resize((min(self.pattern_results.shape[0] * 2, self.total_max_patterns),
                                     self.pattern_results.shape[1]), refcheck=False)

        if self.current_max_patterns != self.patterns.shape[0]:
            self.current_max_patterns = self.patterns.shape[0]
            return True
        else:
            self.current_max_patterns = self.patterns.shape[0]
            return False

    def increase_nr_of_columns(self):
        self.pattern_results.resize((self.pattern_results.shape[0],
                                    min(self.pattern_results.shape[1] * 2, self.total_max_tiles)), refcheck=False)
        if self.current_max_tiles != (self.pattern_results.shape[1]):
            self.current_max_tiles = (self.pattern_results.shape[1])
            return True
        else:
            return False

    def is_data_set_deterministic(self):
        raise np.all(np.sum(self.pattern_results[:len(self.known_patterns), :] > 0, 1) <= 1)

    def get_data_set(self):
        """ :return: pattern, occurrence count per class
        """
        data = np.zeros((len(self.known_patterns), self.patterns.shape[1]+1))
        data[:, :-1] = self.patterns[:len(self.known_patterns), :]
        data[:, -1] = np.argmax(self.pattern_results[:len(self.known_patterns), :],1)
        return data
        #return self.patterns[:len(self.known_patterns), :], \
        #       self.pattern_results[:len(self.known_patterns), :self.tile_set.nr_of_known_tiles()]

    def get_random_batch(self, batch_size):
        idx = random.sample(range(len(self.known_patterns)), min(len(self.known_patterns), batch_size))
        return self.patterns[idx, :], \
               self.pattern_results[idx, :self.tile_set.nr_of_known_tiles()]

    def write_to_file(self, filename):
        with open(filename, "wb") as file:
            pickle.dump(self.base_mask, file),

            pickle.dump(self.observation_shape, file)
            pickle.dump(self.mask_array, file)
            pickle.dump(self.span, file),

            pickle.dump(self.padded_array, file)
            pickle.dump(self.flattened_padded_array, file)
            pickle.dump(self.observed_patterns, file)
            pickle.dump(self.patterns, file)
            pickle.dump(self.pattern_results, file),

            pickle.dump(self.current_max_patterns, file)
            pickle.dump(self.total_max_patterns, file)
            pickle.dump(self.current_max_tiles, file)
            pickle.dump(self.total_max_tiles, file)
            pickle.dump(self.known_patterns, file)

    @classmethod
    def load_from_file(cls, filename):
        with open(filename, "rb") as file:
            base_mask = pickle.load(file)

            observation_shape = pickle.load(file)
            mask_array = pickle.load(file)
            span = pickle.load(file)

            padded_array = pickle.load(file)
            flattened_padded_array = pickle.load(file)
            observed_patterns = pickle.load(file)
            patterns = pickle.load(file)
            pattern_results = pickle.load(file)

            current_max_patterns = pickle.load(file)
            total_max_patterns = pickle.load(file)
            current_max_tiles = pickle.load(file)
            total_max_tiles = pickle.load(file)
            known_patterns = pickle.load(file)

        lfm_data = cls(base_mask, observation=None, initial_pattern_count=100, max_size=total_max_patterns,
                       tile_set=None, initial_tile_size=current_max_tiles, max_tiles=total_max_tiles)

        lfm_data.observation_shape = observation_shape
        lfm_data.mask_array = mask_array
        lfm_data.span = span

        lfm_data.padded_array = padded_array
        lfm_data.flattened_padded_array = flattened_padded_array
        lfm_data.observed_patterns = observed_patterns
        lfm_data.patterns = patterns
        lfm_data.pattern_results = pattern_results

        lfm_data.current_max_patterns = current_max_patterns
        lfm_data.total_max_patterns = total_max_patterns
        lfm_data.current_max_tiles = current_max_tiles
        lfm_data.total_max_tiles = total_max_tiles
        lfm_data.known_patterns = known_patterns

        return lfm_data

    def get_prediction(self, state, action):
        # todo outdated
        patterns = self.get_patterns(state, action)
        try:
            return np.array([np.argmax(self.pattern_results[self.known_patterns[tuple(x)], :]) for x in patterns]).reshape(state.shape)
        except Exception:
            print([tuple(x) for x in patterns if tuple(x) not in self.known_patterns])

        return None


@njit
def extract_observed_patterns(prev_observation, padded_array, span, mask_array, data_set, action):
    padded_array[span[0]:(span[0] + prev_observation.shape[0]), span[1]:(span[1] + prev_observation.shape[1])] = prev_observation
    a = padded_array.flatten()

    for i in prange(len(mask_array)):
        data_set[i, :-1] = a[mask_array[i]]
    data_set[:, -1] = action

    return data_set
