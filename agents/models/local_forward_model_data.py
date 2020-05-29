import numpy as np
from numba import njit, prange
import pickle
import random


class LocalForwardModelData:

    def __init__(self, mask, observation=None, initial_pattern_count=100, max_size=100000):
        self.base_mask = mask
        self.observation_shape, self.mask_array, self.span = None, None, None
        self.padded_array, self.flattened_padded_array, self.observed_patterns = None, None, None

        if observation is not None:
            self.initialize(observation)

        # data storage
        self.patterns = np.zeros((initial_pattern_count, np.sum(mask)+2))  # mask + action and target columns
        self.max_patterns = initial_pattern_count
        self.max_size = max_size
        self.known_patterns = set()

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
            self.observed_patterns = np.zeros((observation.size, np.sum(self.base_mask)+2))

        # else: do nothing in case the observation shape stays the same

    def add_observation(self, prev_observation, action, observation):
        if self.observation_shape != prev_observation.shape:
            raise AssertionError("LocalForwardModelData has not been correctly initialized. "
                                 "Please call initialize everytime the observation shape changes.")
        if len(self.known_patterns) < self.max_patterns:
            extract_observed_patterns(prev_observation, self.padded_array, self.span, self.mask_array, self.observed_patterns, action, observation)

            for i, pattern in enumerate(self.observed_patterns):
                el = tuple(pattern.tolist())
                if el not in self.known_patterns:
                    self.patterns[len(self.known_patterns), :] = pattern
                    self.known_patterns.add(el)
                    if len(self.known_patterns) == self.max_patterns:
                        if not self.increase_data_set_size():
                            break
        return self.observed_patterns

    def get_patterns(self, observation, action):
        return extract_patterns_to_predict(observation, self.padded_array, self.span, self.mask_array, self.observed_patterns, action)[:,:-1]

    def increase_data_set_size(self):
        self.patterns.resize((min(self.patterns.shape[0]*2, self.max_size), self.patterns.shape[1]), refcheck=False)
        if self.max_patterns != self.patterns.shape[0]:
            self.max_patterns = self.patterns.shape[0]
            return True
        else:
            self.max_patterns = self.patterns.shape[0]
            return False

    def get_data_set(self):
        return self.patterns[:len(self.known_patterns), :]

    def get_random_batch(self, batch_size):
        idx = random.sample(range(len(self.known_patterns)), min(len(self.known_patterns), batch_size))
        return self.patterns[idx, :]

    def write_to_file(self, filename):
        with open(filename, "wb") as file:
            pickle.dump(self.base_mask, file)

            pickle.dump(self.observation_shape, file)
            pickle.dump(self.mask_array, file)
            pickle.dump(self.span, file)

            pickle.dump(self.padded_array, file)
            pickle.dump(self.flattened_padded_array, file)
            pickle.dump(self.observed_patterns, file)

            pickle.dump(self.patterns, file)
            pickle.dump(self.max_patterns, file)
            pickle.dump(self.max_size, file)
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
            max_patterns = pickle.load(file)
            max_size = pickle.load(file)
            known_patterns = pickle.load(file)

        lfm_data = cls(base_mask, observation=None, initial_pattern_count=1, max_size=max_size)

        lfm_data.observation_shape = observation_shape
        lfm_data.mask_array = mask_array
        lfm_data.span = span
        lfm_data.padded_array = padded_array
        lfm_data.flattened_padded_array = flattened_padded_array
        lfm_data.observed_patterns = observed_patterns
        lfm_data.patterns = patterns
        lfm_data.max_patterns = max_patterns
        lfm_data.known_patterns = known_patterns

        return lfm_data


@njit
def extract_observed_patterns(prev_observation, padded_array, span, mask_array, data_set, action, observation):
    padded_array[span[0]:(span[0] + prev_observation.shape[0]), span[1]:(span[1] + prev_observation.shape[1])] = prev_observation
    a = padded_array.flatten()

    for i in prange(len(mask_array)):
        data_set[i, :-2] = a[mask_array[i]]
    data_set[:, -2] = action
    data_set[:, -1] = observation.flatten()

    return data_set


@njit
def extract_patterns_to_predict(prev_observation, padded_array, span, mask_array, data_set, action):
    padded_array[span[0]:(span[0] + prev_observation.shape[0]), span[1]:(span[1] + prev_observation.shape[1])] = prev_observation
    a = padded_array.flatten()

    for i in prange(len(mask_array)):
        data_set[i, :-2] = a[mask_array[i]]
    data_set[:, -2] = action

    return data_set

