from abc import ABC, abstractmethod
import numpy as np
from typing import Union
from abstractclasses.AbstractGameState import GridGameState


class AbstractNeighborhoodPattern(ABC):
    """
    Implementation of Local Forward Model Patterns. The get_pattern functions are deprecated since their processing
    in the LocalForwardModelData object is much more efficient.
    """
    def __init__(self, span):
        super().__init__()
        self._span = span
        self._pattern_mask = self.get_mask()
        self._ext_pattern_mask = np.transpose(np.array([self._pattern_mask, self._pattern_mask, self._pattern_mask]),
                                              (1, 2, 0))

        self._pattern_elements = np.sum(self._pattern_mask)
        self._width = span*2+1
        self._height = span*2+1

    @abstractmethod
    def get_mask(self):
        """
        :return: np.ndarray containing a true for each field that is part of the center tile's local neighborhood
        """
        pass

    def get_width(self):
        return self._width

    def get_height(self):
        return self._height

    def get_num_elements(self):
        return self._pattern_elements

    def get_pattern(self, game_state_grid: np.ndarray, x: int, y: int):
        ext_game_state_grid = np.pad(game_state_grid, self._span, "constant", constant_values="x")
        return self.get_pattern_of_extended_game_state(ext_game_state_grid, x, y)

    def get_pattern_of_extended_game_state(self, ext_game_state_grid: np.ndarray, x: int, y: int,
                                           include_z: bool = False):
        if not include_z:
            return ext_game_state_grid[self._span + x - self._span: self._span + x + self._span + 1,
                                       self._span + y - self._span: self._span + y + self._span + 1][self._pattern_mask]
        else:
            return \
                ext_game_state_grid[self._span + x - self._span: self._span + x + self._span + 1,
                                    self._span + y - self._span: self._span + y + self._span + 1, :][
                    self._ext_pattern_mask].flatten()

    def pattern_to_img(self, pattern):
        image = np.zeros((self._span*2+1, self._span*2+1, 3), dtype=np.uint8)
        image[self._ext_pattern_mask] = pattern
        return image

    def get_patterns_for_position_list(self, game_state_grid: np.ndarray, positions):
        if len(game_state_grid.shape) == 3:  # image array
            ext_game_state_grid = np.pad(game_state_grid, [(self._span, self._span), (self._span, self._span), (0, 0)],
                                         "constant", constant_values=0)
            patterns = np.zeros((len(positions), self._pattern_elements*3), dtype=ext_game_state_grid.dtype)
            for idx, (x, y) in enumerate(positions):
                patterns[idx, :] = self.get_pattern_of_extended_game_state(ext_game_state_grid, x, y, True)
        else:
            ext_game_state_grid = np.pad(game_state_grid, self._span, "constant", constant_values="x")
            patterns = np.zeros((len(positions), self._pattern_elements), dtype=ext_game_state_grid.dtype)
            for idx, (x, y) in enumerate(positions):
                patterns[idx, :] = self.get_pattern_of_extended_game_state(ext_game_state_grid, x, y)
        return patterns

    def get_all_patterns(self, game_state: Union[GridGameState, np.ndarray], action=None,
                         next_game_state: Union[GridGameState, np.ndarray, None] = None,
                         differential=False):
        """
        returns the square-neighborhood-environment
        second to last column is the action
        last column is the outcome
        """
        if isinstance(game_state, GridGameState):
            game_state_grid = game_state.get_tile_map()
        else:
            game_state_grid = game_state
        if next_game_state is not None and isinstance(game_state, GridGameState):
            target = next_game_state.get_tile_map()
        else:
            target = next_game_state

        patterns = self.get_patterns_for_position_list(game_state_grid, [(x, y) for x in range(game_state_grid.shape[0])
                                                                         for y in range(game_state_grid.shape[1])])

        if action is not None and next_game_state is not None:
            # one row per cell, one column per element in the neighborhood + action + target
            train_data = np.zeros((game_state_grid.shape[1] * game_state_grid.shape[0],
                                   patterns.shape[1] + 2), dtype=game_state_grid.dtype)
            train_data[:, :-2] = patterns
            train_data[:, -2] = action
            if differential:
                differential_target = target.copy()
                differential_target[:, :] = 'y'
                mask = target != game_state_grid
                differential_target[mask] = target[mask]
                train_data[:, -1] = differential_target.flatten()

            else:
                train_data[:, -1] = target.flatten()
        elif action is not None or next_game_state is not None:
            # one row per cell, one column per element in the neighborhood + (action or target)
            train_data = np.zeros((game_state_grid.shape[1] * game_state_grid.shape[0],
                                   patterns.shape[1] + 1), dtype=game_state_grid.dtype)
            train_data[:, :-1] = patterns
            if action is not None:
                train_data[:, -1] = action
            else:
                if differential:
                    differential_target = target.copy()
                    differential_target[:, :] = 'y'
                    mask = target != game_state_grid
                    differential_target[mask] = target[mask]
                    train_data[:, -1] = differential_target.flatten()

                else:
                    train_data[:, -1] = target.flatten()
        else:
            # one row per cell, one column per element in the neighborhood + action
            return patterns

        return train_data

    def get_all_patterns_and_rotations(self, game_state: Union[GridGameState, np.ndarray], action=None,
                                       next_game_state: Union[GridGameState, np.ndarray, None] = None):
        if isinstance(game_state, GridGameState):
            game_state_grid = game_state.get_tile_map()
        else:
            game_state_grid = game_state
        if next_game_state is not None and isinstance(game_state, GridGameState):
            target = next_game_state.get_tile_map()
        else:
            target = next_game_state

        patterns = [self.get_all_patterns(game_state_grid, action, target)]

        # add rotations

        game_state_grid, action, target = np.rot90(game_state_grid), (action % 4) + 1, np.rot90(target)
        patterns.append(self.get_all_patterns(game_state_grid, action, target))

        game_state_grid, action, target = np.rot90(game_state_grid), (action % 4) + 1, np.rot90(target)
        patterns.append(self.get_all_patterns(game_state_grid, action, target))

        game_state_grid, action, target = np.rot90(game_state_grid), (action % 4) + 1, np.rot90(target)
        patterns.append(self.get_all_patterns(game_state_grid, action, target))

        return np.vstack(patterns)

    def get_all_patterns_and_flipped_rotations(self, game_state: Union[GridGameState, np.ndarray], action=None,
                                               next_game_state: Union[GridGameState, np.ndarray, None] = None,
                                               differential=False):
        if isinstance(game_state, GridGameState):
            game_state_grid = game_state.get_tile_map()
        else:
            game_state_grid = game_state
        if next_game_state is not None and isinstance(game_state, GridGameState):
            target = next_game_state.get_tile_map()
        else:
            target = next_game_state

        patterns = []

        # add rotations
        for i in range(4):
            patterns.append(self.get_all_patterns(game_state_grid, action, target, differential))
            game_state_grid, action, target = np.rot90(game_state_grid), (action % 4) + 1, np.rot90(target)

        # add flipped rotations
        game_state_grid, target = np.flip(game_state_grid, 0), np.flip(target, 0)
        if action in {2, 4}:
            horizonally_flipped = {2: 4, 4: 2, 1: 3, 3: 1}
        else:
            horizonally_flipped = {2: 2, 4: 4, 1: 1, 3: 3}

        for i in range(4, 8):
            patterns.append(self.get_all_patterns(game_state_grid, horizonally_flipped[action], target))

            # from sokoban.sokoban import SokobanConstants
            # from visualization.TileMapVisualizer import TileMapVisualizer
            # tsv.visualize_observation_grid(game_state_grid, game_state_grid.shape[0], game_state_grid.shape[1])
            # plt.title(horizonally_flipped[action])
            # plt.show()

            game_state_grid, action, target = np.rot90(game_state_grid), (action % 4) + 1, np.rot90(target)

        return np.vstack(patterns)


class SquareNeighborhoodPattern(AbstractNeighborhoodPattern):
    """
    Square-shaped neighborhood pattern including all elements with manhattan distance < span.
    """
    def __init__(self, span):
        super().__init__(span)
        self._span = span

    def get_mask(self):
        return np.array([[True for _ in range(self._span*2+1)] for _ in range(self._span*2+1)])


class CrossNeighborhoodPattern(AbstractNeighborhoodPattern):
    """
    Cross-shaped neighborhood pattern as shown in the paper.
    """
    def __init__(self, span):
        super().__init__(span)

    def get_mask(self):
        return np.array([[True if x == self._span or y == self._span else False
                          for x in range(self._span * 2+1)] for y in range(self._span * 2 + 1)])


class GeneralizedNeighborhoodPattern(AbstractNeighborhoodPattern):
    """
    Generalized neighborhood pattern using the Minkowski distance family.
    """
    def __init__(self, span, k: float = 2):
        self.k = k
        super().__init__(span)

    def get_mask(self):
        return np.array([[True if x == self._span or y == self._span or
                        ((abs(self._span-x)/self._span)**self.k +
                         (abs(self._span-y)/self._span)**self.k)**(1/self.k) <= 1
                          else False for x in range(self._span*2+1)] for y in range(self._span*2+1)])
