from numba import njit
import numba
from numba.typed import List
import numpy as np
import pickle


class TileSet:

    def __init__(self, frame, tile_size, tile_set=None, threshold=0.85):
        self.tile_size = tile_size
        self.threshold = min(0.99, threshold)
        if tile_set is None:
            self.tile_set = create_unique_tile_set(frame[:, :, :3], tile_size, threshold)
        else:
            self.tile_set = tile_set

    @classmethod
    def load_from_file(cls, filename):
        with open(filename, "rb") as file:
            tile_size = pickle.load(file)
            tile_set = List.empty_list(numba.types.Array(numba.uint8, 3, "A"))
            tile_set_list = pickle.load(file)
            for el in tile_set_list:
                tile_set.append(el)
        return cls(None, tile_size, tile_set)

    def write_to_file(self, filename):
        with open(filename, "wb") as file:
            pickle.dump(self.tile_size, file)
            pickle.dump(list(self.tile_set), file)

    def add_new_frame(self, frame):
        self.tile_set = add_unique_tiles(frame, self.tile_size, self.tile_set, self.threshold)

    def map_frame_to_lfm_state(self, frame, add_frame=False):
        frame = frame[:, :, :3]
        if add_frame:
            self.add_new_frame(frame)
        return map_frame_to_lfm_state(frame, self.tile_size, self.tile_set, self.threshold)

    def classify_frame_to_lfm_state(self, frame):
        frame = frame[:, :, :3]
        return classify_frame_to_lfm_state(frame, self.tile_size, self.tile_set)

    def map_lfm_state_to_frame(self, lfm_state):
        target_array = np.zeros((lfm_state.shape[0] * self.tile_size,
                                 lfm_state.shape[1] * self.tile_size, self.tile_set[0].shape[2]), dtype=np.int)
        return map_lfm_state_to_frame(target_array, lfm_state, self.tile_size, self.tile_set, self.threshold)

    def nr_of_known_tiles(self):
        return len(self.tile_set)

    def plot_tile_dict(self, filename = None):
        import matplotlib.pyplot as plt
        import math

        rows = math.ceil(math.sqrt(len(self.tile_set)))
        cols = math.ceil(len(self.tile_set)/rows)
        fig, axs = plt.subplots(rows, cols)

        for tile, ax in zip(self.tile_set, axs.flatten()):
            ax.imshow(tile)
            ax.set_axis_off()
        for i in range(len(self.tile_set), len(axs.flatten())):
            axs.flatten()[i].set_axis_off()

        if filename is not None:
            plt.savefig(filename)
        plt.show()

    def plot_lfm_state(self, lfm_state, filename=None):
        import matplotlib.pyplot as plt
        target_array = np.zeros((lfm_state.shape[0] * self.tile_size,
                                 lfm_state.shape[1] * self.tile_size, self.tile_set[0].shape[2]), dtype=np.int)
        plt.imshow(map_lfm_state_to_frame(target_array, lfm_state, self.tile_size, self.tile_set))
        plt.gca().set_axis_off()
        if filename is not None:
            plt.savefig(filename)
        plt.show()


def create_unique_tile_set(frame, tile_size, threshold):
    tile_set = List()
    xTiles = frame.shape[0] // tile_size
    yTiles = frame.shape[1] // tile_size
    for x in range(xTiles):
        for y in range(yTiles):
            tile = frame[(x * tile_size):((x + 1) * tile_size),
                         (y * tile_size):((y + 1) * tile_size), :]
            for tmp in tile_set:
                if np.corrcoef(tmp.flatten(), tile.flatten())[1, 0] >= threshold:
                #if np.array_equal(tmp, tile):
                    break
            else:
                tile_set.append(tile)
    return tile_set

@njit
def add_unique_tiles(frame, tile_size, tile_set, threshold):
    xTiles = frame.shape[0] // tile_size
    yTiles = frame.shape[1] // tile_size
    for x in range(xTiles):
        for y in range(yTiles):
            tile = frame[(x * tile_size):((x + 1) * tile_size),
                         (y * tile_size):((y + 1) * tile_size), :]
            for tmp in tile_set:
                #if np.array_equal(tmp, tile):
                if np.corrcoef(tmp.flatten(), tile.flatten())[1, 0] >= threshold:
                    break
            else:
                tile_set.append(tile)
    return tile_set


@njit
def map_frame_to_lfm_state(frame, tile_size, tile_set, threshold):
    state = np.zeros((frame.shape[0]//tile_size, frame.shape[1]//tile_size), dtype=np.uint8)
    for x in range(state.shape[0]):
        for y in range(state.shape[1]):
            tile = frame[(x * tile_size):((x + 1) * tile_size), (y * tile_size):((y + 1) * tile_size), :]
            for i, tmp in enumerate(tile_set):
                #if np.array_equal(tmp, tile):
                if np.corrcoef(tmp.flatten(), tile.flatten())[1, 0] >= threshold:
                    state[x, y] = i
                    break
    return state


@njit
def classify_frame_to_lfm_state(frame, tile_size, tile_set):
    state = np.zeros((frame.shape[0]//tile_size, frame.shape[1]//tile_size), dtype=np.uint8)
    for x in range(state.shape[0]):
        for y in range(state.shape[1]):
            tile = frame[(x * tile_size):((x + 1) * tile_size), (y * tile_size):((y + 1) * tile_size), :]
                #if np.array_equal(tmp, tile):
            a = list()
            for tmp in tile_set:
                a.append(np.corrcoef(tmp.flatten(), tile.flatten())[1, 0])
            state[x, y] = np.argmax(np.array(a))
    return state


@njit
def map_lfm_state_to_frame(target_array, lfm_state, tile_size, tile_set):
    for x in range(lfm_state.shape[0]):
        for y in range(lfm_state.shape[1]):
            target_array[(x * tile_size):((x + 1) * tile_size),
                  (y * tile_size):((y + 1) * tile_size), :] = tile_set[int(lfm_state[x, y])]
    return target_array
