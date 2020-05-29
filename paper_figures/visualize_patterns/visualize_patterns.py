from agents.models.local_forward_model_data import LocalForwardModelData
from abstractclasses.AbstractNeighborhoodPattern import CrossNeighborhoodPattern
import numpy as np
from agents.models.tile_map import TileSet
import matplotlib.pyplot as plt
import imageio


def visualize_pattern(pattern, tile_set):
    target_array = np.zeros((pattern.shape[0] * tile_set.tile_size,
                             pattern.shape[1] * tile_set.tile_size,
                             tile_set.tile_set[0].shape[2]), dtype=np.int)
    target_array[:, :, :] = 255

    for x in range(pattern.shape[0]):
        for y in range(pattern.shape[1]):
            if pattern[x, y] == -1:
                # plot default
                target_array[(x * tile_set.tile_size):((x + 1) * tile_set.tile_size),
                    (y * tile_set.tile_size):((y + 1) * tile_set.tile_size), :] = default_tile[:, :, :3]
                continue
            if pattern[x, y] == -2:
                continue  # plot nothing
            target_array[(x * tile_set.tile_size):((x + 1) * tile_set.tile_size),
                         (y * tile_set.tile_size):((y + 1) * tile_set.tile_size), :] = tile_set.tile_set[int(pattern[x, y])]
    return target_array


def visualize_all_pattern(game_state, patterns, mask, tile_set, filename=None):
    fig, ax = plt.subplots(game_state.shape[0], game_state.shape[1])

    plot_positions = [(x, y) for x in range(game_state.shape[0]) for y in range(game_state.shape[1])]
    for i, (x, y) in enumerate(plot_positions):
        pattern_to_plot = np.zeros(mask.shape)
        pattern_to_plot[:] = -2
        pattern_to_plot[mask] = patterns[i, :-1]

        pattern = visualize_pattern(pattern_to_plot, tile_set)
        ax[x, y].imshow(pattern)
        ax[x, y].axis('off')
    if filename is not None:
        plt.savefig(filename)
    plt.show()

    return fig, ax


if __name__ == "__main__":
    tile_set = TileSet.load_from_file("tile_set_waterpuzzle.bin")
    tile_set.plot_tile_dict()

    # original game-state
    game_state = np.zeros((3, 4))
    game_state[0, 0] = 1
    game_state[0, 1:4] = 1
    game_state[2, 1:4] = 1
    game_state[1, 0] = 3
    game_state[0, 1] = 0
    game_state[1, 1] = 4

    mask = CrossNeighborhoodPattern(1).get_mask()
    lfm_data = LocalForwardModelData(mask, game_state)
    tile_set.plot_lfm_state(game_state, "original_lfm_state.pdf")

    default_tile = imageio.imread('default.png')
    default_tile[default_tile == 0] = 255
    visualize_all_pattern(game_state, lfm_data.get_patterns(game_state, 1), mask, tile_set, "patterns.pdf")

    # predicted game-state
    game_state = np.zeros((3, 4))
    game_state[0, 0] = 1
    game_state[0, 1:4] = 1
    game_state[2, 1:4] = 1
    game_state[1, 0] = 4
    game_state[0, 1] = 0
    game_state[1, 1] = 0
    tile_set.plot_lfm_state(game_state, "resulting_lfm_state.pdf")
