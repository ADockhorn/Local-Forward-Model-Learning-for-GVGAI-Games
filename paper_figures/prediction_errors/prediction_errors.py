from agents.models.tile_map import TileSet
import numpy as np
import matplotlib.pyplot as plt


def prediction_errors():
    pass


def plot_state(lfm_state, filename):
    state = tile_set.map_lfm_state_to_frame(lfm_state)
    plt.imshow(state[::-1, :, :], origin='lower')
    for x in [1, 2, 3, 4]:
        plt.axhline(x * 10-0.5, c="k")
        plt.axvline(x * 10-0.5, c="k")
    plt.axis("off")
    plt.savefig(filename)
    plt.show()


if __name__ == "__main__":
    prediction_errors()

    tile_set = TileSet.load_from_file("tile_set.bin")
    tile_set.plot_tile_dict()

    original = np.zeros((5, 5))
    original[:, :] = 5
    original[0, 0] = 0
    original[0, 1:4] = 1
    original[0, 4] = 2
    original[1:4, 0] = 3
    original[1:4, 4] = 3
    original[4, 0] = 7
    original[4, 1:4] = 1
    original[4, 4] = 8
    original[2, 2] = 18
    plot_state(original, "original_state.pdf")

    true_outcome = original.copy()
    true_outcome[2, 2] = 5
    true_outcome[1, 2] = 18
    plot_state(true_outcome, "true_outcome.pdf")

    wrong_prediction = original.copy()
    wrong_prediction[2, 2] = 5
    wrong_prediction[2, 1] = 18
    plot_state(wrong_prediction, "valid_prediction.pdf")

    invalid_prediction = original.copy()
    invalid_prediction[2, 2] = 5
    invalid_prediction[2, 1] = 18
    invalid_prediction[2, 3] = 18
    invalid_prediction[1, 2] = 18
    invalid_prediction[3, 2] = 18
    plot_state(invalid_prediction, "invalid_prediction.pdf")

