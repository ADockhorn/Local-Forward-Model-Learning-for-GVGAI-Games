from agents.models.tile_map import TileSet
import matplotlib.pyplot as plt
import imageio


def figure_different_tile_sizes():
    initial_frame = imageio.imread('example_game_state.png')

    plt.imshow(initial_frame[::-1, :, :], origin='lower')
    for x in range(0, initial_frame.shape[0] // 10 + 1):
        plt.axhline(min(x * 10, 99), c="r")
    for y in range(0, initial_frame.shape[1] // 10 + 1):
        plt.axvline(min(y * 10, 99), c="r")
    plt.xticks([0, 20, 40, 60, 80, 99], [0, 20, 40, 60, 80, 100], fontsize=16)
    plt.yticks([0, 20, 40, 60, 80, 99], [0, 20, 40, 60, 80, 100], fontsize=16)
    plt.savefig("tiling_tile_size_10.pdf")
    plt.show()

    tile_set_10 = TileSet(initial_frame, 10)
    tile_set_10.add_new_frame(initial_frame)
    tile_set_10.plot_tile_dict("tile_dict_tile_size_10.pdf")

    plt.imshow(initial_frame[::-1, :, :], origin='lower')
    for x in range(0, initial_frame.shape[0] // 20 + 1):
        plt.axhline(min(x * 20, 99), c="r")
    for y in range(0, initial_frame.shape[1] // 20 + 1):
        plt.axvline(min(y * 20, 99), c="r")
    plt.xticks([0, 20, 40, 60, 80, 99], [0, 20, 40, 60, 80, 100], fontsize=16)
    plt.yticks([0, 20, 40, 60, 80, 99], [0, 20, 40, 60, 80, 100], fontsize=16)
    plt.savefig("tiling_tile_size_20.pdf")
    plt.show()

    tile_set_20 = TileSet(initial_frame, 20)
    tile_set_20.add_new_frame(initial_frame)
    tile_set_20.plot_tile_dict("tile_dict_tile_size_20.pdf")


if __name__ == "__main__":
    figure_different_tile_sizes()



