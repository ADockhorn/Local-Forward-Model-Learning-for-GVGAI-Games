import matplotlib.pyplot as plt
import gym
import gvgai


def figure_different_tile_sizes():
    env = gym.make(f'gvgai-golddigger-lvl0-v0', pixel_observations=True, include_semantic_data=False)
    initial_frame_golddigger = env.reset()
    env.close()

    env = gym.make(f'gvgai-treasurekeeper-lvl0-v0', pixel_observations=True, include_semantic_data=False)
    initial_frame_treasurekeeper = env.reset()
    env.close()

    env = gym.make(f'gvgai-waterpuzzle-lvl0-v0', pixel_observations=True, include_semantic_data=False)
    initial_frame_waterpuzzle = env.reset()
    env.close()

    plt.imshow(initial_frame_golddigger[::-1, :, :], origin='lower')
    plt.axis("off")
    plt.savefig(f"initial_state_golddigger.pdf")
    plt.show()

    plt.imshow(initial_frame_treasurekeeper[::-1, :, :], origin='lower')
    plt.axis("off")
    plt.savefig(f"initial_state_treasurekeeper.pdf")
    plt.show()

    plt.imshow(initial_frame_waterpuzzle[::-1, :, :], origin='lower')
    plt.axis("off")
    plt.savefig(f"initial_state_waterpuzzle.pdf")
    plt.show()


if __name__ == "__main__":
    figure_different_tile_sizes()


