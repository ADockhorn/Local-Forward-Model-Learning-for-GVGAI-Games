import gym_gvgai
import random
import matplotlib.pyplot as plt
import numpy as np
import math


class Agent:
    def __init__(self):
        pass

    def act(self, state_obs, actions):
        print(state_obs, actions)
        return random.choice(range(len(actions)))


def estimate_tile_size(frames):
    unique_tiles_per_size = dict()
    for tile_size in range(5, 20):
        if frames[0].shape[0] % tile_size != 0 or frames[1].shape[1] % tile_size != 0:
            continue
        print(tile_size)
        tiles = get_unique_tiles(frames, tile_size)
        unique_tiles_per_size[tile_size] = len(tiles)
    return unique_tiles_per_size


def get_unique_tiles(frames, tile_size):
    tiles = []
    for frame in frames:
        xTiles = frame.shape[0] // tile_size
        yTiles = frame.shape[1] // tile_size
        for x in range(xTiles):
            for y in range(yTiles):
                tile = frame[(x * tile_size):((x + 1) * tile_size), (y * tile_size):((y + 1) * tile_size), :]
                for tmp in tiles:
                    if np.all(np.equal(tmp, tile)):
                        break
                else:
                    tiles.append(tile)

    return tiles


if __name__ == "__main__":
    games = ['gvgai-golddigger', 'gvgai-treasurekeeper', 'gvgai-waterpuzzle']

    validateLevels = ['lvl2-v0']
    repsPerLevel = 5
    ticksPerLevel = 100

    # variables for recording the results
    results = {}

    best_tile_size = dict()
    tile_sets = dict()

    for game in games:
        levelRecord = {}
        state_observations = []
        for level in validateLevels:
            timeRecord = {}
            for tick in range(repsPerLevel):
                env = gym_gvgai.make(game + '-' + level)
                agent = Agent()
                print('Starting ' + env.env.game + " with Level " + str(env.env.lvl))
                stateObs = env.reset()
                plt.imshow(stateObs)
                plt.show()
                env.close()

                state_observations.append(stateObs)
                actions = env.unwrapped.get_action_meanings()
                totalScore = 0
                for tick in range(ticksPerLevel):
                    action_id = agent.act(stateObs, actions)
                    stateObs, diffScore, done, debug = env.step(action_id)
                    totalScore += diffScore
                    env.render()

                    print("Action " + str(action_id) + " tick " + str(tick+1) + " reward " + str(diffScore) + " win " + debug["winner"])
                    if done:
                        break
                timeRecord[tick] = [tick, totalScore, debug["winner"]]
                env.close()
            levelRecord[level] = timeRecord

        a = estimate_tile_size(state_observations)
        plt.plot(list(a.keys()), list(a.values()))
        plt.xlabel("tile size")
        plt.ylabel("nr of unique tiles")
        plt.title(game)
        plt.show()

        tile_size = list(a.keys())[np.argmin(list(a.values()))]

        tiles = get_unique_tiles(state_observations, tile_size)
        unique_tiles = len(tiles)
        fig, axs = plt.subplots(math.ceil(math.sqrt(unique_tiles)), math.ceil(math.sqrt(unique_tiles)), figsize=(8, 10))

        for tile, ax in zip(tiles, axs.flatten()):
            ax.imshow(tile)
            ax.set_axis_off()
        for i in range(len(tiles), len(axs.flatten())):
            axs.flatten()[i].set_axis_off()
        fig.suptitle(f"Unique tiles of the game {game}", fontsize=18)
        plt.show()

        results[game] = levelRecord
