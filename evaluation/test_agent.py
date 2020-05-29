import gym
#import gym_gvgai
import gvgai

import pickle
import tqdm
import time

from replay.sparsereplay import SparseReplay
from agents.models.tile_map import TileSet


def evaluate_agent(agent, game, levels, repetitions=20, tile_set=None, result_folder=None):
    results_per_level = {}

    for level in levels:
        result_per_trial = dict()
        env = gym.make("gvgai-" + game + '-' + level)
        #env = gym_gvgai.make("gvgai-" + game + '-' + level)

        for repetition in range(repetitions):
            state_observation = env.reset()

            if tile_set is None:
                tile_set = TileSet(state_observation, 10)
            else:
                tile_set.add_new_frame(state_observation)

            actions = env.unwrapped.get_action_meanings()
            total_score = 0
            replay = SparseReplay(game, level, tile_set.map_frame_to_lfm_state(state_observation))

            pbar = tqdm.trange(2000, desc=f"evaluation of: " + game + "-" + level)
            for tick in range(2000):
                pbar.update(1)
                action_id = agent.act(state_observation, actions)
                state_observation, diff_score, done, debug = env.step(action_id)
                replay.add_frame(action_id, tile_set.map_frame_to_lfm_state(state_observation),
                                 diff_score, debug["winner"])

                total_score += diff_score
                if done:
                    break
            pbar.close()
            result_per_trial[repetition] = [tick, total_score, debug["winner"], replay]

            if result_folder is not None:
                tile_set.write_to_file(f"{result_folder}\\tile_set.bin")
                replay.write_to_file(f"{result_folder}\\replay_data\\{level}_{repetition}.sreplay")
                with open(f"{result_folder}\\results.txt", "wb") as file:
                    pickle.dump(results_per_level, file)
        results_per_level[level] = result_per_trial
        env.close()

    return results_per_level


def evaluate_lfm_agent(agent, game, levels, repetitions=20, tile_set=None, result_folder=None, max_ticks=2000):
    results_per_level = {}

    for level in levels:
        result_per_trial = dict()
        #env = gym_gvgai.make("gvgai-" + game + '-' + level)
        env = gym.make("gvgai-" + game + '-' + level)

        for repetition in range(repetitions):
            state_observation = env.reset()
            agent_time = 0

            if tile_set is None:
                tile_set = TileSet(state_observation, 10)

            actions = env.unwrapped.get_action_meanings()
            #print(actions)
            total_score = 0
            lfm_state = tile_set.classify_frame_to_lfm_state(state_observation)
            replay = SparseReplay(game, level, lfm_state)
            agent.re_initialize(lfm_state, range(len(actions)))

            pbar = tqdm.trange(2000, desc=f"evaluation of: " + game + "-" + level)
            for tick in range(2000):
                pbar.update(1)

                start = time.time()
                action_id = agent.get_next_action(lfm_state, range(len(actions)))
                end = time.time()
                agent_time += end-start

                state_observation, diff_score, done, debug = env.step(action_id)
                lfm_state = tile_set.classify_frame_to_lfm_state(state_observation)
                replay.add_frame(action_id, lfm_state, diff_score, debug["winner"])
                env.render()
                total_score += diff_score

                if debug["winner"] == 'PLAYER_WINS' or debug["winner"] == 3:
                    diff_score += 1000
                if debug["winner"] == 'PLAYER_LOSES' or (debug["winner"] == 2 and game != "waterpuzzle"):
                    diff_score -= 1000

                start = time.time()
                agent.add_observation(lfm_state, diff_score, range(len(actions)))
                end = time.time()
                agent_time += end - start

                if done:
                    break
            pbar.close()
            result_per_trial[repetition] = [tick, total_score, debug["winner"], replay]
            print(debug["winner"], f"after {tick} ticks with an average decision time of {round(agent_time/max(1, tick),3)}s")

            if result_folder is not None:
                tile_set.write_to_file(f"{result_folder}\\tile_set.bin")
                replay.write_to_file(f"{result_folder}\\replay_data\\{level}_{repetition}.sreplay")
                with open(f"{result_folder}\\results.txt", "wb") as file:
                    pickle.dump(results_per_level, file)
        results_per_level[level] = result_per_trial
        env.close()

        if result_folder is not None:
            tile_set.write_to_file(f"{result_folder}\\tile_set.bin")
            replay.write_to_file(f"{result_folder}\\replay_data\\{level}_{repetition}.sreplay")
            with open(f"{result_folder}\\results.txt", "wb") as file:
                pickle.dump(results_per_level, file)

    return results_per_level

