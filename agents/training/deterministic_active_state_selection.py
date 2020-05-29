from abc import ABC
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pickle
from abstractclasses.AbstractNeighborhoodPattern import CrossNeighborhoodPattern
from agents.models.active_learning_local_forward_model_data import ActiveLearningLocalForwardModelData
from agents.models.score_model_data import ScoreModelData
from agents.models.tile_map import TileSet
from replay.sparsereplay import SparseReplay
import datetime
import os
from Constants import THRESHOLDS, DETERMINISTIC_TEST_GAMES, SPANS
import gym
import gvgai  # optimized framework version by Chris Bamford, not fully compatible due to image compression
from data.additional_training_levels.create_additional_training_levels import load_file, write_file


def plot_level_tag(level_tag, state_shape):
    plt.imshow(np.array(level_tag, dtype=np.int).reshape(state_shape))
    plt.show()


def get_state_tag(state):
    if state is None:
        print()
        return None
    return tuple(state.flatten())


class ActiveStateSelectionAgent(ABC):
    def __init__(self, lfm_data_object, sm_data_object):
        super().__init__()
        self.lfm_data = lfm_data_object
        self.sm_data = sm_data_object
        self.data_set = dict()


class MaxUnknownPatternsStateActionSelection(ActiveStateSelectionAgent):
    def __init__(self, lfm_data_object, sm_data_object):
        super().__init__(lfm_data_object, sm_data_object)
        self.known_states = dict()      # stores which patterns occur per state
        self.known_lfm_patterns = dict()    # stores in which state each pattern patterns occurs
        self.known_sm_patterns = dict()     # stores in which state each pattern patterns occurs
        self.candidate_solution = [-1, ()]
        self.state_shape = None
        self.solution_still_valid = False
        self.terminal_transitions = dict()
        self.sm_pattern_length = 0
        self.states_to_expand = []
        pass

    def set_state_shape(self, state_shape):
        self.solution_still_valid = False
        if not np.array_equal(state_shape, self.state_shape):
            self.state_shape = state_shape
            self.known_states = dict()
            self.states_to_expand = []
            for pattern in self.known_lfm_patterns:
                self.known_lfm_patterns[pattern]["states"] = set()
            for pattern in self.known_sm_patterns:
                self.known_sm_patterns[pattern]["state_action_pairs"] = set()

    def add_state(self, game_state, action_indices, stop_after_x_expansions=None, stop_after_unknown_pattern=False):
        # for each state track the game_state, observable patterns that have not been observed yet,
        # and a list of child_states

        self.states_to_expand.insert(0, game_state)
        unknown_pattern_found = False

        while len(self.states_to_expand) > 0 and (stop_after_x_expansions is None or stop_after_x_expansions > 0):
            current_state = self.states_to_expand.pop(0)

            if current_state is None:
                raise ValueError()
            level_tag = get_state_tag(current_state)

            # never process a state twice
            if level_tag in self.known_states:
                continue
            if stop_after_unknown_pattern and unknown_pattern_found:
                break

            self.known_states[level_tag] = {"unknown_patterns": {x: set() for x in action_indices},
                                            "child_level_tags": {x: None for x in action_indices},
                                            "unknown_sm_pattern": {x: True for x in action_indices}}
            # add patterns -> state links
            patterns = self.lfm_data.get_patterns(current_state, 0)[:, :-1]
            if stop_after_x_expansions is not None:
                stop_after_x_expansions -= 1

            for action in action_indices:
                for pattern in patterns:
                    pattern = (*pattern, action)
                    if pattern in self.known_lfm_patterns:
                        # only add the link in case the pattern is still unknown to the learner
                        if not self.known_lfm_patterns[pattern]["observed"]:
                            self.known_lfm_patterns[pattern]["states"].add(level_tag)
                    else:
                        self.known_lfm_patterns[pattern] = {"observed": False, "states": {level_tag}}

                    # add state -> pattern links
                    # only add the pattern in case the pattern was unobserved and
                    # we have not already looked at this state
                    if self.known_states[level_tag]["child_level_tags"][action] is None:
                        if not self.known_lfm_patterns[pattern]["observed"]:
                            self.known_states[level_tag]["unknown_patterns"][action].add(pattern)
                            unknown_pattern_found = True
                    else:
                        raise ValueError("the level_tag's child states should be None since the state was just added")

            for action in action_indices[::-1]:
                if len(self.known_states[level_tag]["unknown_patterns"][action]) == 0 and \
                        self.known_states[level_tag]["child_level_tags"][action] is None:
                    # the next state can be predicted
                    child_state = self.lfm_data.get_prediction(current_state, action)
                    self.known_states[level_tag]["child_level_tags"][action] = get_state_tag(child_state)

                    sm_pattern = tuple(self.sm_data.get_pattern(current_state, child_state))
                    if sm_pattern not in self.known_sm_patterns:
                        self.known_sm_patterns[sm_pattern] = {"observed": False, "state_action_pairs": set()}

                    if self.known_sm_patterns[sm_pattern]["observed"] is False:
                        self.known_sm_patterns[sm_pattern]["state_action_pairs"].add((level_tag, action))
                        self.known_states[level_tag]["unknown_sm_pattern"][action] = True
                    else:
                        self.known_states[level_tag]["unknown_sm_pattern"][action] = False

                    self.states_to_expand.insert(0, child_state)
                    # self.add_state(child_state, action_indices)

        return get_state_tag(game_state)

    def add_child_state(self, parent_tag, action, child_tag):
        self.known_states[parent_tag]["child_states"][action] = child_tag

    def update_sm_pattern_size(self, current_length):
        if self.sm_pattern_length != current_length:
            self.sm_pattern_length = current_length
            key_set = list(self.known_sm_patterns.keys())
            for old_pattern in key_set:
                new_pattern = old_pattern + (0,)*(current_length - len(old_pattern))
                self.known_sm_patterns[new_pattern] = self.known_sm_patterns[old_pattern]
                del self.known_sm_patterns[old_pattern]
        pass

    def remove_observed_lfm_patterns(self, patterns_to_be_removed, action_indices):
        for pattern in patterns_to_be_removed:
            outcome = pattern[-1]
            pattern = tuple(pattern[:-1])
            action = pattern[-1]

            if pattern in self.known_lfm_patterns:
                self.known_lfm_patterns[pattern]["observed"] = True
                self.known_lfm_patterns[pattern]["outcome"] = outcome
                # if pattern not in self.lfm_data.known_patterns:
                #    self.lfm_data.add_pattern_manually(pattern, outcome)

                for state in self.known_lfm_patterns[pattern]["states"]:
                    self.known_states[state]["unknown_patterns"][action].discard(pattern)

                for state in self.known_lfm_patterns[pattern]["states"]:
                    if len(self.known_states[state]["unknown_patterns"][action]) == 0:
                        self.solution_still_valid = False

                        parent_state = np.array(state).reshape(self.state_shape)
                        child_state = self.lfm_data.get_prediction(parent_state, action)
                        if child_state is None:
                            child_state = self.predict(state, action)
                        if child_state is None:
                            continue
                        self.known_states[state]["child_level_tags"][action] = get_state_tag(child_state)

                        sm_pattern = tuple(self.sm_data.get_pattern(parent_state, child_state))
                        if sm_pattern not in self.known_sm_patterns:
                            self.known_sm_patterns[sm_pattern] = {"observed": False, "state_action_pairs": set()}

                        if self.known_sm_patterns[sm_pattern]["observed"] is False:
                            self.known_sm_patterns[sm_pattern]["state_action_pairs"].add((state, action))
                            self.known_states[state]["unknown_sm_pattern"][action] = True
                        else:
                            self.known_states[state]["unknown_sm_pattern"][action] = False

                        self.add_state(child_state, action_indices, None, False)

                self.known_lfm_patterns[pattern]["states"] = set()
            else:
                raise ValueError("The pattern should already be known, but not observed")
        return

    def remove_observed_sm_patterns(self, parent_tag, last_action, pattern_to_be_removed):
        pattern_to_be_removed = tuple(pattern_to_be_removed)
        if pattern_to_be_removed in self.known_sm_patterns and \
                not self.known_sm_patterns[pattern_to_be_removed]["observed"]:
            self.known_sm_patterns[pattern_to_be_removed]["observed"] = True
            for state, action in self.known_sm_patterns[pattern_to_be_removed]["state_action_pairs"]:
                self.known_states[state]["unknown_sm_pattern"][action] = False
        else:
            self.known_sm_patterns[pattern_to_be_removed] = {"observed": True, "state_action_pairs": set()}
        self.known_states[parent_tag]["unknown_sm_pattern"][last_action] = False

    def predict(self, state, action):
        patterns = self.lfm_data.get_patterns(np.array(state).reshape(self.state_shape), action)
        outcomes = []
        invalid = False
        for pattern in patterns:
            pattern = tuple(pattern)
            if pattern in self.known_lfm_patterns:
                if "outcome" in self.known_lfm_patterns[pattern]:
                    outcomes.append(self.known_lfm_patterns[pattern]["outcome"])
                else:
                    self.known_lfm_patterns[pattern]["states"].add(state)
                    self.known_states[state]["unknown_patterns"][action].add(pattern)
            else:
                invalid = True
                self.known_states[state]["unknown_patterns"][action].add(pattern)
                self.known_lfm_patterns[pattern] = {"observed": False, "states": {state}}

        if invalid:
            return None
        else:
            return np.array(outcomes).reshape(self.state_shape)

    def reset_patterns(self):
        self.known_states = dict()
        for pattern in self.known_lfm_patterns:
            self.known_lfm_patterns[pattern]["states"] = set()
        return

    def select_action(self, state, action_indices):
        self.update_sm_pattern_size(sm_data.get_pattern_lenth())
        # state_tag = self.add_state(state, range(len(action_indices)), 100, False)
        state_tag = self.add_state(state, range(len(action_indices)), None, False)

        # perform graph search
        visited_states = set()
        states_to_visit = [(state_tag, ())]
        if len(self.candidate_solution[1]) > 0:
            most_unknown_patterns, best_action_sequence = self.candidate_solution
        else:
            pass
            most_unknown_patterns = -1
            best_action_sequence = ()
            self.solution_still_valid = False

        if not self.solution_still_valid:
            while len(states_to_visit) > 0:
                current_state, action_sequence = states_to_visit.pop(0)
                if current_state in visited_states:
                    continue    # a child node was added multiple times before being processed
                visited_states.add(current_state)

                if current_state in self.known_states:
                    # plot_level_tag(current_state)
                    for action in action_indices:
                        # print(len(self.known_states[current_state]["unknown_patterns"][action]))
                        nr_unknown_patterns = len(self.known_states[current_state]["unknown_patterns"][action])
                        nr_unknown_patterns += self.known_states[current_state]["unknown_sm_pattern"][action]
                        if nr_unknown_patterns > most_unknown_patterns:
                            # print("new max", len(self.known_states[current_state]["unknown_patterns"][action]),
                            # "before", most_unknown_patterns)
                            best_action_sequence = (*action_sequence, action)
                            most_unknown_patterns = nr_unknown_patterns

                        child_state = self.known_states[current_state]["child_level_tags"][action]
                        if child_state in self.known_states and child_state not in visited_states and\
                                not (current_state in self.terminal_transitions and action in
                                     self.terminal_transitions[current_state]):
                            states_to_visit.append((child_state, (*action_sequence, action)))

                else:
                    raise ValueError("either the search did not stop despite not knowing the child state "
                                     "or the starting state was not properly added")

        self.candidate_solution = [most_unknown_patterns, best_action_sequence[1:]]
        self.solution_still_valid = True
        return best_action_sequence[0], most_unknown_patterns

    def add_observation(self, parent_state, child_state, last_action, observed_lfm_patterns,
                        observed_sm_patterns, action_indices, done):
        parent_tag = get_state_tag(parent_state)
        child_tag = get_state_tag(child_state)
        if self.known_states[parent_tag]["child_level_tags"][last_action] is not None:
            if self.known_states[parent_tag]["child_level_tags"][last_action] != child_tag:
                # plot_level_tag(parent_tag)
                # plot_level_tag(child_tag)
                # plot_level_tag(self.known_states[parent_tag]["child_level_tags"][last_action])
                self.known_states[parent_tag]["child_level_tags"][last_action] = child_tag
                self.candidate_solution = [-1, ()]
                self.solution_still_valid = False

        self.remove_observed_lfm_patterns(observed_lfm_patterns.copy(), action_indices)
        self.update_sm_pattern_size(sm_data.get_pattern_lenth())
        self.remove_observed_sm_patterns(parent_tag, last_action, observed_sm_patterns)
        if done:
            if parent_tag in self.terminal_transitions:
                self.terminal_transitions[parent_tag].add(last_action)
            else:
                self.terminal_transitions[parent_tag] = {last_action}


def actively_train_agent_model(level_generator, lfm_data, sm_data, tile_set_threshold, tile_set=None, target_folder=None,
                               max_ticks=2000, reps_per_level=20):
    global environment_fail
    results = dict()

    skip_level = None
    agent = MaxUnknownPatternsStateActionSelection(lfm_data, sm_data)
    for training_run, (env, game, level, initial_frame) in enumerate(level_generator):
        if level == skip_level:
            if target_folder is not None and ((training_run + 1) % reps_per_level) == 0:
                tile_set.write_to_file(f"{target_folder}\\checkpoints\\tile_set_{level.split('-')[0]}.bin")
                lfm_data.write_to_file(f"{target_folder}\\checkpoints\\lfm_data_{level.split('-')[0]}.bin")
                sm_data.write_to_file(f"{target_folder}\\checkpoints\\sm_data_{level.split('-')[0]}.bin")
            continue
        else:
            skip_level = None

        if tile_set is None:
            tile_set = TileSet(initial_frame, 10, None, tile_set_threshold)
            sm_data.set_tile_set(tile_set)

        prev_lfm_state = tile_set.map_frame_to_lfm_state(initial_frame, add_frame=True)
        lfm_data.initialize(prev_lfm_state)
        replay = SparseReplay(game, level, prev_lfm_state)
        actions = env.unwrapped.get_action_meanings()

        # agent.reset_patterns()
        agent.candidate_solution = [-1, ()]
        agent.set_state_shape(prev_lfm_state.shape)

        # play the level
        done = False
        unknown_patterns = -1

        total_score = 0
        ticks = 0
        pbar = tqdm.trange(max_ticks, desc=f"training run {training_run}: playing {game}-{level}")

        while not done and unknown_patterns != 0:
            # select and apply action
            action_id, unknown_patterns = agent.select_action(prev_lfm_state, range(len(actions)))
            if unknown_patterns == 0 and ticks == 0:
                # the level has nothing new to offer. skip until another level is given to the agent
                skip_level = level
                #tile_set.plot_tile_dict()
                break

            try:
                frame, diff_score, done, debug = env.step(action_id)
            except Exception:
                environment_fail = True
                break
            ticks += 1

            # update models
            score = diff_score
            if done:
                print(debug["winner"])
                print()

            if debug["winner"] == 'PLAYER_WINS' or debug["winner"] == 3:
                score += 1000
            if debug["winner"] == 'PLAYER_LOSES' or debug["winner"] == 2:
                score -= 1000

            lfm_state = tile_set.map_frame_to_lfm_state(frame, add_frame=True)
            lfm_patterns = lfm_data.add_observation(prev_lfm_state, action_id, lfm_state)
            sm_pattern = sm_data.add_observation(prev_lfm_state, lfm_state, score)
            agent.add_observation(prev_lfm_state, lfm_state, action_id, lfm_patterns, sm_pattern,
                                  range(len(actions)), done)
            env.render()

            # update records
            pbar.update(1)
            prev_lfm_state = lfm_state
            total_score += diff_score
            replay.add_frame(action_id, lfm_state, len(agent.known_lfm_patterns), debug["winner"])
            if ticks > max_ticks:
                break

        pbar.close()

        results[training_run] = [replay.length, total_score, debug["winner"]]

        if target_folder is not None and ticks > 0:
            tile_set.write_to_file(f"{target_folder}\\tile_set.bin")
            lfm_data.write_to_file(f"{target_folder}\\lfm_data.bin")
            sm_data.write_to_file(f"{target_folder}\\sm_data.bin")
            replay.write_to_file(f"{target_folder}\\replay_data\\{training_run}_{game}_{level}.sreplay")
            # replay.create_animation(tile_set, f'..\\replay\\replay_video\\{training_run}_{game}_{level}.mp4')
            with open(f"{result_folder}\\results.txt", "wb") as file:
                pickle.dump(results, file)

        if target_folder is not None and ((training_run+1) % reps_per_level) == 0:
            tile_set.write_to_file(f"{target_folder}\\checkpoints\\tile_set_{level.split('-')[0]}.bin")
            lfm_data.write_to_file(f"{target_folder}\\checkpoints\\lfm_data_{level.split('-')[0]}.bin")
            sm_data.write_to_file(f"{target_folder}\\checkpoints\\sm_data_{level.split('-')[0]}.bin")
        tile_set.plot_tile_dict()

    return results, lfm_data, sm_data, tile_set


def training_level_generator(game, levels, repetitions, symmetry_file_path=None):
    global environment_fail
    for level in levels:
        print(f'Starting {game}-{level}')
        env = gym.make(game + '-' + level, pixel_observations=True, include_semantic_data=True)
        for rep in range(repetitions):
            print(f'Reset {game}-{level}')
            frame = env.reset()
            yield env, game, level, frame
            if environment_fail and rep < repetitions-1:
                environment_fail = False
                env.close()
                env = gym.make(game + '-' + level, pixel_observations=True, include_semantic_data=True)
        env.close()
        if symmetry_file_path is not None:
            raise NotImplementedError
    return


def extended_training_level_generator(game, level_folder, target_file, repetitions, training_levels, plot_levels=False):
    global environment_fail
    import os
    levels = os.listdir(level_folder)
    for level_file in levels:
        if "backup" in level_file or sum([x in level_file for x in training_levels]) == 0:
            continue

        level = load_file(level_folder+level_file)
        write_file(target_file, level)
        new_level = True

        #env = gym_gvgai.make(game + '-lvl2-v0')
        env = gym.make(game + '-lvl2-v0', pixel_observations=True, include_semantic_data=True)
        for rep in range(repetitions):
            frame = env.reset()

            if plot_levels and new_level:
                plt.imshow(frame)
                plt.axis('off')
                plt.show()
                new_level = False

            yield env, game, level_file, frame

            if environment_fail and rep < repetitions-1:
                environment_fail = False
                env = gym.make(game + '-' + level, pixel_observations=True, include_semantic_data=True)
                #env = gym_gvgai.make(game + '-lvl2-v0')

        env.close()
    return


environment_fail = False

if __name__ == "__main__":

    #import gym_gvgai

    for game in DETERMINISTIC_TEST_GAMES:
        environment_fail = False

        #training_levels = ['lvl0-v0', 'lvl1-v0']
        repetitions_per_level = 25
        #level_generator = training_level_generator("gvgai-"+game, training_levels, repetitions_per_level)

        training_levels = ['lvl0', 'lvl1']
        level_generator = extended_training_level_generator("gvgai-"+game, f"..\\..\\data\\additional_training_levels\\gvgai-{game}\\",
                                                            f"D:\\Git Folders\\General Game AI\\GVGAI_GYM_Optimized\\games\\{game}_v0\\{game}_lvl2.txt",
                                                            #f"D:\\GVGAI_GYM\\gym_gvgai\\envs\\games\\{game}_v0\\{game}_lvl2.txt",
                                                            repetitions_per_level, training_levels, plot_levels=False)

        date = datetime.datetime.now()
        #result_folder = f"..\\..\\data\\training_results\\gvgai-{game}\\{date.year}_{date.month}_{date.day}-" \
        #                f"{date.hour}_{date.minute}_{date.second}-active_learning_competition"

        result_folder = f"..\\..\\data\\paper_training\\{game}\\symmetric_active_learning_optimized"
        if os.path.exists(result_folder):
            continue
        else:
            os.makedirs(f"{result_folder}\\replay_data")
            os.makedirs(f"{result_folder}\\checkpoints")

        span = SPANS[game]
        mask = CrossNeighborhoodPattern(span).get_mask()
        lfm_data = ActiveLearningLocalForwardModelData(mask, initial_pattern_count=1000, max_size=1000000)
        sm_data = ScoreModelData()

        _, _, _, tile_set = actively_train_agent_model(level_generator, lfm_data, sm_data, THRESHOLDS[game],
                                                       None, result_folder, reps_per_level=repetitions_per_level)
        tile_set.plot_tile_dict()
