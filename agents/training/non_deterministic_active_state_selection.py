from abc import ABC
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import random


def plot_level_tag(level_tag, state_shape):
    plt.imshow(np.array(level_tag, dtype=np.int).reshape(state_shape))
    plt.show()


def get_state_tag(state):
    if state is None:
        print()
        return None
    return tuple(state.flatten())


class NonDeterministicActiveStateSelectionAgent(ABC):
    def __init__(self, lfm_data_object, sm_data_object):
        super().__init__()
        self.lfm_data = lfm_data_object
        self.sm_data = sm_data_object
        self.data_set = dict()


class MaxUnknownPatternsStateActionSelection(NonDeterministicActiveStateSelectionAgent):
    def __init__(self, lfm_data_object, sm_data_object, random_ticks_until_end=50):
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
        self.random_ticks_until_end = random_ticks_until_end
        self.current_random_ticks_until_end = random_ticks_until_end
        pass

    def set_state_shape(self, state_shape):
        self.solution_still_valid = False
        self.current_random_ticks_until_end = self.random_ticks_until_end
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

            if game_state is None:
                raise ValueError()
            level_tag = get_state_tag(current_state)

            # never process a state twice
            if level_tag in self.known_states:
                return level_tag

            self.known_states[level_tag] = {"unknown_patterns": {x: set() for x in action_indices},
                                            "child_level_tags": {x: None for x in action_indices},
                                            "unknown_sm_pattern": {x: True for x in action_indices}}
            # add patterns -> state links
            patterns = self.lfm_data.get_patterns(current_state, 0)[:, :-1]  # remove action column to reuse the same patterns
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
                    # only add the pattern in case the pattern was unobserved and we have not already looked at this state
                    if self.known_states[level_tag]["child_level_tags"][action] is None:
                        if not self.known_lfm_patterns[pattern]["observed"]:
                            self.known_states[level_tag]["unknown_patterns"][action].add(pattern)
                    else:
                        raise ValueError("the level_tag's child states should be None since the state was just added")

            for action in action_indices:
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

                    #self.add_state(child_state, action_indices)
                    self.states_to_expand.insert(0, child_state)

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

    def remove_observed_lfm_patterns(self, patterns_to_be_removed, observed_state, action_indices):
        observed_state = observed_state.flatten()
        for i, pattern in enumerate(patterns_to_be_removed):
            outcome = observed_state[i]
            pattern = tuple(pattern)
            action = pattern[-1]

            if pattern in self.known_lfm_patterns:
                self.known_lfm_patterns[pattern]["observed"] = True
                self.known_lfm_patterns[pattern]["outcome"] = outcome

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

                        self.add_state(child_state, action_indices)

                self.known_lfm_patterns[pattern]["states"] = set()
            else:
                raise ValueError("The pattern should already be known, but has not been observed")
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
        state_tag = self.add_state(state, range(len(action_indices)))

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
                                not (current_state in self.terminal_transitions and action in self.terminal_transitions[current_state]):
                            states_to_visit.append((child_state, (*action_sequence, action)))

                else:
                    raise ValueError("either the search did not stop despite not knowing the child state "
                                     "or the starting state was not properly added")

        self.candidate_solution = [most_unknown_patterns, best_action_sequence[1:]]
        self.solution_still_valid = True

        if (most_unknown_patterns == 0):
            if self.current_random_ticks_until_end > 0:
                self.current_random_ticks_until_end -= 1
                # todo avoid terminal states
                best_action_sequence = [random.choice(action_indices)]
                return best_action_sequence[0], 1
        else:
            self.current_random_ticks_until_end = self.random_ticks_until_end

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

        self.remove_observed_lfm_patterns(observed_lfm_patterns.copy(), child_state, action_indices)
        self.update_sm_pattern_size(sm_data.get_pattern_lenth())
        self.remove_observed_sm_patterns(parent_tag, last_action, observed_sm_patterns)
        if done:
            if parent_tag in self.terminal_transitions:
                self.terminal_transitions[parent_tag].add(last_action)
            else:
                self.terminal_transitions[parent_tag] = {last_action}


def actively_train_agent_model(level_generator, lfm_data, sm_data, tile_set_threshold, tile_set=None,
                               target_folder=None, max_ticks=1500, reps_per_level=20):
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
            lfm_data.set_tile_set(tile_set)

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
            #action_id, unknown_patterns = agent.select_action(prev_lfm_state, range(len(actions)))
            action_id, unknown_patterns = random.choice(range(len(actions))), 1
            if unknown_patterns == 0 and ticks == 0:
                # the level has nothing new to offer. skip until another level is given to the agent
                skip_level = level
                break

            frame, diff_score, done, debug = env.step(action_id)
            ticks += 1

            # update models
            lfm_state = tile_set.map_frame_to_lfm_state(frame, add_frame=True)
            lfm_patterns = lfm_data.add_observation(prev_lfm_state, action_id, lfm_state)
            sm_pattern = sm_data.add_observation(prev_lfm_state, lfm_state, diff_score)
            #agent.add_observation(prev_lfm_state, lfm_state, action_id, lfm_patterns, sm_pattern,
            #                      range(len(actions)), done)
            env.render()

            # update records
            pbar.update(1)
            prev_lfm_state = lfm_state
            total_score += diff_score
            replay.add_frame(action_id, lfm_state, len(agent.known_lfm_patterns), debug["winner"])
            if ticks > max_ticks:
                break

        pbar.close()

        if target_folder is not None and ticks > 0:
            tile_set.write_to_file(f"{target_folder}tile_set_tmp.bin")
            lfm_data.write_to_file(f"{target_folder}lfm_data_tmp.bin")
            sm_data.write_to_file(f"{target_folder}sm_data_tmp.bin")
            replay.write_to_file(f"{target_folder}replay_data\\{training_run}_{game}_{level}.sreplay")

        if target_folder is not None and ticks > 0:
            tile_set.write_to_file(f"{target_folder}tile_set.bin")
            lfm_data.write_to_file(f"{target_folder}lfm_data.bin")
            sm_data.write_to_file(f"{target_folder}sm_data.bin")
            replay.write_to_file(f"{target_folder}replay_data\\{training_run}_{game}_{level}.sreplay")
            # replay.create_animation(tile_set, f'..\\replay\\replay_video\\{training_run}_{game}_{level}.mp4')

        results[training_run] = [replay.length, total_score, debug["winner"]]
    return results, lfm_data, sm_data, tile_set


if __name__ == "__main__":
    from agents.training.deterministic_active_state_selection import extended_training_level_generator
    from abstractclasses.AbstractNeighborhoodPattern import CrossNeighborhoodPattern
    from agents.models.probabilistic_local_forward_model import ProbabilisticLearningLocalForwardModelData
    from agents.models.score_model_data import ScoreModelData
    from agents.models.tile_map import TileSet
    from replay.sparsereplay import SparseReplay
    from Constants import THRESHOLDS, NON_DETERMINISTIC_TEST_GAMES, SPANS
    import datetime
    import os

    environment_fail = False

    #for game in NON_DETERMINISTIC_TESTGAMES:
    for game in ["golddigger"]:

        #training_levels = ['lvl0-v0', 'lvl1-v0']
        repetitions_per_level = 25
        # level_generator = training_level_generator(game, training_levels, repetitions_per_level)

        training_levels = ['lvl0', 'lvl1']
        level_generator = extended_training_level_generator("gvgai-"+game, f"..\\..\\data\\additional_training_levels\\gvgai-{game}\\",
                                                            f"D:\\Git Folders\\General Game AI\\GVGAI_GYM_Optimized\\games\\{game}_v0\\{game}_lvl2.txt",
                                                            #f"D:\\GVGAI_GYM\\gym_gvgai\\envs\\games\\{game}_v0\\{game}_lvl2.txt",
                                                            repetitions_per_level, training_levels, plot_levels=False)

        date = datetime.datetime.now()
        result_folder = f"..\\..\\data\\paper_training\\{game}\\symmetric_active_learning_optimized\\"
        if os.path.exists(result_folder):
            continue
        else:
            os.makedirs(f"{result_folder}\\replay_data")
            os.makedirs(f"{result_folder}\\checkpoints")

        span = SPANS[game]
        mask = CrossNeighborhoodPattern(span).get_mask()
        lfm_data = ProbabilisticLearningLocalForwardModelData(mask, initial_pattern_count=1000, max_size=1000000)
        sm_data = ScoreModelData()

        training_results, _, _, _ = actively_train_agent_model(level_generator, lfm_data, sm_data, THRESHOLDS[game],
                                                               None, result_folder, reps_per_level=repetitions_per_level)
