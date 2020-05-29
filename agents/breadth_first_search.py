import numpy as np
import logging
import math
import random

from abstractclasses.AbstractForwardModelAgent import AbstractForwardModelAgent


class SimpleBFS(AbstractForwardModelAgent):

    def __init__(self, expansions, discount_factor=1.0, forward_model=None, score_model=None, max_search_depth=300):
        super().__init__(forward_model, score_model)

        self._expansions = expansions
        self._discount_factor = discount_factor
        self._exploration_value = 0.01
        if self._discount_factor == 1:
            self._discount = [1 for x in range(max_search_depth)]
        else:
            self._discount = [math.pow(self._discount_factor, x) for x in range(max_search_depth)]

        self.known_states = dict()
        self.expansion_list = []
        self.selected_solution = None
        self.previous_state_tag = None
        self.previous_state = None
        self.max_search_depth = max_search_depth
        self.start_state = None
        self.recorded_action_sequence = []
        self.recorded_score_sequence = []

    def re_initialize(self, game_state, actions):
        self.selected_solution = None
        self.previous_state_tag = None
        self.previous_state = None

        if self.start_state is None or not np.array_equal(self.start_state, game_state):
            self.start_state = game_state.copy()
            self.known_states = dict()
            self.expansion_list = []
            self._forward_model.initialize(game_state)
            self.sure_solution = False
        else:
            if len(self.recorded_score_sequence) > 0 and self.recorded_score_sequence[-1] >= 1000:
                discounted_score_sequence = [score*self._discount[i] for i, score in
                                             enumerate(self.recorded_score_sequence)]
                self.selected_solution = [None, self.recorded_action_sequence, sum(discounted_score_sequence),
                                          discounted_score_sequence, True]
                self.sure_solution = True
        self.recorded_score_sequence = []
        self.recorded_action_sequence = []

    def get_identifier(self, level):
        return tuple(level.flatten())

    def get_next_action(self, state, actions):
        start_state = state.copy()
        state_tag = self.get_identifier(state)
        self.previous_state = state.copy()
        self.previous_state_tag = state_tag

        #self.check_correctness(state, actions)

        self.expansion_list.insert(0, (state, state_tag))
        if self.selected_solution is None or not self.selected_solution[4]:
            self.create_expansions(actions)

        # find best action_sequence using BFS starting from the current state
        checked_states = set()
        if self.selected_solution is None:
            candidate_solutions = []
            best_score = -math.inf
        else:
            candidate_solutions = [self.selected_solution]
            best_score = self.selected_solution[2]

        open_states = [(state_tag, (), 0, (), False)]

        search_depth = 0
        while len(open_states) > 0 and search_depth < self.max_search_depth and not self.sure_solution:
            state_tag, action_sequence, discounted_score_parent, score_sequence, winning = open_states.pop(0)
            checked_states.add(state_tag)
            if state_tag not in self.known_states:
                continue
            search_depth += 1

            for action in actions:
                child_level, child_tag, diff_score, _ = self.known_states[state_tag][action]
                if child_tag not in checked_states and diff_score > -1000:

                    child_action_sequence = (*action_sequence, action)
                    #discounted_score = discounted_score_parent + diff_score*math.pow(self._discount_factor,
                    #                                                                 len(action_sequence))

                    discounted_score = discounted_score_parent + diff_score*self._discount[len(action_sequence)]

                    state = (child_tag, child_action_sequence, discounted_score,
                             (*score_sequence, diff_score), diff_score >= 1000)
                    if diff_score < 1000:
                        open_states.append(state)
                    if discounted_score == best_score:
                        candidate_solutions.append(state)
                    elif discounted_score > best_score:
                        candidate_solutions = [state]
                        best_score = discounted_score

        if len(candidate_solutions) >= 1:
            # choose a random candidate solution and return first action of its action sequence
            self.selected_solution = random.choice(candidate_solutions)
            return self.selected_solution[1][0]
        else:
            # this should technically never happen
            action = random.choice(actions)
            expected_score = self.known_states[self.previous_state_tag][action][2]
            self.selected_solution = [None, [action], expected_score, [expected_score], False]
            return action

    def create_expansions(self, actions):
        i = 0
        while len(self.expansion_list) > 0:
            parent_level, parent_tag = self.expansion_list.pop(0)

            if parent_tag not in self.known_states:
                self.known_states[parent_tag] = {action: None for action in actions}

            for action in actions:
                if self.known_states[parent_tag][action] is None:
                    i += 1
                    child_level = self._forward_model.predict(parent_level, action).copy()
                    child_tag = self.get_identifier(child_level)
                    score, pattern = self._score_model.predict_and_get_pattern(parent_level, child_level)
                    self.known_states[parent_tag][action] = [child_level, child_tag, score, pattern]
                else:
                    child_level, child_tag, score, _ = self.known_states[parent_tag][action]
                if child_tag not in self.known_states:
                    self.expansion_list.insert(0, (child_level, child_tag))
            if i > self._expansions:
                break

    def check_correctness(self, observed_lfm_state, observed_score, actions):
        if self.selected_solution is None:
            return False
        if self.sure_solution:
            self.selected_solution = (None, self.selected_solution[1][1:],
                                      self.selected_solution[2] - self.selected_solution[3][0],
                                      self.selected_solution[3][1:], self.selected_solution[4])
            return False
        else:
            predicted_level, _, _, _ = self.known_states[self.previous_state_tag][self.selected_solution[1][0]]
            pred_score, score_pattern = self._score_model.predict_and_get_pattern(self.previous_state, observed_lfm_state)
            if observed_score == self.selected_solution[3][0]:
                pattern = self._score_model.add_pattern(self.previous_state, observed_lfm_state, observed_score)
            else:
                # register new instance and correct all mistakes
                pattern = self._score_model.correct_mistake(self.previous_state, observed_lfm_state, observed_score)
                for state_tag in self.known_states:
                    for action in self.known_states[state_tag]:
                        self.known_states[state_tag][action][2] = \
                            self._score_model.predict_pattern(self.known_states[state_tag][action][3])

            if np.array_equal(predicted_level, observed_lfm_state):
                    # nothing to correct, just shift the solution
                    self.selected_solution = (None, self.selected_solution[1][1:],
                                              self.selected_solution[2] - self.selected_solution[3][0],
                                              self.selected_solution[3][1:], self.selected_solution[4])
                    if len(self.selected_solution[1]) == 0:
                        self.selected_solution = None

            else:
                observed_tag = self.get_identifier(observed_lfm_state)
                self.known_states[self.previous_state_tag][self.selected_solution[1][0]] = \
                    [observed_lfm_state, observed_tag, observed_score, score_pattern]
                self.selected_solution = None
                self.expansion_list = []
                self._add_search_tree_leaves_to_expansion_list(observed_lfm_state, observed_tag, actions)

    def add_observation(self, observed_state, observed_score, actions):
        last_action = self.selected_solution[1][0]
        self._score_model.add_pattern(self.previous_state, observed_state, observed_score)
        self.recorded_action_sequence.append(last_action)
        self.recorded_score_sequence.append(observed_score)
        self.check_correctness(observed_state, observed_score, actions)

    def _add_search_tree_leaves_to_expansion_list(self, root_state, root_tag, actions):
        checked_states = set()
        open_states = [root_tag]
        added_states = set()
        while len(open_states) > 0:
            current_state = open_states.pop(0)
            if current_state in checked_states:
                continue
            checked_states.add(current_state)

            if current_state in self.known_states:
                for action in actions:
                    child_state = self.known_states[current_state][action][1]
                    if child_state is not None:
                        open_states.append(child_state)
                    else:
                        if current_state not in added_states:
                            self.expansion_list.append((np.array(current_state).reshape(root_state.shape), current_state))
                            added_states.add(current_state)
            else:
                if current_state not in added_states:
                    self.expansion_list.append((np.array(current_state).reshape(root_state.shape), current_state))
                    added_states.add(current_state)

    def get_agent_name(self) -> str:
        return "BFS Agent"
