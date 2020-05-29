import numpy as np
import math
import random


class SimpleRHEA:

    def __init__(self, forward_model, score_model, rollout_actions_length, mutation_probability, num_evals,
                 use_shift_buffer=True, flip_at_least_one=True, discount=None):

        self.forward_model = forward_model
        self.score_model = score_model

        self._rollout_actions_length = rollout_actions_length
        self._use_shift_buffer = use_shift_buffer
        self._flip_at_least_one = flip_at_least_one
        self._mutation_probability = mutation_probability
        self._num_evals = num_evals
        self._actions = None
        self._solution = None

        self._discount = discount
        if self._discount is not None:
            self.discount_matrix = np.tile(np.array([math.pow(self._discount, x) for x in
                                                     range(self._rollout_actions_length)]), (self._num_evals, 1))
        else:
            self.discount_matrix = None

    def re_initialize(self, lfm_state, actions):
        self._actions = actions

        # Initialize the solution to a random sequence
        if self._use_shift_buffer:
            self._solution = self._random_solution()

    def get_next_action(self, obs, actions):
        """"
        Get the next best action by evaluating a bunch of mutated solutions
        """
        # use rhea to determine action
        if self._use_shift_buffer:
            solution = self._shift_and_append(self._solution)
        else:
            solution = self._random_solution()

        candidate_solutions = self._new_mutate(solution, self._mutation_probability)

        mutated_scores, pred_state = self.evaluate_rollouts(obs, candidate_solutions, self.discount_matrix)
        best_candidates = np.where(mutated_scores == np.max(mutated_scores))

        # best_score_in_evaluations = mutated_scores[best_idx]

        # The next best action is the first action from the solution space
        self._solution = candidate_solutions[random.randint(0, len(best_candidates)-1)]
        action = self._solution[0]
        # print("apply action:", action)

        return action

    def evaluate_rollouts(self, state, candidate_solutions, discount_matrix):
        pred_states = []
        pred_rewards = []
        pred_unknown_patterns = []

        for candidate in candidate_solutions:
            candidate_predicted_states = []
            candidate_predicted_rewards = []
            candidate_predicted_unknown_patterns = []

            current_state = state
            for action in candidate:
                pred_state, unknown_patterns = self.forward_model.predict_and_get_unknown_patterns(current_state, action)
                score = self.score_model.predict(current_state, pred_state)
                candidate_predicted_states.append(pred_state)
                candidate_predicted_rewards.append(score)
                candidate_predicted_unknown_patterns.append(unknown_patterns)
                current_state = pred_state
            pred_states.append(candidate_predicted_states)
            pred_rewards.append(candidate_predicted_rewards)
            pred_unknown_patterns.append(candidate_predicted_unknown_patterns)

        if discount_matrix is not None:
            pred_rewards = np.multiply(np.array(pred_rewards), discount_matrix)

        return np.sum(pred_rewards, axis=1), pred_states

    def _shift_and_append(self, solution):
        """
        Remove the first element and add a random action on the end
        """
        new_solution = np.copy(solution[1:])
        new_solution = np.hstack([new_solution, random.choice(self._actions)])
        return new_solution

    def add_observation(self, a, b, c):
        return

    def _random_solution(self):
        """
        Create a random set fo actions
        """
        return np.array([random.choice(self._actions) for _ in range(self._rollout_actions_length)])

    def _new_mutate(self, solution, mutation_probability):
        """
        Mutate the solution
        """

        candidate_solutions = [solution]
        # Solution here is 2D of rollout_actions x batch_size
        for b in range(self._num_evals-1):
            # Create a set of indexes in the solution that we are going to mutate
            mutation_indexes = set()
            solution_length = len(solution)
            if self._flip_at_least_one:
                index = np.random.randint(solution_length)
                mutation_indexes = mutation_indexes.union(set(range(index, index+1)))

            mutation_indexes = mutation_indexes.union(
                set(np.where(np.repeat(np.random.random([solution_length]), 1) < mutation_probability)[0]))

            # Create the number of mutations that is the same as the number of mutation indexes
            num_mutations = len(mutation_indexes)
            mutations = [random.choice(self._actions) for _ in range(num_mutations)]
            if type(mutations[0]) is np.ndarray:
                mutations = np.concatenate(mutations)

            # Replace values in the solutions with mutated values
            new_solution = np.copy(solution)
            new_solution[list(mutation_indexes)] = mutations
            candidate_solutions.append(new_solution)

        return np.stack(candidate_solutions)
